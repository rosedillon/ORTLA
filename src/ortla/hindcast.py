"""Hindcast a launched service: strip it, predict it, compare.

The validation loop for a real, already-running service:

1. Strip. Every schedule matching
   :class:`~ortla.config.StripRule` is removed from the
   timetable calls, so the feasibility checks and the existing-frequency
   count see the network as it was without the service. With
   ``remove_calls_at`` the strip is calls-only: the matched schedules
   stay in the before-network without their stops at those stations,
   which is the right framing for a new station inserted into existing
   services, or an existing train extended over new ground.
2. Rebuild the service as candidates, one per train active on the
   reference date, using its real calling pattern and public times.
3. Predict. Run ``propose``'s checks and demand model on each train
   against the stripped timetable: BPLAN running times, headway, demand
   by leg and on-board load by segment. Demand is predicted with each
   model in ``config.demand_models`` (``calibrated`` gravity and/or the
   ``legacy`` pivot + footfall-share model), each two ways:

   * ``marginal``: each train as the only addition (``propose``'s
     framing: ``new_frequency = existing + 1``);
   * ``joint``: every stripped train added at once, so a slot gaining
     ``k`` trains uses ``new_frequency = existing + k`` and shares the
     demand.

4. Compare with outturn: the operator's loading data for the same trains
   (matched by UID, else headcode + origin + destination + origin
   departure).

Outputs go to ``config.output_path``; see :func:`run_hindcast`.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

from ortla import loading_data
from ortla.bplan_geography import BplanNetworkLinks, BplanTimingGraph
from ortla.clock import WEEKDAY_INDEX, cif_hhmm, classify_time_period
from ortla.demand import (
    build_fares_context,
    build_gravity_context,
    estimate_loading,
    matrix_stations,
    predict_demand,
    prefetch_matrix_columns,
    shortest_legs,
    stop_time,
    transfer_demand,
)
from ortla.feasibility import check_headway, check_running_times
from ortla.gravity_model import MODEL_FILENAME
from ortla.od_matrix import read_odm
from ortla.service_lookup import (
    crs_to_tiploc_map,
    day_services,
    existing_trains,
    prepare_calls,
)
from ortla.stations import (
    DEFAULT_DESTINATION_GROUPS,
    canonical_crs,
    load_table_1410,
    recode_columns,
    resolve_aliases,
    resolve_recodes,
    tiploc_to_crs_map,
)
from ortla.timetable import read_public_calls
from ortla.track_distance import TrackDistanceCalculator

if TYPE_CHECKING:
    from ortla.config import CandidateService, HindcastConfig, StripRule
    from ortla.demand import GravityContext

logger = logging.getLogger(__name__)

_SCHEDULE_KEY = ["train_uid", "start_date", "end_date", "days_runs"]


# 1. Strip
def _schedules(calls: pd.DataFrame) -> pd.DataFrame:
    """One row per schedule with its ordered public calling pattern."""
    ordered = calls.assign(seq_n=pd.to_numeric(calls["seq"], errors="coerce")).sort_values(
        [*_SCHEDULE_KEY, "seq_n"]
    )
    grouped = ordered.groupby(_SCHEDULE_KEY, sort=False)
    out = grouped.agg(
        atoc_code=("atoc_code", "first"),
        signalling_id=("signalling_id", "first"),
        stations=("crs", list),
        arrivals=("public_arrival", list),
        departures=("public_departure", list),
    ).reset_index()
    out["origin"] = out["stations"].str[0]
    out["destination"] = out["stations"].str[-1]
    return out


def _matches(schedule: pd.Series, rule: StripRule) -> bool:
    if rule.train_uids:
        return schedule["train_uid"] in rule.train_uids
    if rule.atoc_code and schedule["atoc_code"] != rule.atoc_code:
        return False
    if rule.start_date_from and str(schedule["start_date"]) < rule.start_date_from:
        return False
    stations = schedule["stations"]
    if not set(rule.calls_at) <= set(stations):
        return False
    ends = (schedule["origin"], schedule["destination"])
    if rule.origin and rule.destination:
        wanted = {(rule.origin, rule.destination)}
        if rule.both_directions:
            wanted.add((rule.destination, rule.origin))
        if ends not in wanted:
            return False
    if rule.in_order and len(rule.calls_at) > 1:
        positions = [stations.index(s) for s in rule.calls_at]
        ascending = positions == sorted(positions)
        descending = positions == sorted(positions, reverse=True)
        if not (ascending or (rule.both_directions and descending)):
            return False
    return True


def recode_strip_rule(rule: StripRule, recodes: dict[str, str]) -> StripRule:
    """``rule`` with its station codes recoded as the timetable's are.

    A rule naming a second code (SPX for St Pancras, RDZ for Reading)
    then still finds its trains.
    """

    def code(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().upper()
        return recodes.get(value, value)

    return rule.model_copy(
        update={
            "calls_at": [code(c) for c in rule.calls_at],
            "origin": code(rule.origin),
            "destination": code(rule.destination),
            "remove_calls_at": [code(c) for c in rule.remove_calls_at],
        }
    )


def select_schedules(calls: pd.DataFrame, rule: StripRule) -> pd.DataFrame:
    """Every schedule (all date variants) matching ``rule``."""
    schedules = _schedules(calls)
    return schedules[schedules.apply(_matches, axis=1, rule=rule)].reset_index(drop=True)


def _runs_on(schedule: pd.Series, date: pd.Timestamp, day: str) -> bool:
    start = pd.Timestamp(schedule["start_date"])
    end = pd.Timestamp(schedule["end_date"])
    bitmap = str(schedule["days_runs"])
    idx = WEEKDAY_INDEX[day]
    return start <= date <= end and len(bitmap) > idx and bitmap[idx] == "1"


# 2. Rebuild as candidates
def _date_for_day(reference_date: pd.Timestamp, day: str) -> pd.Timestamp:
    """Return the date of ``day`` in ``reference_date``'s week."""
    return reference_date + pd.Timedelta(days=WEEKDAY_INDEX[day] - reference_date.weekday())


def build_candidates(
    schedules: pd.DataFrame,
    reference_date: pd.Timestamp,
    days: list[str],
    traction_code: str,
    capacity: float | None,
) -> list[CandidateService]:
    """One candidate per train UID, on the ``days`` it actually runs.

    For each day the schedule variant active in the reference week is
    used; a train whose times differ between its day variants keeps the
    variant of its first running day (the times are the same in
    practice).
    """
    from ortla.config import (  # noqa: PLC0415 (import cycle)
        CandidateService,
        ServiceStop,
    )

    candidates = []
    for uid, group in schedules.groupby("train_uid", sort=True):
        running: list[str] = []
        chosen = None
        for day in days:
            date = _date_for_day(reference_date, day)
            active = [row for _, row in group.iterrows() if _runs_on(row, date, day)]
            if active:
                running.append(day)
                chosen = chosen if chosen is not None else active[0]
        if chosen is None:
            continue
        n = len(chosen["stations"])
        stops = [
            ServiceStop(
                station=chosen["stations"][i],
                arrival=None if i == 0 else cif_hhmm(chosen["arrivals"][i]) or None,
                departure=None if i == n - 1 else cif_hhmm(chosen["departures"][i]) or None,
            )
            for i in range(n)
        ]
        candidates.append(
            CandidateService(
                name=str(uid),
                operator=chosen["atoc_code"],
                traction_code=traction_code,
                days_of_week=running,
                stops=stops,
                capacity=capacity,
            )
        )
    return candidates


def _joint_counter(
    candidates: list[CandidateService], cutoffs: dict[str, str]
) -> Counter[tuple[str, str, str, str]]:
    """How many candidates serve each leg in each day and period.

    Whole-day totals are stored under the period key ``"day"``. A train
    calling at a station twice serves a pair once, by the leg
    :func:`predict_demand` prices
    (:func:`~ortla.demand.shortest_legs`), timed by that leg's
    departure.
    """
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for cand in candidates:
        stops = cand.stops
        legs = shortest_legs(stops)
        for day in cand.days_of_week:
            for (origin, destination), (i, _j) in legs.items():
                time_i = stop_time(stops[i])
                if time_i is None:
                    continue
                period = classify_time_period(f"{time_i}:00", cutoffs)
                counts[(day, period, origin, destination)] += 1
                counts[(day, "day", origin, destination)] += 1
    return counts


def apply_forecast_footfall(
    config: HindcastConfig, candidates: list[CandidateService]
) -> dict[str, int]:
    """Give each new station its ex-ante footfall forecast; return CRS -> footfall applied.

    The forecast goes onto the candidate's stops as ``estimated_annual_footfall``, which the
    gravity model prefers over table 1410. A station table 1410 does hold keeps its observed
    figure. See ``HindcastConfig.forecast_footfall_path`` for why the file must be a forecast, not
    an outturn.
    """
    if config.forecast_footfall_path is None:
        return {}
    table = pd.read_csv(config.forecast_footfall_path)
    if config.forecast_footfall_year is not None and "footfall_year" in table.columns:
        table = table[table["footfall_year"].astype(str) == config.forecast_footfall_year]
    aliases = resolve_aliases(config.station_aliases)
    forecast = {
        canonical_crs(code, aliases): int(round(value))  # noqa: RUF046 (numpy float)
        for code, value in zip(table["crs"], table["forecast_entries_exits"], strict=True)
        if pd.notna(value) and value > 0
    }
    known = set(load_table_1410(config.table_1410)) if config.table_1410 is not None else set()
    applied: dict[str, int] = {}
    for cand in candidates:
        for stop in cand.stops:
            code = canonical_crs(stop.station, aliases)
            if code in forecast and code not in known:
                stop.estimated_annual_footfall = forecast[code]
                applied[code] = forecast[code]
    if applied:
        logger.info(
            "Forecast footfall for %d called stations absent from table 1410, from %s: %s",
            len(applied),
            config.forecast_footfall_path.name,
            ", ".join(f"{code} {value:,}" for code, value in sorted(applied.items())),
        )
    return applied


def _require_demand_inputs(config: HindcastConfig) -> None:
    """Fail before predicting anything if the demand inputs are missing.

    Without this, a hindcast whose period matrices were never written (a
    baseline built over a date window its loading feed does not cover,
    say) predicts zero for every train and still reports success.
    """
    matrices = config.period_matrices_dir
    if matrices is None:
        raise ValueError("hindcast needs period_matrices_dir.")
    if not matrices.is_dir() or not any(matrices.glob("odm_*.csv")):
        raise ValueError(
            f"No period matrices (odm_*.csv) in {matrices}. Its loading baseline "
            "has not been run, or wrote nothing; check that its date window "
            "overlaps the loading feed."
        )
    model = matrices.parent / MODEL_FILENAME
    if "calibrated" in config.demand_models and not model.exists():
        raise ValueError(
            f"The calibrated demand model needs {model}, which the loading "
            "baseline writes when given route_frequency_path."
        )


def _balance_slots(
    config: HindcastConfig,
    candidates: list[CandidateService],
    remaining: pd.DataFrame,
    reference_date: pd.Timestamp,
    stations: set[str],
    weekdays: list[str],
) -> tuple[Counter, Counter, Counter]:
    """Trains per weekday slot on every pair touching ``stations``.

    Counts candidate trains and the remaining timetable's trains alike,
    so a station another operator also serves (Merseyrail at Headbolt
    Lane) has all its pairs in view, not only the stripped operator's.
    Returns three counters, keyed by canonical codes: all trains per
    ``(day, period, o, d)``, candidate trains per the same key, and all
    trains per ``(day, o, d)``.
    """
    aliases = resolve_aliases(config.station_aliases)
    cutoffs = config.time_periods_cutoffs
    slot_all: Counter = Counter()
    slot_cand: Counter = Counter()
    daily: Counter = Counter()

    def add(day: str, times: list[str | None], codes: list[str], candidate: bool) -> None:
        for i, time_i in enumerate(times):
            if not time_i:
                continue
            period = classify_time_period(f"{time_i}:00", cutoffs)
            for j in range(i + 1, len(codes)):
                o, d = codes[i], codes[j]
                if o == d or (o not in stations and d not in stations):
                    continue
                slot_all[(day, period, o, d)] += 1
                daily[(day, o, d)] += 1
                if candidate:
                    slot_cand[(day, period, o, d)] += 1

    for cand in candidates:
        codes = [canonical_crs(s.station, aliases) for s in cand.stops]
        times = [stop_time(s) for s in cand.stops]
        for day in cand.days_of_week:
            if day in weekdays:
                add(day, times, codes, candidate=True)

    prepared = prepare_calls(remaining)
    for day in weekdays:
        filtered = day_services(prepared, reference_date, day)
        canon = filtered["crs"].map(lambda c: canonical_crs(c, aliases))
        uids = set(filtered.loc[canon.isin(stations), "train_uid"])
        if not uids:
            continue
        rows = filtered[filtered["train_uid"].isin(uids)].assign(canon=canon)
        rows = rows.sort_values(["train_uid", "start_date", "seq"])
        for _, service in rows.groupby(["train_uid", "start_date"], sort=False):
            times = [
                cif_hhmm(dep) or cif_hhmm(arr) or None
                for dep, arr in zip(
                    service["public_departure"], service["public_arrival"], strict=True
                )
            ]
            add(day, times, list(service["canon"]), candidate=False)
    return slot_all, slot_cand, daily


def _station_totals(
    factors: dict[str, float], pairs: dict[tuple[str, str], float], stations: set[str]
) -> dict[str, float]:
    """Each station's scaled throughput over its pairs, both ways."""
    out = dict.fromkeys(stations, 0.0)
    for (o, d), trips in pairs.items():
        scaled = trips * factors.get(o, 1.0) * factors.get(d, 1.0)
        for end in (o, d):
            if end in out:
                out[end] += scaled
    return out


def fit_station_factors(
    pairs: dict[tuple[str, str], float],
    targets: dict[str, float],
    *,
    iterations: int = 200,
    tolerance: float = 1e-9,
) -> dict[str, float]:
    """Factors bringing each target station's throughput to its target.

    A pair's demand is scaled by the factor of each end that has one, so
    a pair joining two target stations (Ashington -> Newsham) takes
    both. Iterative proportional fitting: each pass rescales every
    station to its target given the others' current factors, until no
    factor moves by more than ``tolerance`` in logs. A station with no
    demand keeps a factor of 1.
    """
    stations = set(targets)
    factors = dict.fromkeys(stations, 1.0)
    for _ in range(iterations):
        current = _station_totals(factors, pairs, stations)
        worst = 0.0
        for station in stations:
            if current[station] > 0:
                step = targets[station] / current[station]
                factors[station] *= step
                worst = max(worst, abs(float(np.log(step))))
        if worst < tolerance:
            break
    return factors


def balance_forecast_stations(
    config: HindcastConfig,
    gravity: GravityContext,
    candidates: list[CandidateService],
    remaining: pd.DataFrame,
    reference_date: pd.Timestamp,
    forecast: dict[str, int],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Scale factors holding each new station to its forecast.

    The gravity model's synthetic demand is summed over every weekday slot of every pair the
    station has a direct train to, on any operator, and each station gets the factor that brings it
    to ``forecast x forecast_footfall_weekday_share``. A pair joining two new stations counts at
    both ends, so the factors are fitted jointly by iterative proportional fitting.

    The candidates' share follows from the gravity model's split over destinations and their share
    of each slot's trains. That is what puts most of Headbolt Lane on Merseyrail rather than on the
    Northern trains being tested.

    Returns CRS -> factor and one row per station for ``hindcast_station_balance.csv``.
    """
    from ortla.config import ServiceStop  # noqa: PLC0415 (import cycle)

    share = config.forecast_footfall_weekday_share
    weekdays = [d for d in config.days if WEEKDAY_INDEX[d] < 5]
    targets = {s: f * share * len(weekdays) / 5 for s, f in forecast.items()}
    stations = set(targets)
    slot_all, slot_cand, daily = _balance_slots(
        config, candidates, remaining, reference_date, stations, weekdays
    )

    stops: dict[str, ServiceStop] = {}

    def stop(code: str) -> ServiceStop:
        if code not in stops:
            stops[code] = ServiceStop(station=code, estimated_annual_footfall=forecast.get(code))
        return stops[code]

    pair_total: dict[tuple[str, str], float] = {}
    pair_cand: dict[tuple[str, str], float] = {}
    unpriced = 0
    for (day, period, o, d), n_all in slot_all.items():
        trips, _band = gravity.synthetic_slot_trips(
            stop(o), stop(d), day, period, daily[(day, o, d)]
        )
        if trips is None:
            unpriced += 1
            continue
        pair_total[(o, d)] = pair_total.get((o, d), 0.0) + trips
        pair_cand[(o, d)] = (
            pair_cand.get((o, d), 0.0) + trips * slot_cand[(day, period, o, d)] / n_all
        )
    if unpriced:
        logger.warning(
            "Station balance: %d weekday slots touching %s have no gravity estimate "
            "(no route or footfall) and are left out of the totals.",
            unpriced,
            ", ".join(sorted(stations)),
        )

    ones = dict.fromkeys(stations, 1.0)
    before = _station_totals(ones, pair_total, stations)
    before_cand = _station_totals(ones, pair_cand, stations)
    factors = fit_station_factors(pair_total, targets)
    after = _station_totals(factors, pair_total, stations)
    after_cand = _station_totals(factors, pair_cand, stations)

    table = pd.DataFrame(
        [
            {
                "crs": s,
                "forecast_entries_exits": forecast[s],
                "weekday_share": share,
                "target_weekday_trips": round(targets[s]),
                "modelled_weekday_trips_before": round(before[s]),
                "modelled_on_candidates_before": round(before_cand[s]),
                "candidate_share": before_cand[s] / before[s] if before[s] else np.nan,
                "factor": factors[s] if before[s] else np.nan,
                "modelled_weekday_trips_after": round(after[s]),
                "modelled_on_candidates_after": round(after_cand[s]),
                "pairs": sum(1 for pair in pair_total if s in pair),
            }
            for s in sorted(stations)
        ]
    )
    missing = [s for s in stations if before[s] <= 0]
    if missing:
        logger.warning(
            "Station balance: no priced demand at %s, so nothing to scale.", ", ".join(missing)
        )
    logger.info("Station balance to forecast footfall:\n%s", table.to_string(index=False))
    return {s: f for s, f in factors.items() if before[s] > 0}, table


def apply_station_balance(
    demand: pd.DataFrame, factors: dict[str, float], aliases: dict[str, str]
) -> pd.DataFrame:
    """Scale the ``gravity_synthetic`` legs touching a balanced station.

    A leg between two balanced stations takes both factors. Captured,
    generated and revenue all scale together.
    """
    if demand.empty or not factors:
        return demand
    synthetic = demand["demand_model_used"].astype(str).eq("gravity_synthetic")
    origin = demand["from_station"].map(lambda c: factors.get(canonical_crs(c, aliases), 1.0))
    destination = demand["to_station"].map(lambda c: factors.get(canonical_crs(c, aliases), 1.0))
    factor = (origin * destination).where(synthetic, 1.0)
    out = demand.copy()
    for col in (
        "predicted_annual_trips_captured",
        "predicted_annual_trips_generated",
        "predicted_annual_revenue_gbp",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * factor
    return out


def screen_count_resets(
    segments: pd.DataFrame, min_load: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blank the outturn on train-days whose counts restart mid-journey.

    The signature is at least ``min_load`` passengers arriving at an
    intermediate call, under a fifth of them leaving it, and nobody
    counted boarding. GTR's counts do this at Cambridge South on most of
    its trains (25 board at Cambridge, all 25 "alight" at Cambridge
    South, and the train runs on empty to Foxton) and now and then at
    Gatwick, City Thameslink and Luton Airport Parkway. Every segment
    after the restart is then compared against a load that is not there,
    so the whole train-day is set aside.

    ``segments`` must be in calling order within each train-day, as
    :func:`run_hindcast` builds it. Returns the screened segments and
    one row per flagged train-day.
    """
    keys = ["train_uid", "day"]
    load_in = segments.groupby(keys, sort=False)["actual_load"].shift(1)
    reset = (
        (load_in >= min_load)
        & (segments["actual_load"] < 0.2 * load_in)
        & (segments["actual_on"].fillna(0) <= 0.5)  # noqa: PLR2004
    )
    flagged = (
        segments.loc[reset, [*keys, "from_station"]]
        .assign(load_arriving=load_in[reset], load_leaving=segments.loc[reset, "actual_load"])
        .drop_duplicates(subset=keys)
        .rename(columns={"from_station": "reset_at"})
    )
    if flagged.empty:
        return segments, flagged
    hit = segments.set_index(keys).index.isin(flagged.set_index(keys).index)
    out = segments.copy()
    for col in ("actual_load", "actual_on", "actual_off", "actual_capacity"):
        if col in out.columns:
            out.loc[hit, col] = np.nan
    logger.warning(
        "Outturn screen: %d of %d counted train-days restart their counts mid-journey "
        "and are left out of the comparison (at %s).",
        len(flagged),
        segments.dropna(subset=["actual_load"]).groupby(keys).ngroups + len(flagged),
        ", ".join(f"{k} {v}" for k, v in flagged["reset_at"].value_counts().items()),
    )
    return out, flagged


def _warn_missing_footfall(
    config: HindcastConfig, candidates: list[CandidateService]
) -> list[str]:
    """Name the called stations the gravity model has no footfall for.

    A station that opened after table 1410's year is in neither the
    table nor anywhere else. The calibrated model then returns nothing
    for every leg touching it and the legacy model substitutes the other
    end's footfall, so the run completes with meaningless results unless
    this warns.

    Station codes are canonicalised first (see
    :mod:`ortla.stations`), so a high-speed platform code the
    table does not carry is matched to its domestic entry rather than
    reported missing.
    """
    if config.table_1410 is None:
        return []
    aliases = resolve_aliases(config.station_aliases)
    known = set(load_table_1410(config.table_1410))
    forecast = {
        stop.station.strip().upper()
        for cand in candidates
        for stop in cand.stops
        if stop.estimated_annual_footfall
    }
    called = {stop.station.strip().upper() for cand in candidates for stop in cand.stops}
    missing = sorted(
        code
        for code in called
        if canonical_crs(code, aliases) not in known and code not in forecast
    )
    if missing:
        logger.warning(
            "No footfall for %d of %d called stations: %s. The calibrated model "
            "cannot predict any leg touching these, and the legacy model will "
            "substitute the other end's footfall. These figures have to come from "
            "table 1410 or an ex-ante forecast (`forecast_footfall_path`); a "
            "hand-entered value measured after the service opened would make the "
            "hindcast circular.",
            len(missing),
            len(called),
            ", ".join(missing),
        )
    return missing


def _warn_missing_matrix_stations(
    config: HindcastConfig, candidates: list[CandidateService]
) -> list[str]:
    """Name called stations the period matrices hold no demand under.

    The counterpart to :func:`_warn_unroutable_stations` for the other
    input the demand model joins on. A station absent from the matrices
    returns no cell for every leg touching it, the leg falls through to
    the synthetic gravity estimate, and the real flow, in the matrices
    all along under the station's other CRS code, is lost without a
    word. That is how a 13x under-prediction once reached the results
    with nothing in the log.

    Codes are canonicalised before the check, so this fires on a real
    gap rather than on an alias :mod:`ortla.stations` already
    covers. One representative matrix is read: the day/period files
    share their station coverage.
    """
    if config.period_matrices_dir is None:
        return []

    aliases = resolve_aliases(config.station_aliases)
    called = sorted({stop.station.strip().upper() for cand in candidates for stop in cand.stops})
    matrices = sorted(config.period_matrices_dir.glob("odm_*.csv"))
    if not matrices:
        logger.warning(
            "No odm_*.csv matrices in %s; every leg will fall back to the synthetic estimate.",
            config.period_matrices_dir,
        )
        return called

    covered = matrix_stations(matrices[0])
    missing = [code for code in called if canonical_crs(code, aliases) not in covered]
    if missing:
        logger.warning(
            "%d of %d called stations are absent from the period matrices (%s): %s. "
            "Every leg touching these has no observed demand to pivot on and falls "
            "through to the synthetic gravity estimate, which will read as a "
            "plausible number rather than as a missing join. If the station is "
            "really in the matrices under another CRS code, add it to "
            "`station_aliases`.",
            len(missing),
            len(called),
            matrices[0].name,
            ", ".join(missing),
        )
    return missing


def _warn_unroutable_stations(
    config: HindcastConfig, candidates: list[CandidateService]
) -> TrackDistanceCalculator | None:
    """Name the stations the distance calculator cannot place.

    Returns the calculator it built (``None`` if it could not), so the
    gravity model can reuse it instead of reading the track shapefile
    again.

    The gravity model asks for a track distance before it asks for
    footfall, so a station missing from the station list, or present but
    snapped onto a disconnected piece of the track shapefile, kills
    every leg touching it whatever footfall it has, observed or
    forecast. Without this warning the failure is silent: the legs come
    back ``insufficient_data`` and the affected segments are reported as
    a lower bound.

    Connectivity is judged from the track graph's connected components
    rather than by routing every pair, which would be quadratic in the
    number of called stations (one launched service calls at 100 of them).
    """
    if config.distance is None:
        return None  # a legacy-only config has no track graph to check against
    called = sorted({stop.station.strip().upper() for cand in candidates for stop in cand.stops})
    try:
        calc = TrackDistanceCalculator(config.distance)
    except (OSError, ValueError, KeyError):
        logger.warning("Could not build the distance calculator to pre-check called stations.")
        return None

    unlocated = [code for code in called if code not in calc.station_locations]
    if unlocated:
        logger.warning(
            "No location for %d of %d called stations: %s. Every leg touching these "
            "is dropped before footfall is considered. Add them to the station list "
            "(`distance.stations.path`).",
            len(unlocated),
            len(called),
            ", ".join(unlocated),
        )

    nodes = getattr(calc, "_station_nodes", {})
    component_of: dict[object, int] = {}
    for index, component in enumerate(nx.connected_components(calc.graph)):
        for node in component:
            component_of[node] = index
    grouped: dict[int, list[str]] = {}
    for code in called:
        node = nodes.get(code)
        if node is not None and node in component_of:
            grouped.setdefault(component_of[node], []).append(code)

    stranded: list[str] = []
    if len(grouped) > 1:
        main = max(grouped.values(), key=len)
        stranded = sorted(
            code for group in grouped.values() if group is not main for code in group
        )
        logger.warning(
            "%s %s on a disconnected part of the track shapefile: no route to the "
            "other called stations, so every leg touching %s is dropped. The "
            "shapefile probably predates the service.",
            ", ".join(stranded),
            "is" if len(stranded) == 1 else "are",
            "it" if len(stranded) == 1 else "them",
        )
    return calc


# 4. Outturn
def _outturn_tiploc_map(config: HindcastConfig) -> dict[str, str]:
    """TIPLOC -> CRS for the outturn reader, as for a loading run.

    SWR and EMR publish TIPLOCs rather than CRS codes and their readers
    drop every row they cannot map, so without this an SWR hindcast
    compares against an empty outturn and an EMR one against the wrong
    station codes, and both are reported only as "0 loading rows
    matched", never as an error.
    """
    stations_path = config.distance.stations.path if config.distance else None
    return tiploc_to_crs_map(stations_path, config.timetable_calls_path)


def load_outturn(
    config: HindcastConfig, candidates: list[CandidateService], schedules: pd.DataFrame
) -> pd.DataFrame:
    """Mean observed load per train, day and station."""
    outturn_cfg = config.outturn
    data = loading_data.load_operator(
        outturn_cfg.operator,
        outturn_cfg.path,
        _outturn_tiploc_map(config),
        cache_dir=outturn_cfg.cache_dir,
    )
    start = pd.to_datetime(data["Period Start Date"], errors="coerce")
    if outturn_cfg.date_from:
        data = data[start >= pd.Timestamp(outturn_cfg.date_from)]
    if outturn_cfg.date_to:
        data = data[start <= pd.Timestamp(outturn_cfg.date_to)]
    # The timetable's second codes were rewritten on read; a count at
    # one has to be rewritten the same way to still meet its call.
    data = recode_columns(
        data,
        resolve_recodes(config.station_recodes),
        ["Departure from CRS", "Origin CRS", "Destination CRS"],
    )

    uids = {c.name for c in candidates}
    by_uid = data[data["Unique ID"].isin(uids)].assign(train_uid=lambda d: d["Unique ID"])
    if by_uid.empty:
        # No UIDs in this operator's feed: match on headcode + ends +
        # origin time.
        keys = {}
        for _, sch in schedules[schedules["train_uid"].isin(uids)].iterrows():
            dep = cif_hhmm(sch["departures"][0]) or None
            keys[(sch["signalling_id"], sch["origin"], sch["destination"], dep)] = sch["train_uid"]
        match = [
            keys.get((h, o, d, (t or "")[:5]))
            for h, o, d, t in zip(
                data["Headcode"],
                data["Origin CRS"],
                data["Destination CRS"],
                data["Origin Departure Time"].astype("string").fillna(""),
                strict=True,
            )
        ]
        by_uid = data.assign(train_uid=match).dropna(subset=["train_uid"])

    matched_trains = by_uid["train_uid"].nunique()
    # No match is not a result: every accuracy figure downstream would
    # be computed over an empty join and reported as though the model
    # had been tested. Say so loudly rather than leaving it in an INFO
    # line.
    log = logger.info if matched_trains else logger.warning
    log(
        "Outturn: %d loading rows matched to %d of %d stripped trains.%s",
        len(by_uid),
        matched_trains,
        len(uids),
        ""
        if matched_trains
        else (
            f" No {outturn_cfg.operator} train could be identified in the feed, so there"
            " is nothing to compare against; check the feed's station codes and whether"
            " it publishes CIF train UIDs."
        ),
    )
    return (
        by_uid.groupby(["train_uid", "Service Week Day", "Departure from CRS"])
        .agg(
            actual_load=("Mean Load on Departure", "mean"),
            actual_on=("Mean Passengers On", "mean"),
            actual_off=("Mean Passengers Off", "mean"),
            actual_capacity=("Mean Capacity", "mean"),
            n_periods=("Mean Load on Departure", "count"),
        )
        .reset_index()
        .rename(columns={"Service Week Day": "day", "Departure from CRS": "from_station"})
    )


def compare_odm(
    config: HindcastConfig,
    demand: pd.DataFrame,
    candidates: list[CandidateService],
    remaining: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Predicted and actual journeys per pair, from two ODM years.

    Background growth between the two ODM years is removed with the
    growth of every pair *not* touching the case's stations. For each
    pair the predicted after-demand is ``before * growth + generated``,
    where ``generated`` is the annual trips new to rail the model
    attributes to the added trains, summed over trains, running days and
    time slots.
    """
    odm_cfg = config.outturn_odm
    before = read_odm(odm_cfg.before_path, odm_cfg.nlc_reference_paths)
    after = read_odm(odm_cfg.after_path, odm_cfg.nlc_reference_paths)
    stations = {s.station for c in candidates for s in c.stops}
    focus = set(odm_cfg.pairs_touching) or stations

    def _touches(frame: pd.DataFrame, codes: set[str]) -> pd.Series:
        return frame["origin_tlc"].isin(codes) | frame["destination_tlc"].isin(codes)

    growth = (
        after.loc[~_touches(after, stations), "journeys"].sum()
        / before.loc[~_touches(before, stations), "journeys"].sum()
    )
    logger.info("Background growth between ODM years (pairs away from the case): %.3f", growth)

    served = {
        (c.stops[i].station, c.stops[j].station)
        for c in candidates
        for i in range(len(c.stops))
        for j in range(i + 1, len(c.stops))
    }
    pairs = pd.DataFrame(sorted(served), columns=["origin_tlc", "destination_tlc"])
    pairs = pairs[pairs["origin_tlc"].isin(focus) | pairs["destination_tlc"].isin(focus)]
    idx = ["origin_tlc", "destination_tlc"]
    table = (
        pairs.merge(before.rename(columns={"journeys": "journeys_before"}), on=idx, how="left")
        .merge(after.rename(columns={"journeys": "journeys_after_actual"}), on=idx, how="left")
        .fillna({"journeys_before": 0.0, "journeys_after_actual": 0.0})
    )
    table["background_growth"] = growth
    table["journeys_after_no_change"] = table["journeys_before"] * growth
    table["actual_increment"] = table["journeys_after_actual"] - table["journeys_after_no_change"]
    # Direct trains left once the service is stripped: 0 marks a
    # genuinely new link. The "before" network is today's timetable
    # minus the service, so for pairs that keep other direct trains it
    # may not match the network of the before-ODM year.
    table["remaining_direct_trains"] = [
        len(
            existing_trains(
                remaining,
                reference_date,
                reference_date.day_name(),
                o,
                d,
                config.time_periods_cutoffs,
            )
        )
        for o, d in zip(table["origin_tlc"], table["destination_tlc"], strict=True)
    ]
    table["new_direct_link"] = table["remaining_direct_trains"].eq(0)

    generated = (
        demand.groupby(["demand_model", "mode", "from_station", "to_station"])[
            "predicted_annual_trips_generated"
        ]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"from_station": "origin_tlc", "to_station": "destination_tlc"})
    )
    rows = []
    for (model, mode), grp in generated.groupby(["demand_model", "mode"]):
        merged = table.merge(
            grp[[*idx, "predicted_annual_trips_generated"]], on=idx, how="left"
        ).rename(columns={"predicted_annual_trips_generated": "predicted_increment"})
        merged["journeys_after_predicted"] = (
            merged["journeys_after_no_change"] + merged["predicted_increment"]
        )
        rows.append(merged.assign(demand_model=model, mode=mode))
    return pd.concat(rows, ignore_index=True) if rows else table


# Metrics
def accuracy_metrics(predicted: pd.Series, actual: pd.Series) -> dict[str, float]:
    """Forecast accuracy measures for paired predictions and actuals."""
    mask = predicted.notna() & actual.notna()
    p = predicted[mask].astype(float)
    a = actual[mask].astype(float)
    if mask.sum() == 0:
        return {"n": 0}
    err = p - a
    nonzero = a > 0
    return {
        "n": int(mask.sum()),
        "mean_actual": float(a.mean()),
        "mean_predicted": float(p.mean()),
        "ratio_total_pred_to_actual": float(p.sum() / a.sum()) if a.sum() else np.nan,
        "mean_error": float(err.mean()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape_pct": float((err[nonzero].abs() / a[nonzero]).mean() * 100)
        if nonzero.any()
        else np.nan,
        "geh_mean": float(np.sqrt(2 * err**2 / (p + a).where(p + a > 0)).mean()),
        "pearson_r": float(np.corrcoef(p, a)[0, 1])
        if len(p) > 2 and p.std() and a.std()
        else np.nan,
    }


# Driver
@dataclasses.dataclass
class _Strip:
    """The timetable before and after the stripped trains, and what was taken out."""

    calls: pd.DataFrame  # recoded, with every train
    remaining: pd.DataFrame  # recoded, without the stripped trains or calls
    schedules: pd.DataFrame
    removed_calls: set[str]
    crs_tiploc: dict[str, str]
    train_tiplocs: dict[str, dict[str, str]]


def _strip(config: HindcastConfig) -> _Strip:
    """Take the service out of the timetable and write the timetable without it."""
    out = config.output_path
    raw_calls = read_public_calls(config.timetable_calls_path)
    # Running times are looked up by each stop's own TIPLOC, so the map
    # is taken before the timetable's second codes are rewritten.
    crs_tiploc = crs_to_tiploc_map(raw_calls)
    recodes = resolve_recodes(config.station_recodes)
    calls = recode_columns(raw_calls, recodes, ["crs"])
    rule = recode_strip_rule(config.strip, recodes)
    schedules = select_schedules(calls, rule)
    if schedules.empty:
        raise ValueError(f"No schedules match the strip rule {rule!r}.")
    schedules.to_csv(out / "hindcast_stripped_schedules.csv", index=False)
    uids = set(schedules["train_uid"])
    removed_calls = set(rule.remove_calls_at)
    if removed_calls:
        # Calls-only: the schedules keep running, without their stops at
        # these stations. Legs that touch one are the new ground; legs
        # that do not still see the candidate's own train in
        # `remaining`, so their existing frequency is one too high
        # (flagged per leg in hindcast_demand_by_leg).
        at_target = calls["crs"].astype("string").str.strip().str.upper().isin(removed_calls)
        remaining = calls[~(calls["train_uid"].isin(uids) & at_target)]
    else:
        remaining = calls[~calls["train_uid"].isin(uids)]
    # Reusable by `propose`/`scenario` as a without-the-service
    # timetable, so written with the raw codes they read (and recode)
    # themselves.
    raw_calls.loc[remaining.index].to_csv(out / "timetable_calls_stripped.csv", index=False)
    # Each stripped train's own TIPLOCs, which the recoded stops (LSX
    # now LST) would otherwise look up as the main station's.
    own = calls.loc[calls["train_uid"].isin(uids), ["train_uid", "crs", "tiploc"]]
    own = own.dropna().astype(str)
    train_tiplocs = {
        uid: dict(
            zip(
                group["crs"].str.strip().str.upper(),
                group["tiploc"].str.strip().str.upper(),
                strict=True,
            )
        )
        for uid, group in own.groupby("train_uid")
    }
    logger.info(
        "Stripped %d schedules (%d train UIDs, %d calls) from the timetable%s.",
        len(schedules),
        len(uids),
        len(calls) - len(remaining),
        f"; calls-only at {', '.join(sorted(removed_calls))}" if removed_calls else "",
    )
    return _Strip(calls, remaining, schedules, removed_calls, crs_tiploc, train_tiplocs)


def _rebuild(
    config: HindcastConfig, strip: _Strip, reference_date: pd.Timestamp
) -> tuple[list[CandidateService], dict[str, float], TrackDistanceCalculator | None]:
    """Rebuild the stripped trains as candidates and check every input they need.

    Returns the candidates, any forecast footfall applied to their new stations, and the
    track distance calculator (when the config has one).
    """
    out = config.output_path
    candidates = build_candidates(
        strip.schedules,
        reference_date,
        config.days,
        config.traction_code,
        config.default_capacity,
    )
    if not candidates:
        raise ValueError(
            f"None of the {strip.schedules['train_uid'].nunique()} stripped trains runs on "
            f"{', '.join(config.days)} in the week of {reference_date.date()}."
        )
    logger.info("Rebuilt %d trains running on %s.", len(candidates), ", ".join(config.days))
    forecast_footfall = apply_forecast_footfall(config, candidates)
    if forecast_footfall:
        pd.DataFrame(
            sorted(forecast_footfall.items()), columns=["crs", "forecast_entries_exits"]
        ).to_csv(out / "hindcast_forecast_footfall.csv", index=False)
    _warn_missing_footfall(config, candidates)
    calc = _warn_unroutable_stations(config, candidates)
    _warn_missing_matrix_stations(config, candidates)
    # Every destination column any leg or competing terminal will ask
    # for, parsed once per matrix file rather than once per column.
    aliases = resolve_aliases(config.station_aliases)
    groups = (
        config.destination_groups
        if config.destination_groups is not None
        else DEFAULT_DESTINATION_GROUPS
    )
    prefetch_matrix_columns(
        config.period_matrices_dir,
        {canonical_crs(s.station, aliases) for c in candidates for s in c.stops}
        | {canonical_crs(code, aliases) for members in groups.values() for code in members},
    )
    pd.DataFrame(
        [
            {
                "train_uid": c.name,
                "days": " ".join(c.days_of_week),
                "origin": c.stops[0].station,
                "departure": c.stops[0].departure,
                "destination": c.stops[-1].station,
                "arrival": c.stops[-1].arrival,
                "pattern": "-".join(s.station for s in c.stops),
            }
            for c in candidates
        ]
    ).to_csv(out / "hindcast_candidates.csv", index=False)
    return candidates, forecast_footfall, calc


Variant = tuple[str, str]  # (demand model, "marginal" or "joint")


def _predict(
    config: HindcastConfig,
    strip: _Strip,
    candidates: list[CandidateService],
    reference_date: pd.Timestamp,
    forecast_footfall: dict[str, float],
    calc: TrackDistanceCalculator | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[Variant, list[pd.DataFrame]], list[Variant]]:
    """Check each train's feasibility and predict its demand under every model and mode.

    Returns the running-time and headway checks, the demand by leg for each variant, and
    the variants.
    """
    graph = BplanTimingGraph(config.bplan_output_path)
    try:
        network_links = BplanNetworkLinks(config.bplan_output_path)
    except FileNotFoundError:
        network_links = None
    fares = build_fares_context(config)
    joint = _joint_counter(candidates, config.time_periods_cutoffs)
    model_configs = {
        name: config.model_copy(update={"demand_model": name}) for name in config.demand_models
    }
    gravities = {name: build_gravity_context(cfg, calc) for name, cfg in model_configs.items()}
    # Routed on the full timetable (the one after the change), not on
    # `remaining` (see TransferDemand).
    transfers = transfer_demand(config, strip.calls, reference_date)
    variants = [(name, mode) for name in config.demand_models for mode in ("marginal", "joint")]

    running, headway = [], []
    demand: dict[Variant, list[pd.DataFrame]] = {v: [] for v in variants}
    for i, cand in enumerate(candidates, start=1):
        logger.info("Hindcast train %d/%d: %s", i, len(candidates), cand.name)
        cand_tiploc = {**strip.crs_tiploc, **strip.train_tiplocs.get(cand.name, {})}
        running.append(check_running_times(cand, cand_tiploc, graph).assign(train_uid=cand.name))
        headway.append(
            check_headway(
                config,
                cand,
                strip.remaining,
                reference_date,
                crs_tiploc=cand_tiploc,
                network_links=network_links,
            ).assign(train_uid=cand.name)
        )
        for name, mode in variants:
            added = (
                (lambda day, period, o, d, counts=joint: counts[(day, period, o, d)])
                if mode == "joint"
                else None
            )
            leg = predict_demand(
                model_configs[name],
                cand,
                strip.remaining,
                reference_date,
                gravities[name],
                fares,
                added_trains=added,
                transfers=transfers,
            )
            demand[(name, mode)].append(leg.assign(demand_model=name, mode=mode))

    calibrated = gravities.get("calibrated")
    if (
        forecast_footfall
        and config.forecast_footfall_weekday_share is not None
        and calibrated is not None
        and calibrated.model is not None
    ):
        factors, balance = balance_forecast_stations(
            config, calibrated, candidates, strip.remaining, reference_date, forecast_footfall
        )
        balance.to_csv(config.output_path / "hindcast_station_balance.csv", index=False)
        aliases = resolve_aliases(config.station_aliases)
        for mode in ("marginal", "joint"):
            if ("calibrated", mode) in demand:
                demand[("calibrated", mode)] = [
                    apply_station_balance(frame, factors, aliases)
                    for frame in demand[("calibrated", mode)]
                ]
    if transfers is not None and transfers.summary:
        pd.DataFrame(transfers.summary).to_csv(
            config.output_path / "hindcast_transfer_routes.csv", index=False
        )
    return (
        pd.concat(running, ignore_index=True),
        pd.concat(headway, ignore_index=True),
        demand,
        variants,
    )


def _predicted_segments(
    config: HindcastConfig,
    candidates: list[CandidateService],
    demand: dict[Variant, list[pd.DataFrame]],
    variants: list[Variant],
) -> tuple[pd.DataFrame, list[str]]:
    """Every (train, day, segment) with each variant's predicted load, figure or not."""
    key = ["train_uid", "day", "from_station", "to_station"]
    predicted = pd.DataFrame(
        [
            {
                "train_uid": cand.name,
                "day": day,
                "from_station": stop.station.strip().upper(),
                "to_station": nxt.station.strip().upper(),
                "capacity": cand.capacity or config.default_capacity,
            }
            for cand in candidates
            for day in cand.days_of_week
            for stop, nxt in zip(cand.stops, cand.stops[1:], strict=False)
        ]
    )
    prediction_cols = []
    for name, mode in variants:
        col = f"predicted_load_{name}_{mode}"
        prediction_cols.append(col)
        frame = pd.concat(
            [
                estimate_loading(cand, leg, config)
                for cand, leg in zip(candidates, demand[(name, mode)], strict=True)
            ],
            ignore_index=True,
        )
        if frame.empty:
            predicted[col] = np.nan
            continue
        frame = frame.rename(
            columns={
                "candidate_name": "train_uid",
                "on_board_typical_day": col,
                "incomplete_estimate": f"incomplete_{name}_{mode}",
            }
        )[[*key, col, f"incomplete_{name}_{mode}"]]
        predicted = predicted.merge(frame, on=key, how="left")
    return predicted, prediction_cols


def _odm_rows(
    config: HindcastConfig,
    demand_all: pd.DataFrame,
    candidates: list[CandidateService],
    remaining: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> list[dict]:
    """Score the predicted journeys against the ODM outturn, when one is configured."""
    if config.outturn_odm is None:
        return []
    odm_table = compare_odm(config, demand_all, candidates, remaining, reference_date)
    odm_table.to_csv(config.output_path / "hindcast_odm_pairs.csv", index=False)
    rows = []
    for (name, mode), grp in odm_table.groupby(["demand_model", "mode"]):
        for subset, subset_rows in (
            ("all_pairs", grp),
            ("new_direct_links", grp[grp["new_direct_link"]]),
            ("already_served", grp[~grp["new_direct_link"]]),
        ):
            for measure, pred, act in (
                ("odm_journeys_after", "journeys_after_predicted", "journeys_after_actual"),
                ("odm_increment", "predicted_increment", "actual_increment"),
            ):
                rows.append(
                    {
                        "case": config.case_name,
                        "level": f"{measure}:{subset}",
                        "demand_model": name,
                        "demand_mode": mode,
                        **accuracy_metrics(subset_rows[pred], subset_rows[act]),
                    }
                )
    return rows


def _load_rows(
    config: HindcastConfig,
    segments: pd.DataFrame,
    trains: pd.DataFrame,
    variants: list[Variant],
) -> list[dict]:
    """Score each variant's loads on segments and on trains, on two bases.

    ``all_rows`` includes rows whose predicted load is a lower bound (some leg was
    unpredictable), which drags predicted load toward zero and makes the error look like a
    model result when it is really coverage. ``complete_only`` is the accuracy of the model
    where it actually ran. Quote ``complete_only``, with ``pct_incomplete`` beside it.
    """
    rows = []
    for level, frame in (("segment", segments), ("train", trains)):
        for name, mode in variants:
            predicted_col = frame[f"predicted_load_{name}_{mode}"]
            actual_col = frame["actual_load"]
            inc_col = f"incomplete_{name}_{mode}"
            incomplete = (
                frame[inc_col].fillna(value=False).astype(bool)
                if inc_col in frame.columns
                else pd.Series(False, index=frame.index)
            )
            base = {
                "case": config.case_name,
                "level": level,
                "demand_model": name,
                "demand_mode": mode,
            }
            rows.append(
                {
                    **base,
                    "basis": "all_rows",
                    "n_incomplete": int(incomplete.sum()),
                    "pct_incomplete": (
                        round(float(incomplete.mean() * 100), 1) if len(frame) else np.nan
                    ),
                    **accuracy_metrics(predicted_col, actual_col),
                }
            )
            rows.append(
                {
                    **base,
                    "basis": "complete_only",
                    "n_incomplete": 0,
                    "pct_incomplete": 0.0,
                    **accuracy_metrics(predicted_col[~incomplete], actual_col[~incomplete]),
                }
            )
    return rows


def _feasibility_row(
    config: HindcastConfig,
    candidates: list[CandidateService],
    running: pd.DataFrame,
    headway: pd.DataFrame,
) -> dict:
    """Summarise the running-time and headway checks in one summary row."""
    n_legs = len(running)
    infeasible = running.get("status") == "infeasible"
    row = {
        "case": config.case_name,
        "level": "feasibility",
        "n_trains": len(candidates),
        "running_time_legs": n_legs,
        "running_time_infeasible": int(infeasible.sum()) if n_legs else 0,
        "running_time_unverifiable": int((running.get("status") == "unverifiable").sum())
        if n_legs
        else 0,
        "headway_evaluations": len(headway),
        "headway_conflicts": int(headway["conflict"].sum()) if len(headway) else 0,
        "trains_flagged_any": len(
            set(running.loc[infeasible, "train_uid"])
            | set(headway.loc[headway["conflict"].astype(bool), "train_uid"])
        )
        if n_legs
        else 0,
    }
    if len(headway):
        nearest = headway[["gap_before_minutes", "gap_after_minutes"]].min(axis=1)
        row["headway_conflicts_fixed_mode"] = int(
            (nearest < config.planning_headway_minutes).sum()
        )
        row["planning_headway_minutes"] = config.planning_headway_minutes
    return row


def run_hindcast(config: HindcastConfig) -> None:
    """Strip, rebuild, predict and compare; see the module docstring.

    Writes to ``config.output_path``:

    * ``hindcast_stripped_schedules.csv``: the schedules removed
    * ``hindcast_candidates.csv``: the rebuilt trains (stops and days)
    * ``hindcast_running_times.csv`` and ``hindcast_headway.csv``:
      feasibility
    * ``hindcast_demand_by_leg.csv``: predicted demand, every model and
      mode
    * ``hindcast_segments.csv``: predicted vs actual load per
      train/day/segment
    * ``hindcast_trains.csv``: the same averaged per train/day
    * ``hindcast_transfer_routes.csv``: pairs routed through one change,
      per day
    * ``hindcast_summary.csv``: accuracy metrics and feasibility counts
    """
    out = config.output_path
    out.mkdir(parents=True, exist_ok=True)
    if config.reference_date is None:
        raise ValueError("hindcast needs an explicit reference_date inside the outturn window.")
    reference_date = pd.Timestamp(config.reference_date)
    _require_demand_inputs(config)

    strip = _strip(config)
    candidates, forecast_footfall, calc = _rebuild(config, strip, reference_date)
    running, headway, demand, variants = _predict(
        config, strip, candidates, reference_date, forecast_footfall, calc
    )
    running.to_csv(out / "hindcast_running_times.csv", index=False)
    headway.to_csv(out / "hindcast_headway.csv", index=False)
    demand_all = pd.concat([f for v in variants for f in demand[v]], ignore_index=True)
    demand_by_leg = demand_all.copy()
    if strip.removed_calls:
        # In calls-only mode only these legs are the new ground; the
        # rest keep their own train in the before-network and are not
        # part of the test.
        demand_by_leg["leg_is_new_ground"] = demand_by_leg["from_station"].isin(
            strip.removed_calls
        ) | demand_by_leg["to_station"].isin(strip.removed_calls)
    demand_by_leg.to_csv(out / "hindcast_demand_by_leg.csv", index=False)

    predicted, prediction_cols = _predicted_segments(config, candidates, demand, variants)
    odm_rows = _odm_rows(config, demand_all, candidates, strip.remaining, reference_date)
    if config.outturn is None:
        pd.DataFrame(odm_rows).to_csv(out / "hindcast_summary.csv", index=False)
        logger.info("Hindcast summary:\n%s", pd.DataFrame(odm_rows).to_string(index=False))
        return

    outturn = load_outturn(config, candidates, strip.schedules)
    segments = predicted.merge(outturn, on=["train_uid", "day", "from_station"], how="left")
    if config.outturn.screen_count_resets:
        segments, screened = screen_count_resets(segments, config.outturn.count_reset_min_load)
        screened.to_csv(out / "hindcast_outturn_screened.csv", index=False)
    first_stop = {c.name: c.stops[0].departure for c in candidates}
    segments["train_departure"] = segments["train_uid"].map(first_stop)
    segments["period"] = segments["train_departure"].map(
        lambda t: classify_time_period(f"{t}:00", config.time_periods_cutoffs) if t else None
    )
    # A segment fed by any leg the model could not predict is a lower
    # bound, not an estimate (see `estimate_loading`). The merge above
    # leaves the flags NaN where a variant has no figure; make them
    # booleans.
    incomplete_cols = [
        f"incomplete_{name}_{mode}"
        for name, mode in variants
        if f"incomplete_{name}_{mode}" in segments.columns
    ]
    for col in incomplete_cols:
        segments[col] = segments[col].fillna(value=False).astype(bool)
    segments.to_csv(out / "hindcast_segments.csv", index=False)

    trains = (
        segments.groupby(["train_uid", "day", "train_departure", "period"], dropna=False)
        .agg(
            actual_load=("actual_load", "mean"),
            **{col: (col, "mean") for col in prediction_cols},
            # A train is incomplete if any segment of it is.
            **{col: (col, "max") for col in incomplete_cols},
        )
        .reset_index()
    )
    trains.to_csv(out / "hindcast_trains.csv", index=False)

    rows = _load_rows(config, segments, trains, variants)
    rows.append(_feasibility_row(config, candidates, running, headway))
    rows.extend(odm_rows)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "hindcast_summary.csv", index=False)
    logger.info("Hindcast summary:\n%s", summary.to_string(index=False))
