"""Price changes to a timetable: cuts, retimes, calls added or taken out, trains added.

Each day's timetable is loaded before and after by :func:`ortla.assign.assign_day`, so a change is
on the same footing as the national loads. ``assign`` alone has no response, so a pair whose
direct trains change count takes one first:

* still served: ``(after / before) ** elasticity``;
* last direct train lost: the larger fall of ``exp(-link_loss)`` (from ``min_trains`` a day) and
  the elasticity down to one train;
* first direct train gained: the larger rise of ``exp(link_gain)`` (to ``min_trains`` a day) and
  the elasticity up from one train.

A pair left with no route of at most one change, at any time of day, loses its journeys. Retiming
keeps the count, so it moves passengers between trains and makes none: there's no validated
response to times. The waits and journeys it changes are reported, so a planner can see who gains
and who loses. The accounting is checked; a forecast of a change made this way is not validated.
"""

from __future__ import annotations

import bisect
import collections
import dataclasses
import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ortla import assign, paths
from ortla.bplan_geography import BplanTimingGraph
from ortla.clock import MINUTES_PER_DAY, WEEKDAYS, WEEKDAYS_PER_YEAR, cif_hhmm, hhmm_minutes
from ortla.demand import build_fares_context, candidate_calls
from ortla.network_assignment import PeriodMatrices, demand_pairs, slot_trains
from ortla.stations import canonical_crs, recode_columns, resolve_aliases, resolve_recodes
from ortla.timetable import filter_services, read_public_calls

if TYPE_CHECKING:
    from ortla.config import ChangesConfig, TimetableChange
    from ortla.demand import FaresContext
    from ortla.network_assignment import PeriodChangeRoutes

logger = logging.getLogger(__name__)

_FIRST, _LAST = 6 * 60, 22 * 60  # when passengers turn up, for the journey time with its wait
_PERSON_MINUTES_FLOOR = 1e-9


# The timetable after the change


def _clock(minutes: float) -> str:
    total = round(minutes) % MINUTES_PER_DAY
    return f"{total // 60:02d}{total % 60:02d}"


@dataclasses.dataclass
class _Call:
    row: int
    crs: str
    arrive: float | None
    depart: float | None
    new: bool = False


def _read_calls(rows: pd.DataFrame) -> list[_Call]:
    ordered = rows.assign(order=pd.to_numeric(rows["seq"], errors="coerce")).sort_values("order")
    out: list[_Call] = []
    latest = None
    for index, row in ordered.iterrows():
        times = []
        for value in (row["public_arrival"], row["public_departure"]):
            at = hhmm_minutes(cif_hhmm(value)) if pd.notna(value) else None
            if at is not None:
                while latest is not None and at < latest - MINUTES_PER_DAY / 2:
                    at += MINUTES_PER_DAY
                latest = at
            times.append(at)
        out.append(_Call(index, str(row["crs"]).strip().upper(), *times))
    return out


def _position(calls: list[_Call], station: str, change: TimetableChange) -> int:
    for i, call in enumerate(calls):
        if call.crs == station:
            return i
    raise ValueError(f"{change.train_uid} doesn't call at {station}.")


def _move(calls: list[_Call], by: float) -> None:
    for call in calls:
        call.arrive = None if call.arrive is None else call.arrive + by
        call.depart = None if call.depart is None else call.depart + by


def _typed(value: str | None, near: float) -> float | None:
    at = hhmm_minutes(value)
    if at is None:
        return None
    while at < near - MINUTES_PER_DAY / 2:
        at += MINUTES_PER_DAY
    return at


def _retime(calls: list[_Call], change: TimetableChange, _recodes: dict[str, str]) -> list[_Call]:
    if change.from_station is None:
        _move(calls, change.shift_minutes)
        return calls
    at = _position(calls, canonical_crs(change.from_station, _recodes), change)
    if calls[at].depart is not None:
        calls[at].depart += change.shift_minutes
    _move(calls[at + 1 :], change.shift_minutes)
    return calls


def _add_call(calls: list[_Call], change: TimetableChange, recodes: dict[str, str]) -> list[_Call]:
    station = canonical_crs(change.station or "", recodes)
    at = _position(calls, canonical_crs(change.after_station or "", recodes), change)
    before = calls[at]
    near = before.depart if before.depart is not None else before.arrive or 0.0
    arrive, depart = _typed(change.arrival, near), _typed(change.departure, near)
    last = at + 1 == len(calls)
    if last:
        if before.depart is None and before.arrive is not None:
            before.depart = before.arrive + 1  # it was the end, and now runs on
        depart = None
    elif depart is None:
        raise ValueError(f"{change.train_uid}: a call before the last needs a departure.")
    _move(calls[at + 1 :], change.delay_minutes)
    added = _Call(before.row, station, arrive, depart, new=True)
    return [*calls[: at + 1], added, *calls[at + 1 :]]


def _remove_call(
    calls: list[_Call], change: TimetableChange, recodes: dict[str, str]
) -> list[_Call]:
    at = _position(calls, canonical_crs(change.station or "", recodes), change)
    _move(calls[at + 1 :], change.delay_minutes)
    kept = calls[:at] + calls[at + 1 :]
    if len(kept) > 1:
        kept[0].arrive, kept[-1].depart = None, None
    return kept


_EDITS = {"retime": _retime, "add_call": _add_call, "remove_call": _remove_call}


def _checked(uid: str, calls: list[_Call]) -> None:
    if len(calls) < 2:
        raise ValueError(f"{uid} would have fewer than two calls. Cut it instead.")
    for a, b in itertools.pairwise(calls):
        leaves = a.depart if a.depart is not None else a.arrive
        comes = b.arrive if b.arrive is not None else b.depart
        if leaves is None or comes is None:
            continue
        if comes < leaves or (b.arrive is not None and b.depart is not None and b.depart < comes):
            raise ValueError(f"{uid}: the times run backwards between {a.crs} and {b.crs}.")


def apply_changes(calls: pd.DataFrame, config: ChangesConfig, weekday: str) -> pd.DataFrame:
    """Return the calls as changed on ``weekday``. ``calls`` have recoded station codes."""
    recodes = resolve_recodes(config.station_recodes)
    today = filter_services(calls, config.reference_timestamp, [WEEKDAYS.index(weekday)])
    by_train: dict[str, list[TimetableChange]] = collections.defaultdict(list)
    for change in config.changes:
        by_train[change.train_uid].append(change)
    drop: list[int] = []
    added: list[pd.DataFrame] = []
    for uid, changes in by_train.items():
        own = today[today["train_uid"] == uid]
        if own.empty:
            raise ValueError(f"No passenger train {uid} runs on {weekday} in this timetable.")
        key = ["days_runs", "start_date", "end_date", "svc_origin_dep", "signalling_id"]
        for _, rows in own.groupby(key, dropna=False, sort=False):
            drop.extend(rows.index)
            if any(change.kind == "cut" for change in changes):
                continue
            edited = _read_calls(rows)
            for change in changes:
                edited = _EDITS[change.kind](edited, change, recodes)
            _checked(uid, edited)
            added.append(_written(calls.loc[rows.index], edited))
    for service in config.services:
        if weekday not in service.days_of_week:
            continue
        stops = [
            stop.model_copy(update={"station": canonical_crs(stop.station, recodes)})
            for stop in service.stops
        ]
        new = service.model_copy(update={"stops": stops})
        rows = candidate_calls(new, config.reference_timestamp)
        added.append(rows.assign(days_runs="1111111", departure=None, arrival=None))
    out = calls.drop(index=drop)
    if not added:
        return out.reset_index(drop=True)
    return pd.concat([out, *added], ignore_index=True)


def _written(original: pd.DataFrame, edited: list[_Call]) -> pd.DataFrame:
    columns = original.loc[[call.row for call in edited]].copy().reset_index(drop=True)
    columns["seq"] = [str(i) for i in range(len(edited))]
    columns["crs"] = [call.crs for call in edited]
    for field, column in (("arrive", "public_arrival"), ("depart", "public_departure")):
        values = [getattr(call, field) for call in edited]
        columns[column] = pd.Series(
            [None if at is None else _clock(at) for at in values], dtype=original[column].dtype
        )
    if "tiploc" in columns:
        columns.loc[[call.new for call in edited], "tiploc"] = None
    for end, column in ((0, "svc_origin_crs"), (-1, "svc_dest_crs")):
        last = original.assign(order=pd.to_numeric(original["seq"])).sort_values("order")
        if edited[end].crs != str(last["crs"].iloc[end]).strip().upper():
            columns[column] = edited[end].crs
    return columns


# The response


@dataclasses.dataclass(frozen=True)
class Response:
    """How many journeys a change in a pair's direct trains makes."""

    elasticity: float
    link_gain: float
    link_loss: float
    min_trains: int

    def factor(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Return each pair's demand multiplier from its direct trains a day."""
        before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
        out = np.power(np.maximum(after, 1.0) / np.maximum(before, 1.0), self.elasticity)
        lost = (after == 0) & (before >= self.min_trains)
        out[lost] = np.minimum(out[lost], np.exp(-self.link_loss))
        gained = (before == 0) & (after >= self.min_trains)
        out[gained] = np.maximum(out[gained], np.exp(self.link_gain))
        return out


class _ScaledMatrices:
    """Period matrices with the changed pairs' demand multiplied."""

    def __init__(self, matrices: PeriodMatrices, factors: pd.Series) -> None:
        self.matrices = matrices
        origins, destinations = (factors.index.get_level_values(i).astype(str) for i in (0, 1))
        self.stations = pd.Index(sorted(set(origins) | set(destinations)))
        keys = self._key(np.asarray(origins), np.asarray(destinations))
        order = np.argsort(keys)
        self.keys, self.factors = keys[order], factors.to_numpy(dtype=float)[order]

    def _key(self, origins: np.ndarray, destinations: np.ndarray) -> np.ndarray:
        o, d = self.stations.get_indexer(origins), self.stations.get_indexer(destinations)
        return np.where((o < 0) | (d < 0), -1, o.astype(np.int64) * len(self.stations) + d)

    def cells(
        self, day: str, periods: np.ndarray, origins: np.ndarray, destinations: np.ndarray
    ) -> np.ndarray:
        """Annual trips per period and pair, after the response."""
        out = self.matrices.cells(day, periods, origins, destinations)
        keys = self._key(origins, destinations)
        at = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        found = self.keys[at] == keys
        out[found] *= self.factors[at[found]]
        return out


class Network:
    """What one loading of the timetable saw: direct trains and reachable pairs.

    Given the network ``before`` and a ``response``, it also scales the matrices it is loaded
    from by the response to the change.
    """

    def __init__(self, before: Network | None = None, response: Response | None = None) -> None:
        """Start an observation, of the change from ``before`` where given."""
        self.before, self.response = before, response
        self.trains = pd.Series(dtype=float)
        self.factors = pd.Series(dtype=float)
        self.stations = pd.Index([])
        self.reach = np.zeros((0, 0), dtype=bool)

    def observe(
        self,
        legs: pd.DataFrame,
        origins: np.ndarray,
        destinations: np.ndarray,
        routes: PeriodChangeRoutes | None,
    ) -> None:
        """Keep the day's direct trains by pair and the pairs with a route."""
        slots = slot_trains(legs.assign(o=origins, d=destinations))
        self.trains = slots.groupby(level=["o", "d"]).sum().astype(float)
        self.stations = pd.Index(np.unique(np.concatenate([origins, destinations]))).astype(object)
        n = len(self.stations)
        self.reach = np.zeros((n, n), dtype=bool)
        index = self.stations.get_indexer
        self.reach[index(origins), index(destinations)] = True
        if routes is not None:
            for period in routes.routes:
                o, d = routes.pairs(period)
                self.reach[index(o), index(d)] = True
        if self.before is not None and self.response is not None:
            pairs = self.before.trains.index.union(self.trains.index)
            was = self.before.trains.reindex(pairs).fillna(0.0)
            now = self.trains.reindex(pairs).fillna(0.0)
            factors = pd.Series(self.response.factor(was, now), index=pairs)
            self.factors = factors[factors != 1.0]

    def scaled(self, matrices: PeriodMatrices) -> PeriodMatrices | _ScaledMatrices:
        """Return ``matrices`` after the response."""
        return matrices if self.factors.empty else _ScaledMatrices(matrices, self.factors)

    def count(self, origins: np.ndarray, destinations: np.ndarray) -> np.ndarray:
        """Return the direct trains a day on each pair."""
        pairs = pd.MultiIndex.from_arrays([origins, destinations])
        return self.trains.reindex(pairs).fillna(0.0).to_numpy()

    def reached(self, origins: np.ndarray, destinations: np.ndarray) -> np.ndarray:
        """Whether each pair has a route of at most one change."""
        o, d = self.stations.get_indexer(origins), self.stations.get_indexer(destinations)
        known = (o >= 0) & (d >= 0)
        out = np.zeros(len(o), dtype=bool)
        out[known] = self.reach[o[known], d[known]]
        return out


def journeys(
    config: ChangesConfig, weekday: str, before: Network, after: Network
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return the pairs whose journeys or trains change, and the day's totals."""
    directory = config.journeys_matrices_dir or config.operator_groups()[-1][2]
    matrices = PeriodMatrices(directory, config.odm_open_share)
    periods = ["early", *config.time_periods_cutoffs]
    origins, destinations = demand_pairs(matrices, weekday, periods)
    base = np.zeros(len(origins))
    for period in periods:
        slot = np.full(len(origins), period, dtype=object)
        base += np.nan_to_num(matrices.cells(weekday, slot, origins, destinations))
    base /= WEEKDAYS_PER_YEAR
    was, now = before.count(origins, destinations), after.count(origins, destinations)
    had, has = before.reached(origins, destinations), after.reached(origins, destinations)
    j_before = base * had
    cases = {
        "": config.response(),
        "_least": config.response("least"),
        "_most": config.response("most"),
    }
    after_by_case = {name: base * r.factor(was, now) * has for name, r in cases.items()}
    j_after = after_by_case[""]
    both = had & has
    totals = {
        "journeys_before": float(j_before.sum()),
        "journeys_after": float(j_after.sum()),
        "journeys_change": float((j_after - j_before).sum()),
        "journeys_change_response": float((j_after - j_before)[both].sum()),
        "journeys_lost_no_route": float(-j_before[had & ~has].sum()),
        "journeys_gained_route": float(j_after[has & ~had].sum()),
        "journeys_change_least": float((after_by_case["_least"] - j_before).sum()),
        "journeys_change_most": float((after_by_case["_most"] - j_before).sum()),
        "pairs_losing_every_direct_train": float(((was > 0) & (now == 0) & (base > 0)).sum()),
        "pairs_gaining_a_direct_train": float(((was == 0) & (now > 0) & (base > 0)).sum()),
    }
    moved = (was != now) | (had != has)
    table = pd.DataFrame(
        {
            "from_station": origins[moved],
            "to_station": destinations[moved],
            "trains_before": was[moved],
            "trains_after": now[moved],
            "route_before": had[moved],
            "route_after": has[moved],
            "journeys_before": j_before[moved],
            "journeys_after": j_after[moved],
            "journeys_change": (j_after - j_before)[moved],
        }
    )
    return table, totals


# Who gains and who loses time


def _journey_times(
    calls: pd.DataFrame, config: ChangesConfig, weekday: str, pairs: set[tuple[str, str]]
) -> dict[tuple[str, str], tuple[float, float]]:
    """Each pair's mean (journey with its wait, quickest ride) on direct trains."""
    day = filter_services(calls, config.reference_timestamp, [WEEKDAYS.index(weekday)])
    aliases = resolve_aliases(config.station_aliases)
    wanted = {code for pair in pairs for code in pair}
    codes = day["crs"].astype(str).str.strip().str.upper()
    codes = codes.map({code: canonical_crs(code, aliases) for code in codes.unique()})
    day = day.assign(code=codes, order=pd.to_numeric(day["seq"], errors="coerce"))
    day = day[day["code"].isin(wanted)]
    key = ["train_uid", "days_runs", "start_date", "svc_origin_dep", "signalling_id"]
    service = pd.factorize(pd.MultiIndex.from_frame(day[key].astype(str)))[0]
    at: dict[str, dict[int, tuple[float, float | None, float | None]]] = collections.defaultdict(
        dict
    )
    columns = zip(
        service,
        day["order"],
        day["code"],
        day["public_arrival"],
        day["public_departure"],
        strict=True,
    )
    for train, order, code, arrival, departure in columns:
        arrive = hhmm_minutes(cif_hhmm(arrival)) if pd.notna(arrival) else None
        depart = hhmm_minutes(cif_hhmm(departure)) if pd.notna(departure) else None
        at[code].setdefault(train, (order, arrive, depart))
    out = {}
    for origin, destination in pairs:
        trips = []
        ends = at.get(destination, {})
        for train, (order, _, depart) in at.get(origin, {}).items():
            end = ends.get(train)
            if depart is None or end is None or end[0] <= order:
                continue
            arrive = end[1] if end[1] is not None else end[2]
            if arrive is not None:
                trips.append((depart, depart + (arrive - depart) % MINUTES_PER_DAY))
        if trips:
            out[origin, destination] = _mean_journey(trips)
    return out


def _mean_journey(trips: list[tuple[float, float]]) -> tuple[float, float]:
    """Mean wait and ride for someone turning up at any minute, and the quickest ride."""
    ride = min(arrive - depart for depart, arrive in trips)
    trips = sorted([*trips, *((d + MINUTES_PER_DAY, a + MINUTES_PER_DAY) for d, a in trips)])
    departs = [depart for depart, _ in trips]
    soonest = list(itertools.accumulate([a for _, a in reversed(trips)], min))[::-1]
    spent = [soonest[bisect.bisect_left(departs, t)] - t for t in range(_FIRST, _LAST)]
    return sum(spent) / len(spent), ride


def _touched_pairs(
    frames: list[pd.DataFrame], trains: set[str], config: ChangesConfig, weekday: str
) -> set[tuple[str, str]]:
    """Every pair a changed train serves, before or after."""
    aliases = resolve_aliases(config.station_aliases)
    pairs: set[tuple[str, str]] = set()
    for calls in frames:
        own = calls[calls["train_uid"].isin(trains)]
        if own.empty:
            continue
        own = filter_services(own, config.reference_timestamp, [WEEKDAYS.index(weekday)])
        own = own.assign(order=pd.to_numeric(own["seq"], errors="coerce")).sort_values("order")
        for _, rows in own.groupby("train_uid"):
            codes = [canonical_crs(str(code).strip().upper(), aliases) for code in rows["crs"]]
            pairs |= {(a, b) for a, b in itertools.combinations(codes, 2) if a != b}
    return pairs


def _with_times(
    pairs: pd.DataFrame,
    touched: set[tuple[str, str]],
    times: list[dict[tuple[str, str], tuple[float, float]]],
    fares: FaresContext | None,
) -> pd.DataFrame:
    keys = ["from_station", "to_station"]
    extra = pd.DataFrame(sorted(touched), columns=keys)
    table = pairs.merge(extra, on=keys, how="outer").sort_values(keys).reset_index(drop=True)
    ordered = list(zip(table["from_station"], table["to_station"], strict=True))
    nothing = (np.nan, np.nan)
    for which, found in zip(("before", "after"), times, strict=True):
        values = [found.get(pair, nothing) for pair in ordered]
        table[f"minutes_with_wait_{which}"] = [value[0] for value in values]
        table[f"quickest_minutes_{which}"] = [value[1] for value in values]
    if fares is not None:
        fare = np.array([fares.fare(*pair) or np.nan for pair in ordered], dtype=float)
        table["average_fare_gbp"] = fare
        table["revenue_change_per_day"] = table["journeys_change"].to_numpy(dtype=float) * fare
    return table


def _fill_unchanged(
    table: pd.DataFrame, config: ChangesConfig, weekday: str, network: Network
) -> pd.DataFrame:
    """Give the retimed pairs, whose counts don't change, their journeys."""
    missing = table["journeys_before"].isna().to_numpy()
    if not missing.any():
        return table
    origins = table.loc[missing, "from_station"].to_numpy(dtype=object)
    destinations = table.loc[missing, "to_station"].to_numpy(dtype=object)
    directory = config.journeys_matrices_dir or config.operator_groups()[-1][2]
    matrices = PeriodMatrices(directory, config.odm_open_share)
    daily = np.zeros(len(origins))
    for period in ("early", *config.time_periods_cutoffs):
        slot = np.full(len(origins), period, dtype=object)
        daily += np.nan_to_num(matrices.cells(weekday, slot, origins, destinations))
    reached = network.reached(origins, destinations)
    daily = daily / WEEKDAYS_PER_YEAR * reached
    trains = network.count(origins, destinations)
    for column, values in (
        ("journeys_before", daily),
        ("journeys_after", daily),
        ("journeys_change", np.zeros(len(origins))),
        ("trains_before", trains),
        ("trains_after", trains),
    ):
        table.loc[missing, column] = values
    table.loc[missing, "route_before"] = reached
    table.loc[missing, "route_after"] = reached
    return table


def _time_totals(table: pd.DataFrame) -> dict[str, float]:
    saved = table["minutes_with_wait_before"] - table["minutes_with_wait_after"]
    riders = table[["journeys_before", "journeys_after"]].min(axis=1)
    weighed = (saved * riders).fillna(0.0)
    moved = weighed.abs() > _PERSON_MINUTES_FLOOR
    return {
        "minutes_saved_by_gainers": float(weighed[weighed > 0].sum()),
        "minutes_lost_by_losers": float(-weighed[weighed < 0].sum()),
        "journeys_quicker": float(riders[moved & (weighed > 0)].sum()),
        "journeys_slower": float(riders[moved & (weighed < 0)].sum()),
    }


# Loads


def _compared(before: pd.DataFrame, after: pd.DataFrame, trains: set[str]) -> dict:
    key = ["train_uid", "from_station", "to_station"]
    was, now = (frame.groupby(key)["load"].max() for frame in (before, after))
    segments = pd.concat([was.rename("load_before"), now.rename("load_after")], axis=1)
    segments = segments.fillna(0.0).reset_index()
    segments["load_change"] = segments["load_after"] - segments["load_before"]

    def by_train(frame: pd.DataFrame, which: str) -> pd.DataFrame:
        grouped = frame.groupby("train_uid")
        return pd.DataFrame(
            {
                f"peak_{which}": grouped["load"].max(),
                f"boardings_{which}": grouped["boardings"].sum(),
            }
        )

    each = by_train(before, "before").join(by_train(after, "after"), how="outer").fillna(0.0)
    about = ["atoc_code", "headcode", "train_origin", "train_destination", "train_departure"]
    names = pd.concat([after, before]).drop_duplicates("train_uid").set_index("train_uid")[about]
    each = each.join(names).rename_axis("train_uid").reset_index()
    each["peak_change"] = each["peak_after"] - each["peak_before"]
    each["boardings_change"] = each["boardings_after"] - each["boardings_before"]
    each["changed"] = each["train_uid"].isin(trains)
    moved = each["changed"] | (each["boardings_change"].abs() >= 0.5)  # noqa: PLR2004 (half a person)
    kept = each[moved]

    stations = pd.concat(
        [
            before.groupby("from_station")["boardings"].sum().rename("boardings_before"),
            after.groupby("from_station")["boardings"].sum().rename("boardings_after"),
        ],
        axis=1,
    ).fillna(0.0)
    stations["boardings_change"] = stations["boardings_after"] - stations["boardings_before"]
    stations = stations.rename_axis("station").reset_index()
    return {
        "trains": kept.sort_values("boardings_change").reset_index(drop=True),
        "segments": segments[segments["train_uid"].isin(kept["train_uid"])].reset_index(drop=True),
        "stations": stations[stations["boardings_change"].abs() >= 0.05].reset_index(drop=True),  # noqa: PLR2004
    }


# Paths


def _check_paths(
    config: ChangesConfig,
    after: pd.DataFrame,
    weekday: str,
    checker: paths.CandidatePaths,
) -> pd.DataFrame:
    """Check each changed or added train, clear of the others as changed."""
    working = checker.checker.timetable
    edited = {change.train_uid for change in config.changes}
    number = WEEKDAYS.index(weekday)
    own = collections.defaultdict(list)
    for n, train in enumerate(working.trains):
        if train.uid in edited and train.runs(number):
            own[train.uid].append(n)
    gone = {n for numbers in own.values() for n in numbers}
    cuts = {change.train_uid for change in config.changes if change.kind == "cut"}
    names = sorted(edited - cuts) + [
        service.name for service in config.services if weekday in service.days_of_week
    ]
    traction = {service.name: service.traction_code for service in config.services}
    rows = []
    for name in names:
        train = filter_services(
            after[after["train_uid"] == name], config.reference_timestamp, [number]
        )
        if train.empty:
            continue
        listed = _read_calls(train.drop_duplicates(["seq"]))
        times = []
        for call in listed:
            arrive = call.arrive if call.arrive is not None else call.depart
            depart = call.depart if call.depart is not None else call.arrive
            times.append((arrive or 0.0, depart or 0.0))
        timed = [
            (call.crs, round(2 * arrive), round(2 * depart))
            for call, (arrive, depart) in zip(listed, times, strict=True)
        ]
        ignore = frozenset(gone | set(checker.checker.excluded(own.get(name, []))))
        hints = []
        if own.get(name):
            options = [checker.places.get(call.crs, frozenset()) for call in listed]
            hints = _route_hints(working.trains[own[name][0]].points, options)
        found = checker.find_calls(timed, number, traction.get(name, ""), ignore, hints)
        rows.append(checker.row(found, name, weekday, 0.0, times[0][1]))
        if found.clear:
            checker.occupy(name, weekday, found.points)
    return pd.DataFrame(rows, columns=paths.PATH_COLUMNS)


def _route_hints(points: list, places: list[frozenset[str]]) -> list[list[str]]:
    """Cut a train's own route at its calls, where it passes through every one."""
    at, start = [], 0
    for options in places:
        found = next((i for i in range(start, len(points)) if points[i].tiploc in options), None)
        if found is None:
            return []
        at.append(found)
        start = found + 1
    return [[p.tiploc for p in points[a : b + 1]] for a, b in itertools.pairwise(at)]


@dataclasses.dataclass
class ChangesResult:
    """A change, day by day and over the days asked for."""

    summary: pd.DataFrame
    pairs: pd.DataFrame
    trains: pd.DataFrame
    segments: pd.DataFrame
    stations: pd.DataFrame
    paths: pd.DataFrame


def price_changes(config: ChangesConfig) -> ChangesResult:
    """Load each day before and after the change and compare them."""
    if not config.changes and not config.services:
        logger.warning("changes: nothing is changed, so before and after are the same.")
    logger.info("Loading timetable calls from %s ...", config.timetable_calls_path)
    recodes = resolve_recodes(config.station_recodes)
    calls = recode_columns(read_public_calls(config.timetable_calls_path), recodes, ["crs"])
    inputs = assign.load_inputs(config)
    fares = build_fares_context(config)
    graph = None
    if config.path_check is not None:
        if config.bplan_output_path is None:
            raise ValueError("changes: path_check needs bplan_output_path.")
        graph = BplanTimingGraph(config.bplan_output_path)

    trains = {change.train_uid for change in config.changes} | {s.name for s in config.services}
    found: dict[str, list] = collections.defaultdict(list)
    for weekday in config.days:
        after_calls = apply_changes(calls, config, weekday)
        logger.info("%s: loading the timetable as it is ...", weekday)
        network = Network()
        before = assign.assign_day(calls, weekday, config, **inputs, network=network)
        logger.info("%s: loading the timetable as changed ...", weekday)
        changed = Network(network, config.response())
        after = assign.assign_day(after_calls, weekday, config, **inputs, network=changed)
        table, totals = journeys(config, weekday, network, changed)
        touched = _touched_pairs([calls, after_calls], trains, config, weekday)
        times = [_journey_times(x, config, weekday, touched) for x in (calls, after_calls)]
        table = _with_times(table, touched, times, fares)
        table = _fill_unchanged(table, config, weekday, network)
        totals |= _time_totals(table)
        if "revenue_change_per_day" in table:
            totals["revenue_change"] = float(table["revenue_change_per_day"].sum())
        totals["boardings_change"] = float(after["boardings"].sum() - before["boardings"].sum())
        found["summary"].append({"day": weekday, **totals})
        found["pairs"].append(table.assign(day=weekday))
        for name, frame in _compared(before, after, trains).items():
            found[name].append(frame.assign(day=weekday))
        if config.path_check is not None:
            logger.info("%s: looking for paths ...", weekday)
            checker = paths.CandidatePaths.build(
                config.path_check,
                calls,
                config.reference_timestamp,
                graph,
                config.bplan_output_path,
            )
            found["paths"].append(_check_paths(config, after_calls, weekday, checker))
    tables = {
        name: pd.concat(found[name], ignore_index=True) if found[name] else pd.DataFrame()
        for name in ("pairs", "trains", "segments", "stations", "paths")
    }
    return ChangesResult(pd.DataFrame(found["summary"]), **tables)


def run_changes(config: ChangesConfig) -> ChangesResult:
    """Price the change and write its tables to ``config.output_path``.

    * ``changes_summary.csv``: each day's journeys, their split, time gained and lost, revenue
    * ``changes_pairs.csv``: who gains and who loses, by station pair
    * ``changes_trains.csv``, ``changes_segments.csv``: loads of every train that moves
    * ``changes_stations.csv``: boardings by station
    * ``changes_paths.csv``, with ``path_check``: each changed train's path
    """
    config.output_path.mkdir(parents=True, exist_ok=True)
    result = price_changes(config)
    for name in ("summary", "pairs", "trains", "segments", "stations", "paths"):
        frame = getattr(result, name)
        if name != "paths" or config.path_check is not None:
            frame.to_csv(config.output_path / f"changes_{name}.csv", index=False)
    logger.info("Changes run complete: %d days priced.", len(config.days))
    return result
