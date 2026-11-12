"""Predict the demand a candidate service captures, leg by leg.

Every ordered pair of a candidate's stops is a leg a passenger could
ride. :func:`predict_demand` prices each one from the period matrices:
a pair already served pivots on its observed demand, a pair gaining its
first direct train takes the share of its existing market the new train
wins, and a pair with no observed demand takes the calibrated gravity
model's prediction (see :mod:`ortla.gravity_model`). Journeys
that change trains, the direction a pair's demand travels and how a slot
divides between its trains come from
:mod:`ortla.network_assignment` and
:mod:`ortla.direction_split`. :func:`estimate_loading` turns
the legs into an on-board load for each segment of the train.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ortla import direction_split, models
from ortla.clock import (
    WEEKDAYS,
    WEEKDAYS_PER_YEAR,
    classify_time_period,
    elapsed,
    hhmm_minutes,
    period_matrix_path,
)
from ortla.fares import (
    FLOW_CSV,
    LOCATIONS_CSV,
    TICKETS_CSV,
    FaresTable,
    build_fares_table,
    median_fare,
)
from ortla.gravity_model import MODEL_FILENAME, GravityModel
from ortla.network_assignment import (
    SlotShareModel,
    TransferDemand,
    detour_limit,
    station_km,
)
from ortla.period_matrices import classify_route_type
from ortla.service_lookup import (
    best_existing_minutes,
    day_services,
    existing_journey,
    od_services,
    prepare_calls,
    train_positions,
)
from ortla.stations import (
    DEFAULT_DESTINATION_GROUPS,
    LONDON_TERMINALS,
    canonical_crs,
    load_table_1410,
    resolve_aliases,
    with_alias_keys,
)
from ortla.track_distance import TrackDistanceCalculator

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Iterable, Sequence

    from ortla.config import (
        CandidateService,
        ProposeConfig,
        RouteTypeConfig,
        ServiceStop,
    )

logger = logging.getLogger(__name__)


# Demand/journey-type columns read from a `loading` command's
# {operator}_loading_analysis.csv, matching LoadingConfig's own
# defaults.
_GRAVITY_DEMAND_COLUMN = "Mean Passengers On"


_GRAVITY_JOURNEY_TYPE_COLUMN = "route_type_pax_off"


_DEFAULT_ALIASES = resolve_aliases(None)


def stop_time(stop: ServiceStop) -> str | None:
    """Return a stop's departure time, or its arrival at a terminus."""
    return stop.departure or stop.arrival


def _candidate_leg_minutes(stop_i: ServiceStop, stop_j: ServiceStop) -> float | None:
    """In-train minutes the candidate offers on one leg, or ``None``.

    Measured departure-to-arrival off the candidate's own public times,
    so it is the journey time a passenger choosing between this service
    and the alternatives would actually compare.
    """
    depart = hhmm_minutes(stop_i.departure or stop_i.arrival or "")
    arrive = hhmm_minutes(stop_j.arrival or stop_j.departure or "")
    if depart is None or arrive is None:
        return None
    minutes = arrive - depart if arrive >= depart else arrive - depart + 24 * 60
    # A zero-minute leg is two times that did not parse into a real
    # journey.
    return float(minutes) if minutes > 0 else None


_DIRECTION_CACHE: dict[str, direction_split.DirectionSplitModel | None] = {}


def _direction_split(config: ProposeConfig) -> direction_split.DirectionSplitModel | None:
    """Return the run's fitted direction split, or None if not set.

    Cached on the model path: `predict_demand` is called once per
    candidate and a hindcast runs hundreds of them, so the JSON is read
    once per process.
    """
    path = getattr(config, "direction_split_model", None)
    if path is None:
        return None
    key = str(path)
    if key not in _DIRECTION_CACHE:
        path = models.resolve(path)
        if not path.exists():
            logger.warning(
                "direction_split_model %s does not exist; demand stays symmetric.", path
            )
            _DIRECTION_CACHE[key] = None
        else:
            model = direction_split.DirectionSplitModel.load(path)
            logger.info(
                "Directional split loaded from %s (fitted on %d cells).", path, model.n_cells
            )
            _DIRECTION_CACHE[key] = model
    return _DIRECTION_CACHE[key]


_SHARE_CACHE: dict[str, SlotShareModel | None] = {}


def _slot_share_model(config: ProposeConfig) -> SlotShareModel | None:
    """Return the run's slot share model, or None for equal shares."""
    path = getattr(config, "slot_share_model", None)
    if path is None:
        return None
    key = f"{path}|{getattr(config, 'slot_share_profile', None)}"
    if key not in _SHARE_CACHE:
        path = models.resolve(path)
        if not path.exists():
            logger.warning("slot_share_model %s does not exist; equal shares.", path)
            _SHARE_CACHE[key] = None
        else:
            model = SlotShareModel.load(path)
            profile = getattr(config, "slot_share_profile", None)
            if profile is not None:
                model = model.with_profile(
                    profile,
                    config.time_periods_cutoffs,
                    getattr(config, "slot_share_profile_strength", 0.75),
                    getattr(config, "slot_share_profile_smoothing", 2.0),
                )
            _SHARE_CACHE[key] = model
    return _SHARE_CACHE[key]


def _share_trains(  # noqa: PLR0913
    model: SlotShareModel,
    listings: pd.DataFrame,
    period: str,
    stops: list[ServiceStop],
    i: int,
    j: int,
    candidate_london: bool,
    added: int,
    aliases: dict[str, str] | None = None,
    filtered: pd.DataFrame | None = None,
    coordinates: pd.DataFrame | None = None,
) -> float:
    """Weighted trains sharing the slot, in units of the candidate.

    ``sum(w_existing) / w_candidate + added``: equal weights give
    ``n + added``, the equal share. Added trains other than the
    candidate are weighted as it is. A model with a network also reads
    each train's place in its run from the day's ``filtered`` calls and
    its distances from ``coordinates``; the candidate joins the slot's
    trains for the gaps and count, as it would in the timetable.
    """
    stop_i, stop_j = stops[i], stops[j]
    in_period = listings[listings["time_period"] == period]
    minutes = _listing_minutes(in_period)
    own = _candidate_leg_minutes(stop_i, stop_j)
    known = np.r_[minutes[~np.isnan(minutes)], [own] if own is not None else []]
    fill = float(known.mean()) if len(known) else 0.0
    times = [hhmm_minutes(str(value or "")) for value in in_period["dep_from_origin"]]
    dep = np.array([np.nan if m is None else m for m in times], dtype=float)
    aliases = _DEFAULT_ALIASES if aliases is None else aliases
    o, d = (
        canonical_crs(stop_i.station, aliases),
        canonical_crs(stop_j.station, aliases),
    )  # demand codes
    log_existing = model.log_weight(
        np.full(len(in_period), period, dtype=object),
        dep,
        in_period["svc_destination_crs"].isin(LONDON_TERMINALS).to_numpy(),
        np.nan_to_num(minutes, nan=fill),
        np.full(len(in_period), o, dtype=object),
        np.full(len(in_period), d, dtype=object),
    )
    own_dep = hhmm_minutes(stop_time(stop_i) or "")
    own_dep_value = np.nan if own_dep is None else own_dep
    log_own = model.log_weight(
        np.array([period], dtype=object),
        np.array([own_dep_value]),
        np.array([candidate_london]),
        np.array([fill if own is None else own]),
        np.array([o], dtype=object),
        np.array([d], dtype=object),
    )[0]
    if model.net is not None:
        origin, destination = stop_i.station.strip().upper(), stop_j.station.strip().upper()
        if filtered is not None:
            where = train_positions(
                filtered, in_period["train_uid"].astype(str).tolist(), origin, destination
            )
        else:
            where = pd.DataFrame(
                {"o_is_origin": True, "d_is_terminus": True, "calls_od": 1, "train_calls": 2},
                index=range(len(in_period)),
            )
        first, last = stops[0].station.strip().upper(), stops[-1].station.strip().upper()
        where.loc[len(where)] = (i == 0, j == len(stops) - 1, j - i, len(stops))
        svc_o = [*in_period["svc_origin_crs"].astype(str), first]
        svc_d = [*in_period["svc_destination_crs"].astype(str), last]
        frame = where.assign(
            o=origin,
            d=destination,
            o_key=o,
            d_key=d,
            svc_origin_crs=svc_o,
            svc_dest_crs=svc_d,
            svc_o_key=[canonical_crs(c, aliases) for c in svc_o],
            svc_d_key=[canonical_crs(c, aliases) for c in svc_d],
            count_period=period,
            dep_minutes=np.r_[dep, own_dep_value],
            minutes=np.r_[np.nan_to_num(minutes, nan=fill), fill if own is None else own],
        )
        g = model.net(model.net.features(frame, "count_period", frame, coordinates))
        log_existing, log_own = log_existing + g[:-1], log_own + g[-1]
    return float(np.exp(log_existing - log_own).sum()) + added


def _calibrate_gravity_ratios(
    loading_analysis_path: pathlib.Path,
    footfall: dict[str, int],
    min_observations: int,
) -> dict[str, float]:
    """Trips per unit of footfall by route-type band, for the gravity estimate.

    ``ratio[band]`` = annual boardings in the band / table 1410 footfall of the stations generating
    them: a footfall-share proxy, not a calibrated gravity model. Bands with fewer than
    ``min_observations`` loading rows are dropped. Boardings are averaged over count periods and
    scaled by weeks per year, so the ratio does not grow with the number of periods in the file.
    """
    if not loading_analysis_path.exists():
        return {}
    data = pd.read_csv(loading_analysis_path, dtype=str)
    required = (_GRAVITY_JOURNEY_TYPE_COLUMN, _GRAVITY_DEMAND_COLUMN)
    if any(col not in data.columns for col in required):
        return {}

    demand = pd.to_numeric(
        data[_GRAVITY_DEMAND_COLUMN].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    stations = data["Departure from CRS"].astype(str).str.strip().str.upper()
    n_periods = max(data["RSP Period"].nunique(), 1) if "RSP Period" in data.columns else 1

    ratios: dict[str, float] = {}
    for band, idx in data.groupby(_GRAVITY_JOURNEY_TYPE_COLUMN).groups.items():
        if band in ("Unknown", "nan") or len(idx) < min_observations:
            continue
        total_demand = demand.loc[idx].sum(skipna=True) / n_periods * WEEKDAYS_PER_YEAR
        total_footfall = sum(footfall.get(s, 0) for s in stations.loc[idx].unique())
        if total_footfall > 0:
            ratios[str(band)] = total_demand / total_footfall
    return ratios


def _load_temporal_factors(
    temporal_factors_path: pathlib.Path,
) -> dict[tuple[str, str, str], float]:
    """Map ``(journey_type, day, period)`` to a temporal factor.

    Read from a `loading` command's
    ``{operator}_temporal_factors.csv`` (see
    :func:`~ortla.period_matrices.compute_temporal_factors`).

    Returns an empty dict, not an error, if the file is missing or lacks
    the expected columns; the gravity model then uses its flat annual
    estimate.
    """
    if not temporal_factors_path.exists():
        return {}
    data = pd.read_csv(temporal_factors_path, dtype=str)
    required = ("journey_type", "Service Week Day", "time_period", "factor")
    if any(col not in data.columns for col in required):
        return {}
    factor = pd.to_numeric(data["factor"], errors="coerce")
    return {
        (journey_type, day, period): f
        for journey_type, day, period, f in zip(
            data["journey_type"],
            data["Service Week Day"],
            data["time_period"],
            factor,
            strict=True,
        )
        if pd.notna(f)
    }


def _leg_footfall(
    stop_i: ServiceStop, stop_j: ServiceStop, footfall: dict[str, int]
) -> int | None:
    """Footfall driving a leg's gravity estimate.

    Prefers an explicit new-station estimate at either end, since the
    gravity model is usually there for a stop with no footfall data yet.
    Otherwise it takes the smaller of the two known table_1410
    footfalls, because the quieter end usually limits the flow.
    """
    override = stop_j.estimated_annual_footfall or stop_i.estimated_annual_footfall
    if override:
        return override
    known = [
        f
        for f in (
            footfall.get(stop_i.station.strip().upper()),
            footfall.get(stop_j.station.strip().upper()),
        )
        if f
    ]
    return min(known) if known else None


class GravityContext:
    """Precomputed inputs for the gravity/catchment demand model.

    ``temporal_factors``, when available, rescales the flat annual
    estimate down to the day and time period being evaluated (see
    :func:`_load_temporal_factors`). Without it, every day and period a
    leg is checked for gets the same annual figure.
    """

    def __init__(
        self,
        footfall: dict[str, int],
        ratios: dict[str, float],
        calc: TrackDistanceCalculator,
        route_type: RouteTypeConfig,
        temporal_factors: dict[tuple[str, str, str], float] | None = None,
        model: GravityModel | None = None,
    ) -> None:
        self.footfall = footfall
        self.ratios = ratios
        self.calc = calc
        self.route_type = route_type
        self.temporal_factors = temporal_factors or {}
        self.model = model

    def _slot_factor(self, band: str, day: str, period: str) -> float | None:
        """Return a journey type's share of annual demand in a slot.

        Falls back to the band's mean slot share when that exact slot
        was never observed (so an annual figure is never mistaken for a
        slot figure).
        """
        factor = self.temporal_factors.get((band, day, period))
        if factor is not None:
            return factor
        band_factors = [f for (b, _, _), f in self.temporal_factors.items() if b == band]
        return float(np.mean(band_factors)) if band_factors else None

    def synthetic_slot_trips(
        self,
        stop_i: ServiceStop,
        stop_j: ServiceStop,
        day: str,
        period: str,
        trains_per_day: float,
    ) -> tuple[float | None, str | None]:
        """Gravity annual trips in one slot for a pair with no ODM cell.

        Needs footfall at *both* ends (table 1410 or an explicit
        ``estimated_annual_footfall``) and a track route.
        """
        if self.model is None:
            return None, None
        origin = stop_i.station.strip().upper()
        destination = stop_j.station.strip().upper()
        try:
            miles = self.calc.calculate_distance(origin, destination, unit="miles", plausible=True)
        except ValueError:
            return None, None
        band = classify_route_type(miles, self.route_type)
        f_i = stop_i.estimated_annual_footfall or self.footfall.get(origin)
        f_j = stop_j.estimated_annual_footfall or self.footfall.get(destination)
        factor = self._slot_factor(band, day, period)
        if not f_i or not f_j or factor is None:
            return None, band
        return self.model.predict(f_i, f_j, miles, trains_per_day) * factor, band

    def predict(
        self, stop_i: ServiceStop, stop_j: ServiceStop, day: str, period: str
    ) -> tuple[float | None, str | None]:
        """Estimate annual trips captured in this day/period slot.

        Returns the predicted trips and the route-type band used (or
        ``(None, band)``). Without ``temporal_factors``, or when this
        leg's exact ``(band, day, period)`` has no observations in them,
        this is the leg's flat annual estimate (``footfall * ratio``);
        otherwise that estimate is scaled by the band's share of annual
        demand falling in ``(day, period)``, so a peak-time leg and an
        off-peak leg of the same band no longer get identical figures.
        """
        if self.model is not None:
            # Unreachable with either demand model as configured today:
            # a calibrated run is caught by `_calibrated_leg` (which
            # passes the real post-change frequency), and a legacy run
            # never carries a fitted `model`. The 1.0 is a placeholder,
            # not a modelling choice: it zeroes the `b4 ln f` term. Any
            # new demand model that reaches here must pass its own
            # trains/day instead.
            return self.synthetic_slot_trips(stop_i, stop_j, day, period, 1.0)
        origin = stop_i.station.strip().upper()
        destination = stop_j.station.strip().upper()
        try:
            distance = self.calc.calculate_distance(
                origin, destination, unit="miles", plausible=True
            )
        except ValueError:
            return None, None

        band = classify_route_type(distance, self.route_type)
        ratio = self.ratios.get(band)
        footfall = _leg_footfall(stop_i, stop_j, self.footfall)
        if ratio is None or not footfall:
            return None, band
        annual_total = footfall * ratio
        # A missing (band, day, period) key means that slot was never
        # observed: `compute_temporal_factors` only writes rows for
        # combinations it saw. Treating it as 0.0 would zero the
        # estimate for sparse slots, and those are most common at the
        # new or thinly counted stations this model is for. The leg
        # keeps its flat annual estimate, as it does when no temporal
        # factors are configured.
        factor = self.temporal_factors.get((band, day, period))
        if factor is not None:
            annual_total *= factor
        return annual_total, band


class FaresContext:
    """Average fare per journey per O-D pair, for revenue estimates.

    Built once from a cleaned fares feed (see
    :func:`ortla.fares.build_fares_table`); each leg's
    predicted trips are priced at its pair's median adult fare (see
    :func:`ortla.fares.median_fare`) times ``yield_factor``.
    Results are memoised per pair since the same O-D leg is priced once
    per day the candidate runs.
    """

    def __init__(self, table: FaresTable, yield_factor: float = 1.0) -> None:
        self._table = table
        self._yield_factor = yield_factor
        self._cache: dict[tuple[str, str], float | None] = {}

    def fare(self, origin: str, destination: str) -> float | None:
        """Average fare per journey (GBP) for a pair, or ``None``."""
        key = (origin, destination)
        if key not in self._cache:
            median = median_fare(self._table, origin, destination)
            self._cache[key] = None if median is None else round(median * self._yield_factor, 2)
        return self._cache[key]


_MATRIX_COLUMN_CACHE: dict[tuple[pathlib.Path, str], pd.Series | None] = {}


_MATRIX_STATION_CACHE: dict[pathlib.Path, frozenset[str]] = {}


_MATRIX_EXISTS: dict[pathlib.Path, bool] = {}


_REPORTED_MATRIX_GAPS: set[tuple[pathlib.Path, str]] = set()


def _matrix_exists(path: pathlib.Path) -> bool:
    """Cache ``path.exists()``; hindcasts ask it millions of times."""
    if path not in _MATRIX_EXISTS:
        _MATRIX_EXISTS[path] = path.exists()
    return _MATRIX_EXISTS[path]


def prefetch_matrix_columns(matrices_dir: pathlib.Path, codes: Iterable[str]) -> None:
    """Read the wanted columns of every period matrix, one parse each.

    :func:`matrix_cell` otherwise reads one destination column at a
    time, and each read parses the whole ~50 MB file: about 900 parses
    and half the run time of a small hindcast. Filling its cache up
    front with the same columns, parsed the same way, leaves every
    lookup unchanged. A code the file has no column for is cached as
    missing, as a failed read would be.
    """
    wanted = {str(code).strip().upper() for code in codes}
    for path in sorted(matrices_dir.glob("odm_*.csv")):
        header = [str(c) for c in pd.read_csv(path, nrows=0).columns]
        present = [c for c in header if c in wanted and (path, c) not in _MATRIX_COLUMN_CACHE]
        for code in wanted.difference(header):
            _MATRIX_COLUMN_CACHE.setdefault((path, code), None)
        if not present:
            continue
        frame = pd.read_csv(path, usecols=["origin_tlc", *present]).set_index("origin_tlc")
        for code in present:
            _MATRIX_COLUMN_CACHE[(path, code)] = pd.to_numeric(frame[code], errors="coerce")


def matrix_stations(path: pathlib.Path) -> frozenset[str]:
    """Every station code a period matrix carries, row or column.

    Read once per file (only the header and the ``origin_tlc`` column,
    not the ~50 MB body) and cached. Used to tell "this station is not
    in the matrices at all" apart from "this pair happens to be empty".
    """
    if path not in _MATRIX_STATION_CACHE:
        codes: set[str] = set()
        if path.exists():
            header = pd.read_csv(path, nrows=0)
            codes |= {str(c).strip().upper() for c in header.columns if c != "origin_tlc"}
            origins = pd.read_csv(path, usecols=["origin_tlc"])["origin_tlc"].dropna()
            codes |= set(origins.astype(str).str.strip().str.upper())
        # Not "NAN": that is Nantwich.
        _MATRIX_STATION_CACHE[path] = frozenset(codes - {""})
    return _MATRIX_STATION_CACHE[path]


# Why a leg has no observed demand to pivot on. Only "observed" and
# "observed_zero" mean the matrices were asked and answered; the rest
# are join failures, which must not be read as a real zero.
MATRIX_STATUSES = (
    "observed",  # a real, non-zero annual figure
    "observed_zero",  # the cell is there and is zero
    "no_matrix",  # the day/period file does not exist
    "destination_missing",  # no column for the destination
    "origin_missing",  # no row for the origin
    "cell_suppressed",  # cell present but not a number (the "!" fill value)
)


def matrix_cell(path: pathlib.Path, origin: str, destination: str) -> tuple[float | None, str]:
    """Return a period-matrix cell and its status.

    National period matrices are ~50 MB each and :func:`predict_demand`
    asks for the same destination columns for every leg and day, so the
    column is kept in memory (origin-indexed) after its first read.

    The status (one of :data:`MATRIX_STATUSES`) matters as much as the
    value. A missing destination column raises ``ValueError`` out of
    ``usecols``, and a bare ``None`` would make an absent station look
    like an empty cell, sending the leg to the synthetic estimate
    without a word. A station missing from the matrices is a broken join
    and is logged once per (matrix, station).
    """
    key = (path, destination)
    if key not in _MATRIX_COLUMN_CACHE:
        column = None
        if _matrix_exists(path):
            try:
                df = pd.read_csv(path, usecols=["origin_tlc", destination])
                column = pd.to_numeric(df.set_index("origin_tlc")[destination], errors="coerce")
            except ValueError:
                column = None
        _MATRIX_COLUMN_CACHE[key] = column
    column = _MATRIX_COLUMN_CACHE[key]

    if not _matrix_exists(path):
        return None, "no_matrix"
    if column is None:
        _warn_matrix_gap(path, destination, "destination")
        return None, "destination_missing"
    if origin not in column.index:
        if origin not in matrix_stations(path):
            _warn_matrix_gap(path, origin, "origin")
        return None, "origin_missing"
    value = column.loc[origin]
    if pd.isna(value):
        return None, "cell_suppressed"
    value = float(value)
    return value, "observed" if value > 0 else "observed_zero"


def _warn_matrix_gap(path: pathlib.Path, code: str, end: str) -> None:
    """Log a station the matrices lack, once per matrix and station."""
    key = (path, code)
    if key in _REPORTED_MATRIX_GAPS:
        return
    _REPORTED_MATRIX_GAPS.add(key)
    logger.warning(
        "%s has no %s entry for %s: every leg %s it is treated as unobserved and "
        "falls through to the synthetic estimate. If the demand is really there "
        "under another CRS code, add it to `station_aliases`.",
        path.name,
        end,
        code,
        "from" if end == "origin" else "into",
    )


_DEMAND_COLUMNS = [
    "candidate_name",
    "day",
    "period",
    "from_station",
    "to_station",
    "leg_stop_i",
    "leg_stop_j",
    "existing_annual_trips",
    "existing_transfer_trips",
    "existing_demand_lookup",
    "existing_frequency",
    "predicted_annual_trips_captured",
    "predicted_annual_trips_generated",
    "demand_model_used",
    "average_fare_gbp",
    "predicted_annual_revenue_gbp",
]


_MATRIX_ATTRACTION_CACHE: dict[tuple[pathlib.Path, frozenset[str]], dict[str, float]] = {}


def _matrix_attraction(path: pathlib.Path, codes: frozenset[str]) -> dict[str, float]:
    """Trips arriving at each of ``codes`` in a period matrix.

    The destination-size term of the choice model below: how big a draw
    each competing terminal is in its own right, which is what tells us
    what a new service into one of them is worth when its own O-D cell
    is empty.
    """
    key = (path, codes)
    if key not in _MATRIX_ATTRACTION_CACHE:
        totals: dict[str, float] = {}
        if path.exists():
            present = [c for c in pd.read_csv(path, nrows=0).columns if c in codes]
            if present:
                frame = pd.read_csv(path, usecols=present)
                for code in present:
                    total = pd.to_numeric(frame[code], errors="coerce").sum()
                    if total > 0:
                        totals[code] = float(total)
        _MATRIX_ATTRACTION_CACHE[key] = totals
    return _MATRIX_ATTRACTION_CACHE[key]


def _logit_share(minutes_a: float, minutes_b: float, theta: float) -> float:
    """Binary logit share of option A over B on generalised minutes."""
    # Differenced before exponentiating so a long journey cannot
    # overflow.
    return 1.0 / (1.0 + np.exp(-theta * (minutes_b - minutes_a)))


@dataclass(frozen=True)
class ChoiceTerms:
    """What competing destinations and routes do to one leg's base.

    ``substituted_trips`` is the leg's O-D demand after passengers are
    allowed to switch from the other destinations in its group, and
    replaces the raw matrix cell as the base the frequency pivot works
    on. ``direct_share`` is the share of that demand riding the new
    direct service rather than staying on the interchange journey it
    replaces, and stops a brand-new direct link being credited with
    every trip on the pair.
    """

    substituted_trips: float | None = None
    direct_share: float | None = None
    group_trips: float | None = None
    minutes_before: float | None = None
    minutes_after: float | None = None


class SubstitutionContext:
    """Competing destinations and the logit that splits demand.

    The split is an incremental (pivot-point) logit: observed shares are
    taken as given and only the destination whose service actually
    changed moves, so the group's total demand is conserved and nothing
    is generated by substitution alone. A new link into a city draws on
    that total (the existing Victoria and London Bridge passengers), not
    on the near-empty O-D cell for a service that did not exist yet.
    """

    def __init__(
        self,
        aliases: dict[str, str],
        groups: dict[str, list[str]],
        theta: float,
        interchange_penalty: float,
    ) -> None:
        self.aliases = aliases
        self.theta = theta
        self.interchange_penalty = interchange_penalty
        self.group_of: dict[str, frozenset[str]] = {}
        for members in groups.values():
            codes = frozenset(canonical_crs(code, aliases) for code in members)
            for code in codes:
                self.group_of[code] = codes

    def terms(
        self,
        matrix_path: pathlib.Path,
        filtered: pd.DataFrame,
        origin: str,
        destination: str,
        observed_trips: float,
        minutes_after: float | None,
        *,
        timetable_origin: str | None = None,
        timetable_destination: str | None = None,
    ) -> ChoiceTerms:
        """Substituted demand base and direct-route share for one leg.

        ``origin``/``destination`` are the codes the demand data holds
        the pair under; the timetable searched for the existing journey
        uses the codes it books the service under, which for an aliased
        station is the other one (see :mod:`ortla.stations`).
        Looking the alternative journey up under the canonical code
        finds no service at all and leaves the leg with no comparison to
        make.

        ``minutes_after`` is the candidate's own journey time for the
        leg; with no time to compare against, neither the destination
        split nor the route split has anything to work on and the leg is
        left on its raw cell.
        """
        if minutes_after is None:
            return ChoiceTerms()
        minutes_before, was_direct = existing_journey(
            filtered,
            timetable_origin or origin,
            timetable_destination or destination,
            self.interchange_penalty,
        )
        direct_share = (
            None
            if minutes_before is None
            else _logit_share(minutes_after, minutes_before, self.theta)
        )

        group = self.group_of.get(destination)
        if group is None or len(group) < 2:
            return ChoiceTerms(
                direct_share=direct_share,
                minutes_before=minutes_before,
                minutes_after=minutes_after,
            )

        group_trips = 0.0
        for member in group:
            trips, _status = matrix_cell(matrix_path, origin, member)
            group_trips += trips or 0.0
        if group_trips <= 0 or minutes_before is None:
            return ChoiceTerms(
                direct_share=direct_share,
                group_trips=group_trips or None,
                minutes_before=minutes_before,
                minutes_after=minutes_after,
            )

        # Pivot-point logit: only this destination's generalised time
        # changed, so every other member's exponent cancels and the
        # group total holds. The cell is one of the group's own, so its
        # share cannot exceed one; a figure that says otherwise is not a
        # raw cell, and letting it through would put share * pull + (1 -
        # share) near or below zero.
        share_before = min(observed_trips / group_trips, 1.0)
        # Adding a train can make the destination's best journey faster
        # or leave it as it was, never slower: the quicker existing
        # journey still runs. Pricing a slower candidate at its own time
        # would shrink every stopping train's London demand: by 14%
        # across Thameslink's London legs in the Brent Cross West
        # hindcast, where the calls-only strip leaves each train in the
        # timetable it is compared with, so the pull could only fall.
        best_after = min(minutes_after, minutes_before)
        pull = float(np.exp(-self.theta * (best_after - minutes_before)))
        share_after = share_before * pull / (share_before * pull + (1.0 - share_before))

        if not was_direct:
            # The pivot preserves observed shares, which is right for a
            # destination people already choose and useless for one they
            # cannot: with no direct service the cell is near zero
            # because the service did not exist, not because nobody
            # wants to go there, and pivoting a share of ~0 leaves it ~0
            # however good the new journey is. Fall back on what the
            # market does (terminal size against generalised journey
            # time) and take whichever is larger.
            share_after = max(
                share_after,
                self._market_share(
                    matrix_path,
                    filtered,
                    timetable_origin or origin,
                    destination,
                    group,
                    minutes_after,
                ),
            )

        return ChoiceTerms(
            substituted_trips=group_trips * share_after,
            direct_share=direct_share,
            group_trips=group_trips,
            minutes_before=minutes_before,
            minutes_after=minutes_after,
        )

    def _market_share(
        self,
        matrix_path: pathlib.Path,
        filtered: pd.DataFrame,
        timetable_origin: str,
        destination: str,
        group: frozenset[str],
        minutes_after: float,
    ) -> float:
        """Destination's share of its group on size and time alone.

        A plain multinomial logit over ``ln(attraction) - theta * GJT``:
        the same size-and-impedance trade-off the gravity model makes,
        applied across the competing destinations instead of across
        pairs. It uses no observed share for the origin on purpose: it
        is only reached for a destination whose observed share means
        nothing.
        """
        attraction = _matrix_attraction(matrix_path, group)
        if destination not in attraction:
            return 0.0
        utilities: dict[str, float] = {}
        for member, size in attraction.items():
            minutes = (
                minutes_after
                if member == destination
                else best_existing_minutes(
                    filtered, timetable_origin, member, self.interchange_penalty
                )
            )
            if minutes is not None:
                utilities[member] = float(np.log(size)) - self.theta * minutes
        if destination not in utilities:
            return 0.0
        top = max(utilities.values())
        weights = {m: float(np.exp(u - top)) for m, u in utilities.items()}
        return weights[destination] / sum(weights.values())


def _build_substitution_context(config: ProposeConfig) -> SubstitutionContext:
    """Build the destination substitution context from ``config``."""
    aliases = resolve_aliases(config.station_aliases)
    groups = (
        DEFAULT_DESTINATION_GROUPS
        if config.destination_groups is None
        else config.destination_groups
    )
    return SubstitutionContext(
        aliases, groups, config.gjt_logit_theta, config.interchange_penalty_minutes
    )


def _trains_after(
    existing_od: pd.DataFrame,
    candidate_name: str,
    existing_forward: int,
    existing_reverse: int,
    added_forward: int,
    added_reverse: int,
) -> tuple[int, int]:
    """Trains each way in a leg's slot after the change, for the split.

    A proposed train, or one a hindcast stripped whole, is not in the
    timetable it is compared with, so the trains being added are counted
    on top: the candidates this way (``added_forward``, the candidate
    among them) and those running the other way (``added_reverse``). In
    a calls-only strip the candidate is still in the timetable (only its
    calls at the stripped station are gone), so it is already one of
    ``existing_forward`` and its reverse-running fellows are already in
    ``existing_reverse``. Adding them again would count each candidate
    twice: every leg would tilt towards its own direction, the two
    directions' shares would sum to more than one, and the split would
    create demand (about 14% on a London leg where a third of the trains
    were candidates).
    """
    in_timetable = not existing_od.empty and candidate_name in set(existing_od["train_uid"])
    if in_timetable:
        return existing_forward, existing_reverse
    return existing_forward + added_forward, existing_reverse + added_reverse


def _on_split_direction(
    choice: ChoiceTerms, raw_trips: float | None, split_trips: float | None
) -> ChoiceTerms:
    """Put a substituted demand base on the leg's direction.

    :meth:`SubstitutionContext.terms` works on raw, direction-symmetric
    cells, because the group it compares the pair with is made of raw
    cells. The direction split then scales the pair's own cell from
    ``raw_trips`` to ``split_trips``; this applies the same factor to
    the substituted base, so the split and substitution compose instead
    of the split's figure being read as a share of an unsplit group.
    """
    if choice.substituted_trips is None or not raw_trips or split_trips == raw_trips:
        return choice
    return dataclasses.replace(
        choice, substituted_trips=choice.substituted_trips * (split_trips or 0.0) / raw_trips
    )


def calibrated_leg(
    gravity: GravityContext,
    stop_i: ServiceStop,
    stop_j: ServiceStop,
    day: str,
    period: str,
    *,
    existing_trips: float,
    pair_trips: float | None = None,
    trains_in_period: int,
    trains_per_day: int,
    added_in_period: int,
    added_per_day: int,
    choice: ChoiceTerms | None = None,
    apply_dilution: bool = True,
    pivot_elasticity: float | None = None,
    cell_is_after: bool = False,
    new_link_floor: Literal["synthetic", "median", "none"] = "median",
    share_trains: float | None = None,
    new_link_uplift: float = 0.0,
    new_link_min_trains: int = 4,
    step_share: float = 1.0,
) -> tuple[float | None, float | None, str]:
    """Gravity demand in one leg slot: ``(captured, generated, model)``.

    The slot's demand after the change is (see
    :mod:`ortla.gravity_model`):

    * ``gravity_pivot``, a pair already directly served: its demand base
      x ``(trains_per_day + added_per_day) / trains_per_day) ** e``,
      where ``e`` is ``pivot_elasticity``, or ``b4`` when that is None;
    * ``gravity_new_link``, the first direct trains on a pair people
      already travel by some slower route: the share of that demand the
      new service wins on generalised journey time, or the served-pair
      prediction if larger;
    * ``gravity_synthetic``, no observed demand: the served-pair
      prediction.

    The base is the raw matrix cell unless ``choice`` carries a
    substituted figure, in which case it is the demand the pair holds
    once passengers are allowed to switch to it from competing
    destinations (see :class:`SubstitutionContext`); the label then
    gains a ``_substituted`` suffix. Substitution moves trips between
    destinations without creating any, so generated demand is measured
    against the base rather than against the raw cell.

    ``apply_dilution`` controls the frequency response. Demand grows
    with the pivot's exponent (below 1) while the trains it is shared
    over grow linearly, so per-train load falls as trains are added.
    That is right when forecasting a new service and wrong for a
    hindcast re-inserting a train that already ran and is about to be
    compared with its own observed load. With it off, demand grows in
    proportion to frequency and per-train load is held.

    The new slot total is shared equally across every train in the
    period, so each added train captures ``new / (trains_in_period +
    added_in_period)``. Generated demand is the growth over the base on
    that same per-train basis: the ``new - old`` trips new to rail ride
    the same trains as the rest, so they are divided by the same
    denominator, not by the added trains alone.

    ``cell_is_after`` is for a hindcast whose matrix was observed with
    the stripped trains running
    (``HindcastConfig.matrix_includes_stripped``). The cell is then
    already the demand after the change, including any growth,
    substitution or new-link response the trains caused, so an observed
    cell is shared equally over the trains after (label
    ``observed_share``), as the national assignment shares it. Without
    this the strip removes the train but not its passengers, and the
    pivot hands it ``cell / (n - 1)`` instead of ``cell / n`` (case 20).
    Generated demand cannot be told apart from the cell and is reported
    as zero.

    ``pair_trips`` is the pair's demand before choice changers leave it
    (``existing_trips`` is after). A pair with any takes the new-link
    branch, never the synthetic one, and when the new-link floor binds its
    excess over the time share is abstracted from ``pair_trips`` first;
    only the part above it, or the uplift if larger, is generated.

    ``share_trains`` replaces ``trains_in_period + added_in_period`` as
    the divisor when a slot share model
    (:class:`~ortla.network_assignment.SlotShareModel`)
    weights the slot's trains (``_share_trains``); None is the equal
    share.

    ``new_link_floor`` is the floor on a new link's demand:
    ``synthetic`` (the served-pair prediction, a smeared mean),
    ``median`` (that prediction divided by the smearing factor) or
    ``none``. Taking the larger of an observed figure and a noisy
    estimate selects the estimate's upward errors: on case 20's served
    pairs the smeared floor won 77% of the time at a median 2.95x the
    cell.

    ``new_link_uplift`` is the log growth of a pair's whole market when
    it gains a direct link of at least ``new_link_min_trains`` trains a
    day (case 24: 0.28 from pairs losing one, 0.45 from pairs gaining
    one; a token service of 1-3 a day moved demand 0-8%). It is scaled
    by the direct share, and the growth rides the new trains as
    generated demand; the share moved off the old route is abstracted.
    ``step_share`` is the part of the base that is the pair's own
    market: journeys routed through the leg on the way somewhere else
    gain no direct link, so they take no step.
    """
    model = gravity.model
    if model is None:  # guarded by the caller
        return None, None, "insufficient_data"
    if cell_is_after and existing_trips > 0:
        shared_over = share_trains or trains_in_period + added_in_period
        return existing_trips / shared_over, 0.0, "observed_share"
    after = trains_per_day + added_per_day
    substituted = choice is not None and choice.substituted_trips is not None
    base = choice.substituted_trips if substituted else existing_trips
    # What abstraction can draw on: the pair's demand before any of it
    # changes trains by choice.
    gross = max(base, pair_trips or 0.0)

    if base > 0 and trains_in_period > 0:
        # Demand is shared over the trains in the *period*, so when
        # dilution is suppressed it is the period's frequencies that
        # have to grow in step for per-train load to hold. Using the
        # day's would credit the added train with the whole day's uplift
        # in a period it barely serves.
        ratio = (
            model.frequency_ratio(trains_per_day, after, pivot_elasticity)
            if apply_dilution
            else (trains_in_period + added_in_period) / trains_in_period
        )
        old = base
        new = base * ratio
        label = "gravity_pivot"
    else:
        synthetic, _ = gravity.synthetic_slot_trips(stop_i, stop_j, day, period, after)
        if gross > 0:
            # No direct train in this slot before, so the whole
            # denominator is the trains being added and the cell's
            # demand was reaching the destination some other way.
            # Crediting the added trains with all of it hands every trip
            # on the pair to one train; only the share that prefers it
            # on generalised journey time actually moves.
            no_split = choice is None or choice.direct_share is None
            share = 1.0 if no_split else choice.direct_share
            floor = synthetic or 0.0
            if new_link_floor == "median":
                floor /= model.smearing
            elif new_link_floor == "none":
                floor = 0.0
            # The measured step is a pair gaining its first direct
            # trains of the day; a pair served in other periods is a
            # frequency change instead.
            first_link = trains_per_day == 0 and after >= new_link_min_trains
            uplift = new_link_uplift if first_link else 0.0
            old = base * share  # moved off the old route, not new to rail
            growth = base * step_share * (np.exp(uplift * share) - 1.0)
            new = max(old + growth, floor)
            # When the floor binds, its trips beyond the time share come off
            # the pair's existing demand first; only what exceeds the whole
            # base is new to rail. Otherwise a slow train, whose share
            # collapses, books the floor as generated (case 195, R2).
            old = new - max(growth, new - gross)
            label = "gravity_new_link"
        elif synthetic is None:
            return None, None, "insufficient_data"
        else:
            old = 0.0
            new = synthetic
            label = "gravity_synthetic"

    # Both figures are per train in the period. The slot's `new` trips
    # ride `trains_in_period + added_in_period` trains, and the
    # `new - old` of them that are new to rail are spread over the same
    # trains. Dividing the growth by the added trains alone would report
    # it as (n + k) / k times larger than its share of a train's load.
    # The two denominators coincide on the new-link and synthetic
    # branches, where `trains_in_period` is zero, so only the pivot
    # branch moves.
    trains_after = share_trains or trains_in_period + added_in_period
    captured = new / trains_after
    generated = max(0.0, new - old) / trains_after
    return captured, generated, f"{label}_substituted" if substituted else label


def _synthetic_split_factor(
    direction: direction_split.DirectionSplitModel,
    gravity: GravityContext,
    filtered: pd.DataFrame,
    stop_i: ServiceStop,
    stop_j: ServiceStop,
    day: str,
    period: str,
    cutoffs: dict[str, str],
    *,
    origin_key: str | None = None,
    destination_key: str | None = None,
    trains_forward: int,
    trains_per_day: int,
    added_trains: Callable[[str, str, str, str], int] | None,
    footfall_origin: float | None,
    footfall_destination: float | None,
) -> float:
    """Multiplier putting a ``gravity_synthetic`` leg on its direction.

    The gravity model is near-symmetric, so a new station's legs come
    out the same size both ways in every period, with as many alighting
    as boarding in the morning peak at a commuter station. This splits
    the pair's combined synthetic demand (forward + reverse) with the
    fitted directional model, as the observed legs are split, and
    returns ``split / forward``. The pair's combined demand is
    conserved, so a station's total is unchanged.

    Reverse trains count the other trains being added as well as the
    remaining timetable's: in a calls-only strip every train serving the
    new station is a candidate, so the remaining timetable alone shows
    none running the other way and the split would never fire. Left at 1
    when either direction has no train in the slot, as the observed
    split is.
    """
    origin = stop_i.station.strip().upper()
    destination = stop_j.station.strip().upper()
    reverse_od = od_services(filtered, destination, origin, cutoffs)
    reverse_in_period = (
        int((reverse_od["time_period"] == period).sum()) if not reverse_od.empty else 0
    )
    reverse_per_day = len(reverse_od)
    if added_trains is not None:
        reverse_in_period += added_trains(day, period, destination, origin)
        reverse_per_day += added_trains(day, "day", destination, origin)
    if trains_forward <= 0 or reverse_in_period <= 0:
        return 1.0
    forward, _ = gravity.synthetic_slot_trips(stop_i, stop_j, day, period, trains_per_day)
    reverse, _ = gravity.synthetic_slot_trips(stop_j, stop_i, day, period, reverse_per_day)
    if not forward or reverse is None:
        return 1.0
    share = direction.share(
        period,
        footfall_origin,
        footfall_destination,
        trains_forward,
        reverse_in_period,
        origin=origin_key,
        destination=destination_key,
    )
    return share * (forward + reverse) / forward


_COORDINATE_CACHE: dict[str, pd.DataFrame] = {}


def station_coordinates(config: ProposeConfig) -> pd.DataFrame | None:
    """Station ``lat``/``lon`` by CRS, or None if no list is set."""
    distance = getattr(config, "distance", None)
    stations = distance.stations if distance is not None else None
    if stations is None or not stations.path.exists():
        return None
    key = str(stations.path)
    if key not in _COORDINATE_CACHE:
        frame = pd.read_csv(stations.path)
        coordinates = pd.DataFrame(
            {
                "lat": pd.to_numeric(frame[stations.lat], errors="coerce").to_numpy(),
                "lon": pd.to_numeric(frame[stations.lon], errors="coerce").to_numpy(),
            },
            index=frame[stations.crs].astype(str).str.strip().str.upper(),
        )
        _COORDINATE_CACHE[key] = coordinates[~coordinates.index.duplicated()]
    return _COORDINATE_CACHE[key]


_KM_CACHE: dict[tuple[int, str, str], float] = {}


def _pair_km(coordinates: pd.DataFrame, origin: str, destination: str) -> float:
    """:func:`station_km` for one pair, cached for the process."""
    key = (id(coordinates), origin, destination)
    if key not in _KM_CACHE:
        _KM_CACHE[key] = float(
            station_km(
                coordinates,
                np.array([origin], dtype=object),
                np.array([destination], dtype=object),
            )[0]
        )
    return _KM_CACHE[key]


def shortest_legs(stops: Sequence[ServiceStop]) -> dict[tuple[str, str], tuple[int, int]]:
    """Each pair's candidate leg with the fewest calls between.

    A candidate calling at a station twice serves some pairs by two
    legs; it serves each pair once, by the shortest (earliest on a tie),
    as :func:`ortla.network_assignment.one_leg_per_pair` does.
    """
    best: dict[tuple[str, str], tuple[int, int]] = {}
    codes = [stop.station.strip().upper() for stop in stops]
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            if codes[i] == codes[j]:
                continue  # a loop's two calls at one station are no journey
            key = (codes[i], codes[j])
            if key not in best or j - i < best[key][1] - best[key][0]:
                best[key] = (i, j)
    return best


def _listing_minutes(listings: pd.DataFrame) -> np.ndarray:
    """In-train minutes of each listed train, NaN where unknown."""
    out = np.full(len(listings), np.nan)
    for row, (dep, arr) in enumerate(
        zip(listings["dep_from_origin"], listings["arr_at_destination"], strict=True)
    ):
        depart, arrive = hhmm_minutes(str(dep or "")), hhmm_minutes(str(arr or ""))
        if depart is not None and arrive is not None:
            minutes = elapsed(depart, arrive)
            out[row] = minutes if minutes > 0 else np.nan
    return out


def _drop_slow_listings(
    listings: pd.DataFrame, limit: float, period: str | None, candidate_fast: bool
) -> tuple[pd.DataFrame, bool]:
    """Drop trains slower than ``limit`` where a period has a faster.

    ``candidate_fast`` counts the candidate as a fast option in
    ``period``. Returns the kept listings and whether an existing train
    in ``period`` is within the limit. As
    :func:`ortla.network_assignment.drop_detour_legs`.
    """
    if listings.empty:
        return listings, False
    slow = _listing_minutes(listings) > limit
    periods = listings["time_period"].to_numpy(dtype=object)
    fast_periods = set(periods[~slow])
    existing_fast = period in fast_periods
    if candidate_fast and period is not None:
        fast_periods.add(period)
    drop = slow & np.isin(periods, list(fast_periods))
    if drop.any():
        listings = listings[~drop].reset_index(drop=True)
    return listings, existing_fast


def candidate_calls(candidate: CandidateService, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Write the candidate as calls-table rows on ``reference_date``.

    Journeys that change trains are routed on the timetable *after* the
    change (see
    :class:`~ortla.network_assignment.TransferDemand`), so
    the candidate has to be in it: without it Manchester -> Gatwick can
    never change onto a new Birmingham -> Gatwick train.
    """
    stops = candidate.stops
    date = reference_date.strftime("%Y-%m-%d")

    def hhmm(value: str | None) -> str | None:
        return value.replace(":", "") if value else None

    return pd.DataFrame(
        {
            "train_uid": candidate.name,
            "stp_indicator": "P",
            "atoc_code": candidate.operator or "",
            "days_runs": "".join(
                "1" if day in candidate.days_of_week else "0" for day in WEEKDAYS
            ),
            "start_date": date,
            "end_date": date,
            "train_category": "XX",
            "signalling_id": "1Z99",
            "svc_origin_crs": stops[0].station.strip().upper(),
            "svc_origin_dep": hhmm(stop_time(stops[0])),
            "svc_dest_crs": stops[-1].station.strip().upper(),
            "seq": [str(i) for i in range(len(stops))],
            "tiploc": "",
            "crs": [stop.station.strip().upper() for stop in stops],
            "public_arrival": [hhmm(stop.arrival) for stop in stops],
            "public_departure": [hhmm(stop.departure) for stop in stops],
            "is_public_call": "True",
        }
    )


def transfer_demand(
    config: ProposeConfig, calls: pd.DataFrame, reference_date: pd.Timestamp
) -> TransferDemand | None:
    """Return the run's transfer assignment, or None when it is off.

    ``calls`` must be the timetable *after* the change (see
    :class:`~ortla.network_assignment.TransferDemand`): a
    hindcast's full timetable, or a scenario's with the candidate
    inserted.
    """
    if not config.transfer_assignment or config.period_matrices_dir is None:
        return None
    coordinates = station_coordinates(config)
    if coordinates is None:
        logger.warning(
            "No station coordinates (distance.stations), so journeys routed through a"
            " change, and direct legs, are not held to the detour limit."
        )
    return TransferDemand(
        calls,
        reference_date,
        config.time_periods_cutoffs,
        resolve_aliases(config.station_aliases),
        config.period_matrices_dir,
        interchange_penalty=config.interchange_penalty_minutes,
        detour_base_minutes=config.transfer_detour_base_minutes,
        detour_minutes_per_km=config.transfer_detour_minutes_per_km,
        coordinates=coordinates,
        direct_detour_limit=config.direct_detour_limit,
        routing=config.transfer_routing,
        cache_dir=getattr(config, "transfer_cache_dir", None),
        scale=getattr(config, "transfer_scale", 1.0),
        london_scale=getattr(config, "transfer_scale_london", None),
        choice_theta=getattr(config, "transfer_choice_theta", None),
    )


def predict_demand(
    config: ProposeConfig,
    candidate: CandidateService,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    gravity: GravityContext | None = None,
    fares: FaresContext | None = None,
    added_trains: Callable[[str, str, str, str], int] | None = None,
    substitution: SubstitutionContext | None = None,
    transfers: TransferDemand | None = None,
) -> pd.DataFrame:
    """Predict the demand the candidate captures on each leg, one row per leg and running day.

    Every ordered pair of the candidate's stops is a leg, labelled by the day and period of the
    earlier stop's departure.

    * Where the pair has trips in the period matrices and an existing train in the slot, a pivot
      applies: ``new_total = existing * (new_frequency / existing_frequency) ** elasticity``, and
      the candidate takes a frequency share of it.
    * Otherwise, with ``gravity``, a footfall-share gravity estimate applies, scaled by the slot's
      temporal factor. It is meant for new or barely served stops.
    * Otherwise the leg is left as ``insufficient_data`` rather than guessed.

    ``predicted_annual_trips_generated`` is the part new to rail: the growth of the slot's total
    under the pivot, every captured trip under the gravity estimate. Under the calibrated model a
    destination that competes with others (a London terminal) is priced on its whole group's demand
    (:class:`SubstitutionContext`), and a first direct service on a pair already travelled by a
    slower route wins only the share that prefers it on generalised journey time.

    Codes are canonicalised before every lookup in the matrices, table 1410 and the fares feed (see
    :mod:`ortla.stations`); ``from_station`` and ``to_station`` stay as the timetable has them.
    ``existing_demand_lookup`` records what the matrices said, so a station missing from them can
    be told apart from a zero cell.

    ``fares`` prices each leg at its pair's average fare into ``predicted_annual_revenue_gbp``
    (revenue carried, not necessarily new). ``added_trains(day, period, origin, destination)`` is
    how many trains, this one included, join the slot at once (default 1). ``transfers`` adds the
    trips of journeys that change trains to a leg's cell before the direction split, as the
    national assignment does.
    """
    if config.period_matrices_dir is None:
        logger.info("No period_matrices_dir configured; skipping demand prediction.")
        return pd.DataFrame(columns=_DEMAND_COLUMNS)

    calls = prepare_calls(calls)
    aliases = resolve_aliases(config.station_aliases)
    if substitution is None:
        substitution = _build_substitution_context(config)
    direction = _direction_split(config)
    split_legs = getattr(config, "direction_split_legs", "observed")
    split_observed = direction is not None and split_legs in ("observed", "all")
    split_synthetic = direction is not None and split_legs in ("synthetic", "all")
    direction_footfall = gravity.footfall if gravity is not None else {}
    stops = candidate.stops
    shortest = shortest_legs(stops)
    share_model = _slot_share_model(config)
    candidate_london = bool(stops) and stops[-1].station.strip().upper() in LONDON_TERMINALS
    serves_london = bool(stops) and bool(
        {stops[0].station.strip().upper(), stops[-1].station.strip().upper()} & LONDON_TERMINALS
    )
    coordinates = (
        station_coordinates(config) if getattr(config, "direct_detour_limit", True) else None
    )
    rows = []

    for day in candidate.days_of_week:
        filtered = day_services(calls, reference_date, day)

        for i in range(len(stops)):
            time_i = stop_time(stops[i])
            if time_i is None:
                continue
            period = classify_time_period(f"{time_i}:00", config.time_periods_cutoffs)

            for j in range(i + 1, len(stops)):
                # Reported against the timetable's own codes so a leg
                # still matches the operator's loading data; looked up
                # against the codes the demand data holds the same
                # station under.
                origin = stops[i].station.strip().upper()
                destination = stops[j].station.strip().upper()
                if shortest.get((origin, destination)) != (i, j):
                    continue  # the same station, or a pair a shorter leg of this train serves
                origin_key = canonical_crs(origin, aliases)
                destination_key = canonical_crs(destination, aliases)

                matrix_path = period_matrix_path(config.period_matrices_dir, day, period)
                existing_trips, lookup = matrix_cell(matrix_path, origin_key, destination_key)
                # Journeys that change trains ride this leg as well;
                # they join the cell before the slot's demand is shared
                # over its trains.
                transfer = (
                    transfers.trips(day, period, origin_key, destination_key, serves_london)
                    if transfers is not None and existing_trips is not None
                    else 0.0
                )
                # The share of the pair's own journeys that change
                # trains by choice (case 88; 0 unless
                # `transfer_choice_theta` is set) leaves its direct
                # legs.
                changing = (
                    transfers.direct_share(day, period, origin_key, destination_key)
                    if transfers is not None and existing_trips is not None
                    else 0.0
                )
                added_in_period = (
                    max(1, added_trains(day, period, origin, destination)) if added_trains else 1
                )

                existing_od = od_services(
                    filtered, origin, destination, config.time_periods_cutoffs
                )
                # The detour limit on direct legs: a leg far slower than
                # the straight line allows (the long way round a loop)
                # takes no share where its slot has a leg within the
                # limit.
                limit = None
                if coordinates is not None:
                    km = _pair_km(coordinates, origin, destination)
                    if np.isfinite(km):
                        limit = float(
                            detour_limit(
                                km,
                                config.transfer_detour_base_minutes,
                                config.transfer_detour_minutes_per_km,
                            )
                        )
                if limit is not None:
                    minutes = _candidate_leg_minutes(stops[i], stops[j])
                    candidate_slow = minutes is not None and minutes > limit
                    existing_od, existing_fast = _drop_slow_listings(
                        existing_od, limit, period, candidate_fast=not candidate_slow
                    )
                    if candidate_slow and existing_fast:
                        continue
                existing_frequency = (
                    int((existing_od["time_period"] == period).sum())
                    if not existing_od.empty
                    else 0
                )
                # Trains the slot's demand is shared over, weighted by
                # the share model.
                share_trains = (
                    _share_trains(
                        share_model,
                        existing_od,
                        period,
                        stops,
                        i,
                        j,
                        candidate_london,
                        added_in_period,
                        aliases,
                        filtered,
                        station_coordinates(config),
                    )
                    if share_model is not None and existing_frequency > 0
                    else None
                )

                # The ODM's own cell: destination substitution compares
                # it with the other cells of its group, which are raw
                # ODM cells too (see below).
                cell_trips = existing_trips
                # The leg's demand before any direction split, transfers
                # included.
                raw_trips = (
                    existing_trips * (1.0 - changing) + transfer
                    if existing_trips is not None
                    else None
                )
                existing_trips = raw_trips
                direction_factor = 1.0
                # The ODM is exactly direction-symmetric, so without
                # this the pair's two directions hold identical slot
                # demand and the only directional signal left is the
                # frequency divisor below, which points the wrong way.
                # Redistribute the pair's combined demand between its
                # directions instead; the total is conserved.
                if split_observed and existing_trips:
                    reverse_od = od_services(
                        filtered, destination, origin, config.time_periods_cutoffs
                    )
                    if limit is not None:
                        reverse_od, _ = _drop_slow_listings(
                            reverse_od, limit, period, candidate_fast=False
                        )
                    reverse_frequency = (
                        int((reverse_od["time_period"] == period).sum())
                        if not reverse_od.empty
                        else 0
                    )
                    reverse_trips, _ = matrix_cell(matrix_path, destination_key, origin_key)
                    # The split reflects service provision *after* the
                    # change.
                    forward_after, reverse_after = _trains_after(
                        existing_od,
                        candidate.name,
                        existing_frequency,
                        reverse_frequency,
                        added_in_period,
                        added_trains(day, period, destination, origin) if added_trains else 0,
                    )
                    # The factor comes from the pair's own cells and
                    # scales the transfer trips too, as the national
                    # assignment applies it. A pair whose only demand is
                    # routed trips takes the symmetric factor.
                    forward_cell, reverse_cell = (
                        (cell_trips, reverse_trips or 0.0) if cell_trips else (1.0, 1.0)
                    )
                    factor = (
                        direction.split(
                            forward_cell,
                            reverse_cell,
                            period,
                            direction_footfall.get(origin_key),
                            direction_footfall.get(destination_key),
                            forward_after,
                            reverse_after,
                            origin=origin_key,
                            destination=destination_key,
                        )
                        / forward_cell
                    )
                    existing_trips = raw_trips * factor
                    direction_factor = factor
                # The pair's demand before choice changers leave it. They
                # are computed with the candidate in the timetable, so a
                # slow candidate can send nearly all of them off and leave
                # a pair with real demand looking empty (case 196).
                pair_trips = ((cell_trips or 0.0) + transfer) * direction_factor

                generated = None
                # A hindcast whose matrix saw the stripped trains
                # running: an observed cell is already the demand after
                # the change.
                cell_is_after = getattr(config, "matrix_includes_stripped", False)
                if (
                    config.demand_model == "calibrated"
                    and gravity is not None
                    and gravity.model is not None
                    and cell_is_after
                    and existing_trips
                    and existing_trips > 0
                ):
                    # Substitution is a response the cell already holds,
                    # so it is not looked up.
                    captured, generated, model_used = calibrated_leg(
                        gravity,
                        stops[i],
                        stops[j],
                        day,
                        period,
                        existing_trips=existing_trips,
                        trains_in_period=existing_frequency,
                        trains_per_day=len(existing_od),
                        added_in_period=added_in_period,
                        added_per_day=1,
                        cell_is_after=True,
                        share_trains=share_trains,
                    )
                elif (
                    config.demand_model == "calibrated"
                    and gravity is not None
                    and gravity.model is not None
                ):
                    # Substitution reads the other destinations' raw
                    # cells, so it must see this pair's raw cell too:
                    # handed the split figure (up to twice the cell) it
                    # can find a destination holding more than its whole
                    # group, and the pivot logit then divides by nearly
                    # zero (Colchester -> Liverpool Street came out 52
                    # times too big). Its result is put on the leg's
                    # direction afterwards by the same factor the split
                    # applied to the cell.
                    choice = substitution.terms(
                        matrix_path,
                        filtered,
                        origin_key,
                        destination_key,
                        cell_trips or 0.0,
                        _candidate_leg_minutes(stops[i], stops[j]),
                        timetable_origin=origin,
                        timetable_destination=destination,
                    )
                    # Routed trips are riding this leg on the way
                    # somewhere else, so no other destination competes
                    # for them: they join the substituted base as they
                    # are.
                    if transfer and choice.substituted_trips is not None:
                        choice = dataclasses.replace(
                            choice, substituted_trips=choice.substituted_trips + transfer
                        )
                    choice = _on_split_direction(choice, raw_trips, existing_trips)
                    added_per_day = (
                        max(1, added_trains(day, "day", origin, destination))
                        if added_trains
                        else 1
                    )
                    captured, generated, model_used = calibrated_leg(
                        gravity,
                        stops[i],
                        stops[j],
                        day,
                        period,
                        existing_trips=existing_trips or 0.0,
                        pair_trips=pair_trips,
                        trains_in_period=existing_frequency,
                        trains_per_day=len(existing_od),
                        added_in_period=added_in_period,
                        added_per_day=added_per_day,
                        choice=choice,
                        apply_dilution=config.apply_frequency_dilution,
                        pivot_elasticity=config.served_pair_frequency_elasticity,
                        new_link_floor=config.new_link_floor,
                        share_trains=share_trains,
                        new_link_uplift=config.new_link_uplift,
                        new_link_min_trains=config.new_link_min_trains,
                        step_share=(cell_trips or 0.0) / raw_trips if raw_trips else 1.0,
                    )
                    if split_synthetic and captured and model_used == "gravity_synthetic":
                        factor = _synthetic_split_factor(
                            direction,
                            gravity,
                            filtered,
                            stops[i],
                            stops[j],
                            day,
                            period,
                            config.time_periods_cutoffs,
                            origin_key=origin_key,
                            destination_key=destination_key,
                            trains_forward=existing_frequency + added_in_period,
                            trains_per_day=len(existing_od) + added_per_day,
                            added_trains=added_trains,
                            footfall_origin=(
                                stops[i].estimated_annual_footfall
                                or direction_footfall.get(origin_key)
                            ),
                            footfall_destination=(
                                stops[j].estimated_annual_footfall
                                or direction_footfall.get(destination_key)
                            ),
                        )
                        captured *= factor
                        generated = generated * factor if generated is not None else None
                elif existing_trips and existing_trips > 0 and cell_is_after:
                    shared_over = share_trains or existing_frequency + added_in_period
                    captured = existing_trips / shared_over
                    generated = 0.0
                    model_used = "observed_share"
                elif existing_trips and existing_trips > 0 and existing_frequency > 0:
                    new_frequency = existing_frequency + added_in_period
                    new_total = (
                        existing_trips
                        * (new_frequency / existing_frequency) ** config.frequency_elasticity
                    )
                    captured = new_total / (share_trains or new_frequency)
                    # Most captured trips move off existing trains; only
                    # the growth in the slot's total is new to rail.
                    generated = (new_total - existing_trips) / added_in_period
                    model_used = "pivot"
                elif gravity is not None:
                    captured, _band = gravity.predict(stops[i], stops[j], day, period)
                    if captured is not None:
                        captured /= added_in_period
                        generated = captured
                    model_used = "gravity" if captured is not None else "insufficient_data"
                else:
                    captured = None
                    model_used = "insufficient_data"

                fare = fares.fare(origin_key, destination_key) if fares else None
                revenue = (
                    round(captured * fare, 2)
                    if captured is not None and fare is not None
                    else None
                )

                rows.append(
                    {
                        "candidate_name": candidate.name,
                        "day": day,
                        "period": period,
                        "from_station": origin,
                        "to_station": destination,
                        "leg_stop_i": i,
                        "leg_stop_j": j,
                        "existing_annual_trips": existing_trips,
                        "existing_transfer_trips": transfer,
                        "existing_demand_lookup": lookup,
                        "existing_frequency": existing_frequency,
                        "predicted_annual_trips_captured": captured,
                        "predicted_annual_trips_generated": generated,
                        "demand_model_used": model_used,
                        "average_fare_gbp": fare,
                        "predicted_annual_revenue_gbp": revenue,
                    }
                )

    return pd.DataFrame(rows, columns=_DEMAND_COLUMNS)


_LOADING_COLUMNS = [
    "candidate_name",
    "day",
    "from_station",
    "to_station",
    "on_board_typical_day",
    "capacity",
    "voc",
    "incomplete_estimate",
]


def estimate_loading(
    candidate: CandidateService,
    demand: pd.DataFrame,
    config: ProposeConfig,
) -> pd.DataFrame:
    """On-board load and load over capacity for each segment, one row per segment and day.

    A leg from stop i to stop j is on board for every segment between them. Trips are annual, so
    they are divided by :data:`ortla.clock.WEEKDAYS_PER_YEAR` for a typical day before dividing by
    capacity (the candidate's own, else ``config``'s fallback). A segment fed by any leg with no
    estimate is flagged ``incomplete_estimate``: its load is a lower bound.
    """
    capacity = candidate.capacity or config.default_capacity
    stops = candidate.stops
    n_segments = len(stops) - 1
    if n_segments < 1 or demand.empty:
        return pd.DataFrame(columns=_LOADING_COLUMNS)

    rows = []
    for day, day_demand in demand.groupby("day"):
        loads = [0.0] * n_segments
        incomplete = [False] * n_segments
        for _, leg in day_demand.iterrows():
            i, j = int(leg["leg_stop_i"]), int(leg["leg_stop_j"])
            captured = leg["predicted_annual_trips_captured"]
            for segment in range(i, j):
                if pd.isna(captured):
                    incomplete[segment] = True
                else:
                    loads[segment] += captured

        for segment in range(n_segments):
            typical_day_load = loads[segment] / WEEKDAYS_PER_YEAR
            voc = typical_day_load / capacity if capacity else None
            rows.append(
                {
                    "candidate_name": candidate.name,
                    "day": day,
                    "from_station": stops[segment].station.strip().upper(),
                    "to_station": stops[segment + 1].station.strip().upper(),
                    "on_board_typical_day": round(typical_day_load, 2),
                    "capacity": capacity,
                    "voc": round(voc, 4) if voc is not None else None,
                    "incomplete_estimate": incomplete[segment],
                }
            )

    return pd.DataFrame(rows, columns=_LOADING_COLUMNS)


def build_gravity_context(
    config: ProposeConfig, calc: TrackDistanceCalculator | None = None
) -> GravityContext | None:
    """Build the gravity model from ``config``, or ``None`` if unset.

    Requires ``table_1410``, ``distance`` and ``period_matrices_dir`` to
    all be set, and at least one route-type band to be calibrated from
    the sibling ``{operator}_loading_analysis.csv``. Also loads the
    sibling ``{operator}_temporal_factors.csv`` if present, so estimates
    are scaled by day/period (see :class:`GravityContext`); its absence
    is not fatal, just less precise.
    """
    if not (config.table_1410 and config.distance and config.period_matrices_dir):
        return None

    operator = config.period_matrices_dir.name.removesuffix("_period_matrices")
    # Keyed by the timetable's codes as well as table 1410's own, so a
    # leg into a high-speed platform group finds the station's footfall
    # instead of falling out of the model for want of a row (see
    # :mod:`ortla.stations`).
    footfall = with_alias_keys(
        load_table_1410(config.table_1410), resolve_aliases(config.station_aliases)
    )
    model = None
    ratios: dict[str, float] = {}
    if config.demand_model == "calibrated":
        model_path = config.period_matrices_dir.parent / MODEL_FILENAME
        if not model_path.exists():
            logger.warning(
                "No %s beside %s (run `loading` with route_frequency_path); "
                "legs without observed demand will be left insufficient_data.",
                MODEL_FILENAME,
                config.period_matrices_dir,
            )
            return None
        model = GravityModel.load(model_path)
        logger.info(
            "Calibrated gravity model: %d pairs, R^2 %.3f, frequency coefficient %.3f.",
            model.n_pairs,
            model.r_squared,
            model.frequency_coefficient,
        )
        if config.frequency_coefficient_override is not None:
            fitted = model.frequency_coefficient
            model = model.with_frequency_coefficient(config.frequency_coefficient_override)
            logger.warning(
                "Frequency coefficient overridden: b4 %.3f -> %.3f. Every other "
                "coefficient is as fitted, so this is not a refit: the intercept "
                "still carries the fitted b4's contribution.",
                fitted,
                model.frequency_coefficient,
            )
    else:
        loading_analysis_path = (
            config.period_matrices_dir.parent / f"{operator}_loading_analysis.csv"
        )
        ratios = _calibrate_gravity_ratios(
            loading_analysis_path, footfall, config.gravity_min_loading_observations
        )
        if not ratios:
            logger.warning(
                "No gravity ratios could be calibrated from %s; new/thin-data legs "
                "will be left insufficient_data.",
                loading_analysis_path,
            )
            return None
        logger.info("Footfall-share model calibrated for route types: %s", ratios)

    temporal_factors_path = config.period_matrices_dir.parent / f"{operator}_temporal_factors.csv"
    temporal_factors = _load_temporal_factors(temporal_factors_path)
    if temporal_factors:
        logger.info(
            "Gravity/catchment model will scale by day/period temporal factors from %s.",
            temporal_factors_path,
        )
    else:
        logger.warning(
            "No temporal factors found at %s; gravity/catchment estimates will be "
            "flat annual figures, not day/period-specific.",
            temporal_factors_path,
        )

    if calc is None:
        calc = TrackDistanceCalculator(config.distance)
    return GravityContext(footfall, ratios, calc, config.route_type, temporal_factors, model)


def build_fares_context(config: ProposeConfig) -> FaresContext | None:
    """Build the fares pricing context from ``config``, or ``None``.

    Requires ``fares_dir`` to be set and to hold the cleaned fares CSVs
    a `fares` command writes. Without it, demand legs are left unpriced
    (no ``predicted_annual_revenue_gbp``); that is not an error.
    """
    if config.fares_dir is None:
        return None
    missing = [
        name
        for name in (
            FLOW_CSV,
            LOCATIONS_CSV,
            TICKETS_CSV,
        )
        if not (config.fares_dir / name).exists()
    ]
    if missing:
        logger.warning(
            "fares_dir %s is missing %s; demand legs will be left unpriced. Run "
            "the `fares` command into that directory first.",
            config.fares_dir,
            ", ".join(missing),
        )
        return None

    logger.info("Loading cleaned fares feed from %s for revenue pricing ...", config.fares_dir)
    table = build_fares_table(config.fares_dir)
    return FaresContext(table, config.fare_yield_factor)
