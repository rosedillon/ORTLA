"""The trains that run between two stations on a given day.

The ``services`` command lists the timetabled trains for one origin and
destination: it reads the cleaned ``timetable_calls.csv``, finds every
service that makes a public call at the origin and later in its run at
the destination, and reports each train's operator, headcode, departure
from the origin and arrival at the destination.

The service set is filtered as the route frequency analysis filters it
(permanent passenger public schedules active on the reference date), so
the trains listed here reconcile with the ``route_frequency`` counts.

By default a single representative weekday is used, so a train counts
once if it runs that day. Set ``reference_weekdays`` to a list of
weekdays (Monday to Friday, say) to average across them instead: each
train is then weighted by the share of those days it runs, so a
Friday-only train counts 0.2 across Monday to Friday. An optional time
filter narrows the listing to a named period, a clock hour or an
``HH:MM`` range, applied to each train's departure from the origin.

The same lookups serve the feasibility and demand code, which ask for
them tens of thousands of times in a hindcast. :func:`day_services`,
:func:`od_services` and :func:`existing_journey` memoise them per
timetable frame, and :func:`clear_timetable_caches` drops the memos.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ortla.clock import (
    WEEKDAY_INDEX,
    cif_hhmm,
    cif_time_minutes,
    classify_time_period,
    elapsed,
    hhmm_minutes,
)
from ortla.stations import LONDON_TERMINALS, canonical_crs, resolve_aliases
from ortla.timetable import SERVICE_KEY, filter_services, resolve_reference_date

if TYPE_CHECKING:
    from collections.abc import Callable

    from ortla.config import ServiceLookupConfig

logger = logging.getLogger(__name__)


# A time_range filter is a (start, end) pair.
_TIME_RANGE_LEN = 2


# Valid weekday indices (0=Mon .. 6=Sun) for the reference-weekday
# filter.
_MAX_WEEKDAY = 6


# Columns carried through onto each listed train.
_OUTPUT_COLUMNS = [
    "origin",
    "destination",
    "operator",
    "headcode",
    "train_uid",
    "dep_from_origin",
    "arr_at_destination",
    "dep_hour",
    "time_period",
    "days_per_week",
    "weekday_fraction",
    "svc_origin_crs",
    "svc_destination_crs",
]


def _selected_weekdays(config: ServiceLookupConfig) -> list[int]:
    """Return the weekdays to average over.

    ``reference_weekdays`` when set, else the single weekday.
    """
    weekdays = config.reference_weekdays or [config.reference_weekday]
    weekdays = sorted(set(weekdays))
    if not weekdays or any(not 0 <= w <= _MAX_WEEKDAY for w in weekdays):
        raise ValueError(
            f"reference weekday(s) must be integers 0 (Mon) .. 6 (Sun); got {weekdays}."
        )
    return weekdays


def extract_od_services(
    calls: pd.DataFrame,
    origin: str,
    destination: str,
    cutoffs: dict[str, str],
    n_weekdays: int,
) -> pd.DataFrame:
    """Every service calling at ``origin`` then ``destination``, one row per train.

    ``calls`` is the filtered, weekday-weighted public-call subset
    (:func:`ortla.timetable.filter_services`) with numeric ``seq``. Each train's earliest origin
    call pairs with the first destination call after it, so only trains carrying passengers forward
    are listed. A train calling twice at either station (a loop, a reversal) is listed once, by its
    shortest leg, the one :func:`ortla.network_assignment.one_leg_per_pair` keeps. ``n_weekdays``
    turns ``weekday_fraction`` back into days per week. Rows are sorted by departure from the
    origin.
    """
    # Only origin/destination calls matter, which shrinks the group-by
    # hugely.
    pair = calls[calls["crs"].isin({origin, destination})]
    records = []
    for _, group in pair.groupby(SERVICE_KEY, sort=False):
        grp = group.sort_values("seq")
        origin_calls = grp[grp["crs"] == origin]
        dest_calls = grp[grp["crs"] == destination]
        if origin_calls.empty or dest_calls.empty:
            continue
        best = None
        for _, d_call in dest_calls.iterrows():
            before = origin_calls[origin_calls["seq"] < d_call["seq"]]
            if before.empty:
                continue  # destination is only served before the origin here
            o_call = before.iloc[-1]
            span = int((grp["seq"].between(o_call["seq"], d_call["seq"])).sum())
            if best is None or span < best[0]:
                best = (span, o_call, d_call)
        if best is None:
            continue
        _, o, d = best

        dep = cif_hhmm(o["public_departure"])
        dep_minutes = hhmm_minutes(dep)
        fraction = float(o["weekday_fraction"])
        records.append(
            {
                "origin": origin,
                "destination": destination,
                "operator": o["atoc_code"],
                "headcode": o["signalling_id"],
                "train_uid": o["train_uid"],
                "dep_from_origin": dep,
                "arr_at_destination": cif_hhmm(d["public_arrival"]),
                "dep_hour": dep_minutes // 60 if dep_minutes is not None else None,
                "time_period": classify_time_period(f"{dep}:00", cutoffs) if dep else "Unknown",
                "days_per_week": round(fraction * n_weekdays),
                "weekday_fraction": round(fraction, 3),
                "svc_origin_crs": o["svc_origin_crs"],
                "svc_destination_crs": o["svc_dest_crs"],
            }
        )

    services = pd.DataFrame(records, columns=_OUTPUT_COLUMNS)
    if services.empty:
        return services
    services = _collapse_variants(services, n_weekdays)
    return services.sort_values("dep_from_origin", na_position="last").reset_index(drop=True)


def _collapse_variants(services: pd.DataFrame, n_weekdays: int) -> pd.DataFrame:
    """Merge a train's seasonal variants into one weighted row.

    The same train can appear as several permanent schedule records that
    each run on part of the week (e.g. a Mon-Tue variant and a Wed-Fri
    variant). Grouping on the train's identity and departure sums their
    weekday shares back into one row, so the listing shows one train per
    service with its total ``days_per_week`` and the distinct-train
    count reconciles with the single-day view.
    """
    keys = [
        "origin",
        "destination",
        "operator",
        "headcode",
        "dep_from_origin",
        "arr_at_destination",
        "dep_hour",
        "time_period",
        "svc_origin_crs",
        "svc_destination_crs",
    ]
    collapsed = services.groupby(keys, dropna=False, as_index=False).agg(
        train_uid=("train_uid", "first"),
        weekday_fraction=("weekday_fraction", "sum"),
    )
    # Two variants should partition the week, but cap at 1.0 in case any
    # overlap.
    fraction = collapsed["weekday_fraction"].clip(upper=1.0).round(3)
    collapsed = collapsed.assign(
        weekday_fraction=fraction,
        days_per_week=(fraction * n_weekdays).round().astype(int),
    )
    return collapsed[_OUTPUT_COLUMNS]


def _apply_time_filter(
    services: pd.DataFrame, config: ServiceLookupConfig
) -> tuple[pd.DataFrame, str]:
    """Narrow ``services`` by the time filter; return rows and label.

    Precedence: explicit ``time_range`` beats ``hour`` beats ``period``;
    with none set the whole day is returned. The label describes the
    applied filter for logging.
    """
    if services.empty:
        return services, "whole day"

    if config.time_range:
        if len(config.time_range) != _TIME_RANGE_LEN:
            raise ValueError(
                f"time_range must be two 'HH:MM' values (start, end); got {config.time_range!r}."
            )
        start, end = config.time_range
        lo, hi = hhmm_minutes(start), hhmm_minutes(end)
        dep_minutes = services["dep_from_origin"].map(hhmm_minutes)
        if lo <= hi:
            keep = dep_minutes.notna() & (dep_minutes >= lo) & (dep_minutes < hi)
        else:
            keep = dep_minutes.notna() & ((dep_minutes >= lo) | (dep_minutes < hi))
        return services[keep].reset_index(drop=True), f"{start}-{end}"

    if config.hour is not None:
        keep = services["dep_hour"] == config.hour
        return services[keep].reset_index(drop=True), f"the {config.hour:02d}:00 hour"

    if config.period:
        keep = services["time_period"] == config.period
        return services[keep].reset_index(drop=True), f"period {config.period}"

    return services, "whole day"


def _log_summary(
    services: pd.DataFrame,
    config: ServiceLookupConfig,
    origin: str,
    destination: str,
    n_weekdays: int,
) -> None:
    """Log the listed trains plus per-period and per-hour totals.

    With one weekday the totals are plain train counts; with several
    they are the mean trains per weekday (each train weighted by the
    days it runs).
    """
    if services.empty:
        logger.warning("No trains found for %s -> %s with the given filter.", origin, destination)
        return

    averaged = n_weekdays > 1
    columns = [
        "origin",
        "destination",
        "dep_from_origin",
        "arr_at_destination",
        "operator",
        "headcode",
        "time_period",
        "svc_origin_crs",
        "svc_destination_crs",
    ]
    if averaged:
        columns.insert(6, "days_per_week")

    if averaged:
        logger.info(
            "%d distinct %s -> %s trains, mean %.1f per weekday across %d days:\n%s",
            len(services),
            origin,
            destination,
            services["weekday_fraction"].sum(),
            n_weekdays,
            services[columns].to_string(index=False),
        )
    else:
        logger.info(
            "%d trains %s -> %s (typical weekday):\n%s",
            len(services),
            origin,
            destination,
            services[columns].to_string(index=False),
        )

    unit = "mean trains/weekday" if averaged else "trains"
    by_period = services.groupby("time_period")["weekday_fraction"].sum()
    order = [p for p in config.period_order if p in by_period.index]
    order += [p for p in by_period.index if p not in order]
    logger.info(
        "%s per period:\n%s",
        unit.capitalize(),
        by_period.reindex(order).round(2).to_string(),
    )
    by_hour = (
        services.dropna(subset=["dep_hour"])
        .groupby(services["dep_hour"].astype("Int64"))["weekday_fraction"]
        .sum()
        .sort_index()
    )
    logger.info(
        "%s per departure hour:\n%s",
        unit.capitalize(),
        by_hour.rename_axis("hour").round(2).to_string(),
    )


def run_service_lookup(config: ServiceLookupConfig) -> None:
    """List the timetabled trains for one station pair, or every pair among ``station_codes``.

    The calls table is read and filtered once (permanent passenger public services on the reference
    date) and reused for every pair; each pair is written to ``services_<ORIGIN>_<DEST>.csv``.
    """
    config.output_path.mkdir(parents=True, exist_ok=True)
    pairs = config.pairs()
    weekdays = _selected_weekdays(config)

    logger.info("Loading cleaned calls from %s ...", config.timetable_calls_path)
    calls = pd.read_csv(config.timetable_calls_path, dtype=str)

    reference_date = resolve_reference_date(calls, config)
    weekday_labels = ", ".join(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][w] for w in weekdays
    )
    logger.info(
        "Snapshot around %s; averaging over weekday(s): %s.",
        reference_date.date(),
        weekday_labels,
    )

    calls = filter_services(calls, reference_date, weekdays)
    calls.loc[:, "crs"] = calls["crs"].astype(str).str.strip().str.upper()
    calls.loc[:, "seq"] = pd.to_numeric(calls["seq"], errors="coerce")
    calls = calls.dropna(subset=["seq"])

    logger.info("Listing services for %d station pair(s).", len(pairs))
    for origin, destination in pairs:
        _run_service_lookup_pair(config, calls, origin, destination, weekdays)


def _run_service_lookup_pair(
    config: ServiceLookupConfig,
    calls: pd.DataFrame,
    origin: str,
    destination: str,
    weekdays: list[int],
) -> None:
    """List one pair's trains from an already filtered ``calls``."""
    services = extract_od_services(
        calls, origin, destination, config.time_periods_cutoffs, len(weekdays)
    )
    services, label = _apply_time_filter(services, config)

    _log_summary(services, config, origin, destination, len(weekdays))

    out_path = config.output_path / f"services_{origin}_{destination}.csv"
    services.to_csv(out_path, index=False)
    logger.info(
        "Listed %d %s -> %s trains (%s) -> %s",
        len(services),
        origin,
        destination,
        label,
        out_path,
    )


def crs_to_tiploc_map(calls: pd.DataFrame) -> dict[str, str]:
    """Build a CRS -> TIPLOC map from the cleaned calls table.

    Every call row already carries both codes for its own station (see
    ``timetable._CALL_FIELDS``), so this avoids re-streaming the raw CIF
    just to resolve a candidate's station codes to TIPLOCs for the BPLAN
    lookup. A station with several TIPLOCs (Clapham Junction, Victoria,
    Waterloo) maps to the one most calls use, rather than whichever came
    last, which BPLAN might not route and the running-time check would
    skip as unverifiable.
    """
    pairs = calls[["crs", "tiploc"]].dropna()
    pairs = pd.DataFrame(
        {
            "crs": pairs["crs"].astype(str).str.strip().str.upper(),
            "tiploc": pairs["tiploc"].astype(str).str.strip().str.upper(),
        }
    )
    pairs = pairs[pairs["crs"].ne("") & pairs["tiploc"].ne("")]
    busiest = pairs.value_counts().reset_index().drop_duplicates("crs")
    return dict(zip(busiest["crs"], busiest["tiploc"], strict=True))


# Per-run memo of timetable filtering. Keys use id() of the source
# frame, and each entry keeps a reference to that frame so the id can't
# be reused.
_TIMETABLE_CACHE: dict[tuple, tuple[pd.DataFrame, pd.DataFrame]] = {}


def _memo(key: tuple, source: pd.DataFrame, build: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    hit = _TIMETABLE_CACHE.get(key)
    if hit is not None and hit[0] is source:
        return hit[1]
    result = build()
    _TIMETABLE_CACHE[key] = (source, result)
    return result


def clear_timetable_caches() -> None:
    """Drop the memos keyed on timetable frames, freeing them.

    Harmless within one run; a discovery sweep builds new frames for
    every candidate, and holding them all grew each process to 7.5 GB by
    its 50th pair.
    """
    for cache in (_TIMETABLE_CACHE, _JOURNEY_TIME_CACHE, _EXISTING_JOURNEY_CACHE):
        cache.clear()


def prepare_calls(calls: pd.DataFrame) -> pd.DataFrame:
    """Normalise the columns :func:`existing_trains` needs."""
    if calls.attrs.get("prepared"):
        return calls

    def build() -> pd.DataFrame:
        out = calls.copy()
        out.loc[:, "crs"] = out["crs"].astype(str).str.strip().str.upper()
        out.loc[:, "seq"] = pd.to_numeric(out["seq"], errors="coerce")
        out = out.dropna(subset=["seq"])
        out.attrs["prepared"] = True
        return out

    return _memo(("prepared", id(calls)), calls, build)


def day_services(calls: pd.DataFrame, reference_date: pd.Timestamp, weekday: str) -> pd.DataFrame:
    """Services running on one weekday, filtered once per timetable.

    See :func:`~ortla.timetable.filter_services`.
    """
    return _memo(
        ("day", id(calls), reference_date, weekday),
        calls,
        lambda: filter_services(calls, reference_date, [WEEKDAY_INDEX[weekday]]),
    )


class _DayIndex:
    """One day's calls laid out for fast origin -> destination lookups.

    :func:`extract_od_services` scans the whole day's calls and runs a
    pandas group-by for every pair it is asked about, about 86 ms a
    pair. A hindcast asks for tens of thousands of (day, pair)
    combinations, so that lookup dominated its run time. This sorts the
    calls by service and calling order once and keeps each station's row
    positions, so a pair lookup touches only the two stations' calls.
    :meth:`od_services` returns exactly what :func:`extract_od_services`
    does with ``n_weekdays=1``.
    """

    def __init__(self, filtered: pd.DataFrame) -> None:
        frame = filtered.reset_index(drop=True)
        codes, _ = pd.factorize(
            pd.MultiIndex.from_frame(frame[SERVICE_KEY].astype("string").fillna("<NA>"))
        )
        seq = frame["seq"].to_numpy(dtype=float)
        order = np.lexsort((seq, codes))
        self.row = order  # original position, for first-appearance ordering
        self.svc = codes[order]
        self.seq = seq[order]
        pick = frame.iloc[order]
        self.dep = pick["public_departure"].to_numpy(dtype=object)
        self.arr = pick["public_arrival"].to_numpy(dtype=object)
        self.operator = pick["atoc_code"].to_numpy(dtype=object)
        self.headcode = pick["signalling_id"].to_numpy(dtype=object)
        self.uid = pick["train_uid"].to_numpy(dtype=object)
        self.fraction = pick["weekday_fraction"].to_numpy(dtype=float)
        self.svc_origin = pick["svc_origin_crs"].to_numpy(dtype=object)
        self.svc_dest = pick["svc_dest_crs"].to_numpy(dtype=object)
        crs = pick["crs"].to_numpy(dtype=object)
        positions = pd.Series(np.arange(len(crs))).groupby(crs, sort=False).indices
        self.by_crs = {code: np.sort(idx) for code, idx in positions.items()}

    def od_services(self, origin: str, destination: str, cutoffs: dict[str, str]) -> pd.DataFrame:
        """:func:`extract_od_services` for this day alone."""
        origin_rows = self.by_crs.get(origin)
        dest_rows = self.by_crs.get(destination)
        if origin_rows is None or dest_rows is None:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)
        # Rows are sorted by (service, seq), so row positions order a
        # service's calls. Each destination call pairs with the
        # service's latest origin call before it, and a service is
        # listed once, by its shortest such leg (the leg
        # `network_assignment.one_leg_per_pair` keeps). For a train
        # calling at each station once it is the only leg.
        prior = np.searchsorted(origin_rows, dest_rows) - 1
        paired_origin = origin_rows[np.maximum(prior, 0)]
        valid = (prior >= 0) & (self.svc[paired_origin] == self.svc[dest_rows])
        if not valid.any():
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)
        leg_dest, leg_origin = dest_rows[valid], paired_origin[valid]
        shortest_first = np.lexsort((leg_dest - leg_origin, self.svc[leg_dest]))
        services, first = np.unique(self.svc[leg_dest][shortest_first], return_index=True)
        dest_pos = leg_dest[shortest_first][first]
        origin_pos = leg_origin[shortest_first][first]
        # extract_od_services emits services in order of first
        # appearance among the origin and destination calls; keep that
        # order so ties sort alike.
        first_seen = np.full(len(services), np.iinfo(np.int64).max)
        for rows in (origin_rows, dest_rows):
            codes = self.svc[rows]
            slot = np.searchsorted(services, codes)
            in_range = slot < len(services)
            found = in_range & (services[np.minimum(slot, len(services) - 1)] == codes)
            np.minimum.at(first_seen, slot[found], self.row[rows[found]])
        ordered = np.argsort(first_seen, kind="stable")
        records = []
        for o, d in zip(origin_pos[ordered], dest_pos[ordered], strict=True):
            dep = cif_hhmm(self.dep[o])
            dep_minutes = hhmm_minutes(dep)
            fraction = float(self.fraction[o])
            records.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "operator": self.operator[o],
                    "headcode": self.headcode[o],
                    "train_uid": self.uid[o],
                    "dep_from_origin": dep,
                    "arr_at_destination": cif_hhmm(self.arr[d]),
                    "dep_hour": dep_minutes // 60 if dep_minutes is not None else None,
                    "time_period": classify_time_period(f"{dep}:00", cutoffs)
                    if dep
                    else "Unknown",
                    "days_per_week": round(fraction * 1),
                    "weekday_fraction": round(fraction, 3),
                    "svc_origin_crs": self.svc_origin[o],
                    "svc_destination_crs": self.svc_dest[o],
                }
            )
        listing = _collapse_variants(pd.DataFrame(records, columns=_OUTPUT_COLUMNS), 1)
        return listing.sort_values("dep_from_origin", na_position="last").reset_index(drop=True)

    def positions(self, uids: list[str], origin: str, destination: str) -> pd.DataFrame:
        """Find where each train's ``origin -> destination`` leg sits.

        One row per uid: ``o_is_origin``, ``d_is_terminus``,
        ``calls_od`` (calls from origin to destination) and
        ``train_calls``, for the leg :meth:`od_services` lists (the
        shortest). A uid not found gives a two-call run of its own.
        """
        if not hasattr(self, "_spans"):
            starts = np.flatnonzero(np.r_[True, self.svc[1:] != self.svc[:-1]])
            ends = np.r_[starts[1:], len(self.svc)] - 1
            spans: dict[str, tuple[int, int]] = {}
            for start, end in zip(starts, ends, strict=True):
                spans.setdefault(self.uid[start], (int(start), int(end)))
            self._spans = spans
        empty = np.empty(0, dtype=int)
        o_rows = self.by_crs.get(origin, empty)
        d_rows = self.by_crs.get(destination, empty)
        out = []
        for uid in uids:
            start, end = self._spans.get(uid, (0, -1))
            o = o_rows[(o_rows >= start) & (o_rows <= end)]
            d = d_rows[(d_rows >= start) & (d_rows <= end)]
            prior = np.searchsorted(o, d) - 1
            legs = [(int(o[k]), int(r)) for k, r in zip(prior, d, strict=True) if k >= 0]
            if not legs:
                out.append((True, True, 1, 2))
                continue
            i, j = min(legs, key=lambda leg: leg[1] - leg[0])
            out.append((i == start, j == end, j - i, end - start + 1))
        return pd.DataFrame(
            out, columns=["o_is_origin", "d_is_terminus", "calls_od", "train_calls"]
        )


def train_positions(
    filtered: pd.DataFrame, uids: list[str], origin: str, destination: str
) -> pd.DataFrame:
    """:meth:`_DayIndex.positions` on one day's filtered calls."""
    return _day_index(filtered).positions(uids, origin, destination)


def _day_index(filtered: pd.DataFrame) -> _DayIndex:
    """Return the :class:`_DayIndex` for one day, built once."""
    key = ("day_index", id(filtered))
    hit = _TIMETABLE_CACHE.get(key)
    if hit is not None and hit[0] is filtered:
        return hit[1]
    index = _DayIndex(filtered)
    _TIMETABLE_CACHE[key] = (filtered, index)
    return index


def od_services(
    filtered: pd.DataFrame, origin: str, destination: str, cutoffs: dict[str, str]
) -> pd.DataFrame:
    """:func:`extract_od_services` for one day, once per pair."""
    return _memo(
        ("od", id(filtered), origin, destination, tuple(cutoffs.items())),
        filtered,
        lambda: _day_index(filtered).od_services(origin, destination, cutoffs),
    )


def existing_trains(
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    weekday: str,
    origin: str,
    destination: str,
    cutoffs: dict[str, str],
) -> pd.DataFrame:
    """Existing trains running ``origin -> destination`` on ``weekday``.

    :func:`ortla.timetable.filter_services` and
    :func:`ortla.service_lookup.extract_od_services` for a
    single weekday, so ``weekday_fraction`` is 0 or 1.
    """
    filtered = day_services(calls, reference_date, weekday)
    return od_services(filtered, origin, destination, cutoffs)


# Destination substitution and route choice
# Times are read off the timetable once per (day-table, station) and
# reused for every leg and competing terminal, so the two-leg
# interchange search below costs one pass over the day's calls rather
# than one per pair.
_JOURNEY_TIME_CACHE: dict[tuple, tuple[object, dict[str, float]]] = {}


# A one-change journey riding longer than this, both trains together, is
# no real alternative to a direct service. It was 120, which took the
# cap for a wait between trains (which this search does not see) and
# made every long-distance pair, Euston - Shrewsbury or Edinburgh -
# Liverpool, unserved, so a new link there saved nobody any time (case
# 27).
_MAX_CONNECTION_MINUTES = 360


# Crossing London between terminals by Underground or on foot (Euston ->
# Victoria), wait included, charged on top of the interchange penalty.
# Without it Birmingham -> Brighton had no journey through London and a
# direct train "saved" 70 minutes.
_CROSS_LONDON_MINUTES = 30.0


_DEFAULT_ALIASES = resolve_aliases(None)


def _leg_times(filtered: pd.DataFrame, station: str, *, outward: bool) -> dict[str, float]:
    """Fastest direct minutes from ``station`` to each place it serves.

    ``outward`` gives ``station -> X`` (departing ``station``),
    otherwise ``X -> station`` (arriving at ``station``). Only public
    calls count, since a passenger cannot board or alight at a passing
    point.
    """
    key = ("journey", id(filtered), station, outward)
    hit = _JOURNEY_TIME_CACHE.get(key)
    if hit is not None and hit[0] is filtered:
        return hit[1]

    public = filtered
    if "is_public_call" in filtered.columns:
        public = filtered[filtered["is_public_call"].astype(str).str.lower() == "true"]
    serving = set(public.loc[public["crs"] == station, "train_uid"])
    rows = public[public["train_uid"].isin(serving)].sort_values(["train_uid", "seq"])

    times: dict[str, float] = {}
    for _uid, service in rows.groupby("train_uid", sort=False):
        stations = list(service["crs"])
        if station not in stations:
            continue
        first, last = stations.index(station), len(stations) - 1 - stations[::-1].index(station)
        pivot = first if outward else last
        pivot_time = cif_time_minutes(
            service["public_departure"].iloc[pivot]
            if outward
            else service["public_arrival"].iloc[pivot]
        )
        if pivot_time is None:
            continue
        reach = range(pivot + 1, len(stations)) if outward else range(pivot)
        for i in reach:
            other = stations[i]
            other_time = cif_time_minutes(
                service["public_arrival"].iloc[i]
                if outward
                else service["public_departure"].iloc[i]
            )
            if other_time is None or other == station:
                continue
            minutes = (
                elapsed(pivot_time, other_time) if outward else elapsed(other_time, pivot_time)
            )
            if minutes > 0 and minutes < times.get(other, float("inf")):
                times[other] = float(minutes)

    _JOURNEY_TIME_CACHE[key] = (filtered, times)
    return times


_EXISTING_JOURNEY_CACHE: dict[tuple, tuple[object, tuple[float | None, bool]]] = {}


def existing_journey(
    filtered: pd.DataFrame, origin: str, destination: str, interchange_penalty: float
) -> tuple[float | None, bool]:
    """Find the best existing journey and whether it is direct.

    A direct service if there is one; otherwise the quickest single
    change, charged ``interchange_penalty`` generalised minutes for the
    change itself. ``(None, False)`` when neither exists: the pair is
    unserved, so a new link has nothing to abstract from.

    A change can also be across London: into one terminal, across to
    another in ``_CROSS_LONDON_MINUTES``, out of that one.

    Memoised per pair: the interchange search runs for every competing
    destination of every leg, and a candidate calling at 100 stations
    asks for the same pairs tens of thousands of times.
    """
    key = ("existing", id(filtered), origin, destination, interchange_penalty)
    hit = _EXISTING_JOURNEY_CACHE.get(key)
    if hit is not None and hit[0] is filtered:
        return hit[1]

    origin = canonical_crs(origin, _DEFAULT_ALIASES)
    destination = canonical_crs(destination, _DEFAULT_ALIASES)
    outward = _station_leg_times(filtered, origin, outward=True)
    direct = outward.get(destination)
    if direct is not None:
        result: tuple[float | None, bool] = (direct, True)
    else:
        inbound = _station_leg_times(filtered, destination, outward=False)
        options = [
            outward[via] + interchange_penalty + inbound[via]
            for via in outward.keys() & inbound.keys()
            if outward[via] + inbound[via] <= _MAX_CONNECTION_MINUTES
        ]
        options += [
            outward[into] + _CROSS_LONDON_MINUTES + interchange_penalty + inbound[out_of]
            for into in outward.keys() & LONDON_TERMINALS
            for out_of in inbound.keys() & LONDON_TERMINALS
            if into != out_of and outward[into] + inbound[out_of] <= _MAX_CONNECTION_MINUTES
        ]
        result = (min(options), False) if options else (None, False)

    _EXISTING_JOURNEY_CACHE[key] = (filtered, result)
    return result


def _station_leg_times(filtered: pd.DataFrame, station: str, *, outward: bool) -> dict[str, float]:
    """:func:`_leg_times` over every code ``station`` is booked under.

    St Pancras's Kent trains call at SPX, so Ashford -> STP alone finds
    none.
    """
    # Simplification: the default aliases only; a run's own `station_aliases`
    # are not seen here.
    times: dict[str, float] = {}
    codes = {station} | {a for a, c in _DEFAULT_ALIASES.items() if c == station}
    for code in codes:
        for other, minutes in _leg_times(filtered, code, outward=outward).items():
            key = canonical_crs(other, _DEFAULT_ALIASES)
            times[key] = min(minutes, times.get(key, minutes))
    return times


def best_existing_minutes(
    filtered: pd.DataFrame, origin: str, destination: str, interchange_penalty: float
) -> float | None:
    """Generalised minutes of the best existing journey."""
    return existing_journey(filtered, origin, destination, interchange_penalty)[0]
