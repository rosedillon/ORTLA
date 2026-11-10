"""Check whether a candidate service fits the timetable.

Two independent checks, both against data already in the model:

* Running time. For each leg of the candidate's stop-by-stop timetable,
  is the proposed running time achievable? It is compared with the
  minimum sectional running time from BPLAN Geography (see
  :mod:`ortla.bplan_geography`) for the candidate's traction
  code, summed along the shortest path between the two calling points,
  which are often not adjacent in the geography graph.
* Headway. For each leg, does the candidate's departure clash with an
  existing train on the same station-pair segment? The model has no
  signalling-headway data (BPLAN Geography gives running times, not
  train separation), so this is an empirical proxy: the tightest gap
  already scheduled between two trains on that segment, less a safety
  margin, floored at a minimum. It says whether the candidate fits
  alongside what is already timetabled, not whether the infrastructure
  can carry it.

:func:`search_departure_window` and :func:`search_flexible_slot` look
for departures that pass both checks, and
:func:`check_historic_performance` reports how reliably comparable
trains on the corridor ran, from the Rail Delivery Group's Historic
Service Performance (HSP) API.

HSP
---
HSP holds historical Darwin data and is queried live over HTTPS through
its ``serviceMetrics`` endpoint, documented at
https://wiki.openraildata.com/index.php/HSP. It returns a punctuality
summary (the share of historic services that arrived within a chosen
tolerance of their scheduled time) for every departure pattern between
two stations, over a date range, time window and day type. It
authenticates with HTTP Basic Auth on a National Rail Data Portal
account (https://opendata.nationalrail.co.uk) with the HSP subscription
ticked, the same account the fares feed uses. Credentials fall back to
the ``NRDP_USERNAME``/``NRDP_PASSWORD`` environment variables when
unset.

HSP only returns a service if its public departure and arrival times and
TOC match exactly, it is not marked deleted, and (for a called-at
location) it has a recorded actual arrival time. "Rules for Inclusion of
Trains" on the wiki page has the full list.
"""

from __future__ import annotations

import base64
import bisect
import itertools
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from ortla.clock import MINUTES_PER_DAY, cif_time_minutes, format_hhmm, hhmm_minutes
from ortla.demand import predict_demand, stop_time
from ortla.service_lookup import existing_trains, prepare_calls
from ortla.stations import canonical_crs, resolve_recodes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ortla.bplan_geography import BplanNetworkLinks, BplanTimingGraph
    from ortla.config import (
        CandidateService,
        DepartureWindow,
        ProposeConfig,
        ServiceStop,
    )
    from ortla.demand import FaresContext, GravityContext
    from ortla.network_assignment import TransferDemand

logger = logging.getLogger(__name__)


# Candidate day-of-week -> HSP's day-type bucket. HSP cannot distinguish
# individual weekdays, only Mon-Fri as a whole.
_HSP_DAY_BUCKET = {
    "Monday": "WEEKDAY",
    "Tuesday": "WEEKDAY",
    "Wednesday": "WEEKDAY",
    "Thursday": "WEEKDAY",
    "Friday": "WEEKDAY",
    "Saturday": "SATURDAY",
    "Sunday": "SUNDAY",
}


# Slack before a leg is called infeasible. Candidate times are whole
# public minutes, while BPLAN (like the working timetable) times in half
# minutes, so a real train's public times can undershoot the BPLAN
# minimum by up to 0.5 min.
_RUNNING_TIME_TOLERANCE_MINUTES = 0.55


# An empirical minimum gap needs at least two existing trains on a
# segment to measure a gap from; with fewer, only the floor applies.
_MIN_TRAINS_FOR_EMPIRICAL_GAP = 2


def _clear_intervals(
    existing_minutes: Sequence[float], min_gap: float
) -> list[tuple[float, float]]:
    """Every interval of the day ``min_gap`` clear of existing trains.

    Bounded to one day (``[0, 1440]``), with the day's last train
    padding the early hours and its first padding late evening, since
    the timetable repeats across midnight. Used by
    :func:`_nearest_feasible_departure` for a single segment and by
    :func:`search_flexible_slot`, intersected across a leg's running
    days. The module docstring explains why this is only an empirical
    proxy.
    """
    if not existing_minutes:
        return [(0.0, float(MINUTES_PER_DAY))]
    times = sorted(existing_minutes)
    bounds = [times[-1] - MINUTES_PER_DAY, *times, times[0] + MINUTES_PER_DAY]
    intervals = []
    for lo, hi in itertools.pairwise(bounds):
        lo_pad = max(lo + min_gap, 0.0)
        hi_pad = min(hi - min_gap, float(MINUTES_PER_DAY))
        if hi_pad >= lo_pad:
            intervals.append((lo_pad, hi_pad))
    return intervals


def _neighbours(
    existing_minutes: Sequence[float], candidate_dep: float
) -> tuple[float | None, float | None]:
    """Find the departures either side of ``candidate_dep``.

    Both are on ``candidate_dep``'s own clock, so the train before a
    00:10 departure can be yesterday's 23:50, returned as ``-10``.
    ``existing_minutes`` must be sorted.
    """
    if not existing_minutes:
        return None, None
    day_start = candidate_dep - candidate_dep % MINUTES_PER_DAY
    idx = bisect.bisect_right(existing_minutes, candidate_dep - day_start)
    before = existing_minutes[idx - 1] if idx else existing_minutes[-1] - MINUTES_PER_DAY
    after = (
        existing_minutes[idx]
        if idx < len(existing_minutes)
        else existing_minutes[0] + MINUTES_PER_DAY
    )
    return day_start + before, day_start + after


def _intersect_intervals(
    a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Every overlap between an interval in ``a`` and one in ``b``."""
    result = []
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            if lo <= hi:
                result.append((lo, hi))
    return result


def _nearest_point_in_intervals(
    intervals: Sequence[tuple[float, float]], desired: float
) -> float | None:
    """Find the point in ``intervals`` nearest ``desired``."""
    if not intervals:
        return None

    def _distance(interval: tuple[float, float]) -> float:
        lo, hi = interval
        return 0.0 if lo <= desired <= hi else min(abs(desired - lo), abs(desired - hi))

    best_lo, best_hi = min(intervals, key=_distance)
    return min(max(desired, best_lo), best_hi)


def _nearest_feasible_departure(
    existing_minutes: Sequence[float], desired: float, min_gap: float
) -> float:
    """Find the nearest time ``min_gap`` clear of every existing one.

    Suggests a conflict-free departure close to the one the candidate
    asked for. It rests on the same empirical proxy as the headway check
    (see the module docstring), so treat it as a starting point to check
    by hand.
    """
    intervals = _clear_intervals(existing_minutes, min_gap)
    point = _nearest_point_in_intervals(intervals, desired)
    return desired if point is None else point


def recode_stops(candidate: CandidateService, recodes: dict[str, str]) -> CandidateService:
    """``candidate`` with each second code replaced by its station's.

    The calls are recoded the same way in :func:`run_propose`, so a stop
    typed as PDX finds the trains booked at PAD and PDX alike and takes
    its demand from PAD's cells. The running-time check keeps the stop
    as typed, since it looks up that platform group's own TIPLOC.
    """
    stops = [
        stop.model_copy(update={"station": canonical_crs(stop.station, recodes)})
        for stop in candidate.stops
    ]
    return candidate.model_copy(update={"stops": stops})


_RUNNING_TIME_COLUMNS = [
    "candidate_name",
    "from_station",
    "to_station",
    "proposed_minutes",
    "minimum_minutes",
    "shortfall_minutes",
    "status",
    "traction_exact_share",
    "suggested_departure",
    "suggested_arrival",
]


def check_running_times(
    candidate: CandidateService,
    crs_tiploc: dict[str, str],
    graph: BplanTimingGraph,
) -> pd.DataFrame:
    """Check each leg's proposed running time against the BPLAN minimum.

    One row per leg with both times. ``status`` is ok, infeasible, or unverifiable (no BPLAN route
    between the TIPLOCs). An infeasible leg also gets ``suggested_departure`` (keeping the arrival)
    and ``suggested_arrival`` (keeping the departure) that meet the minimum.
    """
    if candidate.new_infrastructure:
        # BPLAN Geography only knows today's physical network, so it
        # can't validate a running time over track that doesn't exist
        # yet.
        return pd.DataFrame(columns=_RUNNING_TIME_COLUMNS)

    rows = []
    stops = candidate.stops
    for prev, nxt in itertools.pairwise(stops):
        # An intermediate stop with only an arrival set (no explicit
        # dwell) is assumed to depart immediately, same convention as
        # predict_demand.
        dep_min = hhmm_minutes(stop_time(prev))
        arr_min = hhmm_minutes(nxt.arrival)
        if dep_min is None or arr_min is None:
            continue

        proposed = arr_min - dep_min
        if proposed < 0:
            proposed += 24 * 60  # candidate leg crosses midnight

        from_crs = prev.station.strip().upper()
        to_crs = nxt.station.strip().upper()
        from_tiploc = crs_tiploc.get(from_crs)
        to_tiploc = crs_tiploc.get(to_crs)

        minimum = exact_share = None
        if from_tiploc and to_tiploc:
            minimum, exact_share = graph.running_time_detail(
                from_tiploc, to_tiploc, candidate.traction_code
            )

        if minimum is None:
            status = "unverifiable"
        elif proposed < minimum - _RUNNING_TIME_TOLERANCE_MINUTES:
            status = "infeasible"
        else:
            status = "ok"

        suggested_departure = None
        suggested_arrival = None
        if status == "infeasible" and minimum is not None:
            shortfall = minimum - proposed  # minutes still needed, > 0
            suggested_departure = format_hhmm(dep_min - shortfall)
            suggested_arrival = format_hhmm(arr_min + shortfall)

        rows.append(
            {
                "candidate_name": candidate.name,
                "from_station": from_crs,
                "to_station": to_crs,
                "proposed_minutes": round(proposed, 2),
                "minimum_minutes": None if minimum is None else round(minimum, 2),
                "shortfall_minutes": None if minimum is None else round(proposed - minimum, 2),
                "status": status,
                "traction_exact_share": None if exact_share is None else round(exact_share, 3),
                "suggested_departure": suggested_departure,
                "suggested_arrival": suggested_arrival,
            }
        )

    return pd.DataFrame(
        rows,
        columns=_RUNNING_TIME_COLUMNS,
    )


_HEADWAY_COLUMNS = [
    "candidate_name",
    "day",
    "from_station",
    "to_station",
    "candidate_departure",
    "existing_trains_on_segment",
    "empirical_min_gap_minutes",
    "effective_min_gap_minutes",
    "gap_before_minutes",
    "gap_after_minutes",
    "conflict",
    "conflicting_train",
    "data_basis",
    "segment_running_lines",
    "suggested_departure",
    "shift_minutes",
]


def _segment_running_lines(
    origin: str,
    destination: str,
    crs_tiploc: dict[str, str] | None,
    network_links: BplanNetworkLinks | None,
) -> int | None:
    """BPLAN running lines between two CRS codes, or ``None``."""
    if network_links is None or crs_tiploc is None:
        return None
    from_tiploc = crs_tiploc.get(origin)
    to_tiploc = crs_tiploc.get(destination)
    if not from_tiploc or not to_tiploc:
        return None
    return network_links.running_line_count(from_tiploc, to_tiploc)


def check_headway(
    config: ProposeConfig,
    candidate: CandidateService,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    crs_tiploc: dict[str, str] | None = None,
    network_links: BplanNetworkLinks | None = None,
) -> pd.DataFrame:
    """Check each leg's departure against trains on the same segment, per running day.

    The empirical minimum gap is the tightest gap between existing trains on the segment; the
    candidate conflicts if it comes closer than that less ``headway_safety_margin_minutes``,
    floored at ``min_headway_floor_minutes``. With fewer than two existing trains only the floor
    applies (``data_basis="floor_only"``). A conflicting row gets ``suggested_departure``, the
    nearest time clearing every train that day, and ``shift_minutes`` (positive is later).

    This is a proxy, not signalling headways: it knows nothing of platforms or lines. With
    ``network_links`` and ``crs_tiploc``, rows carry ``segment_running_lines`` from BPLAN as
    context for a reviewer; it does not clear a conflict, because the line counts have not been
    checked well enough.
    """
    if candidate.new_infrastructure:
        # Track that doesn't exist yet has no existing service.
        return pd.DataFrame(columns=_HEADWAY_COLUMNS)

    calls = prepare_calls(calls)
    rows = []
    stops = candidate.stops

    for prev, nxt in itertools.pairwise(stops):
        # Same zero-dwell fallback as check_running_times: an
        # intermediate stop with only an arrival set is assumed to
        # depart immediately.
        prev_time = stop_time(prev)
        candidate_dep = hhmm_minutes(prev_time)
        if candidate_dep is None:
            continue
        origin = prev.station.strip().upper()
        destination = nxt.station.strip().upper()
        segment_running_lines = _segment_running_lines(
            origin, destination, crs_tiploc, network_links
        )

        for weekday in candidate.days_of_week:
            existing = existing_trains(
                calls,
                reference_date,
                weekday,
                origin,
                destination,
                config.time_periods_cutoffs,
            )
            existing_minutes = sorted(
                m
                for m in (hhmm_minutes(t) for t in existing.get("dep_from_origin", []))
                if m is not None
            )

            effective_min_gap, empirical_min_gap, data_basis = _headway_rule(
                existing_minutes, config
            )

            before, after = _neighbours(existing_minutes, candidate_dep)
            gap_before = None if before is None else candidate_dep - before
            gap_after = None if after is None else after - candidate_dep

            conflict = False
            conflicting_train = None
            tightest_gap = None
            for gap, neighbour in ((gap_before, before), (gap_after, after)):
                if gap is None or gap >= effective_min_gap:
                    continue
                conflict = True
                if tightest_gap is not None and gap >= tightest_gap:
                    continue
                tightest_gap = gap
                dep_minutes = existing["dep_from_origin"].map(hhmm_minutes)
                match = existing[dep_minutes == neighbour % MINUTES_PER_DAY]
                if not match.empty:
                    row = match.iloc[0]
                    conflicting_train = (
                        f"{row['operator']} {row['headcode']} dep {row['dep_from_origin']}"
                    )

            suggested_departure = None
            shift_minutes = None
            if conflict:
                fixed = _nearest_feasible_departure(
                    existing_minutes, candidate_dep, effective_min_gap
                )
                suggested_departure = format_hhmm(fixed)
                shift_minutes = round(fixed - candidate_dep, 1)

            rows.append(
                {
                    "candidate_name": candidate.name,
                    "day": weekday,
                    "from_station": origin,
                    "to_station": destination,
                    "candidate_departure": prev_time,
                    "existing_trains_on_segment": len(existing_minutes),
                    "empirical_min_gap_minutes": empirical_min_gap,
                    "effective_min_gap_minutes": round(effective_min_gap, 2),
                    "gap_before_minutes": gap_before,
                    "gap_after_minutes": gap_after,
                    "conflict": conflict,
                    "conflicting_train": conflicting_train,
                    "data_basis": data_basis,
                    "segment_running_lines": segment_running_lines,
                    "suggested_departure": suggested_departure,
                    "shift_minutes": shift_minutes,
                }
            )

    return pd.DataFrame(rows, columns=_HEADWAY_COLUMNS)


_METRICS_URL = "https://hsp-prod.rockshore.net/api/v1/serviceMetrics"


_REQUEST_TIMEOUT_SECONDS = 60


_VALID_DAYS = {"WEEKDAY", "SATURDAY", "SUNDAY"}


_METRICS_COLUMNS = [
    "origin_location",
    "destination_location",
    "gbtt_ptd",
    "gbtt_pta",
    "toc_code",
    "matched_services",
    "rids",
    "tolerance_value",
    "num_not_tolerance",
    "num_tolerance",
    "percent_tolerance",
    "global_tolerance",
]


def _resolve_credentials(auth: tuple[str | None, str | None] | None) -> tuple[str, str]:
    username, password = auth or (None, None)
    username = username or os.environ.get("NRDP_USERNAME")
    password = password or os.environ.get("NRDP_PASSWORD")
    if not username or not password:
        raise ValueError(
            "No NRDP credentials given: set username and password under "
            "historic_performance in the propose config, or the "
            "NRDP_USERNAME/NRDP_PASSWORD environment "
            "variables, for a National Rail Data Portal account with the "
            "'HSP' subscription enabled (https://opendata.nationalrail.co.uk)."
        )
    return username, password


def _auth_header(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return f"Basic {token}"


def _post(url: str, body: dict, auth: tuple[str | None, str | None] | None) -> dict:
    username, password = _resolve_credentials(auth)
    request = urllib.request.Request(  # noqa: S310 (fixed https URL, not user input)
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": _auth_header(username, password),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
        return json.loads(response.read())


def fetch_service_metrics(
    from_loc: str,
    to_loc: str,
    from_time: str,
    to_time: str,
    from_date: str,
    to_date: str,
    days: str,
    toc_filter: list[str] | None = None,
    tolerance: list[str] | None = None,
    auth: tuple[str | None, str | None] | None = None,
) -> dict:
    """Call HSP's ``serviceMetrics`` for punctuality between two stations; return the raw JSON.

    Times are ``"HHMM"``, dates ``"YYYY-MM-DD"``, ``days`` WEEKDAY, SATURDAY or SUNDAY, and
    ``tolerance`` up to three minute values (HSP's defaults when unset). Missing credentials fall
    back to ``NRDP_USERNAME`` and ``NRDP_PASSWORD``. Raises ValueError for a bad ``days`` or no
    credentials.
    """
    days = days.strip().upper()
    if days not in _VALID_DAYS:
        raise ValueError(f"days must be one of {sorted(_VALID_DAYS)}, got {days!r}.")

    body = {
        "from_loc": from_loc,
        "to_loc": to_loc,
        "from_time": from_time,
        "to_time": to_time,
        "from_date": from_date,
        "to_date": to_date,
        "days": days,
    }
    if toc_filter:
        body["toc_filter"] = list(toc_filter)
    if tolerance:
        body["tolerance"] = list(tolerance)

    logger.info(
        "Querying HSP serviceMetrics %s -> %s, %s-%s, %s to %s ...",
        from_loc,
        to_loc,
        from_time,
        to_time,
        from_date,
        to_date,
    )
    return _post(_METRICS_URL, body, auth)


def parse_service_metrics(response: dict) -> pd.DataFrame:
    """Flatten a ``serviceMetrics`` response: one row per departure pattern and tolerance.

    ``rids`` is joined with ``;``. Patterns without metrics are dropped.
    """
    rows = []
    for service in response.get("Services", []) or []:
        attrs = service.get("serviceAttributesMetrics", {}) or {}
        base = {
            "origin_location": attrs.get("origin_location"),
            "destination_location": attrs.get("destination_location"),
            "gbtt_ptd": attrs.get("gbtt_ptd"),
            "gbtt_pta": attrs.get("gbtt_pta"),
            "toc_code": attrs.get("toc_code"),
            "matched_services": pd.to_numeric(attrs.get("matched_services"), errors="coerce"),
            "rids": ";".join(attrs.get("rids") or []),
        }
        for metric in service.get("Metrics", []) or []:
            rows.append(
                {
                    **base,
                    "tolerance_value": pd.to_numeric(
                        metric.get("tolerance_value"), errors="coerce"
                    ),
                    "num_not_tolerance": pd.to_numeric(
                        metric.get("num_not_tolerance"), errors="coerce"
                    ),
                    "num_tolerance": pd.to_numeric(metric.get("num_tolerance"), errors="coerce"),
                    "percent_tolerance": pd.to_numeric(
                        metric.get("percent_tolerance"), errors="coerce"
                    ),
                    "global_tolerance": metric.get("global_tolerance"),
                }
            )

    return pd.DataFrame(rows, columns=_METRICS_COLUMNS)


_HISTORIC_PERFORMANCE_COLUMNS = [
    "candidate_name",
    "day_type",
    "from_station",
    "to_station",
    "candidate_departure",
    "nearest_existing_departure",
    "toc_code",
    "tolerance_minutes",
    "matched_historic_services",
    "percent_within_tolerance",
    "status",
]


def check_historic_performance(config: ProposeConfig, candidate: CandidateService) -> pd.DataFrame:
    """Compare each leg's departure with the punctuality of services HSP tracks.

    Not a feasibility check: HSP has nothing on a new service, only on what runs near its departure
    on the same corridor. Days map to HSP's WEEKDAY, SATURDAY and SUNDAY, and the existing pattern
    nearest the candidate's departure within ``window_minutes`` is the comparison. ``status`` is
    ok, poor_reliability, insufficient_sample, no_historic_data or lookup_failed. Empty without
    ``config.historic_performance``.
    """
    hsp_cfg = config.historic_performance
    if hsp_cfg is None:
        return pd.DataFrame(columns=_HISTORIC_PERFORMANCE_COLUMNS)

    day_types = sorted(
        {_HSP_DAY_BUCKET[d] for d in candidate.days_of_week if d in _HSP_DAY_BUCKET}
    )
    rows: list[dict[str, object]] = []
    for prev, nxt in itertools.pairwise(candidate.stops):
        dep_min = hhmm_minutes(stop_time(prev))
        if dep_min is None:
            continue
        origin = prev.station.strip().upper()
        destination = nxt.station.strip().upper()
        # HSP's window is a time of day, so it stops at midnight rather
        # than wrapping (00:10 +/- 30 minutes would otherwise ask for
        # 23:40 to 00:40).
        from_time = format_hhmm(max(dep_min - hsp_cfg.window_minutes, 0)).replace(":", "")
        to_time = format_hhmm(min(dep_min + hsp_cfg.window_minutes, MINUTES_PER_DAY - 1)).replace(
            ":", ""
        )

        for day_type in day_types:
            base_row = {
                "candidate_name": candidate.name,
                "day_type": day_type,
                "from_station": origin,
                "to_station": destination,
                "candidate_departure": format_hhmm(dep_min),
                "nearest_existing_departure": None,
                "toc_code": None,
                "tolerance_minutes": hsp_cfg.tolerance_minutes,
                "matched_historic_services": None,
                "percent_within_tolerance": None,
                "status": None,
            }
            try:
                response = fetch_service_metrics(
                    origin,
                    destination,
                    from_time,
                    to_time,
                    hsp_cfg.from_date,
                    hsp_cfg.to_date,
                    day_type,
                    toc_filter=[candidate.operator] if candidate.operator else None,
                    tolerance=[hsp_cfg.tolerance_minutes],
                    auth=(hsp_cfg.username, hsp_cfg.password),
                )
            except (OSError, ValueError) as exc:
                logger.warning(
                    "HSP lookup failed for %s -> %s (%s): %s",
                    origin,
                    destination,
                    day_type,
                    exc,
                )
                rows.append({**base_row, "status": "lookup_failed"})
                continue

            metrics = parse_service_metrics(response)
            metrics = metrics[
                metrics["tolerance_value"] == float(hsp_cfg.tolerance_minutes)
            ].copy()
            metrics.loc[:, "dep_minutes"] = metrics["gbtt_ptd"].map(cif_time_minutes)
            metrics = metrics.dropna(subset=["dep_minutes"])

            if metrics.empty:
                rows.append({**base_row, "status": "no_historic_data"})
                continue

            nearest = metrics.loc[(metrics["dep_minutes"] - dep_min).abs().idxmin()]
            matched = nearest["matched_services"]
            percent = nearest["percent_tolerance"]
            if pd.isna(matched) or matched < hsp_cfg.min_sample_size:
                status = "insufficient_sample"
            elif pd.isna(percent) or percent < hsp_cfg.poor_reliability_percent:
                status = "poor_reliability"
            else:
                status = "ok"

            rows.append(
                {
                    **base_row,
                    "nearest_existing_departure": (format_hhmm(nearest["dep_minutes"])),
                    "toc_code": nearest["toc_code"],
                    "matched_historic_services": None if pd.isna(matched) else int(matched),
                    "percent_within_tolerance": None if pd.isna(percent) else float(percent),
                    "status": status,
                }
            )

    return pd.DataFrame(rows, columns=_HISTORIC_PERFORMANCE_COLUMNS)


_SEARCH_COLUMNS = [
    "candidate_name",
    "shift_minutes",
    "stop_0_departure",
    "feasible",
    "worst_gap_minutes",
    "predicted_annual_trips_captured",
    "predicted_annual_trips_generated",
    "predicted_annual_revenue_gbp",
    "notes",
]


def _shift_stop_time(value: str | None, shift_minutes: float) -> str | None:
    minutes = hhmm_minutes(value)
    if minutes is None:
        return None
    return format_hhmm(minutes + shift_minutes)


def shift_candidate(candidate: CandidateService, shift_minutes: float) -> CandidateService:
    """Copy ``candidate`` with every stop's times shifted alike.

    Gives :func:`predict_demand` real clock times to classify into
    periods at a trial shift, since a shift can move a departure across
    a period boundary (06:55 "early" -> 07:05 "am_peak", say).
    """
    shifted_stops = [
        stop.model_copy(
            update={
                "arrival": _shift_stop_time(stop.arrival, shift_minutes),
                "departure": _shift_stop_time(stop.departure, shift_minutes),
            }
        )
        for stop in candidate.stops
    ]
    return candidate.model_copy(update={"stops": shifted_stops})


def _existing_minutes_by_leg_day(
    config: ProposeConfig,
    candidate: CandidateService,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    legs: Sequence[tuple[ServiceStop, ServiceStop]],
) -> dict[tuple[int, str], list[float]]:
    """Sorted existing-train departures for each (leg, running weekday).

    Used by :func:`search_departure_window` and
    :func:`search_flexible_slot`. The lookups depend only on the segment
    and day, never on a trial shift or slot, so both searches build the
    dict once.
    """
    calls_prepared = prepare_calls(calls)
    existing_by_leg_day: dict[tuple[int, str], list[float]] = {}
    for leg_idx, (prev, nxt) in enumerate(legs):
        origin = prev.station.strip().upper()
        destination = nxt.station.strip().upper()
        for weekday in candidate.days_of_week:
            existing = existing_trains(
                calls_prepared,
                reference_date,
                weekday,
                origin,
                destination,
                config.time_periods_cutoffs,
            )
            existing_by_leg_day[(leg_idx, weekday)] = sorted(
                m
                for m in (hhmm_minutes(t) for t in existing.get("dep_from_origin", []))
                if m is not None
            )
    return existing_by_leg_day


def search_departure_window(
    config: ProposeConfig,
    candidate: CandidateService,
    crs_tiploc: dict[str, str],
    graph: BplanTimingGraph,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    gravity: GravityContext | None = None,
    fares: FaresContext | None = None,
    transfers: TransferDemand | None = None,
) -> pd.DataFrame:
    """Try every shift in ``candidate.search_window``; one row per shift.

    Every stop moves by the same amount, so each leg keeps its typed running time, and
    :func:`check_running_times` runs once: if any leg fails at zero shift, every shift does and the
    search stops. Only headway varies, and its existing-train lookups depend only on segment and
    weekday, so they are made once and reused. Demand is computed only for shifts that pass, since
    it is the costly part.
    """
    window = candidate.search_window
    if window is None:
        return pd.DataFrame(columns=_SEARCH_COLUMNS)

    # Running-time feasibility does not change with the shift (see the
    # docstring), so check the template once and stop if it can never
    # work.
    running = check_running_times(candidate, crs_tiploc, graph)
    if not running.empty and (running["status"] == "infeasible").any():
        worst = running.loc[running["status"] == "infeasible"].iloc[0]
        return pd.DataFrame(
            [
                {
                    "candidate_name": candidate.name,
                    "shift_minutes": None,
                    "stop_0_departure": None,
                    "feasible": False,
                    "worst_gap_minutes": None,
                    "predicted_annual_trips_captured": None,
                    "notes": (
                        f"Running-time infeasible regardless of shift: "
                        f"{worst['from_station']} -> {worst['to_station']} needs "
                        f"{worst['minimum_minutes']:.1f} min, template only gives "
                        f"{worst['proposed_minutes']:.1f} min."
                    ),
                }
            ],
            columns=_SEARCH_COLUMNS,
        )

    # The headway and demand lookups below match the recoded calls.
    candidate = recode_stops(candidate, resolve_recodes(config.station_recodes))
    stops = candidate.stops
    legs = list(itertools.pairwise(stops))
    raw_dep = [hhmm_minutes(stop_time(prev)) for prev, _nxt in legs]
    if not legs or any(d is None for d in raw_dep):
        return pd.DataFrame(columns=_SEARCH_COLUMNS)
    nominal_dep: list[float] = [d for d in raw_dep if d is not None]

    # Each (leg, weekday)'s existing departures depend on the segment
    # and day, not on the trial shift, so look them up once.
    existing_by_leg_day = (
        {}
        if candidate.new_infrastructure
        else _existing_minutes_by_leg_day(config, candidate, calls, reference_date, legs)
    )

    step = max(window.step_minutes, 0.5)
    results = []
    delta = window.earliest_shift_minutes
    while delta <= window.latest_shift_minutes + 1e-9:
        worst_gap = None
        feasible = True
        if not candidate.new_infrastructure:
            for leg_idx in range(len(legs)):
                candidate_dep = nominal_dep[leg_idx] + delta
                for weekday in candidate.days_of_week:
                    leg_feasible, gap = _leg_day_feasibility(
                        existing_by_leg_day[(leg_idx, weekday)], candidate_dep, config
                    )
                    if gap is not None and (worst_gap is None or gap < worst_gap):
                        worst_gap = gap
                    feasible = feasible and leg_feasible

        results.append(
            {
                "candidate_name": candidate.name,
                "shift_minutes": round(delta, 1),
                "stop_0_departure": format_hhmm(nominal_dep[0] + delta),
                "feasible": feasible,
                "worst_gap_minutes": None if worst_gap is None else round(worst_gap, 1),
                "predicted_annual_trips_captured": None,
                "predicted_annual_revenue_gbp": None,
                "notes": "",
            }
        )
        delta += step

    results_df = pd.DataFrame(results, columns=_SEARCH_COLUMNS)
    _fill_search_demand(
        results_df, config, candidate, calls, reference_date, gravity, fares, transfers
    )
    return results_df


def _leg_day_feasibility(
    existing_minutes: list[float], candidate_dep: float, config: ProposeConfig
) -> tuple[bool, float | None]:
    """Whether ``candidate_dep`` clears every train on a segment, and the tightest gap.

    :func:`check_headway`'s inner test, called once per leg, weekday and trial shift.
    """
    effective_min_gap = _effective_min_gap(existing_minutes, config)

    tightest = None
    feasible = True
    for neighbour in _neighbours(existing_minutes, candidate_dep):
        if neighbour is None:
            continue
        gap = abs(candidate_dep - neighbour)
        if tightest is None or gap < tightest:
            tightest = gap
        if gap < effective_min_gap:
            feasible = False
    return feasible, tightest


def _headway_rule(
    existing_minutes: Sequence[float], config: ProposeConfig
) -> tuple[float, float | None, str]:
    """Return the gap a new train needs on a segment, and its basis.

    ``headway_mode="empirical"``: the tightest gap already scheduled
    between existing trains, less ``headway_safety_margin_minutes``,
    floored at ``min_headway_floor_minutes``. Because that gap reflects
    today's *frequency* rather than signalling capacity, it rejects any
    train that raises a segment's frequency (a 30-minute service implies
    a ~29-minute "headway"). ``headway_mode="fixed"`` instead applies
    ``planning_headway_minutes`` everywhere, the way timetable planning
    rules specify a signalling headway.
    """
    ordered = sorted(existing_minutes)
    empirical = (
        min(b - a for a, b in itertools.pairwise(ordered))
        if len(ordered) >= _MIN_TRAINS_FOR_EMPIRICAL_GAP
        else None
    )
    if config.headway_mode == "fixed":
        return config.planning_headway_minutes, empirical, "fixed"
    if empirical is None:
        return config.min_headway_floor_minutes, None, "floor_only"
    gap = max(config.min_headway_floor_minutes, empirical - config.headway_safety_margin_minutes)
    return gap, empirical, "empirical"


def _effective_min_gap(existing_minutes: Sequence[float], config: ProposeConfig) -> float:
    """Return the gap :func:`check_headway` requires on a segment."""
    return _headway_rule(existing_minutes, config)[0]


def _leg_clear_intervals(
    leg_idx: int,
    days_of_week: Sequence[str],
    existing_by_leg_day: dict[tuple[int, str], list[float]],
    config: ProposeConfig,
) -> list[tuple[float, float]]:
    """Intervals clear of every train on this leg, every running day.

    Intersects each day's :func:`_clear_intervals` (each with its own
    empirical minimum gap) instead of pooling the days' trains, so a
    slot counts as clear only if it is clear on every day the candidate
    runs.
    """
    intervals: list[tuple[float, float]] | None = None
    for weekday in days_of_week:
        existing_minutes = existing_by_leg_day[(leg_idx, weekday)]
        min_gap = _effective_min_gap(existing_minutes, config)
        day_intervals = _clear_intervals(existing_minutes, min_gap)
        intervals = (
            day_intervals if intervals is None else _intersect_intervals(intervals, day_intervals)
        )
    return intervals or []


def _leg_slot_bounds(
    leg_idx: int,
    nominal_dep: Sequence[float],
    nominal_arr: Sequence[float],
    new_arr: Sequence[float],
    window: DepartureWindow,
) -> tuple[float, float, float]:
    """``(desired, lo_bound, hi_bound)`` for one leg's flexible slot.

    Leg 0 is bounded by the search window's shift around its own typed
    departure; every later leg is bounded by the dwell window around the
    previous leg's already-adjusted arrival.
    """
    if leg_idx == 0:
        return (
            nominal_dep[0],
            nominal_dep[0] + window.earliest_shift_minutes,
            nominal_dep[0] + window.latest_shift_minutes,
        )
    desired = new_arr[leg_idx - 1] + (nominal_dep[leg_idx] - nominal_arr[leg_idx - 1])
    return (
        desired,
        new_arr[leg_idx - 1] + window.min_dwell_minutes,
        new_arr[leg_idx - 1] + window.max_dwell_minutes,
    )


def _leg_tightest_gap(
    leg_idx: int,
    point: float,
    days_of_week: Sequence[str],
    existing_by_leg_day: dict[tuple[int, str], list[float]],
    config: ProposeConfig,
) -> float | None:
    """Tightest clearance from ``point`` on any running day."""
    tightest: float | None = None
    for weekday in days_of_week:
        existing_minutes = existing_by_leg_day[(leg_idx, weekday)]
        _feasible, gap = _leg_day_feasibility(existing_minutes, point, config)
        if gap is not None and (tightest is None or gap < tightest):
            tightest = gap
    return tightest


def _apply_leg_slots_to_stops(
    stops: Sequence[ServiceStop], new_dep: Sequence[float], new_arr: Sequence[float]
) -> list[ServiceStop]:
    """Copy ``stops`` with their times set from the per-leg slots.

    ``new_dep[k]``/``new_arr[k]`` are leg ``k``'s departure/arrival; a
    stop's own arrival comes from the leg ending there and its departure
    from the leg starting there, with the origin/terminus having only
    one of each.
    """
    shifted_stops = []
    for i, stop in enumerate(stops):
        if i == 0:
            arrival, departure = None, new_dep[0]
        elif i == len(stops) - 1:
            arrival, departure = new_arr[-1], None
        else:
            arrival, departure = new_arr[i - 1], new_dep[i]
        shifted_stops.append(
            stop.model_copy(
                update={
                    "arrival": None if arrival is None else format_hhmm(arrival),
                    "departure": (None if departure is None else format_hhmm(departure)),
                }
            )
        )
    return shifted_stops


@dataclass
class FlexibleSlotResult:
    """One slot-first, dwell-flexed schedule for a candidate.

    See :func:`search_flexible_slot`.
    """

    candidate: CandidateService
    feasible: bool
    worst_gap_minutes: float | None
    notes: str


def search_flexible_slot(
    config: ProposeConfig,
    candidate: CandidateService,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> FlexibleSlotResult | None:
    """Find each leg its clearest departure, bridged by flexible dwell.

    :func:`search_departure_window` shifts every stop together, so a multi-leg candidate passes
    only if one shift clears every leg. For LIV, WGW, BON the first leg might be clear 5 minutes
    late and the second only 15 minutes late; a real timetable would hold the train longer at WGW.

    This walks the legs in order and finds each its nearest departure clear on every running day.
    The first leg stays within the window's shift bounds; each later leg departs within
    ``min_dwell_minutes`` to ``max_dwell_minutes`` of the previous arrival. Running times stay as
    typed. A leg with no clear slot takes the nearest bound and the result is marked infeasible but
    still returned. None without a search window, with fewer than two stops, or with a stop lacking
    a usable time.
    """
    window = candidate.search_window
    if window is None:
        return None

    stops = candidate.stops
    legs = list(itertools.pairwise(stops))
    if not legs:
        return None

    nominal_dep = [hhmm_minutes(stop_time(prev)) for prev, _nxt in legs]
    nominal_arr = [hhmm_minutes(nxt.arrival) for _prev, nxt in legs]
    if any(t is None for t in (*nominal_dep, *nominal_arr)):
        return None

    running_minutes: list[float] = []
    for dep, arr in zip(nominal_dep, nominal_arr, strict=True):
        minutes = arr - dep
        if minutes < 0:
            minutes += MINUTES_PER_DAY
        running_minutes.append(minutes)

    existing_by_leg_day = (
        {}
        if candidate.new_infrastructure
        else _existing_minutes_by_leg_day(config, candidate, calls, reference_date, legs)
    )

    new_dep = [0.0] * len(legs)
    new_arr = [0.0] * len(legs)
    worst_gap: float | None = None
    all_clear = True
    conflict_notes: list[str] = []

    for leg_idx, (prev, nxt) in enumerate(legs):
        if candidate.new_infrastructure:
            # Track that doesn't exist yet has no existing service to
            # work around, so the leg stays as typed.
            new_dep[leg_idx] = nominal_dep[leg_idx]
            new_arr[leg_idx] = nominal_arr[leg_idx]
            continue

        intervals = _leg_clear_intervals(
            leg_idx, candidate.days_of_week, existing_by_leg_day, config
        )
        desired, lo_bound, hi_bound = _leg_slot_bounds(
            leg_idx, nominal_dep, nominal_arr, new_arr, window
        )
        bounded = _intersect_intervals(intervals, [(lo_bound, hi_bound)])
        point = _nearest_point_in_intervals(bounded, desired)
        if point is None:
            point = min(max(desired, lo_bound), hi_bound)
            all_clear = False
            bound_kind = "the search window" if leg_idx == 0 else "the dwell range"
            conflict_notes.append(
                f"{prev.station.strip().upper()} -> {nxt.station.strip().upper()}: no "
                f"fully clear slot within {bound_kind}; nearest is "
                f"{format_hhmm(point)} (still conflicts)."
            )

        gap = _leg_tightest_gap(
            leg_idx, point, candidate.days_of_week, existing_by_leg_day, config
        )
        if gap is not None and (worst_gap is None or gap < worst_gap):
            worst_gap = gap

        new_dep[leg_idx] = point
        new_arr[leg_idx] = point + running_minutes[leg_idx]

    shifted_stops = _apply_leg_slots_to_stops(stops, new_dep, new_arr)

    return FlexibleSlotResult(
        candidate=candidate.model_copy(update={"stops": shifted_stops}),
        feasible=all_clear,
        worst_gap_minutes=None if worst_gap is None else round(worst_gap, 1),
        notes=" ".join(conflict_notes),
    )


def _fill_search_demand(
    results_df: pd.DataFrame,
    config: ProposeConfig,
    candidate: CandidateService,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    gravity: GravityContext | None,
    fares: FaresContext | None = None,
    transfers: TransferDemand | None = None,
) -> None:
    """Add predicted demand and revenue to each feasible search row."""
    if config.period_matrices_dir is None:
        return
    for idx in results_df.index[results_df["feasible"]]:
        shift = results_df.loc[idx, "shift_minutes"]
        shifted_candidate = shift_candidate(candidate, shift)
        demand = predict_demand(
            config,
            shifted_candidate,
            calls,
            reference_date,
            gravity,
            fares,
            transfers=transfers,
        )
        if not demand.empty:
            total_captured = demand["predicted_annual_trips_captured"].sum(skipna=True)
            results_df.loc[idx, "predicted_annual_trips_captured"] = round(total_captured, 1)
            total_generated = demand["predicted_annual_trips_generated"].sum(skipna=True)
            results_df.loc[idx, "predicted_annual_trips_generated"] = round(total_generated, 1)
            total_revenue = demand["predicted_annual_revenue_gbp"].sum(skipna=True)
            results_df.loc[idx, "predicted_annual_revenue_gbp"] = round(total_revenue, 2)
