"""CIF timetable cleaning and route analysis for the UK rail network.

The source is a Network Rail CIF schedule export in newline-delimited
JSON: one record per line, of type ``TiplocV1`` (location reference),
``JsonScheduleV1`` (a train schedule with a list of calling points),
``JsonAssociationV1`` (join / split links) or a single
``JsonTimetableV1`` header. Schedules address stations by TIPLOC; the
``TiplocV1`` records carry the TIPLOC -> CRS mapping.

Cleaning keeps every schedule that starts at, stops at, or passes
through one of our stations (the valid CRS list), and for each writes
its full calling pattern (every North stop plus every public stop
anywhere, so a Leeds->London train's London terminus is recorded and
reachable), dropping only non-North operational passes. An ``is_north``
flag marks our stations, so the route analysis can count routes with at
least one North endpoint while the service lookup can reach any recorded
stop. Wholly external schedules are dropped.

The file is far too large to load whole, so it is streamed line by line,
and a cheap substring test skips the JSON parse for record types we do
not need.
"""

from __future__ import annotations

import csv
import itertools
import logging
import pathlib  # noqa: TC003
import re
from collections import Counter
from typing import TYPE_CHECKING, Protocol

import pandas as pd

from ortla import cif_archive, wtt_workbook
from ortla.stations import read_station_list

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ortla.config import TimetableConfig


class _ReferenceDateConfig(Protocol):
    """A config that can resolve a snapshot reference date.

    Satisfied by both :class:`~ortla.config.TimetableConfig`
    and :class:`~ortla.config.ProposeConfig`, which share
    these fields.
    """

    reference_date: str | None
    reference_weekday: int


logger = logging.getLogger(__name__)

# CIF train categories starting with these letters are passenger
# services (O = ordinary passenger, X = express); B = bus, S = ship, E =
# empty stock.
PASSENGER_CATEGORIES = ("O", "X")
# Columns that together identify one schedule within the calls table.
SERVICE_KEY = ["train_uid", "start_date", "end_date", "svc_origin_dep", "signalling_id"]
# Merseyrail's ATOC code, used by
# ``TimetableConfig.exclude_merseyrail``.
_MERSEYRAIL_ATOC_CODE = "ME"

# Service-level columns copied onto every call row (denormalised so the
# routes analysis is one flat table).
_SERVICE_FIELDS = [
    "train_uid",
    "stp_indicator",
    "atoc_code",
    "days_runs",
    "start_date",
    "end_date",
    "train_category",
    "power_type",
    "timing_load",
    "speed",
    "signalling_id",
    "svc_origin_crs",
    "svc_origin_tiploc",
    "svc_origin_dep",
    "svc_dest_crs",
    "svc_dest_tiploc",
    "svc_dest_arr",
    "n_our_stops",
]
# Per-call columns. A kept service's full calling pattern is written
# (every public stop plus every North stop); ``is_north`` flags whether
# the stop is one of our stations.
_CALL_FIELDS = [
    "seq",
    "tiploc",
    "crs",
    "location_type",
    "arrival",
    "departure",
    "pass",
    "public_arrival",
    "public_departure",
    "is_public_call",
    "is_north",
]
_CALLS_HEADER = _SERVICE_FIELDS + _CALL_FIELDS


# Qualifiers the CIF appends to a station's name when it publishes
# several TIPLOCs for one station, one per platform group or running
# line. Stripping one of these and re-matching the station name recovers
# the station. Nothing else is stripped, so a rail-replacement bus stop,
# ferry pier or depot ("OKEHAMPTON WEST STREET", "GOUROCK PIER", "LONDON
# ROAD DEPOT") never matches a station name and is left unmapped rather
# than inventing train service there.
_PLATFORM_GROUP_SUFFIXES = (
    "(E)",
    "(W)",
    "(N)",
    "(S)",
    "(EAST)",
    "(WEST)",
    "(NORTH)",
    "(SOUTH)",
    "(TOWN PLAT)",
    "(MAIN)",
    "D.C.",
    "DC",
    "ELL",
    "NLL",
    "HIGH LEVEL",
    "LOW LEVEL",
    "MAIN",
    "STATION",
)
_PLATFORM_NUMBERS = re.compile(r"\b(PLATS?|PLATFORMS?)\s*[\d\s-]*$")
_MAX_SUFFIX_STRIPS = 3


def _normalise_station_name(name: str) -> str:
    """Upper-case, expand ``&`` and keep single-spaced letters."""
    text = str(name).upper().replace("&", " AND ")
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", text).split())


def _load_station_names(station_reference: pathlib.Path) -> dict[str, str]:
    """Normalised station name -> CRS, for unambiguous names.

    A name shared by two CRS codes is dropped rather than guessed.
    """
    frame = pd.read_csv(station_reference)
    if not {"stationName", "crsCode"}.issubset(frame.columns):
        logger.warning(
            "%s has no stationName/crsCode columns; cannot recover missing CRS by name.",
            station_reference,
        )
        return {}
    by_name: dict[str, set[str]] = {}
    for name, crs in zip(frame["stationName"], frame["crsCode"], strict=True):
        if pd.isna(name) or pd.isna(crs):
            continue
        by_name.setdefault(_normalise_station_name(name), set()).add(str(crs).strip().upper())
    return {name: next(iter(codes)) for name, codes in by_name.items() if len(codes) == 1}


def _crs_from_description(description: str, names: dict[str, str]) -> str:
    """Recover a CRS from a TIPLOC's ``tps_description``, or ``""``.

    Matches the description against station names exactly, then re-tries
    after stripping one platform-group qualifier at a time. Only an
    exact match on the whole remaining name is accepted.
    """
    text = description.strip().upper()
    for _ in range(_MAX_SUFFIX_STRIPS + 1):
        crs = names.get(_normalise_station_name(text))
        if crs:
            return crs
        stripped = _PLATFORM_NUMBERS.sub("", text).strip()
        for suffix in _PLATFORM_GROUP_SUFFIXES:
            if stripped.endswith(suffix) and len(stripped) > len(suffix):
                stripped = stripped[: -len(suffix)].strip(" -")
                break
        if stripped == text:
            return ""
        text = stripped
    return ""


def load_tiploc_crs_map(
    cif_path: pathlib.Path,
    station_reference: pathlib.Path | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """TIPLOC -> CRS from the CIF's ``TiplocV1`` records ("" where none), upper-cased.

    Only ``TiplocV1`` lines are parsed, so this is a cheap first pass. The CIF publishes some
    platform TIPLOCs with a null ``crs_code`` but a ``tps_description`` naming the station
    (WATRLMN, "LONDON WATERLOO"). Calls with no CRS are dropped downstream, which once removed
    Waterloo, Victoria, London Bridge, Clapham Junction and Vauxhall from a national run. With
    ``station_reference`` those TIPLOCs are recovered from the description; ``overrides`` win over
    both, for descriptions too abbreviated to match ("STOURBRIDGE JN (TOWN PLAT)").
    """
    logger.info("Scanning TIPLOC reference from %s ...", cif_path)
    tiploc_to_crs: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    for record in _iter_tiplocs(cif_path):
        tiploc = (record.get("tiploc_code") or "").strip().upper()
        if not tiploc:
            continue
        crs = record.get("crs_code") or ""
        tiploc_to_crs[tiploc] = crs.strip().upper() if crs else ""
        if not tiploc_to_crs[tiploc]:
            descriptions[tiploc] = (record.get("tps_description") or "").strip()
    n_published = sum(1 for c in tiploc_to_crs.values() if c)

    n_recovered = 0
    if station_reference is not None:
        names = _load_station_names(station_reference)
        for tiploc, description in descriptions.items():
            if not description:
                continue
            crs = _crs_from_description(description, names)
            if crs:
                tiploc_to_crs[tiploc] = crs
                n_recovered += 1
        logger.info(
            "Recovered a CRS for %d of %d TIPLOCs the CIF publishes without one, "
            "by matching tps_description against %s.",
            n_recovered,
            len(descriptions),
            station_reference,
        )

    for tiploc, crs in (overrides or {}).items():
        tiploc_to_crs[tiploc.strip().upper()] = str(crs).strip().upper()

    logger.info(
        "Loaded %d TIPLOCs (%d with a CRS: %d published, %d recovered, %d overridden).",
        len(tiploc_to_crs),
        sum(1 for c in tiploc_to_crs.values() if c),
        n_published,
        n_recovered,
        len(overrides or {}),
    )
    return tiploc_to_crs


def valid_tiplocs_for(tiploc_to_crs: dict[str, str], valid_crs: set[str]) -> set[str]:
    """Return the TIPLOCs whose CRS is one of our valid stations."""
    return {tiploc for tiploc, crs in tiploc_to_crs.items() if crs in valid_crs}


def _iter_tiplocs(
    cif_path: pathlib.Path, tiploc_cif_path: pathlib.Path | None = None
) -> Iterator[dict]:
    """Yield each location-reference record, from either CIF format.

    The archive extracts (the published timetable for a past period, and
    so the only way to build an era-matched "before" network) come in
    the original fixed-width CIF rather than the daily feed's
    newline-delimited JSON. :mod:`ortla.cif_archive` reads
    that format into the same record shapes, so everything below this
    point is format-blind.
    """
    if wtt_workbook.is_wtt_workbook_set(cif_path):
        # The books hold no location records; they name locations in
        # words.
        yield from _iter_tiplocs(_require_tiploc_reference(tiploc_cif_path))
        return
    yield from cif_archive.iter_tiploc_records(cif_path)


def _require_tiploc_reference(path: pathlib.Path | None) -> pathlib.Path:
    """Return the TIPLOC reference path, or say why one is needed."""
    if path is None:
        raise ValueError(
            "A Working Timetable book folder names its locations rather than "
            "coding them, so it needs tiploc_cif_path set to a CIF extract to "
            "resolve those names."
        )
    return path


def _iter_schedules(
    cif_path: pathlib.Path, tiploc_cif_path: pathlib.Path | None = None
) -> Iterator[dict]:
    """Yield each ``JsonScheduleV1``-shaped record from any source."""
    if wtt_workbook.is_wtt_workbook_set(cif_path):
        reference = _require_tiploc_reference(tiploc_cif_path)
        names = wtt_workbook.load_name_map(reference)
        operators = wtt_workbook.load_service_operators(reference)
        yield from wtt_workbook.iter_schedule_records(cif_path, names, operators)
        return
    yield from cif_archive.iter_schedule_records(cif_path)


def _service_summary(record: dict, tiploc_to_crs: dict[str, str], n_our_stops: int) -> dict:
    """Read the service-level fields, including its origin and end."""
    segment = record.get("schedule_segment") or {}
    locations = segment.get("schedule_location") or []
    origin = locations[0] if locations else {}
    dest = locations[-1] if locations else {}
    origin_tiploc = (origin.get("tiploc_code") or "").strip().upper()
    dest_tiploc = (dest.get("tiploc_code") or "").strip().upper()
    return {
        "train_uid": record.get("CIF_train_uid"),
        "stp_indicator": record.get("CIF_stp_indicator"),
        "atoc_code": record.get("atoc_code"),
        "days_runs": record.get("schedule_days_runs"),
        "start_date": record.get("schedule_start_date"),
        "end_date": record.get("schedule_end_date"),
        "train_category": segment.get("CIF_train_category"),
        "power_type": segment.get("CIF_power_type"),
        "timing_load": segment.get("CIF_timing_load"),
        "speed": segment.get("CIF_speed"),
        "signalling_id": segment.get("signalling_id"),
        "svc_origin_crs": tiploc_to_crs.get(origin_tiploc, ""),
        "svc_origin_tiploc": origin_tiploc,
        "svc_origin_dep": origin.get("public_departure") or origin.get("departure"),
        "svc_dest_crs": tiploc_to_crs.get(dest_tiploc, ""),
        "svc_dest_tiploc": dest_tiploc,
        "svc_dest_arr": dest.get("public_arrival") or dest.get("arrival"),
        "n_our_stops": n_our_stops,
    }


def _call_rows(
    record: dict,
    tiploc_to_crs: dict[str, str],
    valid_tiplocs: set[str] | None,
) -> list[dict] | None:
    """Return the call rows of a service at a wanted station, or None.

    ``None`` means the service never touches a valid station and is
    dropped (a wholly southern run, say). With ``valid_tiplocs``
    ``None`` every station is valid (national mode). Otherwise the
    service's full calling pattern is kept: every North stop plus every
    public stop anywhere (so a Leeds->London train's London terminus is
    recorded and reachable), dropping only non-North operational passes.
    Each row carries the denormalised service fields, this call's
    details, and an ``is_north`` flag.
    """
    segment = record.get("schedule_segment") or {}
    locations = segment.get("schedule_location") or []
    indexed = [
        (i, loc, (loc.get("tiploc_code") or "").strip().upper()) for i, loc in enumerate(locations)
    ]
    if valid_tiplocs is None:
        # National mode: every station counts as one of ours.
        valid_tiplocs = {tip for _, _, tip in indexed if tiploc_to_crs.get(tip)}
    n_north = sum(1 for _, _, tip in indexed if tip in valid_tiplocs)
    if not n_north:
        return None

    summary = _service_summary(record, tiploc_to_crs, n_north)
    rows = []
    for seq, loc, tiploc in indexed:
        public_arrival = loc.get("public_arrival")
        public_departure = loc.get("public_departure")
        # A public call means the train picks up / sets down here, as
        # opposed to passing through or a non-public operational stop.
        is_public = bool(public_arrival or public_departure)
        is_north = tiploc in valid_tiplocs
        # Keep North calls and public calls anywhere; drop non-North
        # operational passes, which are useless for a station-to-station
        # lookup.
        if not (is_north or is_public):
            continue
        rows.append(
            {
                **summary,
                "seq": seq,
                "tiploc": tiploc,
                "crs": tiploc_to_crs.get(tiploc, ""),
                "location_type": loc.get("location_type"),
                "arrival": loc.get("arrival"),
                "departure": loc.get("departure"),
                "pass": loc.get("pass"),
                "public_arrival": public_arrival,
                "public_departure": public_departure,
                "is_public_call": is_public,
                "is_north": is_north,
            }
        )
    return rows


def clean_timetable(config: TimetableConfig) -> dict[str, int]:
    """Filter the CIF to the wanted stations and write ``timetable_calls.csv``.

    One pass over the CIF after a quick TIPLOC pass; returns counts of schedules, kept schedules
    and call rows.
    """
    config.output_path.mkdir(parents=True, exist_ok=True)

    tiploc_to_crs = load_tiploc_crs_map(
        config.tiploc_cif_path or config.cif_path,
        station_reference=config.crs_reference or config.station_list,
        overrides=config.tiploc_crs_overrides,
    )
    valid_tiplocs: set[str] | None = None
    if config.station_list is None:
        logger.info("No station_list set: keeping every station's services (national).")
    else:
        valid_crs, _ = read_station_list(config.station_list)
        valid_tiplocs = valid_tiplocs_for(tiploc_to_crs, valid_crs)
        logger.info("%d valid CRS map to %d valid TIPLOCs.", len(valid_crs), len(valid_tiplocs))
        if not valid_tiplocs:
            logger.error("No valid TIPLOCs resolved; nothing to filter.")
            return {"schedules": 0, "kept": 0, "call_rows": 0}

    calls_path = config.output_path / "timetable_calls.csv"
    n_schedules = n_kept = n_rows = 0
    with calls_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CALLS_HEADER)
        writer.writeheader()
        for record in _iter_schedules(config.cif_path, config.tiploc_cif_path):
            n_schedules += 1
            rows = _call_rows(record, tiploc_to_crs, valid_tiplocs)
            if rows is None:
                continue
            n_kept += 1
            writer.writerows(rows)
            n_rows += len(rows)
            if n_schedules % 100_000 == 0:
                logger.info("Processed %d schedules, kept %d ...", n_schedules, n_kept)

    logger.info(
        "Cleaning complete: kept %d of %d schedules (%.1f%%), %d call rows -> %s",
        n_kept,
        n_schedules,
        100 * n_kept / n_schedules if n_schedules else 0.0,
        n_rows,
        calls_path,
    )
    if not wtt_workbook.is_wtt_workbook_set(config.cif_path):
        write_associations(config.cif_path, tiploc_to_crs, config.output_path)
    return {"schedules": n_schedules, "kept": n_kept, "call_rows": n_rows}


ASSOCIATION_COLUMNS = [
    "main_uid",
    "portion_uid",
    "category",
    "tiploc",
    "crs",
    "start_date",
    "end_date",
    "days",
    "CIF_stp_indicator",
]


def write_associations(
    cif_path: pathlib.Path, tiploc_to_crs: dict[str, str], output_path: pathlib.Path
) -> pathlib.Path:
    """Write the CIF's joins (JJ) and divides (VV) to ``timetable_associations.csv``.

    ``assign`` reads it as ``associations_path`` and keeps the permanent records running on its
    reference date, as it does schedules.
    """
    rows = [
        {
            "main_uid": r.get("main_train_uid"),
            "portion_uid": r.get("assoc_train_uid"),
            "category": r.get("category"),
            "tiploc": r.get("location"),
            "crs": tiploc_to_crs.get(str(r.get("location") or "").upper(), ""),
            "start_date": r.get("assoc_start_date"),
            "end_date": r.get("assoc_end_date"),
            "days": r.get("assoc_days"),
            "CIF_stp_indicator": r.get("CIF_stp_indicator"),
        }
        for r in cif_archive.iter_association_records(cif_path)
        if r.get("category") in {"JJ", "VV"}
    ]
    path = output_path / "timetable_associations.csv"
    pd.DataFrame(rows, columns=ASSOCIATION_COLUMNS).to_csv(path, index=False)
    logger.info("Wrote %d joins and divides -> %s", len(rows), path)
    return path


# Columns the feasibility/demand code reads from the calls table.
_FEASIBILITY_COLUMNS = [
    "train_uid",
    "stp_indicator",
    "atoc_code",
    "days_runs",
    "start_date",
    "end_date",
    "train_category",
    "signalling_id",
    "svc_origin_crs",
    "svc_origin_dep",
    "svc_dest_crs",
    "seq",
    "tiploc",
    "crs",
    "public_arrival",
    "public_departure",
    "is_public_call",
]


def read_public_calls(calls_path: pathlib.Path) -> pd.DataFrame:
    """Read the calls table, keeping only the columns used.

    Every consumer (headway, existing-service frequency, CRS -> TIPLOC
    lookup) filters to permanent, public passenger calls anyway, so the
    national table is filtered to those rows on read, a fraction of its
    full in-memory size.
    """
    chunks = []
    for chunk in pd.read_csv(
        calls_path, dtype=str, usecols=_FEASIBILITY_COLUMNS, chunksize=1_000_000
    ):
        category = chunk["train_category"].fillna("").str[:1]
        keep = (
            chunk["stp_indicator"].eq("P")
            & chunk["is_public_call"].eq("True")
            & category.isin(PASSENGER_CATEGORIES)
            # A public call at a location with no CRS can't be matched
            # to anything.
            & chunk["crs"].notna()
        )
        chunks.append(chunk[keep])
    calls = pd.concat(chunks, ignore_index=True)
    logger.info("Read %d permanent public passenger calls from %s", len(calls), calls_path)
    return calls


def _departure_hours(calls: pd.DataFrame) -> pd.Series:
    """Best available departure clock-hour for each call row.

    Prefers the public departure, falling back through the working
    departure and the arrival times so an origin, an intermediate stop
    and a terminus all get an hour where one exists.
    """
    times = pd.Series("", index=calls.index)
    for col in ("public_departure", "departure", "public_arrival", "arrival"):
        candidate = calls[col].fillna("").astype(str)
        times = times.where(times.str.len() > 0, candidate)
    return pd.to_numeric(times.str[:2], errors="coerce")


def resolve_reference_date(calls: pd.DataFrame, config: _ReferenceDateConfig) -> pd.Timestamp:
    """Pick the snapshot date the frequency is measured on.

    Uses ``config.reference_date`` when set; otherwise takes the middle
    of the data's date span and shifts to ``config.reference_weekday``
    so we land on a representative (weekday) date the bulk of schedules
    are active on.
    """
    if config.reference_date:
        return pd.Timestamp(config.reference_date)
    starts = pd.to_datetime(calls["start_date"], errors="coerce")
    ends = pd.to_datetime(calls["end_date"], errors="coerce")
    low, high = starts.min(), ends.max()
    middle = low + (high - low) / 2
    shift = (config.reference_weekday - middle.weekday()) % 7
    return (middle + pd.Timedelta(days=shift)).normalize()


def _weekday_fraction(bitmap: str, weekdays: list[int]) -> float:
    """Fraction of ``weekdays`` a service runs, from ``days_runs``.

    ``days_runs`` is a 7-char ``"1"``/``"0"`` string (Mon..Sun). Returns
    e.g. 1.0 for a train running every selected day, 0.2 for a
    Friday-only train across a Mon-Fri selection.
    """
    runs = sum(1 for w in weekdays if len(bitmap) > w and bitmap[w] == "1")
    return runs / len(weekdays)


def filter_services(
    calls: pd.DataFrame, reference_date: pd.Timestamp, weekdays: list[int]
) -> pd.DataFrame:
    """Keep permanent public passenger calls on ``reference_date``.

    Adds a ``weekday_fraction`` column (share of ``weekdays`` the
    service runs) and drops services that run on none of the selected
    weekdays.
    """
    days = calls["days_runs"].fillna("").astype(str)
    category = calls["train_category"].fillna("").astype(str)
    starts = pd.to_datetime(calls["start_date"], errors="coerce")
    ends = pd.to_datetime(calls["end_date"], errors="coerce")
    keep = (
        (calls["stp_indicator"] == "P")
        & (calls["is_public_call"].astype(str) == "True")
        & category.str[:1].isin(PASSENGER_CATEGORIES)
        & (starts <= reference_date)
        & (ends >= reference_date)
    )
    out = calls[keep].copy()
    # At most 128 distinct bitmaps, against millions of calls.
    fractions = {bitmap: _weekday_fraction(bitmap, weekdays) for bitmap in days[keep].unique()}
    out.loc[:, "weekday_fraction"] = days[keep].map(fractions)
    return out[out["weekday_fraction"] > 0].copy()


def _count_route_hours(calls: pd.DataFrame) -> Counter:
    """Count typical-weekday trains per pair and departure hour.

    For each service, every ordered pair of the public stops it makes
    (earlier stop -> later stop) contributes one train, timed by the
    earlier stop's departure hour, so a route counts every train that
    carries you along it, not only end-to-end services. A train that
    calls at a station twice (the Wirral loop at James Street) still
    counts once per pair, timed by its first departure. Only routes with
    at least one North endpoint are counted (a Leeds->London or
    Warrington->London leg counts; a purely external leg such as
    London->Peterborough on the same train does not).
    """
    ordered = calls.sort_values([*SERVICE_KEY, "seq"])
    columns = [*SERVICE_KEY, "crs", "is_north", "hour"]
    counts: Counter = Counter()
    n_key = len(SERVICE_KEY)

    for _, rows in itertools.groupby(
        ordered[columns].itertuples(index=False, name=None),
        key=lambda row: row[:n_key],
    ):
        # (crs, is_north, hour) in calling-point order.
        stops = [(row[-3], row[-2], row[-1]) for row in rows]
        seen = set()
        for (origin, o_north, hour), (dest, d_north, _) in itertools.combinations(stops, 2):
            if (
                origin
                and dest
                and origin != dest
                and (origin, dest) not in seen
                and pd.notna(hour)
                and (o_north or d_north)
            ):
                seen.add((origin, dest))
                counts[(origin, dest, int(hour))] += 1
    return counts


def analyse_routes(config: TimetableConfig) -> pd.DataFrame:
    """Typical-weekday route frequencies from the calls; returns the summary.

    Writes ``route_frequency_hourly.csv`` (origin, destination, hour, trains) and
    ``route_frequency_summary.csv`` (daily trains, peak trains per hour and its hour, mean daytime
    trains per hour, first and last departure hours).
    """
    calls_path = config.output_path / "timetable_calls.csv"
    logger.info("Loading cleaned calls from %s ...", calls_path)
    calls = pd.read_csv(calls_path, dtype=str)

    if config.exclude_merseyrail:
        is_merseyrail = calls["atoc_code"].fillna("").str.upper() == _MERSEYRAIL_ATOC_CODE
        logger.info("Excluding %d Merseyrail call rows.", int(is_merseyrail.sum()))
        calls = calls[~is_merseyrail]

    reference_date = resolve_reference_date(calls, config)
    logger.info(
        "Frequency snapshot for %s (%s).",
        reference_date.date(),
        reference_date.day_name(),
    )
    calls = filter_services(calls, reference_date, [reference_date.weekday()])
    calls.loc[:, "seq"] = pd.to_numeric(calls["seq"], errors="coerce")
    calls.loc[:, "hour"] = _departure_hours(calls)
    # Older calls files (North-only) have no is_north column; every stop
    # is North.
    if "is_north" in calls.columns:
        calls.loc[:, "is_north"] = calls["is_north"].astype(str).eq("True")
    else:
        calls.loc[:, "is_north"] = True
    calls = calls.dropna(subset=["seq", "hour"])
    logger.info("Counting routes over %d qualifying calls ...", len(calls))

    counts = _count_route_hours(calls)
    if not counts:
        logger.warning("No qualifying passenger services found; no routes written.")
        return pd.DataFrame()

    hourly = pd.DataFrame(
        [(o, d, h, n) for (o, d, h), n in counts.items()],
        columns=["origin_crs", "destination_crs", "hour", "trains"],
    ).sort_values(["origin_crs", "destination_crs", "hour"])
    hourly_path = config.output_path / "route_frequency_hourly.csv"
    hourly.to_csv(hourly_path, index=False)
    logger.info("Hourly route frequencies written to %s", hourly_path)

    summary = _summarise_routes(hourly, config.daytime_hours)
    summary_path = config.output_path / "route_frequency_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Route summary written to %s", summary_path)

    top = summary.head(15)
    logger.info(
        "Top routes by daily trains (typical weekday):\n%s",
        top[["origin_crs", "destination_crs", "daily_trains", "peak_tph", "peak_hour"]].to_string(
            index=False
        ),
    )
    return summary


def _summarise_routes(hourly: pd.DataFrame, daytime_hours: tuple[int, int]) -> pd.DataFrame:
    """Collapse the hourly counts into one row per route."""
    day_start, day_end = daytime_hours
    records = []
    for (origin, dest), grp in hourly.groupby(["origin_crs", "destination_crs"]):
        by_hour = grp.set_index("hour")["trains"]
        peak_hour = int(by_hour.idxmax())
        daytime = by_hour[(by_hour.index >= day_start) & (by_hour.index < day_end)]
        mean_daytime = daytime.sum() / max(day_end - day_start, 1)
        records.append(
            {
                "origin_crs": origin,
                "destination_crs": dest,
                "daily_trains": int(by_hour.sum()),
                "peak_tph": int(by_hour.max()),
                "peak_hour": peak_hour,
                "mean_tph_daytime": round(float(mean_daytime), 2),
                "first_dep_hour": int(by_hour.index.min()),
                "last_dep_hour": int(by_hour.index.max()),
            }
        )
    return (
        pd.DataFrame(records).sort_values("daily_trains", ascending=False).reset_index(drop=True)
    )


def run_timetable(config: TimetableConfig) -> None:
    """Clean the CIF unless reused, and count route frequencies."""
    calls_path = config.output_path / "timetable_calls.csv"
    if config.reuse_existing_calls and calls_path.exists():
        logger.info("Reusing existing cleaned calls at %s (skipping CIF clean).", calls_path)
    else:
        clean_timetable(config)
    analyse_routes(config)
