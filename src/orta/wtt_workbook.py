"""Read the published Working Timetable books (spreadsheets).

Two of the timetable eras the loading counts need, December 2024 and
June 2025, were never issued as an archive CIF. What was published
instead is the Working Timetable itself: a folder of Excel books, one
per route section, each holding a conventional WTT grid with trains down
the columns and locations down the rows.

Between them those two eras are the correct timetable for 43% of the
dated loading rows, so reading them is worth a parser. This one yields
the same ``JsonScheduleV1``-shaped records as
:mod:`orta.cif_archive`, so every consumer downstream is
unchanged.

Three things make a WTT book harder than a CIF:

*   A train is split across books. A book covers one route section, so a
    long-distance service appears in each book its route passes through,
    with only that section's calls. The full schedule is the union over
    books, keyed on the train UID.
*   The times are working times, not public times. A WTT is the
    operational document: it records when a train is planned to arrive
    and leave, to the half minute, not the time printed for passengers.
    The public time is taken here as the working time with the half
    minute dropped, which is the usual relationship but not a guaranteed
    one. Anything matching these schedules to a passenger-facing time
    should allow a minute of slack.
*   Locations are named, not coded. The books print the TPS description,
    so a TIPLOC table from any CIF extract resolves them; see
    :func:`load_name_map`.

Usage::

    from orta import wtt_workbook

    names = wtt_workbook.load_name_map(some_cif_path)
    records = wtt_workbook.iter_schedule_records(book_folder, names)
    for schedule in records:
        ...
"""

from __future__ import annotations

import collections
import logging
import pathlib  # noqa: TC003
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import TYPE_CHECKING

from orta import cif_archive

if TYPE_CHECKING:
    from collections.abc import Iterator

LOG = logging.getLogger(__name__)

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_LABEL_COLUMN = "B"
_LOCATION_COLUMN = "A"
_FIRST_LOCATION_ROW = 9

# The header block above every column of trains.
_HEADER_ROWS = {
    1: "headcode",
    2: "uid",
    3: "origin",
    4: "destination",
    5: "timing_load",
    6: "dates",
    7: "days",
    8: "service_code",
}

# A working time is HHMM with a separator that says whether the train
# stops: "T" for a stop, "/" for a pass, and a half minute is written as
# a trailing fraction. A letter after the separator is an activity code,
# not a digit.
_TIME = re.compile(r"^(\d{2})(?P<sep>[T/])[A-Z]*(\d{2})(?P<half>[½])?$")
_STOP_SEPARATOR = "T"
_HALF_MINUTE = "½"
_NO_TIME = ".."

# The books print a TIPLOC's description, falling back to this when the
# description is missing, in which case the TIPLOC itself is the first
# word.
_LONG_NAME_MISSING = "(LONG NAME NOT FOUND)"

# WTT running-day codes, as a Monday-to-Sunday mask.
_ALL_WEEK = "1111111"
_DAY_CODES = {
    "SX": "1111100",
    "SO": "0000010",
    "SU": "0000001",
    "EWD": "1111100",
    "MO": "1000000",
    "TO": "0100000",
    "WO": "0010000",
    "THO": "0001000",
    "FO": "0000100",
    "MX": "0111100",
    "TX": "1011100",
    "WX": "1101100",
    "THX": "1110100",
    "FX": "1111000",
    "MSX": "0111100",
    "FSX": "1111000",
}
_DATE_RANGE = re.compile(r"(\d{2})/(\d{2})/(\d{4})\s+to\s+(\d{2})/(\d{2})/(\d{4})")
_PERMANENT_STP = "P"
MINUTES_PER_DAY = 1440
# A schedule needs an origin and a destination to be a journey at all.
_MINIMUM_CALLS = 2
# Two entries at one location closer than this are the same call,
# printed twice because two route-section books overlap there.
_SAME_CALL_GAP = 30
# The books carry no CIF train category, but the headcode class says the
# same thing: class 1 is an express passenger service, class 2 an
# ordinary stopping one, class 9 another passenger service. Downstream
# keeps categories starting "X" (express) or "O" (ordinary), so the
# class has to be translated.
_CATEGORY_FOR_CLASS = {"1": "XX", "2": "OO", "9": "XX"}
# The books name the London terminals without their "LONDON" prefix.
_LONDON_PREFIX = "LONDON"


def _books(path: pathlib.Path) -> list[pathlib.Path]:
    """List the Working Timetable books under ``path``."""
    return sorted(p for p in path.glob("*/*/*.xlsx") if not p.name.startswith((".", "~$")))


def is_wtt_workbook_set(path: pathlib.Path) -> bool:
    """Whether ``path`` is a folder of Working Timetable books."""
    return path.is_dir() and bool(_books(path))


def _sheet_cells(book: zipfile.ZipFile, sheet: str, strings: list[str]) -> dict:
    """Return ``{(column, row): value}`` for one worksheet."""
    root = ET.fromstring(book.read(sheet))  # noqa: S314
    cells = {}
    for cell in root.iter(_NS + "c"):
        value = cell.find(_NS + "v")
        if value is None or value.text is None:
            continue
        text = strings[int(value.text)] if cell.get("t") == "s" else value.text
        text = text.strip()
        if not text:
            continue
        reference = re.match(r"([A-Z]+)(\d+)", cell.get("r", ""))
        if reference is None:
            continue
        cells[reference.group(1), int(reference.group(2))] = text
    return cells


def _shared_strings(book: zipfile.ZipFile) -> list[str]:
    """Read the workbook's shared strings, one entry per ``<si>``.

    A rich-text string splits its text over several runs (``<r>``), each
    with its own ``<t>``, so the runs are joined; phonetic hints
    (``<rPh>``) are not part of the text.
    """
    try:
        table = book.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    strings = []
    for item in ET.fromstring(table).iter(_NS + "si"):  # noqa: S314
        runs = (run.find(_NS + "t") for run in item.findall(_NS + "r"))
        nodes = [item.find(_NS + "t"), *runs]
        strings.append("".join(node.text or "" for node in nodes if node is not None))
    return strings


def _parse_time(text: str) -> tuple[str, bool] | None:
    """``(HHMM, stops)`` for a working time, or ``None``."""
    if not text or text == _NO_TIME:
        return None
    match = _TIME.match(text)
    if match is None:
        return None
    hours, minutes = match.group(1), match.group(3)
    return f"{hours}{minutes}", match.group("sep") == _STOP_SEPARATOR


def _train_category(headcode: str) -> str | None:
    """Return the CIF category a passenger headcode implies."""
    if not headcode:
        return None
    return _CATEGORY_FOR_CLASS.get(headcode[0])


def _is_passenger(headcode: str) -> bool:
    """Return whether a headcode belongs to a passenger service.

    The books carry freight, empty stock and engineering movements
    alongside passenger trains. Those stop at places a passenger cannot
    board, so giving them public times would invent calls. The first
    character of the headcode is the train class: 1 and 2 are passenger,
    9 is a channel tunnel or other passenger service, and everything
    else is not.
    """
    return _train_category(headcode) is not None


def _days_mask(text: str) -> str:
    """Return a Monday-to-Sunday mask for a WTT running-day code."""
    return _DAY_CODES.get(text.strip().upper().replace(" ", ""), _ALL_WEEK)


def _dates(text: str) -> tuple[str, str] | None:
    """``(from, to)`` ISO dates from a 'dd/mm/yyyy to dd/mm/yyyy'."""
    match = _DATE_RANGE.search(text or "")
    if match is None:
        return None
    day1, month1, year1, day2, month2, year2 = match.groups()
    return f"{year1}-{month1}-{day1}", f"{year2}-{month2}-{day2}"


def load_name_map(cif_path: pathlib.Path) -> dict[str, tuple[str, str | None]]:
    """``{upper-case name: (tiploc, crs)}`` from a CIF's TIPLOCs.

    The books name locations rather than coding them, so resolving them
    needs a TIPLOC table. Any extract will do; one from the same era is
    best.
    """
    scored: dict[str, tuple[tuple[int, int], tuple[str, str | None]]] = {}
    for record in cif_archive.iter_tiploc_records(cif_path):
        tiploc = record.get("tiploc_code")
        if not tiploc:
            continue
        crs = record.get("crs_code")
        short = (record.get("description") or "").strip()
        # Dozens of TIPLOCs can share a description: the station, the
        # signals around it, the depot, the individual platform groups.
        # Only a public station carries the short 16-character
        # description *and* a CRS, so that pair is what picks London
        # Bridge out of its signals and Reading out of its platform
        # groups.
        rank = (1 if crs else 0, 1 if short else 0)
        for field in ("tps_description", "description"):
            value = record.get(field)
            if not value:
                continue
            key = value.strip().upper()
            existing = scored.get(key)
            if existing is None or rank > existing[0]:
                scored[key] = (rank, (tiploc, crs))
    return {key: entry for key, (_, entry) in scored.items()}


def load_service_operators(cif_path: pathlib.Path) -> dict[str, str]:
    """Return ``{service code: ATOC code}`` from a CIF extract.

    The books print a service code but never an ATOC code, and every
    consumer that selects one operator's trains reads the ATOC code. A
    CIF extract carries both, and the relationship is exact: across the
    June 2024 extract no service code maps to more than one operator.
    Rebuilding the map from an extract is therefore lossless, and far
    better than guessing an operator from the route a book covers.
    """
    operators: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for schedule in cif_archive.iter_schedule_records(cif_path):
        segment = schedule.get("schedule_segment") or {}
        code = segment.get("CIF_train_service_code")
        atoc = schedule.get("atoc_code")
        if code and atoc:
            operators[code][atoc] += 1
    return {code: counts.most_common(1)[0][0] for code, counts in operators.items()}


def _resolve(name: str, names: dict) -> tuple[str, str | None] | None:
    """Return the TIPLOC and CRS for a printed location name."""
    key = name.strip().upper()
    entry = names.get(key)
    if entry is not None:
        return entry
    # The books print "WATERLOO" where the TIPLOC table says "LONDON
    # WATERLOO".
    entry = names.get(f"{_LONDON_PREFIX} {key}")
    if entry is not None:
        return entry
    if _LONG_NAME_MISSING in key:
        # The book prints the bare TIPLOC when the description is
        # missing.
        tiploc = key.split()[0]
        return tiploc, None
    return None


def _iter_book_columns(path: pathlib.Path, names: dict) -> Iterator[tuple[str, dict]]:
    """Yield ``(uid, partial schedule)`` for each train in a book."""
    with zipfile.ZipFile(path) as book:
        strings = _shared_strings(book)
        sheets = [n for n in book.namelist() if n.startswith("xl/worksheets/sheet")]
        for sheet in sorted(sheets):
            try:
                cells = _sheet_cells(book, sheet, strings)
            except ET.ParseError:
                LOG.warning("Could not parse %s in %s", sheet, path.name)
                continue
            yield from _iter_sheet_columns(cells, names)


def _iter_sheet_columns(cells: dict, names: dict) -> Iterator[tuple[str, dict]]:
    """Yield ``(uid, partial schedule)`` for each train in a sheet."""
    header_columns = (_LOCATION_COLUMN, _LABEL_COLUMN)
    columns = {column for column, _ in cells if column not in header_columns}
    # The location a row belongs to is the last name printed in column
    # A.
    rows = sorted({row for _, row in cells if row >= _FIRST_LOCATION_ROW})
    location_of = {}
    current = None
    for row in rows:
        current = cells.get((_LOCATION_COLUMN, row), current)
        location_of[row] = current

    for column in columns:
        header = {field: cells.get((column, row), "") for row, field in _HEADER_ROWS.items()}
        uid = header["uid"]
        if not uid or uid == "UID":
            continue
        calls = _column_calls(cells, column, rows, location_of, names)
        if calls:
            yield uid, {"header": header, "calls": calls}


def _column_calls(
    cells: dict,
    column: str,
    rows: list[int],
    location_of: dict,
    names: dict,
) -> list[dict]:
    """Return the calls one train column records, in book order."""
    calls: dict[str, dict] = {}
    for row in rows:
        label = cells.get((_LABEL_COLUMN, row), "").lower()
        if label not in ("arr", "dep", "pass"):
            continue
        raw = cells.get((column, row))
        if not raw:
            continue
        parsed = _parse_time(raw)
        if parsed is None:
            continue
        working, stops = parsed
        name = location_of.get(row)
        resolved = _resolve(name, names) if name else None
        if resolved is None:
            continue
        tiploc, crs = resolved
        call = calls.setdefault(
            f"{tiploc}@{row}",
            {"tiploc_code": tiploc, "crs_code": crs, "row": row},
        )
        half = _HALF_MINUTE in raw
        call[label] = working
        call[f"{label}_half"] = half
        call["stops"] = call.get("stops", False) or stops
    return [calls[key] for key in sorted(calls, key=lambda k: calls[k]["row"])]


def _merge_calls(calls: list[dict]) -> list[dict]:
    """Combine the entries two overlapping books make of one call.

    Route sections overlap at their ends, so a station on a boundary is
    printed in both books, typically as the arrival that closes one
    section and the departure that opens the next. Left alone these
    count as two calls, which inflates the busiest interchanges. Entries
    at the same location within half an hour of each other are the same
    call; a train that really returns to a location does so far later
    than that.
    """
    by_location: dict[str, list[dict]] = collections.defaultdict(list)
    for call in calls:
        by_location[call["tiploc_code"]].append(call)

    merged = []
    for location_calls in by_location.values():
        groups: list[dict] = []
        for call in sorted(location_calls, key=_call_minutes):
            minute = _call_minutes(call)
            if groups and minute - _call_minutes(groups[-1]) <= _SAME_CALL_GAP:
                target = groups[-1]
                for field in ("arr", "dep", "pass"):
                    target[field] = target.get(field) or call.get(field)
                target["stops"] = target.get("stops", False) or call.get("stops", False)
                target["crs_code"] = target.get("crs_code") or call.get("crs_code")
            else:
                groups.append(dict(call))
        merged.extend(groups)
    return merged


def _call_time(call: dict) -> str | None:
    """Return the time a call is sequenced by."""
    return call.get("arr") or call.get("dep") or call.get("pass")


def _call_minutes(call: dict) -> int:
    """Return a call's time as minutes after midnight, for ordering."""
    time = _call_time(call)
    return _minutes(time) if time else 0


def _minutes(time: str) -> int:
    """Return a HHMM time as minutes after midnight."""
    return int(time[:2]) * 60 + int(time[2:])


def _order_calls(calls: list[dict], origin: str | None) -> list[dict]:
    """Return the merged calls in running order, past midnight.

    Books are concatenated in no particular order, so a train's calls
    have to be put back in sequence by time. Sorting on the clock alone
    breaks every service that runs past midnight. The Euston to
    Inverness sleeper leaves at 2030 and calls at 0500, and a clock sort
    puts the arrival first, which then reads as the train's origin.

    So the sequence is anchored on the origin the book itself states:
    anything timed earlier than the departure from there belongs to the
    following day. Without a usable origin the earliest call is the
    anchor, which is right for any service that does not cross midnight.
    """
    timed = []
    for call in calls:
        if _call_time(call):
            timed.append(call)
    if not timed:
        return []
    start = None
    if origin is not None:
        starts = [_call_minutes(c) for c in timed if c["tiploc_code"] == origin]
        if starts:
            start = min(starts)
    if start is None:
        start = min(_call_minutes(c) for c in timed)

    def sequence(call: dict) -> int:
        minute = _call_minutes(call)
        return minute + (MINUTES_PER_DAY if minute < start else 0)

    return sorted(timed, key=sequence)


def _schedule_location(call: dict, position: str, passenger: bool) -> dict:
    """One ``schedule_location`` entry in the JSON feed's shape."""
    arrival = call.get("arr")
    departure = call.get("dep")
    passing = call.get("pass")
    # A public time exists only where a passenger train stops at a
    # station. The WTT keeps half minutes and the public timetable does
    # not, so the fraction is dropped (see the module docstring).
    public = passenger and call.get("stops") and call.get("crs_code")
    return {
        "location_type": position,
        "record_identity": {"LO": "LO", "LI": "LI", "LT": "LT"}[position],
        "tiploc_code": call["tiploc_code"],
        "crs_code": call.get("crs_code"),
        "arrival": arrival,
        "departure": departure,
        "pass": passing,
        "platform": None,
        "public_arrival": arrival if public else None,
        "public_departure": departure if public else None,
    }


def _to_schedule(
    uid: str,
    header: dict,
    calls: list[dict],
    operators: dict[str, str],
    names: dict | None = None,
) -> dict | None:
    """One ``JsonScheduleV1``-shaped record, or ``None``."""
    origin = _resolve(header.get("origin", ""), names or {})
    ordered = _order_calls(calls, origin[0] if origin else None)
    if len(ordered) < _MINIMUM_CALLS:
        return None
    dates = _dates(header.get("dates", ""))
    if dates is None:
        return None
    headcode = header.get("headcode", "")
    passenger = _is_passenger(headcode)
    locations = []
    for index, call in enumerate(ordered):
        if index == 0:
            position = "LO"
        elif index == len(ordered) - 1:
            position = "LT"
        else:
            position = "LI"
        locations.append(_schedule_location(call, position, passenger))
    return {
        "CIF_train_uid": uid,
        "CIF_stp_indicator": _PERMANENT_STP,
        "schedule_start_date": dates[0],
        "schedule_end_date": dates[1],
        "schedule_days_runs": header.get("days_mask") or _days_mask(header.get("days", "")),
        "train_status": "P" if passenger else "F",
        "transaction_type": "Create",
        "atoc_code": operators.get(header.get("service_code") or ""),
        "applicable_timetable": "Y",
        "schedule_segment": {
            "signalling_id": header.get("headcode") or None,
            "CIF_train_category": _train_category(headcode),
            "CIF_headcode": header.get("headcode") or None,
            "CIF_train_service_code": header.get("service_code") or None,
            "CIF_power_type": None,
            "CIF_timing_load": header.get("timing_load") or None,
            "CIF_speed": None,
            "schedule_location": locations,
        },
    }


def _widest_header(current: dict, candidate: dict) -> dict:
    """Return the header covering the wider span of the two.

    Books disagree slightly about when a train runs, so the schedule is
    dated by the widest range any book gives it.
    """
    if not current:
        return candidate
    here, there = _dates(current.get("dates", "")), _dates(candidate.get("dates", ""))
    if here is None or there is None:
        return current
    merged = dict(current)
    start, end = min(here[0], there[0]), max(here[1], there[1])
    day, month, year = start[8:10], start[5:7], start[:4]
    day2, month2, year2 = end[8:10], end[5:7], end[:4]
    merged["dates"] = f"{day}/{month}/{year} to {day2}/{month2}/{year2}"
    return merged


def _day_family(mask: str) -> str:
    """Return which of the three books a running-day mask belongs to.

    The books are published as Mondays-to-Fridays, Saturdays and
    Sundays, and that is the real division: a UID means a different
    train in each. Within the weekday book the sections disagree about
    the finer code (one prints SX where the next prints MSX for the same
    train), so the finer code cannot be part of the key without
    splitting the train into fragments.
    """
    if mask == _DAY_CODES["SO"]:
        return "SATURDAY"
    if mask == _DAY_CODES["SU"]:
        return "SUNDAY"
    return "WEEKDAY"


def _schedule_key(uid: str, header: dict) -> tuple[str, str]:
    """Return what identifies one schedule, rather than one train UID.

    A UID is reused across the week, so the day-type book is part of the
    key; the date range and the finer running-day code are not, because
    books disagree about both for the same train and keying on either
    splits a London to Glasgow service into truncated fragments, one per
    book.
    """
    return uid, _day_family(_days_mask(header.get("days", "")))


def iter_schedule_records(
    path: pathlib.Path,
    names: dict[str, tuple[str, str | None]],
    operators: dict[str, str] | None = None,
) -> Iterator[dict]:
    """Yield each train in a book folder, merged across books.

    A train's calls are collected from every book its route passes
    through and emitted once, in running order.
    """
    books = _books(path)
    LOG.info("Reading %d Working Timetable books from %s", len(books), path)
    merged: dict[tuple, dict] = collections.defaultdict(
        lambda: {"header": {}, "calls": [], "masks": collections.Counter()}
    )
    for index, book in enumerate(books, start=1):
        if index % 100 == 0:
            LOG.info("  %d/%d books", index, len(books))
        try:
            columns = list(_iter_book_columns(book, names))
        except (zipfile.BadZipFile, KeyError) as error:
            LOG.warning("Skipping %s: %s", book.name, error)
            continue
        for uid, partial in columns:
            entry = merged[_schedule_key(uid, partial["header"])]
            entry["header"] = _widest_header(entry["header"], partial["header"])
            entry["masks"][_days_mask(partial["header"].get("days", ""))] += 1
            entry["calls"].extend(partial["calls"])

    LOG.info("Merged %d distinct schedules", len(merged))
    for (uid, _), entry in merged.items():
        schedule = _to_schedule(
            uid,
            entry["header"] | {"days_mask": entry["masks"].most_common(1)[0][0]},
            _merge_calls(entry["calls"]),
            operators or {},
            names,
        )
        if schedule is not None:
            yield schedule
