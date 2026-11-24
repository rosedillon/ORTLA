"""Read the published Working Timetable books (spreadsheets).

Two of the timetable eras the loading counts need, December 2024 and
June 2025, were never issued as an archive CIF. What was published
instead is the Working Timetable itself: a folder of Excel books, one
per route section, each holding a conventional WTT grid with trains down
the columns and locations down the rows.

Between them those two eras are the correct timetable for 43% of the
dated loading rows, so reading them is worth a parser. This one yields
the same ``JsonScheduleV1``-shaped records as
:mod:`ortla.cif_archive`, so every consumer downstream is
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

    from ortla import wtt_workbook

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

from ortla import cif_archive

if TYPE_CHECKING:
    from collections.abc import Iterator

LOG = logging.getLogger(__name__)

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_LABEL_COLUMN = "B"
_LOCATION_COLUMN = "A"
_FIRST_LOCATION_ROW = 9

# The header block above every column of trains, as the rows the books
# printed up to December 2025. From the June 2026 books an Operator row
# follows the UID and pushes the rest down one, so the rows are found
# by their label in column B and these positions are only the fallback.
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
_HEADER_LABELS = {
    "TID": "headcode",
    "UID": "uid",
    "Origin": "origin",
    "Destination": "destination",
    "Timing Load": "timing_load",
    "Dates of Operation": "dates",
    "Running Days": "days",
    "Service Code": "service_code",
}

# A working time is HHMM with the train's activity codes between the hours
# and the minutes, and a half minute is written as a trailing fraction:
# "15T45½", "06T RM24", "05OPRM01", "23TF-U37". "/" is a pass. A time
# with no code ("1557") is the departure of a stop whose arrival carries
# the codes. Of the CIF activity codes only these make a call public: T
# (take up and set down), TB and TF (train begins and finishes), D (set
# down only), U (take up only) and R (request stop). The others (OP, RM,
# RR, A, C, X, W, -D, -U and so on) are operational on their own.
_TIME = re.compile(r"^(\d{2})(?P<act>[^\d]*)(\d{2})(?P<half>[½])?$")
_ACTIVITY = re.compile(r"-[A-Z]|TB|TF|TW|OP|RM|RR|PR|[A-Z]")
_PUBLIC_ACTIVITY = {"T", "TB", "TF", "D", "U", "R"}
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
    # "Each weekday" is Monday to Saturday: 2,110 of 2,165 EWD trains in the
    # June 2026 books run Monday to Saturday in the CIF.
    "EWD": "1111110",
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
# Any other code is days run together, then O (only those days) or X
# (the weekday book's Monday to Friday except those days): SUO, TWThO,
# WSX. Th and Su come before T and S so "TThO" reads as Tue and Thu.
_DAY_TOKEN = re.compile(r"SU|TH|M|T|W|F|S")
_DAY_COMPOUND = re.compile(r"(?:SU|TH|M|T|W|F|S)+[OX]")
_DAY_INDEX = {"M": 0, "T": 1, "W": 2, "TH": 3, "F": 4, "S": 5, "SU": 6}
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
    # Most books sit two folders down (CJ/CJ01/...), but not all (CJ/CJ02B ...).
    return sorted(p for p in path.rglob("*.xlsx") if not p.name.startswith((".", "~$")))


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
    codes = set(_ACTIVITY.findall(match.group("act").replace(" ", "")))
    return f"{hours}{minutes}", bool(codes & _PUBLIC_ACTIVITY)


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
    code = text.strip().upper().replace(" ", "")
    if code in _DAY_CODES:
        return _DAY_CODES[code]
    if not _DAY_COMPOUND.fullmatch(code):
        return _ALL_WEEK
    days = {_DAY_INDEX[token] for token in _DAY_TOKEN.findall(code[:-1])}
    if code.endswith("O"):
        return "".join("1" if i in days else "0" for i in range(7))
    return "".join("1" if i < 5 and i not in days else "0" for i in range(7))


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
        # The books print TPS descriptions, so where one station's TPS
        # description is another's short description ("SWINTON" is
        # Swinton, Greater Manchester, in TPS and Swinton, South Yorkshire,
        # in short) the TPS one wins.
        for field in ("tps_description", "description"):
            value = record.get(field)
            if not value:
                continue
            rank = (1 if crs else 0, 1 if short else 0, 1 if field == "tps_description" else 0)
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


def _header_rows(cells: dict) -> dict[int, str]:
    """Return ``{row: field}`` for a sheet's header block, read from its labels.

    Falls back to the pre-2026 positions when a sheet doesn't label
    every field.
    """
    found = {
        row: _HEADER_LABELS[str(label).strip()]
        for (column, row), label in cells.items()
        if column == _LABEL_COLUMN and row < _FIRST_LOCATION_ROW + 2
        and str(label).strip() in _HEADER_LABELS
    }
    return found if len(found) == len(_HEADER_LABELS) else _HEADER_ROWS


def _iter_sheet_columns(cells: dict, names: dict) -> Iterator[tuple[str, dict]]:
    """Yield ``(uid, partial schedule)`` for each train in a sheet."""
    header_columns = (_LOCATION_COLUMN, _LABEL_COLUMN)
    columns = {column for column, _ in cells if column not in header_columns}
    header_rows = _header_rows(cells)
    first_location_row = max(header_rows) + 1
    # The location a row belongs to is the last name printed in column
    # A.
    rows = sorted({row for _, row in cells if row >= first_location_row})
    location_of = {}
    current = None
    for row in rows:
        current = cells.get((_LOCATION_COLUMN, row), current)
        location_of[row] = current

    for column in sorted(columns):  # a set: sorted so merges (and so builds) repeat
        header = {field: cells.get((column, row), "") for row, field in header_rows.items()}
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


def _absorb(target: dict, call: dict) -> None:
    """Fold a second entry for the same call into the first."""
    for field in ("arr", "dep", "pass"):
        target[field] = target.get(field) or call.get(field)
    target["stops"] = target.get("stops", False) or call.get("stops", False)
    target["crs_code"] = target.get("crs_code") or call.get("crs_code")


def _merge_calls(calls: list[dict]) -> list[dict]:
    """Combine the entries two overlapping books make of one call.

    Route sections overlap at their ends, so a station on a boundary is
    printed in both books, typically as the arrival that closes one
    section and the departure that opens the next. Left alone these
    count as two calls, which inflates the busiest interchanges. Entries
    at the same location within half an hour of each other, counted round
    midnight, with no stop at another station between them, are the same
    call. A train can come back sooner (one reversing at Eastbourne calls
    at Hampden Park either side of it within minutes), but only after
    stopping somewhere else; a junction passed in between doesn't count.
    """
    by_location: dict[str, list[dict]] = collections.defaultdict(list)
    for call in calls:
        by_location[call["tiploc_code"]].append(call)
    # Only a stop at another station separates two entries; a junction
    # passed between a boundary station's arrival and departure doesn't.
    stops = [(_call_minutes(c), c["tiploc_code"]) for c in calls if c.get("crs_code") and not c.get("pass")]

    def gap(start: int, end: int) -> int:
        return (end - start) % MINUTES_PER_DAY

    def same_call(location: str, start: int, end: int) -> bool:
        return gap(start, end) <= _SAME_CALL_GAP and not any(
            0 < gap(start, m) < gap(start, end) and loc != location for m, loc in stops
        )

    merged = []
    for location_calls in by_location.values():
        groups: list[dict] = []
        for call in sorted(location_calls, key=_call_minutes):
            if groups and same_call(call["tiploc_code"], _call_minutes(groups[-1]), _call_minutes(call)):
                _absorb(groups[-1], call)
            else:
                groups.append(dict(call))
        # an arrival before midnight and a departure after it
        if len(groups) > 1 and same_call(groups[0]["tiploc_code"], _call_minutes(groups[-1]), _call_minutes(groups[0])):
            _absorb(groups[-1], groups.pop(0))
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


def _order_calls(calls: list[dict], origin: str | None, origin_time: str | None = None) -> list[dict]:
    """Return the merged calls in running order, past midnight.

    Books are concatenated in no particular order, so a train's calls
    have to be put back in sequence by time. Sorting on the clock alone
    breaks every service that runs past midnight. The Euston to
    Inverness sleeper leaves at 2030 and calls at 0500, and a clock sort
    puts the arrival first, which then reads as the train's origin.

    So the sequence is anchored on the departure time the header states
    for the origin (or, without one, the earliest call at the origin it
    names): anything timed earlier belongs to the following day. Without a usable origin (the books' sections may not
    reach it) the anchor is the call after the longest gap on the clock:
    a train runs well under a day, so the overnight gap between its last
    call and its next day's first is the longest.
    """
    timed = []
    for call in calls:
        if _call_time(call):
            timed.append(call)
    if not timed:
        return []
    start = None
    if origin_time is not None:
        # the header's departure time: a train that returns to its origin
        # (a loop out of Waterloo) is there again later on the clock or,
        # past midnight, earlier
        start = _minutes(origin_time)
    elif origin is not None:
        starts = [_call_minutes(c) for c in timed if c["tiploc_code"] == origin]
        if starts:
            start = min(starts)
    if start is None:
        clock = sorted({_call_minutes(c) for c in timed})
        gaps = [(b - a, b) for a, b in zip(clock, clock[1:])]
        gaps.append((clock[0] + MINUTES_PER_DAY - clock[-1], clock[0]))
        start = max(gaps)[1]

    def sequence(call: dict) -> int:
        minute = _call_minutes(call)
        return minute + (MINUTES_PER_DAY if minute < start else 0)

    return sorted(timed, key=sequence)


def _schedule_location(call: dict, position: str, passenger: bool) -> dict:
    """One ``schedule_location`` entry in the JSON feed's shape."""
    arrival = call.get("arr")
    departure = call.get("dep")
    passing = call.get("pass")
    # A public time exists only where a passenger train stops. The CRS is
    # not required here: platform TIPLOCs ("STOURBRIDGE JN (TOWN PLAT)",
    # "SHOTTON HIGH LEVEL") have none in the TIPLOC table and get theirs
    # later, from the timetable build's description match and overrides.
    # The WTT keeps half minutes and the public timetable does not, so the
    # fraction is dropped (see the module docstring).
    public = passenger and call.get("stops")
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


def _header_ends(header: dict, calls: list[dict], names: dict) -> list[dict]:
    """The origin and destination a header names but no printed row carries.

    Some lines are in no book (the Northumberland Line), so a train from
    Ashington is printed only from Heaton, yet its header still reads
    "ASHINGTON\n14:07". Where that end is a station the train's rows don't
    reach, it becomes the train's first or last call, at the header's time.
    """
    tiplocs = {call["tiploc_code"] for call in calls}
    stations = {call.get("crs_code") for call in calls}
    ends = []
    for field, time_field in (("origin", "dep"), ("destination", "arr")):
        name, _, time = header.get(field, "").partition("\n")
        end = _resolve(name, names) if name.strip() else None
        clock = re.fullmatch(r"(\d{2}):(\d{2})", time.strip())
        if end and end[1] and clock and end[0] not in tiplocs and end[1] not in stations:
            ends.append({"tiploc_code": end[0], "crs_code": end[1], "row": -1,
                         time_field: clock[1] + clock[2], "stops": True})
    return ends


def _to_schedule(
    uid: str,
    header: dict,
    calls: list[dict],
    operators: dict[str, str],
    names: dict | None = None,
) -> dict | None:
    """One ``JsonScheduleV1``-shaped record, or ``None``."""
    # The books print the origin as its name, a line break and the time.
    origin = _resolve(header.get("origin", "").split("\n")[0], names or {})
    headcode = header.get("headcode", "")
    passenger = _is_passenger(headcode)
    if passenger:
        calls = calls + _header_ends(header, calls, names or {})
    clock = re.fullmatch(r"(\d{2}):(\d{2})", header.get("origin", "").partition("\n")[2].strip())
    ordered = _order_calls(calls, origin[0] if origin else None, clock[1] + clock[2] if clock else None)
    if len(ordered) < _MINIMUM_CALLS:
        return None
    dates = _dates(header.get("dates", ""))
    if dates is None:
        return None
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


def _date_groups(parts: list[dict]) -> list[list[dict]]:
    """Split one UID's book entries into schedules by date range.

    Books disagree slightly about a train's dates, so overlapping ranges
    are one schedule. Ranges that don't overlap are versions of the train
    for different parts of the timetable (Portsmouth Harbour at 04:30 to
    18 September, 04:26 from 21 September), and merging them would mix
    their times. An entry with no dates goes with the first group.
    """
    dated = sorted(
        ((d, p) for p in parts if (d := _dates(p["header"].get("dates", ""))) is not None),
        key=lambda pair: pair[0],
    )
    groups: list[list[dict]] = []
    end = ""
    for (start, finish), part in dated:
        if groups and start <= end:
            groups[-1].append(part)
            end = max(end, finish)
        else:
            groups.append([part])
            end = finish
    undated = [p for p in parts if _dates(p["header"].get("dates", "")) is None]
    if not groups:
        return [undated] if undated else []
    groups[0].extend(undated)
    return groups


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
    partials: dict[tuple, list[dict]] = collections.defaultdict(list)
    for index, book in enumerate(books, start=1):
        if index % 100 == 0:
            LOG.info("  %d/%d books", index, len(books))
        try:
            columns = list(_iter_book_columns(book, names))
        except (zipfile.BadZipFile, KeyError) as error:
            LOG.warning("Skipping %s: %s", book.name, error)
            continue
        for uid, partial in columns:
            partials[_schedule_key(uid, partial["header"])].append(partial)

    merged = [(uid, group) for (uid, _), parts in partials.items() for group in _date_groups(parts)]
    LOG.info("Merged %d distinct schedules", len(merged))
    for uid, parts in merged:
        header: dict = {}
        masks: collections.Counter = collections.Counter()
        for partial in parts:
            header = _widest_header(header, partial["header"])
            masks[_days_mask(partial["header"].get("days", ""))] += 1
        schedule = _to_schedule(
            uid,
            header | {"days_mask": masks.most_common(1)[0][0]},
            _merge_calls([call for partial in parts for call in partial["calls"]]),
            operators or {},
            names,
        )
        if schedule is not None:
            yield schedule
