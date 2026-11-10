"""Read a fixed-width (RSPS5046) CIF as if it were the JSON feed.

Network Rail publishes the daily schedule extract in newline-delimited
JSON, which is what :mod:`ortla.timetable` parses. The archive
extracts (the published timetable for a past period, and the only way to
build an era-matched "before" network) are issued in the original
80-column fixed-width CIF. This module reads that format and yields the
same record shapes, so every consumer downstream is unchanged. Given the
JSON feed instead, its readers pass the records through, so a caller
need not know which format it holds.

The two formats carry the same fields under different names. The mapping
is exact for everything the model reads, and
``scripts/verify_cif_archive.py`` tests it by parsing the fixed-width
and the JSON issue of the same extract and comparing them record by
record.

Record types handled: ``HD`` (header), ``TI``/``TA``/``TD`` (location
reference), ``BS`` (basic schedule), ``BX`` (basic schedule extra),
``LO``/``LI``/``LT`` (calling points), ``AA`` (associations). ``CR`` (change en route) is
read and discarded, matching the JSON feed's single ``schedule_segment``; ``ZZ`` (trailer) is
skipped.
"""

from __future__ import annotations

import json
import logging
import pathlib  # noqa: TC003
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# A fixed-width CIF opens with an 80-column HD record; the JSON feed
# opens with a ``JsonTimetableV1`` object.
_HEADER_RECORD_ID = "HD"
_RECORD_WIDTH = 80
# CIF dates are YYMMDD. The format predates the millennium and has no
# century, so a year at or above this pivot is read as 19xx. Archive
# extracts run from the 1990s to the present, and nothing in the feed
# reaches 2060.
_CENTURY_PIVOT = 60
# The CIF writes "no public call" as four zeroes rather than as blanks,
# and an unknown STANOX as five zeroes; the JSON feed writes both as
# null.
_NO_PUBLIC_TIME = "0000"
_NO_STANOX = "00000"
# CIF dates are six digits, YYMMDD.
_DATE_WIDTH = 6

_TRANSACTION_TYPES = {"N": "Create", "D": "Delete", "R": "Update"}
# The daily feed always fills the STP indicator (C cancellation, N new,
# O overlay, P permanent). A published timetable extract carries the
# permanent plan only and leaves the field blank, as every BS record in
# the Dec-25 and May-26 extracts does. Blank is read as permanent,
# because every filter downstream selects permanent schedules by "P" and
# would otherwise discard the whole extract.
_PERMANENT_STP = "P"


def is_fixed_width_cif(path: pathlib.Path) -> bool:
    """Whether *path* is a fixed-width CIF rather than the JSON feed.

    Decided on the first record: a fixed-width extract opens with an
    80-column ``HD`` header, the JSON feed with a ``JsonTimetableV1``
    object.
    """
    with path.open(encoding="latin-1") as handle:
        first = handle.readline().rstrip("\r\n")
    return first.startswith(_HEADER_RECORD_ID) and len(first) == _RECORD_WIDTH


def _iter_json_records(path: pathlib.Path, record_type: str) -> Iterator[dict]:
    """Yield each ``record_type`` record of the JSON feed.

    Only lines naming the type are JSON-parsed, which keeps a pass
    cheap.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if f'"{record_type}"' in line:
                yield json.loads(line)[record_type]


def _text(line: str, start: int, end: int) -> str | None:
    """Return a stripped field, or None when it is blank."""
    value = line[start:end].strip()
    return value or None


def _date(line: str, start: int) -> str | None:
    """Return a YYMMDD field as an ISO date."""
    raw = line[start : start + 6].strip()
    if len(raw) != _DATE_WIDTH or not raw.isdigit():
        return None
    year = int(raw[:2])
    century = 1900 if year >= _CENTURY_PIVOT else 2000
    return f"{century + year:04d}-{raw[2:4]}-{raw[4:6]}"


def _working_time(line: str, start: int) -> str | None:
    """Return a 5-character working time, keeping any ``H``.

    The JSON feed publishes ``"1026H"`` for a half minute and ``"1026"``
    otherwise, which is what this returns.
    """
    return _text(line, start, start + 5)


def _public_time(line: str, start: int) -> str | None:
    """Return a 4-character public time, or None if not public."""
    raw = line[start : start + 4].strip()
    if not raw or raw == _NO_PUBLIC_TIME:
        return None
    return raw


def _location(line: str) -> tuple[str, str | None]:
    """Split the location field into TIPLOC and instance suffix."""
    tiploc = line[2:9].strip()
    instance = line[9:10].strip() or None
    return tiploc, instance


def iter_tiploc_records(path: pathlib.Path) -> Iterator[dict]:
    """Yield one ``TiplocV1``-shaped dict per location record."""
    if not is_fixed_width_cif(path):
        yield from _iter_json_records(path, "TiplocV1")
        return
    with path.open(encoding="latin-1") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line[:2] not in {"TI", "TA", "TD"}:
                continue
            tiploc = line[2:9].strip()
            if not tiploc:
                continue
            yield {
                "transaction_type": _TRANSACTION_TYPES.get(line[:2][1], "Create"),
                "tiploc_code": tiploc,
                "nalco": _text(line, 11, 17),
                "stanox": None if line[44:49] == _NO_STANOX else _text(line, 44, 49),
                "crs_code": _text(line, 53, 56),
                "description": _text(line, 56, 72),
                "tps_description": _text(line, 18, 44),
            }


def _basic_schedule(line: str) -> dict:
    """Parse a ``BS`` record into the feed's schedule fields."""
    return {
        "CIF_bank_holiday_running": _text(line, 28, 29),
        "CIF_stp_indicator": _text(line, 79, 80) or _PERMANENT_STP,
        "CIF_train_uid": line[3:9].strip(),
        "applicable_timetable": None,
        "atoc_code": None,
        "new_schedule_segment": {"traction_class": "", "uic_code": ""},
        "schedule_days_runs": _text(line, 21, 28),
        "schedule_end_date": _date(line, 15),
        "schedule_start_date": _date(line, 9),
        "train_status": _text(line, 29, 30),
        "transaction_type": _TRANSACTION_TYPES.get(line[2:3], "Create"),
    }


def _schedule_segment(line: str) -> dict:
    """Parse the segment fields carried on the same ``BS`` record."""
    return {
        "signalling_id": _text(line, 32, 36),
        "CIF_train_category": _text(line, 30, 32),
        "CIF_headcode": _text(line, 36, 40),
        "CIF_course_indicator": _text(line, 40, 41),
        "CIF_train_service_code": _text(line, 41, 49),
        "CIF_business_sector": _text(line, 49, 50),
        "CIF_power_type": _text(line, 50, 53),
        "CIF_timing_load": _text(line, 53, 57),
        "CIF_speed": _text(line, 57, 60),
        "CIF_operating_characteristics": _text(line, 60, 66),
        "CIF_train_class": _text(line, 66, 67),
        "CIF_sleepers": _text(line, 67, 68),
        "CIF_reservations": _text(line, 68, 69),
        "CIF_connection_indicator": _text(line, 69, 70),
        "CIF_catering_code": _text(line, 70, 74),
        "CIF_service_branding": _text(line, 74, 78),
        "schedule_location": [],
    }


def _origin_location(line: str) -> dict:
    tiploc, instance = _location(line)
    return {
        "location_type": "LO",
        "record_identity": "LO",
        "tiploc_code": tiploc,
        "tiploc_instance": instance,
        "departure": _working_time(line, 10),
        "public_departure": _public_time(line, 15),
        "platform": _text(line, 19, 22),
        "line": _text(line, 22, 25),
        "engineering_allowance": _text(line, 25, 27),
        "pathing_allowance": _text(line, 27, 29),
        "performance_allowance": _text(line, 41, 43),
    }


def _intermediate_location(line: str) -> dict:
    tiploc, instance = _location(line)
    return {
        "location_type": "LI",
        "record_identity": "LI",
        "tiploc_code": tiploc,
        "tiploc_instance": instance,
        "arrival": _working_time(line, 10),
        "departure": _working_time(line, 15),
        "pass": _working_time(line, 20),
        "public_arrival": _public_time(line, 25),
        "public_departure": _public_time(line, 29),
        "platform": _text(line, 33, 36),
        "line": _text(line, 36, 39),
        "path": _text(line, 39, 42),
        "engineering_allowance": _text(line, 54, 56),
        "pathing_allowance": _text(line, 56, 58),
        "performance_allowance": _text(line, 58, 60),
    }


def _terminating_location(line: str) -> dict:
    tiploc, instance = _location(line)
    return {
        "location_type": "LT",
        "record_identity": "LT",
        "tiploc_code": tiploc,
        "tiploc_instance": instance,
        "arrival": _working_time(line, 10),
        "public_arrival": _public_time(line, 15),
        "platform": _text(line, 19, 22),
        "path": _text(line, 22, 25),
    }


_LOCATION_PARSERS = {
    "LO": _origin_location,
    "LI": _intermediate_location,
    "LT": _terminating_location,
}


def iter_schedule_records(path: pathlib.Path) -> Iterator[dict]:
    """Yield one ``JsonScheduleV1``-shaped dict per schedule.

    A schedule spans several records (``BS``, an optional ``BX``, then
    its calling points), so one is emitted when the next ``BS`` opens or
    the file ends. Deletion records (``BS`` with transaction type ``D``)
    carry no locations and are yielded as the JSON feed yields them,
    with an empty calling pattern.
    """
    if not is_fixed_width_cif(path):
        yield from _iter_json_records(path, "JsonScheduleV1")
        return
    current: dict | None = None
    segment: dict | None = None

    with path.open(encoding="latin-1") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            identity = line[:2]

            if identity == "BS":
                if current is not None:
                    yield current
                segment = _schedule_segment(line)
                current = _basic_schedule(line)
                current["schedule_segment"] = segment
            elif current is None:
                continue
            elif identity == "BX":
                current["atoc_code"] = _text(line, 11, 13)
                current["applicable_timetable"] = _text(line, 13, 14)
                current["new_schedule_segment"] = {
                    "traction_class": line[2:6].strip(),
                    "uic_code": line[6:11].strip(),
                }
            elif identity in _LOCATION_PARSERS and segment is not None:
                segment["schedule_location"].append(_LOCATION_PARSERS[identity](line))

    if current is not None:
        yield current


def iter_association_records(path: pathlib.Path) -> Iterator[dict]:
    """Yield one ``JsonAssociationV1``-shaped dict per association (join, divide, next)."""
    if not is_fixed_width_cif(path):
        for record in _iter_json_records(path, "JsonAssociationV1"):
            for field in ("assoc_start_date", "assoc_end_date"):
                record[field] = (record.get(field) or "")[:10] or None
            yield record
        return
    with path.open(encoding="latin-1") as handle:
        for raw in handle:
            if not raw.startswith("AA"):
                continue
            line = raw.rstrip("\r\n")
            yield {
                "transaction_type": _TRANSACTION_TYPES.get(line[2], line[2]),
                "main_train_uid": line[3:9].strip(),
                "assoc_train_uid": line[9:15].strip(),
                "assoc_start_date": _date(line, 15),
                "assoc_end_date": _date(line, 21),
                "assoc_days": line[27:34],
                "category": _text(line, 34, 36),
                "date_indicator": _text(line, 36, 37),
                "location": _text(line, 37, 44),
                "CIF_stp_indicator": _text(line, 79, 80) or _PERMANENT_STP,
            }
