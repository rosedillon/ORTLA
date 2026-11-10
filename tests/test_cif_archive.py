"""Checks on the fixed-width CIF reader.

The archive extracts, the only source of a timetable as it was published
for a past period and so the only way to build an era-matched "before"
network, come in 80-column fixed-width CIF rather than the daily feed's
JSON. A wrong column offset there fails silently: every field still
parses one character out, and the run completes with slightly wrong
times.

The records below are real lines taken from the May-26 extract, whose
JSON issue was compared field by field against this reader by
``scripts/verify_cif_archive.py`` (244,618 schedules; the only
disagreements were schedules the two extracts, taken four days apart,
genuinely revised).
"""

from __future__ import annotations

import pathlib

import pandas as pd

from ortla import cif_archive
from ortla.timetable import load_tiploc_crs_map

# One test train, Kidsgrove to Longport: a real schedule from that
# extract, with public times written onto its origin and terminus (the
# real ones are a test run, so they carry none) to exercise the
# public-call path as well.
_SCHEDULE = [
    "BSNC012772605182612071000000 FDT2Q14    152495119 DMUS   075QY                 P",
    "BX         ZZY                                                                  ",
    "LOKIDSGRV 1522 1522          TB                                                 ",
    "LIBRDWJN            1527 00000000   GL                                          ",
    "LILNGP573 1528H1530H     00000000         RM                                    ",
    "LTLNGPLRC 1556 1556 2    TF                                                     ",
]
# A location record with a published CRS, and one the CIF leaves without
# one.
_TIPLOCS = [
    "TAABGLELE00244800AABERGELE & PENSARN        40073   0AGLABERGELE & PENSN        ",
    "TAABCWM  00385964VABERCWMBOI                78128   0                           ",
]
_HEADER = "HDTPS.UDFROC1.PD2609111109262005DFROC1P       FA110926110927" + " " * 20


def _write(path: pathlib.Path, lines: list[str]) -> pathlib.Path:
    path.write_text("\r\n".join(lines) + "\r\n")
    return path


def test_fixed_width_extract_is_told_apart_from_the_json_feed(
    tmp_path: pathlib.Path,
) -> None:
    """The reader goes by the first record, not by the file name."""
    archive = _write(tmp_path / "archive", [_HEADER, *_TIPLOCS, *_SCHEDULE])
    feed = tmp_path / "feed.json"
    feed.write_text('{"JsonTimetableV1":{"classification":"public"}}\n')

    assert cif_archive.is_fixed_width_cif(archive)
    assert not cif_archive.is_fixed_width_cif(feed)


def test_schedule_fields_land_in_the_right_columns(tmp_path: pathlib.Path) -> None:
    """Every field the model reads, against one hand-decoded record."""
    path = _write(tmp_path / "archive", [_HEADER, *_SCHEDULE])

    (schedule,) = cif_archive.iter_schedule_records(path)

    assert schedule["CIF_train_uid"] == "C01277"
    assert schedule["CIF_stp_indicator"] == "P"
    assert schedule["schedule_start_date"] == "2026-05-18"
    assert schedule["schedule_end_date"] == "2026-12-07"
    assert schedule["schedule_days_runs"] == "1000000"
    assert schedule["train_status"] == "F"
    # The ATOC code comes off the BX record that follows the BS, not the
    # BS.
    assert schedule["atoc_code"] == "ZZ"

    segment = schedule["schedule_segment"]
    assert segment["signalling_id"] == "2Q14"
    assert segment["CIF_train_category"] == "DT"
    assert segment["CIF_power_type"] == "DMU"
    assert segment["CIF_speed"] == "075"


def test_calling_points_keep_working_public_and_half_minute_times(
    tmp_path: pathlib.Path,
) -> None:
    """Each of the three location layouts gives its own fields."""
    path = _write(tmp_path / "archive", [_HEADER, *_SCHEDULE])

    (schedule,) = cif_archive.iter_schedule_records(path)
    origin, junction, half, terminus = schedule["schedule_segment"]["schedule_location"]

    assert origin["location_type"] == "LO"
    assert origin["tiploc_code"] == "KIDSGRV"
    assert origin["departure"] == "1522"
    assert origin["public_departure"] == "1522"

    # A pass, not a call: the CIF writes no public time as four zeroes,
    # which the JSON feed publishes as null, and the model reads as "not
    # a call".
    assert junction["pass"] == "1527"
    assert junction["public_arrival"] is None
    assert junction["public_departure"] is None

    # Half minutes carry an H suffix, which the model's time parsing
    # expects.
    assert half["arrival"] == "1528H"
    assert half["departure"] == "1530H"

    assert terminus["location_type"] == "LT"
    assert terminus["arrival"] == "1556"
    assert terminus["public_arrival"] == "1556"
    assert terminus["platform"] == "2"


def test_location_reference_reads_published_and_missing_crs(
    tmp_path: pathlib.Path,
) -> None:
    """The TIPLOC -> CRS map builds from an archive as from the feed."""
    path = _write(tmp_path / "archive", [_HEADER, *_TIPLOCS, *_SCHEDULE])

    tiploc_to_crs = load_tiploc_crs_map(path)

    assert tiploc_to_crs["ABGLELE"] == "AGL"
    # Published without a CRS, and with no station reference to recover
    # it from.
    assert tiploc_to_crs["ABCWM"] == ""


def test_an_unknown_stanox_reads_as_missing_not_as_zero(
    tmp_path: pathlib.Path,
) -> None:
    """The CIF writes a missing STANOX as 00000; the feed, null."""
    path = _write(tmp_path / "archive", [_HEADER, *_TIPLOCS])

    records = {r["tiploc_code"]: r for r in cif_archive.iter_tiploc_records(path)}

    assert records["ABGLELE"]["stanox"] == "40073"
    assert records["ABGLELE"]["tps_description"] == "ABERGELE & PENSARN"


def test_a_published_timetable_extract_reads_as_permanent(
    tmp_path: pathlib.Path,
) -> None:
    """A blank STP indicator means permanent, not "no indicator".

    The daily feed fills the field; a published timetable extract
    carries the permanent plan only and leaves it blank. Reading that as
    missing drops every schedule at the first filter, which selects
    ``"P"``.
    """
    published = _SCHEDULE[0][:79] + " "
    path = _write(tmp_path / "archive", [_HEADER, published, *_SCHEDULE[1:]])

    (schedule,) = cif_archive.iter_schedule_records(path)

    assert schedule["CIF_stp_indicator"] == "P"


def test_joins_and_divides_are_written_with_their_dates(tmp_path):
    from ortla import timetable

    cif = tmp_path / "x.cif"
    header = "HD" + " " * 78
    divide = "AANC14853C148852605172612120000100VVSCNR     TP".ljust(80)
    nextp = "AANC00202Y149382605172612060000001NPSKNGX     TO".ljust(80)
    cif.write_text("\n".join([header, divide, nextp]) + "\n", encoding="latin-1")
    path = timetable.write_associations(cif, {"CNR": "CNR"}, tmp_path)
    rows = pd.read_csv(path, dtype=str)
    assert rows.to_dict("records") == [
        {
            "main_uid": "C14853",
            "portion_uid": "C14885",
            "category": "VV",
            "tiploc": "CNR",
            "crs": "CNR",
            "start_date": "2026-05-17",
            "end_date": "2026-12-12",
            "days": "0000100",
            "CIF_stp_indicator": "P",
        }
    ]
