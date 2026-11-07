"""Checks on the network inputs the discovery scan depends on.

Both cover defects that silently produced plausible-looking but wrong
national results: TIPLOCs the CIF publishes without a CRS (which deleted
Waterloo, Victoria, London Bridge, Clapham Junction and Vauxhall from a
national run), and an existing-service test applied one direction at a
time.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

from orta.discovery import _drop_well_served_pairs
from orta.timetable import load_tiploc_crs_map

_STATIONS = [
    ("London Waterloo", "WAT"),
    ("London Victoria", "VIC"),
    ("Clapham Junction", "CLJ"),
    ("Harrow & Wealdstone", "HRW"),
    ("New Cross Gate", "NXG"),
    ("Shotton", "SHT"),
    ("Okehampton", "OKE"),
    ("Gourock", "GRK"),
    ("Stourbridge Junction", "SBJ"),
]

# Each entry is a TIPLOC code, its published crs_code, and its
# tps_description.
_TIPLOCS = [
    ("WATRLOO", "WAT", "LONDON WATERLOO"),  # published normally
    ("WATRLMN", None, "LONDON WATERLOO"),  # exact name match
    ("VICTRIE", None, "LONDON VICTORIA (E)"),  # bracketed platform group
    ("CLPHMJ1", None, "CLAPHAM JUNCTION PLATS 0-2"),  # platform numbers
    ("HROWDC", None, "HARROW & WEALDSTONE D.C."),  # ampersand + D.C. line
    ("NEWXGEL", None, "NEW CROSS GATE STATION ELL"),  # two qualifiers
    ("SHOTTHL", None, "SHOTTON HIGH LEVEL"),  # high/low level
    ("OKHMPWS", None, "OKEHAMPTON WEST STREET"),  # rail-replacement bus stop
    ("GOURPIE", None, "GOUROCK PIER"),  # ferry pier
    ("LRDDEAC", None, "LONDON ROAD DEPOT"),  # depot
    ("STRBDG1", None, "STOURBRIDGE JN (TOWN PLAT)"),  # too abbreviated to match
]


def _write_cif(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "cif.json"
    with path.open("w") as handle:
        handle.write(json.dumps({"JsonTimetableV1": {"classification": "public"}}) + "\n")
        for tiploc, crs, description in _TIPLOCS:
            handle.write(
                json.dumps(
                    {
                        "TiplocV1": {
                            "tiploc_code": tiploc,
                            "crs_code": crs,
                            "tps_description": description,
                        }
                    }
                )
                + "\n"
            )
    return path


def _write_stations(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "stations.csv"
    pd.DataFrame(_STATIONS, columns=["stationName", "crsCode"]).to_csv(path, index=False)
    return path


def test_platform_group_tiplocs_recover_their_crs(tmp_path: pathlib.Path) -> None:
    """A station with only platform-group TIPLOCs must survive."""
    cif = _write_cif(tmp_path)

    without = load_tiploc_crs_map(cif)
    # The defect: every platform-group variant is unmapped, so a station
    # whose only TIPLOCs are variants disappears from the timetable
    # entirely.
    assert without["WATRLMN"] == ""
    assert without["VICTRIE"] == ""
    assert not any(crs == "VIC" for crs in without.values())

    recovered = load_tiploc_crs_map(cif, station_reference=_write_stations(tmp_path))
    assert recovered["WATRLMN"] == "WAT"
    assert recovered["VICTRIE"] == "VIC"
    assert recovered["CLPHMJ1"] == "CLJ"
    assert recovered["HROWDC"] == "HRW"
    assert recovered["NEWXGEL"] == "NXG"
    assert recovered["SHOTTHL"] == "SHT"
    # A published CRS is never disturbed.
    assert recovered["WATRLOO"] == "WAT"


def test_bus_ferry_and_depot_points_are_not_given_a_station_crs(
    tmp_path: pathlib.Path,
) -> None:
    """Mapping these would invent train service that does not exist."""
    recovered = load_tiploc_crs_map(
        _write_cif(tmp_path), station_reference=_write_stations(tmp_path)
    )
    assert recovered["OKHMPWS"] == ""  # "OKEHAMPTON WEST STREET" is a bus stop
    assert recovered["GOURPIE"] == ""  # "GOUROCK PIER" is a ferry pier
    assert recovered["LRDDEAC"] == ""  # "LONDON ROAD DEPOT" is a depot


def test_abbreviated_description_needs_an_explicit_override(
    tmp_path: pathlib.Path,
) -> None:
    """Abbreviations are not expanded; overrides cover the rest."""
    cif = _write_cif(tmp_path)
    stations = _write_stations(tmp_path)

    # "STOURBRIDGE JN" is not "Stourbridge Junction", and expanding
    # abbreviations would risk mapping things that are not stations, so
    # it is left unmapped.
    assert load_tiploc_crs_map(cif, station_reference=stations)["STRBDG1"] == ""

    overridden = load_tiploc_crs_map(cif, station_reference=stations, overrides={"STRBDG1": "SBJ"})
    assert overridden["STRBDG1"] == "SBJ"


def test_existing_service_is_judged_on_the_busier_direction() -> None:
    """A corridor is served in both directions or neither.

    Chichester -> London Victoria has one through working on a typical
    Wednesday against 32 the other way, so testing ordered pairs on
    their own reports a well-served corridor as an unserved market.
    """
    frequency = {("CCH", "VIC"): 1.0, ("VIC", "CCH"): 32.0}
    pairs = [("CCH", "VIC", 56.4), ("VIC", "CCH", 56.4)]

    symmetric = _drop_well_served_pairs(pairs, frequency, 1.0)
    assert symmetric == []

    one_way = _drop_well_served_pairs(pairs, frequency, 1.0, both_directions=False)
    assert [(o, d) for o, d, _, _ in one_way] == [("CCH", "VIC")]


def test_genuinely_unserved_pairs_survive_the_symmetric_test() -> None:
    """The symmetric test must not swallow a real gap."""
    frequency = {("GTW", "VIC"): 142.0}
    kept = _drop_well_served_pairs([("GTW", "BHM", 117.8)], frequency, 1.0)
    # Absent from the summary in both directions: no direct service,
    # kept at 0.
    assert kept == [("GTW", "BHM", 117.8, 0.0)]
