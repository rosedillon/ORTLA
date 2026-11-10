"""Checks on the midnight wrap, codes, return workings and loops."""

from __future__ import annotations

import datetime
import io
import json
import pathlib
import types
import zipfile

import pandas as pd
import pytest

from ortla import (
    appraisal,
    cif_archive,
    clock,
    commercial_case,
    fares,
    feasibility,
    hindcast,
    scenario,
    stations,
    timetable,
    wtt_workbook,
)
from ortla.clock import interval_time_cutoffs
from ortla.config import CandidateService, ServiceStop
from ortla.service_lookup import crs_to_tiploc_map

CUTOFFS = {"am_peak": "7:00", "interpeak": "9:30", "pm_peak": "16:00", "evening": "19:00"}


def _candidate(stops: list[ServiceStop]) -> CandidateService:
    return CandidateService(name="test", traction_code="158", days_of_week=["Monday"], stops=stops)


def test_headway_neighbours_wrap_at_midnight() -> None:
    # A 23:55 departure sits 10 minutes before tomorrow's 00:05.
    assert feasibility._neighbours([5.0, 600.0], 1435.0) == (600.0, 1445.0)
    # Shifted past midnight, it is still 10 minutes after yesterday's
    # 23:55.
    assert feasibility._neighbours([600.0, 1435.0], 1445.0) == (1435.0, 2040.0)


def test_clear_intervals_pad_both_ends_of_the_day() -> None:
    assert feasibility._clear_intervals([10.0, 1430.0], 30.0) == [(40.0, 1400.0)]
    assert feasibility._clear_intervals([], 30.0) == [(0.0, 1440.0)]


def test_crs_to_tiploc_map_takes_the_busiest_tiploc() -> None:
    calls = pd.DataFrame(
        {"crs": ["CLJ", "CLJ", "CLJ"], "tiploc": ["CLPHMJN", "CLPHMJN", "CLPHMJW"]}
    )
    assert crs_to_tiploc_map(calls) == {"CLJ": "CLPHMJN"}


def test_recode_stops_uses_the_odm_code() -> None:
    candidate = _candidate(
        [
            ServiceStop(station="pdx", departure="10:00"),
            ServiceStop(station="RDG", arrival="10:30"),
        ]
    )
    recoded = feasibility.recode_stops(candidate, {"PDX": "PAD"})
    assert [stop.station for stop in recoded.stops] == ["PAD", "RDG"]


def test_return_working_with_one_time_at_a_stop() -> None:
    candidate = _candidate(
        [
            ServiceStop(station="AAA", departure="10:00"),
            ServiceStop(station="BBB", arrival="10:20"),
            ServiceStop(station="CCC", departure="10:40"),
        ]
    ).model_copy(update={"return_after_minutes": 20})
    stops = candidate.return_working().stops
    assert [(s.station, s.arrival, s.departure) for s in stops] == [
        ("CCC", None, "11:00"),
        ("BBB", "11:20", "11:20"),
        ("AAA", "11:40", None),
    ]


def test_loop_train_counts_once_per_pair() -> None:
    # Hamilton Square -> James Street -> Moorfields -> James Street, as
    # the Wirral loop calls.
    stops = [("HAM", 8), ("JST", 8), ("MRF", 8), ("JST", 9)]
    calls = pd.DataFrame(
        {
            "train_uid": "X",
            "start_date": "d",
            "end_date": "d",
            "svc_origin_dep": "0800",
            "signalling_id": "2X01",
            "seq": range(len(stops)),
            "crs": [crs for crs, _ in stops],
            "is_north": True,
            "hour": [hour for _, hour in stops],
        }
    )
    counts = timetable._count_route_hours(calls)
    assert counts[("HAM", "JST", 8)] == 1
    assert counts[("HAM", "JST", 9)] == 0
    assert sum(counts.values()) == 4  # HAM-JST, HAM-MRF, JST-MRF, MRF-JST


def test_period_minutes_follow_the_cutoffs() -> None:
    assert clock.period_minutes(CUTOFFS) == clock.PERIOD_MINUTES
    half_hours = clock.period_minutes(interval_time_cutoffs(30))
    assert len(half_hours) == 48
    assert set(half_hours.values()) == {30.0}


def test_joint_counter_takes_a_loop_trains_shortest_leg() -> None:
    candidate = _candidate(
        [
            ServiceStop(station="HAM", departure="08:00"),
            ServiceStop(station="JST", arrival="08:05", departure="08:05"),
            ServiceStop(station="MRF", arrival="08:10", departure="08:10"),
            ServiceStop(station="JST", arrival="08:20", departure="08:20"),
            ServiceStop(station="BKC", arrival="08:30"),
        ]
    )
    counts = hindcast._joint_counter([candidate], {"am_peak": "7:00", "late": "8:15"})
    assert counts[("Monday", "day", "HAM", "JST")] == 1
    # JST -> BKC is priced from the second call at James Street, at
    # 08:20.
    assert counts[("Monday", "late", "JST", "BKC")] == 1
    assert counts[("Monday", "am_peak", "JST", "BKC")] == 0


def test_crowding_multiplier_is_flat_beyond_crush() -> None:
    assert appraisal._crowding_multiplier(2.0) == 1.90
    assert appraisal._crowding_multiplier(12.0) == 1.90


def test_no_payback_for_a_loss_maker_without_capital() -> None:
    cashflow = pd.DataFrame(
        {"year": [2026, 2027, 2028, 2029], "cumulative_net_cashflow": [0.0, 0.0, -1.0, -2.0]}
    )
    assert commercial_case._payback_year(cashflow, "cumulative_net_cashflow") is None
    cashflow["cumulative_net_cashflow"] = [0.0, 0.0, 1.0, 2.0]
    assert commercial_case._payback_year(cashflow, "cumulative_net_cashflow") == 2028


def test_import_from_propose_reads_captured_trips(tmp_path) -> None:
    pd.DataFrame(
        {
            "predicted_annual_trips_captured": [100.0, 300.0],
            "predicted_annual_revenue_gbp": [1000.0, 3000.0],
        }
    ).to_csv(tmp_path / "propose_svc_demand_by_leg.csv", index=False)
    [service] = commercial_case.import_from_propose(tmp_path)
    assert (service.name, service.annual_passengers, service.avg_fare) == ("svc", 400, 10.0)


def test_shared_strings_join_rich_text_runs() -> None:
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    xml = (
        f'<sst xmlns="{ns}"><si><t>1A01</t></si>'
        "<si><r><t>Lee</t></r><r><t>ds</t></r><rPh><t>x</t></rPh></si>"
        "<si><t>0905</t></si></sst>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as book:
        book.writestr("xl/sharedStrings.xml", xml)
    with zipfile.ZipFile(buffer) as book:
        assert wtt_workbook._shared_strings(book) == ["1A01", "Leeds", "0905"]


def test_cif_readers_pass_the_json_feed_through(tmp_path) -> None:
    path = tmp_path / "feed.json"
    path.write_text(
        json.dumps({"JsonTimetableV1": {}})
        + "\n"
        + json.dumps({"TiplocV1": {"tiploc_code": "LEEDS", "crs_code": "LDS"}})
        + "\n"
        + json.dumps({"JsonScheduleV1": {"CIF_train_uid": "C1"}})
        + "\n"
    )
    assert [r["crs_code"] for r in cif_archive.iter_tiploc_records(path)] == ["LDS"]
    assert [r["CIF_train_uid"] for r in cif_archive.iter_schedule_records(path)] == ["C1"]


def test_flow_file_keeps_only_flows_valid_on_the_date(tmp_path) -> None:
    def flow(flow_id: str, end: str, start: str) -> str:
        return f"RF12345678ROUTE   0S{end}{start}TOCN  {flow_id}\n"

    def fare(flow_id: str, pence: str) -> str:
        return f"RT{flow_id}SDS{pence}  \n"

    path = tmp_path / "RJFA.FFL"
    path.write_text(
        flow("0000001", "28022026", "01012020")
        + fare("0000001", "00001000")
        + flow("0000002", "31122999", "01032026")
        + fare("0000002", "00001100")
    )
    before = fares.parse_flow_file(path, datetime.date(2026, 2, 1))
    after = fares.parse_flow_file(path, datetime.date(2026, 3, 1))
    assert before["fare_pence"].tolist() == [1000]
    assert after["fare_pence"].tolist() == [1100]


def test_route_cost_survives_an_unroutable_leg() -> None:
    def distance(a: str, b: str, *, plausible: bool) -> float:
        if "RDZ" in (a, b):
            raise ValueError("no location")
        return 10.0

    gravity = types.SimpleNamespace(calc=types.SimpleNamespace(calculate_distance=distance))
    assumptions = scenario.ScenarioAssumptions(
        tag=None, operating_cost_per_train_km=1e6 / 52, annual_operating_cost_m=0.0
    )
    candidate = _candidate(
        [
            ServiceStop(station="RDG", departure="10:00"),
            ServiceStop(station="RDZ", arrival="10:05", departure="10:06"),
            ServiceStop(station="WOK", arrival="10:30", departure="10:31"),
            ServiceStop(station="GLD", arrival="10:40"),
        ]
    )
    # Three legs, one routed at 10 km: the two through RDZ are costed at
    # 10 km too.
    costed = scenario.with_route_cost(assumptions, candidate, gravity)
    assert costed.annual_operating_cost_m == pytest.approx(30.0)


def test_hourly_matrix_names_have_no_colon(tmp_path: pathlib.Path) -> None:
    # Windows cannot hold "07:00" in a file name.
    path = clock.period_matrix_path(tmp_path, "Monday", "07:00")
    assert path.name == "odm_Monday_0700.csv"
    label = path.stem.split("_", 2)[2]
    assert clock.period_from_file_label(label) == "07:00"
    assert clock.period_from_file_label("am_peak") == "am_peak"


def test_station_list_keeps_nantwich(tmp_path: pathlib.Path) -> None:
    # Nantwich's CRS is NAN, which is not a missing value.
    path = tmp_path / "stations.csv"
    path.write_text("stationName,crsCode,tiploc\nNantwich,NAN,NANTWCH\nNowhere,,NOWHERE\n")
    codes, tiploc_to_crs = stations.read_station_list(path)
    assert codes == {"NAN"}
    assert tiploc_to_crs == {"NANTWCH": "NAN"}
