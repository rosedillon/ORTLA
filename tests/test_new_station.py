"""SDFT port: schedule resolution, catchments and the trip end."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ortla import new_station as ns


def _calls(rows):
    cols = ["train_uid", "stp_indicator", "start_date", "end_date", "days_runs", "crs"]
    return pd.DataFrame(rows, columns=cols)


def test_overlay_replaces_permanent():
    wednesday = pd.Timestamp("2026-06-17")
    calls = _calls(
        [
            ("A1", "P", "2026-05-17", "2026-12-12", "1111100", "LDS"),
            ("A1", "O", "2026-06-15", "2026-06-19", "1111100", "YRK"),
            ("B2", "N", "2026-06-01", "2026-06-30", "0010000", "HGT"),
            ("C3", "P", "2026-05-17", "2026-12-12", "1111100", "SHF"),
            ("C3", "C", "2026-06-17", "2026-06-17", "0010000", "SHF"),
        ]
    )
    running = ns.running_calls(calls, wednesday)
    assert sorted(running["crs"]) == ["HGT", "YRK"]


def test_catchment_keeps_everyone_within_750m():
    stations = pd.DataFrame(
        {
            "e": [0.0, 400.0],
            "n": [0.0, 0.0],
            "freq": [50.0, 10.0],
            **{k: [0.0, 0.0] for k in ("cat_f", "cctv", "car_spaces", "ticket_machine", "bus")},
        },
        index=["AAA", "BBB"],
    )
    people = pd.DataFrame({"e": [100.0, 300.0], "n": [0.0, 0.0], "pop": [60.0, 40.0]})
    # With two stations the ten nearest pad with repeats, so give the
    # tree enough points to look up.
    padded = pd.concat(
        [stations]
        + [
            stations.iloc[[0]].assign(e=1e7 + i).rename(index=lambda _, i=i: f"Z{i}")
            for i in range(10)
        ]
    )
    pop = ns.catchments(padded, people)
    assert np.isclose(pop[["AAA", "BBB"]].sum(), 100.0, rtol=1e-6)
    assert pop["AAA"] > pop["BBB"]


def test_trip_end_by_hand():
    row = pd.DataFrame(
        {
            "prw_pop": [9999.0],
            "freq": [np.e**4],
            "jobs": [0.0],
            "car_spaces": [0.0],
            "electric": [1.0],
            "travelcard": [0.0],
            "terminus": [0.0],
        }
    )
    expected = 3.5 + 0.3663 * np.log(10000) + 1.1245 * 4 + 0.2366
    assert np.isclose(ns.trip_end(row, 3.5).iloc[0], expected)


def test_borrowing_leans_on_near_stations_and_never_on_itself():
    cal = pd.DataFrame({"e": [0.0, 1000.0, 50_000.0], "n": 0.0}, index=["A", "B", "C"])
    new = pd.DataFrame({"e": [0.0], "n": [0.0]}, index=["A"])
    weights = ns.distance_weights(new, cal, 2.0)
    assert weights[0, 0] == 0.0  # its own residual
    assert weights[0, 2] < 1e-10  # 50 km away
    lift = ns.borrow(weights, np.array([9.0, 1.0, 9.0]), 0.25)
    w = np.exp(-0.5)
    assert np.isclose(lift[0], w / (0.25 + w))


def test_two_stations_at_one_point_share_the_catchment_equally():
    attrs = {k: [0.0] * 12 for k in ("cat_f", "cctv", "car_spaces", "ticket_machine", "bus")}
    stations = pd.DataFrame(
        {
            "e": [0.0, 0.0, *[1e7 + i for i in range(10)]],
            "n": [0.0] * 12,
            "freq": [50.0] * 12,
            **attrs,
        },
        index=["AAA", "BBB", *[f"Z{i}" for i in range(10)]],
    )
    people = pd.DataFrame({"e": [300.0, -2000.0], "n": [0.0, 500.0], "pop": [60.0, 40.0]})
    pop = ns.catchments(stations, people)
    assert np.isclose(pop["AAA"], pop["BBB"])


def test_warnings_flag_an_empty_place_and_inputs_past_the_calibration_stations():
    cal = pd.DataFrame({"freq": [10.0, 200.0], "car_spaces": [0.0, 500.0]})
    rows = pd.DataFrame({"freq": [64.0, 2000.0], "car_spaces": [100.0, 100000.0]})
    notes = ns._warnings(rows, cal, np.array([False, True]), np.array([1e5, 2e7]), 1.5e6)
    assert notes[0] == ""
    assert "forecast 0" in notes[1] and "trains a day" in notes[1]
    assert "car spaces" in notes[1] and "size domain" in notes[1]


def test_frequency_disagreements_flag_beyond_factor_two():
    ours = pd.Series({"DWN": 4.0, "LDS": 400.0, "HDW": 8.0, "XXX": 10.0})
    sram = pd.Series({"DWN": 65.0, "LDS": 420.0, "HDW": 7.0})
    bad = ns.frequency_disagreements(ours, sram)
    assert list(bad.index) == ["DWN"]
