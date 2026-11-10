"""Tests for the national train-level assignment (`ortla assign`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_network_assignment import CUTOFFS, WEDNESDAY, _train

from ortla import assign, loading_data
from ortla import network_assignment as na
from ortla.clock import WEEKDAYS_PER_YEAR
from ortla.config import AssignConfig


def _calls() -> pd.DataFrame:
    """XX trains AAA -> BBB -> CCC twice, a YY AAA -> CCC; am peak."""
    return pd.DataFrame(
        _train(
            "X1", "1X01", [("AAA", None, "0800"), ("BBB", "0810", "0811"), ("CCC", "0820", None)]
        )
        + _train(
            "X2", "1X02", [("AAA", None, "0830"), ("BBB", "0840", "0841"), ("CCC", "0850", None)]
        )
        + _train("Y1", "1Y01", [("AAA", None, "0815"), ("CCC", "0835", None)], atoc="YY")
    )


@pytest.fixture
def world(tmp_path):
    """Calls and matrices: 10 A-C, 4 A-B, 6 B-C trips a day."""
    _calls().to_csv(tmp_path / "calls.csv", index=False)
    codes = ["AAA", "BBB", "CCC"]
    daily = pd.DataFrame(0.0, index=codes, columns=codes)
    for o, d, trips in (("AAA", "CCC", 10), ("AAA", "BBB", 4), ("BBB", "CCC", 6)):
        daily.loc[o, d] = daily.loc[d, o] = trips
    (tmp_path / "mx").mkdir()
    for period in ("am_peak", "interpeak"):
        (daily * WEEKDAYS_PER_YEAR).rename_axis("origin_tlc").to_csv(
            tmp_path / "mx" / f"odm_Wednesday_{period}.csv"
        )
    return tmp_path


def _config(world, **kw) -> AssignConfig:
    settings = {
        "output_path": world / "out",
        "timetable_calls_path": world / "calls.csv",
        "reference_date": str(WEDNESDAY.date()),
        "operators": {"xx": {"atoc_codes": ["XX"], "period_matrices_dir": world / "mx"}},
        "station_recodes": {},
        "time_periods_cutoffs": CUTOFFS,
    }
    return AssignConfig(**{**settings, **kw})


def _day(world, **kw) -> pd.DataFrame:
    config = _config(world, **kw)
    return assign.assign_day(
        _calls(),
        "Wednesday",
        config,
        share_model=assign._share_model(config),
        split=None,
        footfall={},
        coordinates=None,
    )


def test_each_slot_is_shared_over_every_train_serving_it(world):
    seg = _day(world).set_index(["train_uid", "from_station"])
    # AAA -> CCC: 10 a day over X1, X2 and Y1 (Y1 counted, not loaded).
    # AAA -> BBB: 4 over X1 and X2; BBB -> CCC: 6 over X1 and X2.
    assert seg.loc[("X1", "AAA"), "load"] == pytest.approx(10 / 3 + 4 / 2)
    assert seg.loc[("X1", "BBB"), "load"] == pytest.approx(10 / 3 + 6 / 2)
    assert seg.loc[("X1", "AAA"), "boardings"] == pytest.approx(10 / 3 + 4 / 2)
    assert set(seg["operator"]) == {"xx"}
    assert "Y1" not in seg.index.get_level_values(0)


def test_other_operators_load_from_their_own_matrices(world):
    seg = _day(world, other_period_matrices_dir=world / "mx").set_index("train_uid")
    assert seg.loc["Y1", "operator"] == assign.OTHER
    assert seg.loc["Y1", "load"] == pytest.approx(10 / 3)


def test_the_packaged_slot_share_moves_load_between_a_slot_s_trains(world):
    equal = _day(world, transfer_assignment=False).set_index(["train_uid", "from_station"])
    weighted = _day(world, transfer_assignment=False, slot_share_model="m3").set_index(
        ["train_uid", "from_station"]
    )
    # The slot's total stays; its split between X1 and X2 changes.
    for station in ("AAA", "BBB"):
        total = sum(equal.loc[(u, station), "load"] for u in ("X1", "X2"))
        moved = sum(weighted.loc[(u, station), "load"] for u in ("X1", "X2"))
        assert moved == pytest.approx(total, rel=0.35)
    assert weighted.loc[("X1", "AAA"), "load"] != pytest.approx(equal.loc[("X1", "AAA"), "load"])


def test_a_neural_model_needs_no_more_than_the_timetable(world):
    seg = _day(world, slot_share_model="neural")
    assert np.isfinite(seg["load"]).all() and (seg["load"] > 0).all()


def test_no_operator_and_no_other_matrices_is_an_error(world):
    with pytest.raises(ValueError, match="operator"):
        _day(world, operators={})


def test_counts_match_by_uid_or_by_headcode_and_ends():
    segments = pd.DataFrame(
        {
            "day": "Wednesday",
            "service": [0, 0, 1],
            "atoc_code": "XX",
            "train_uid": ["X1", "X1", "X2"],
            "headcode": ["1X01", "1X01", "1X02"],
            "train_origin": "AAA",
            "train_destination": "CCC",
            "train_departure": ["08:00", "08:00", "08:30"],
        }
    )
    counts = pd.DataFrame(
        {
            "Service Week Day": "Wednesday",
            "Unique ID": ["X1", "X1", None],
            "Headcode": ["1X01", "1X01", "1X02"],
            "Origin CRS": "AAA",
            "Destination CRS": "CCC",
            "Origin Departure Time": ["08:00", "08:00", "08:30:00"],
            "Departure from CRS": ["AAA", "AAA", "AAA"],
            "Call Sequence": 1,
            "RSP Period": ["1", "2", "1"],
            "Mean Load on Departure": [10.0, 20.0, 7.0],
            "Mean Passengers On": [10.0, 20.0, 7.0],
            "Mean Passengers Off": [0.0, 0.0, 0.0],
        }
    )
    matched = assign.match_counts(counts, segments, ["XX"]).set_index("train_uid")
    assert matched.loc["X1", "actual_load"] == pytest.approx(15.0)
    assert matched.loc["X1", "n_periods"] == 2
    assert matched.loc["X2", "actual_load"] == pytest.approx(7.0)


def test_undated_feed_reads_each_file_as_a_period(tmp_path):
    row = {
        "Unique ID": "X1",
        "Train ID (Headcode)": "1X01",
        "Day of Count": "Wednesday",
        "Origin of Service (CRS)": "AAA",
        "Destination of Service (CRS)": "CCC",
        "Timetabled Departure from Origin": "08:00",
        "Station CRS": "AAA",
        "Station Scheduled Arrival Time": "08:00",
        "Number of Counts": "4",
        "Standard Seated Capacity": "300",
        "First Class Seated Capacity": "20",
        "Total Load (Mean)": "50",
    }
    for name in ("a.csv", "b.csv"):
        pd.DataFrame([row]).to_csv(tmp_path / name, index=False)
    counts = loading_data.read_awc(tmp_path)
    assert counts["RSP Period"].tolist() == ["1", "2"]
    assert loading_data.read_awc(tmp_path / "b.csv")["RSP Period"].tolist() == ["1"]


def test_score_reports_level_and_log_error():
    frame = pd.DataFrame(
        {
            "train_uid": ["a", "a", "b"],
            "day": "Wednesday",
            "load": [20.0, 5.0, 0.0],
            "actual_load": [10.0, 10.0, 4.0],
        }
    )
    s = assign.score(frame)
    assert s["segments"] == 3 and s["trains"] == 2
    assert s["ratio"] == pytest.approx(25 / 24)
    assert s["rmse_log"] == pytest.approx(np.log(2))
    assert s["zero_predictions"] == 1


def test_the_command_writes_its_tables(world):
    _config(world, other_period_matrices_dir=world / "mx").run()
    out = world / "out"
    segments = pd.read_csv(out / "assign_segments.csv")
    stations = pd.read_csv(out / "assign_stations.csv")
    trains = pd.read_csv(out / "assign_trains.csv")
    links = pd.read_csv(out / "assign_links.csv")
    assert set(trains["train_uid"]) == {"X1", "X2", "Y1"}
    # Everyone who boards alights.
    assert stations["boardings"].sum() == pytest.approx(stations["alightings"].sum())
    assert stations["boardings"].sum() == pytest.approx(segments["boardings"].sum())
    assert links["passengers"].sum() == pytest.approx(segments["load"].sum())
    assert not (out / "assign_scores.csv").exists()


def test_split_factors_conserve_a_pair_s_two_directions():
    calls = pd.DataFrame(
        _train("O1", "1O01", [("AAA", None, "0800"), ("CCC", "0820", None)])
        + _train("O2", "1O02", [("AAA", None, "0810"), ("CCC", "0830", None)])
        + _train("R1", "1R01", [("CCC", None, "0800"), ("AAA", "0820", None)])
    )
    legs = na.build_legs(na.day_calls(calls, WEDNESDAY, "Wednesday"), CUTOFFS)

    class Model:
        def footfall_slope(self, period):
            return 0.0

    o, d = legs["o"].to_numpy(), legs["d"].to_numpy()
    f = assign.split_factors(legs, o, d, Model(), {}, ["early", *CUTOFFS])
    by = dict(zip(legs["train_uid"], f, strict=True))
    # Two trains one way, one back: 2/3 and 1/3 of the pair's demand.
    assert by["O1"] == pytest.approx(2 * 2 / 3)
    assert by["R1"] == pytest.approx(2 * 1 / 3)


def test_the_charts_draw_from_the_command_s_tables(world):
    pytest.importorskip("plotly")
    from ortla.gui import charts

    _config(world, other_period_matrices_dir=world / "mx").run()
    out = world / "out"
    segments = pd.read_csv(out / "assign_segments.csv")
    segments["actual_load"] = segments["load"] * 1.5
    segments.to_csv(out / "assign_segments.csv", index=False)
    scores = [{"operator": op, **assign.score(g)} for op, g in segments.groupby("operator")]
    scores.append({"operator": "all", **assign.score(segments)})
    pd.DataFrame(scores).to_csv(out / "assign_scores.csv", index=False)
    stations = pd.DataFrame(
        {"lat": [53.0, 53.1, 53.2], "lon": [-2.0, -2.1, -2.2]}, index=["AAA", "BBB", "CCC"]
    )
    titles = [t for t, _ in charts.build_figures("assign", out, stations=stations)]
    for title in (
        "Scorecard: error by operator",
        "Predicted against counted",
        "Boardings by hour",
        "Busiest trains",
        "Map: station boardings",
    ):
        assert title in titles
