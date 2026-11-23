"""Timetable changes and forecast ranges: their arithmetic."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pydantic
import pytest

from ortla import changes, config, paths, ranges

RESPONSE = changes.Response(0.1, 0.28, 0.284, 4)


def _calls() -> pd.DataFrame:
    rows = [("AAA", None, "0800"), ("BBB", "0820", "0822"), ("CCC", "0840", None)]
    return pd.DataFrame(
        [
            {
                "train_uid": "T1",
                "stp_indicator": "P",
                "atoc_code": "XX",
                "days_runs": "1111100",
                "start_date": "2026-05-17",
                "end_date": "2026-12-12",
                "train_category": "OO",
                "signalling_id": "2A01",
                "svc_origin_crs": "AAA",
                "svc_origin_dep": "0800",
                "svc_dest_crs": "CCC",
                "seq": str(i),
                "tiploc": crs + "X",
                "crs": crs,
                "public_arrival": arrive,
                "public_departure": depart,
                "is_public_call": "True",
            }
            for i, (crs, arrive, depart) in enumerate(rows)
        ]
    )


def _changed(*edits: dict) -> pd.DataFrame:
    wanted = types.SimpleNamespace(
        changes=[config.TimetableChange(**edit) for edit in edits],
        services=[],
        station_recodes={},
        reference_timestamp=pd.Timestamp("2026-06-17"),
    )
    return changes.apply_changes(_calls(), wanted, "Wednesday")


def test_cutting_the_last_train_never_adds_journeys() -> None:
    for before in (1, 3, 4, 10, 30, 200):
        after = np.arange(before, -1, -1)
        factors = RESPONSE.factor(np.full(len(after), before), after)
        assert (np.diff(factors) <= 1e-12).all()
        assert factors[0] == 1


def test_more_trains_never_lose_journeys() -> None:
    after = np.arange(0, 100)
    assert (np.diff(RESPONSE.factor(np.zeros(100), after)) >= -1e-12).all()


def test_the_steps_are_the_measured_ones_at_four_trains() -> None:
    assert RESPONSE.factor(np.array([0]), np.array([4]))[0] == pytest.approx(np.exp(0.28))
    assert RESPONSE.factor(np.array([4]), np.array([0]))[0] == pytest.approx(np.exp(-0.284))


def test_a_cut_train_is_gone() -> None:
    assert _changed({"kind": "cut", "train_uid": "T1"}).empty


def test_a_retime_moves_from_the_station_given() -> None:
    after = _changed(
        {"kind": "retime", "train_uid": "T1", "shift_minutes": 5, "from_station": "BBB"}
    )
    assert list(after["public_departure"])[:2] == ["0800", "0827"]
    assert list(after["public_arrival"])[1:] == ["0820", "0845"]


def test_an_added_call_delays_what_follows() -> None:
    after = _changed(
        {
            "kind": "add_call",
            "train_uid": "T1",
            "station": "DDD",
            "after_station": "AAA",
            "arrival": "08:10",
            "departure": "08:11",
            "delay_minutes": 3,
        }
    )
    assert list(after["crs"]) == ["AAA", "DDD", "BBB", "CCC"]
    assert list(after["public_arrival"])[1:] == ["0810", "0823", "0843"]
    assert pd.isna(after["tiploc"].iloc[1])


def test_taking_out_the_last_call_ends_the_train_sooner() -> None:
    after = _changed({"kind": "remove_call", "train_uid": "T1", "station": "CCC"})
    assert list(after["crs"]) == ["AAA", "BBB"]
    assert pd.isna(after["public_departure"].iloc[-1])
    assert set(after["svc_dest_crs"]) == {"BBB"}


def test_times_running_backwards_are_refused() -> None:
    with pytest.raises(ValueError, match="backwards"):
        _changed(
            {"kind": "retime", "train_uid": "T1", "shift_minutes": -30, "from_station": "BBB"}
        )


def test_an_unknown_train_is_refused() -> None:
    with pytest.raises(ValueError, match="No passenger train"):
        _changed({"kind": "cut", "train_uid": "NOPE"})


def test_the_mean_journey_counts_the_wait() -> None:
    hourly = [(60.0 * hour, 60.0 * hour + 20) for hour in range(24)]
    spent, ride = changes._mean_journey(hourly)
    assert ride == 20
    assert spent == pytest.approx(20 + 29.5)


def test_scaled_matrices_touch_only_changed_pairs() -> None:
    plain = types.SimpleNamespace(cells=lambda *_: np.array([10.0, 10.0, 10.0]))
    pairs = pd.MultiIndex.from_tuples([("AAA", "BBB")])
    scaled = changes._ScaledMatrices(plain, pd.Series([0.5], index=pairs))
    origins = np.array(["AAA", "BBB", "ZZZ"], dtype=object)
    destinations = np.array(["BBB", "AAA", "BBB"], dtype=object)
    found = scaled.cells("Wednesday", np.array(["am_peak"] * 3), origins, destinations)
    assert list(found) == [5.0, 10.0, 10.0]


def test_call_times_run_past_midnight() -> None:
    stops = [
        types.SimpleNamespace(arrival=None, departure="23:50"),
        types.SimpleNamespace(arrival="00:10", departure=None),
    ]
    assert paths.call_times(stops) == [(2860, 2860), (2900, 2900)]


def test_range_ends_are_checked() -> None:
    with pytest.raises(pydantic.ValidationError):
        config.RangesConfig(load_low=1.2, load_high=1.5)


def test_the_central_case_is_the_run_itself() -> None:
    run = types.SimpleNamespace(
        ranges=config.RangesConfig(load_low=0.6, load_high=1.7),
        served_pair_frequency_elasticity=0.1,
        new_link_uplift=0.28,
        model_copy=lambda update: update,
    )
    found = {name: (load, update) for name, load, update in ranges.cases(run)}
    assert found["central"] == (
        1.0,
        {"served_pair_frequency_elasticity": 0.1, "new_link_uplift": 0.28},
    )
    assert found["low"][0] == 0.6
    assert found["high"][1]["served_pair_frequency_elasticity"] == 0.18


def test_a_dead_end_platform_group_gives_way_to_the_main_one() -> None:
    def find(calls: list, *_: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(clear=calls[1][0] == {"MAIN"}, points=[])

    station = paths.CandidatePaths(
        types.SimpleNamespace(find=find),
        {"A": frozenset({"A"}), "B": frozenset({"BAY", "MAIN"}), "C": frozenset({"C"})},
        {"A": frozenset({"A"}), "B": frozenset({"MAIN"}), "C": frozenset({"C"})},
    )
    assert station.find_calls([("A", 0, 0), ("B", 10, 12), ("C", 20, 20)], 2).clear
