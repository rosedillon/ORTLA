"""Checks on the modelling maths used for the paper."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import types

import numpy as np
import pandas as pd
import pytest

from ortla import (
    demand,
    direction_split,
    gravity_model,
    loading_data,
    stations,
)
from ortla.bplan_geography import BplanTimingGraph
from ortla.demand import (
    calibrated_leg,
    matrix_cell,
)
from ortla.feasibility import _headway_rule
from ortla.od_matrix import rebalance_to_station_totals
from ortla.service_lookup import best_existing_minutes


def _calls(loads: list[float | None], stations: list[str], destination: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Train Operator": "gwr",
            "Unique ID": "X1",
            "Headcode": "1A00",
            "Service Week Day": "Monday",
            "Origin CRS": stations[0],
            "Destination CRS": destination,
            "Origin Departure Time": "08:00:00",
            "Departure from CRS": stations,
            "Call Sequence": range(len(stations)),
            "Period Start Date": pd.Timestamp("2026-06-01"),
            "Mean Load on Departure": loads,
            "Mean Passengers On": np.nan,
            "Mean Passengers Off": np.nan,
        }
    )
    return loading_data._canonical(df)


def test_boardings_from_load_changes_with_terminus() -> None:
    df = loading_data.derive_on_off(_calls([10, 25, 15, 99], ["A", "B", "C", "D"], "D"))
    assert df["Mean Passengers On"].tolist() == [10, 15, 0, 0]
    assert df["Mean Passengers Off"].tolist() == [0, 0, 10, 15]


def test_uncounted_call_uses_last_known_load_and_missing_terminus_row() -> None:
    df = loading_data.derive_on_off(_calls([10, None, 30], ["A", "B", "C"], "Z"))
    # C is compared with A (last counted), not with an assumed empty
    # train.
    assert df["Mean Passengers On"].tolist()[2] == 20
    # C is not the destination, so nobody is assumed to alight there.
    assert df["Mean Passengers Off"].tolist()[2] == 0


def test_republished_windows_are_dropped(tmp_path: pathlib.Path) -> None:
    for name, cols in [
        ("20240602 to 20240629 - Average Passenger Counts.csv", "Average Loading"),
        ("20240602 to 20240630 - Average Passenger Counts.csv", "Average Loading (Uncapped)"),
        ("20260301 to 20260328 - Average Passenger Counts.csv", "Average Loading"),
        ("20260301 to 20260331 - Average Passenger Counts.csv", "Average Loading"),
        ("20260329 to 20260425 - Average Passenger Counts.csv", "Average Loading"),
    ]:
        (tmp_path / name).write_text(cols + "\n")
    kept = [p.name[:22] for p in loading_data.select_count_windows(sorted(tmp_path.iterdir()))]
    assert kept == ["20240602 to 20240630 -", "20260301 to 20260328 -", "20260329 to 20260425 -"]


def test_count_periods_dated_from_file_names(tmp_path: pathlib.Path) -> None:
    se = tmp_path / "se"
    se.mkdir()
    pd.DataFrame(
        {
            "Start Time": ["08:00"],
            "Origin CRS": ["CHX"],
            "Destination CRS": ["TON"],
            "Day of Week": ["Monday"],
            "CRS": ["CHX"],
            "Order": [1],
            "Start Date": ["09/02/2025"],
            "End Date": ["08/03/2025"],
            "Observations": [3],
            "Average Loading": [50],
        }
    ).to_csv(se / "20250209 to 20250308 - Average Passenger Counts.csv", index=False)
    df = loading_data.READERS["southeastern"](se)
    assert df["Period Start Date"].iloc[0] == pd.Timestamp("2025-02-09")
    assert df["Period End Date"].iloc[0] == pd.Timestamp("2025-03-08")

    swr = tmp_path / "swr"
    swr.mkdir()
    names = [
        "Location",
        "Rail Period",
        "Train UID",
        "Headcode",
        "ODT",
        "Origin",
        "Destination",
        "Origin Time",
        "Dep",
        "Arr",
        "Number of Readings",
        "Standard Seated",
        "First Class Seats",
        "Average Train Load",
    ]
    row = [
        "WATRLMN",
        "RY25 P13",
        "W1",
        "1A00",
        "Monday",
        "WATRLMN",
        "WOKING",
        "08:00",
        "08:00",
        None,
        3,
        100,
        0,
        40,
    ]
    with pd.ExcelWriter(swr / "SWR Passenger Loadings - RY26 P03.xlsx") as book:
        pd.DataFrame([["RY26 P03 : 25th May 2026 - 21st Jun 2025"]]).to_excel(
            book, sheet_name="Notes", header=False, index=False
        )
        pd.DataFrame([names, [None] * len(names), row]).to_excel(
            book, sheet_name="Loadings Data", header=False, index=False
        )
    df = loading_data.READERS["swr"](swr, tiploc_to_crs={"WATRLMN": "WAT", "WOKING": "WOK"})
    assert df["Period Start Date"].iloc[0] == pd.Timestamp("2025-06-01")
    assert df["Period End Date"].iloc[0] == pd.Timestamp("2025-06-28")
    assert df["RSP Period"].iloc[0] == "RY26 P03"


def test_headway_rule_modes() -> None:
    empirical = types.SimpleNamespace(
        headway_mode="empirical",
        min_headway_floor_minutes=2.0,
        headway_safety_margin_minutes=1.0,
        planning_headway_minutes=3.0,
    )
    assert _headway_rule([0, 30, 60], empirical) == (29.0, 30, "empirical")
    assert _headway_rule([0], empirical) == (2.0, None, "floor_only")
    fixed = types.SimpleNamespace(**{**vars(empirical), "headway_mode": "fixed"})
    assert _headway_rule([0, 30, 60], fixed) == (3.0, 30, "fixed")


def test_gravity_calibration_recovers_known_coefficients() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    codes = [f"S{i:04d}" for i in range(n)]
    footfall = {c: float(rng.integers(1e4, 1e7)) for c in codes}
    origins = rng.choice(codes, n)
    dests = rng.choice(codes, n)
    miles = rng.uniform(2, 200, n)
    trains = rng.integers(1, 60, n).astype(float)
    true = [-9.0, 0.6, 0.55, -0.8, 0.3]
    ln_j = (
        true[0]
        + true[1] * np.log([footfall[o] for o in origins])
        + true[2] * np.log([footfall[d] for d in dests])
        + true[3] * np.log(miles)
        + true[4] * np.log(trains)
        + rng.normal(0, 0.1, n)
    )
    odm = pd.DataFrame(
        {
            "origin_tlc": origins,
            "destination_tlc": dests,
            "journeys": np.exp(ln_j),
            "distance_miles": miles,
        }
    )
    freq = dict(zip(zip(origins, dests, strict=True), trains, strict=True))
    model = gravity_model.calibrate(odm, footfall, freq, "synthetic")
    fitted = list(model.coefficients.values())
    assert fitted == pytest.approx(true, abs=0.05)
    assert model.frequency_ratio(2, 6) == pytest.approx(3 ** fitted[4])


def test_calibrated_leg_pivots_observed_demand() -> None:
    model = gravity_model.GravityModel(
        {
            "intercept": 0,
            "ln_footfall_origin": 0,
            "ln_footfall_destination": 0,
            "ln_distance": 0,
            "ln_trains": 0.5,
        },
        n_pairs=1,
        r_squared=1,
        residual_std=0,
        source_odm="x",
    )
    gravity = types.SimpleNamespace(model=model)
    # 3 trains/day -> 4: slot demand 100 * (4/3)^0.5; 1 existing + 1 new
    # train in the slot.
    captured, generated, label = calibrated_leg(
        gravity,
        None,
        None,
        "Monday",
        "am_peak",
        existing_trips=100.0,
        trains_in_period=1,
        trains_per_day=3,
        added_in_period=1,
        added_per_day=1,
    )
    new = 100 * (4 / 3) ** 0.5
    assert label == "gravity_pivot"
    assert captured == pytest.approx(new / 2)
    # Generated demand is per train in the period, the same basis as
    # captured: the growth rides the existing train as well as the added
    # one.
    assert generated == pytest.approx((new - 100) / 2)


def test_running_time_uses_start_and_stop_timings(tmp_path: pathlib.Path) -> None:
    rows = []
    for a, b in [("A", "J"), ("J", "B")]:
        for entry, exit_, minutes in [(-1, -1, 4.0), (0, -1, 5.0), (-1, 0, 4.5), (0, 0, 6.0)]:
            rows.append(
                {
                    "from_tiploc": a,
                    "to_tiploc": b,
                    "running_line": "",
                    "traction_code": "802-5D",
                    "timing_load_speed_mph": 125,
                    "entry_speed": entry,
                    "exit_speed": exit_,
                    "start_date": "",
                    "end_date": "",
                    "sectional_running_time_minutes": minutes,
                }
            )
    rows.append(
        {
            "from_tiploc": "A",
            "to_tiploc": "J",
            "running_line": "",
            "traction_code": "150",
            "timing_load_speed_mph": 75,
            "entry_speed": -1,
            "exit_speed": -1,
            "start_date": "",
            "end_date": "",
            "sectional_running_time_minutes": 3.0,
        }
    )
    pd.DataFrame(rows).to_csv(tmp_path / "bplan_timing_links.csv", index=False)
    graph = BplanTimingGraph(tmp_path)
    # Start timing on A-J (5.0) + stop timing on J-B (4.5); not the pass
    # times (8.0).
    minutes, exact = graph.running_time_detail("A", "B", "802-5D")
    assert minutes == pytest.approx(9.5)
    assert exact == pytest.approx(1.0)
    # A code BPLAN doesn't time falls back to the fastest traction per
    # link.
    minutes, exact = graph.running_time_detail("A", "B", "999")
    assert exact == pytest.approx(0.0)


def test_bus_and_underground_timings_are_not_track(tmp_path: pathlib.Path) -> None:
    rows = [
        {
            "from_tiploc": a,
            "to_tiploc": b,
            "traction_code": code,
            "entry_speed": -1,
            "exit_speed": -1,
            "sectional_running_time_minutes": minutes,
        }
        for a, b, code, minutes in [
            ("A", "B", "150", 30.0),
            ("A", "X", "BUS", 2.0),
            ("X", "B", "BUSXC", 2.0),
            ("A", "B", "LUL", 1.0),
        ]
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "bplan_timing_links.csv", index=False)
    graph = BplanTimingGraph(tmp_path)
    # Neither the bus shortcut nor the Underground timing is available
    # to a train.
    assert graph.minimum_running_time("A", "B", "999") == pytest.approx(30.0)
    assert graph.fastest_path("A", "B", "999") == ["A", "B"]


def _timetable_calls(rows: list[dict], path: pathlib.Path) -> pathlib.Path:
    """Write a minimal calls CSV with the columns the matcher reads."""
    frame = pd.DataFrame(rows)
    defaults = {"atoc_code": "NT", "stp_indicator": "P", "is_public_call": "True"}
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
    frame.to_csv(path, index=False)
    return path


def _loading_rows(rows: list[dict]) -> pd.DataFrame:
    """Build loading rows in the schema the imputer keys on."""
    defaults = {
        "Train Operator": "northern",
        "Headcode": "1A01",
        "Origin CRS": "MAN",
        "Destination CRS": "LDS",
        "Origin Departure Time": "08:00:00",
        "Departure from CRS": "HUD",
        "Departure Time": None,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_departure_time_published_wins_and_origin_is_last_resort(
    tmp_path: pathlib.Path,
) -> None:
    calls = _timetable_calls(
        [
            {
                "signalling_id": "1A01",
                "svc_origin_crs": "MAN",
                "svc_dest_crs": "LDS",
                "svc_origin_dep": "0800",
                "crs": "HUD",
                "public_departure": "0835",
            }
        ],
        tmp_path / "calls.csv",
    )
    data = _loading_rows(
        [
            {"Departure Time": "08:40:00"},  # published: kept as published
            {},  # no published time: matched exactly
            # No schedule on this route at this time: origin fallback.
            {"Headcode": "9Z99", "Origin Departure Time": "14:00:00"},
        ]
    )
    out = loading_data.resolve_departure_times(data, calls)
    assert list(out[loading_data.SOURCE_COLUMN]) == [
        "published",
        "timetable_exact",
        "origin_fallback",
    ]
    assert list(out[loading_data.RESOLVED_COLUMN]) == [
        "08:40:00",
        "08:35:00",
        "14:00:00",
    ]


def test_ambiguous_timetable_key_is_not_guessed(tmp_path: pathlib.Path) -> None:
    """Two departures under one key fall through rather than guess."""
    calls = _timetable_calls(
        [
            {
                "signalling_id": "1A01",
                "svc_origin_crs": "MAN",
                "svc_dest_crs": "LDS",
                "svc_origin_dep": "0800",
                "crs": "HUD",
                "public_departure": dep,
            }
            for dep in ("0835", "0905")
        ],
        tmp_path / "calls.csv",
    )
    out = loading_data.resolve_departure_times(_loading_rows([{}]), calls)
    assert out[loading_data.SOURCE_COLUMN].iloc[0] == "origin_fallback"


def test_arrival_time_operators_prefer_the_timetabled_departure(
    tmp_path: pathlib.Path,
) -> None:
    """Avanti gives arrivals; the timetable's departure should win."""
    calls = _timetable_calls(
        [
            {
                "atoc_code": "VT",
                "signalling_id": "1A01",
                "svc_origin_crs": "MAN",
                "svc_dest_crs": "EUS",
                "svc_origin_dep": "0800",
                "crs": "SPT",
                "public_departure": "0832",
            }
        ],
        tmp_path / "calls.csv",
    )
    data = _loading_rows(
        [
            {
                "Train Operator": "awc",
                "Destination CRS": "EUS",
                "Departure from CRS": "SPT",
                "Departure Time": "08:30:00",
            },
            {  # no schedule to match: keeps its own arrival, not the origin time
                "Train Operator": "awc",
                "Headcode": "9Z99",
                "Destination CRS": "EUS",
                "Departure from CRS": "SPT",
                "Origin Departure Time": "14:00:00",
                "Departure Time": "08:30:00",
            },
        ]
    )
    out = loading_data.resolve_departure_times(data, calls)
    assert list(out[loading_data.SOURCE_COLUMN]) == [
        "timetable_exact",
        "published_arrival",
    ]
    assert out[loading_data.RESOLVED_COLUMN].iloc[0] == "08:32:00"


def test_nearest_origin_rung_respects_its_tolerance(tmp_path: pathlib.Path) -> None:
    calls = _timetable_calls(
        [
            {
                "signalling_id": "1A01",
                "svc_origin_crs": "MAN",
                "svc_dest_crs": "LDS",
                "svc_origin_dep": "0803",
                "crs": "HUD",
                "public_departure": "0838",
            }
        ],
        tmp_path / "calls.csv",
    )
    data = _loading_rows([{}])
    within = loading_data.resolve_departure_times(data, calls, tolerance_minutes=5)
    assert within[loading_data.SOURCE_COLUMN].iloc[0] == "timetable_nearest_origin"
    assert within[loading_data.RESOLVED_COLUMN].iloc[0] == "08:38:00"

    outside = loading_data.resolve_departure_times(data, calls, tolerance_minutes=1)
    assert outside[loading_data.SOURCE_COLUMN].iloc[0] == "origin_fallback"


def test_synthetic_headcode_still_matches_on_route_and_origin_time(
    tmp_path: pathlib.Path,
) -> None:
    """Southeastern and Greater Anglia rows carry made-up headcodes.

    The route, the origin departure minute and the calling station
    should still pin the schedule.
    """
    calls = _timetable_calls(
        [
            {
                "atoc_code": "SE",
                "signalling_id": "2W45",
                "svc_origin_crs": "CHX",
                "svc_dest_crs": "ORP",
                "svc_origin_dep": "0001",
                "crs": "LBG",
                "public_departure": "0012",
            }
        ],
        tmp_path / "calls.csv",
    )
    data = _loading_rows(
        [
            {
                "Train Operator": "southeastern",
                "Headcode": "0001CHXORP",  # synthesised, matches no signalling ID
                "Origin CRS": "CHX",
                "Destination CRS": "ORP",
                "Origin Departure Time": "00:01:00",
                "Departure from CRS": "LBG",
            }
        ]
    )
    out = loading_data.resolve_departure_times(data, calls)
    assert out[loading_data.SOURCE_COLUMN].iloc[0] == "timetable_no_headcode"
    assert out[loading_data.RESOLVED_COLUMN].iloc[0] == "00:12:00"


def _gravity(exponent: float = 0.5) -> types.SimpleNamespace:
    """A gravity context with only the frequency response live."""
    model = gravity_model.GravityModel(
        {
            "intercept": 0,
            "ln_footfall_origin": 0,
            "ln_footfall_destination": 0,
            "ln_distance": 0,
            "ln_trains": exponent,
        },
        n_pairs=1,
        r_squared=1,
        residual_std=0,
        source_odm="x",
    )
    return types.SimpleNamespace(model=model, synthetic_slot_trips=lambda *a, **k: (None, None))


def test_alias_map_collapses_chains_and_can_be_disabled() -> None:
    aliases = stations.resolve_aliases({"A": "B", "B": "C", "C": "C"})
    assert aliases == {"A": "C", "B": "C"}
    assert stations.canonical_crs(" spx ", stations.resolve_aliases(None)) == "STP"
    assert stations.resolve_aliases({}) == {}
    assert stations.canonical_crs("SPX", {}) == "SPX"


def test_alias_keys_carry_the_canonical_footfall() -> None:
    expanded = stations.with_alias_keys({"STP": 100}, {"SPX": "STP", "EBF": "EBD"})
    assert expanded["SPX"] == 100
    assert "EBF" not in expanded  # nothing to copy from


def test_matrix_lookup_tells_a_missing_station_from_an_empty_cell(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "odm_Monday_am_peak.csv"
    path.write_text("origin_tlc,STP,VIC\nMDE,1200,0\nAFK,!,5\n")
    assert demand.matrix_stations(path) == frozenset({"STP", "VIC", "MDE", "AFK"})
    assert matrix_cell(path, "MDE", "STP") == (1200.0, "observed")
    # A zero cell and a station the matrix has never heard of used to be
    # the same `None`; the whole point of defect 4 is that they are not.
    assert matrix_cell(path, "MDE", "VIC") == (0.0, "observed_zero")
    assert matrix_cell(path, "MDE", "SPX") == (None, "destination_missing")
    assert matrix_cell(path, "CBW", "STP") == (None, "origin_missing")
    assert matrix_cell(path, "AFK", "STP") == (None, "cell_suppressed")
    assert matrix_cell(tmp_path / "nope.csv", "MDE", "STP") == (None, "no_matrix")


def test_suppressing_frequency_dilution_holds_per_train_load() -> None:
    # 2 trains/day and 2 in the period, one more added: with the fitted
    # b4 the slot total grows by (3/2)**0.5 while the trains it is
    # shared over grow by 3/2, so each train carries less than it did. A
    # hindcast is re-inserting a train that really ran, so that is a
    # loss it never took.
    args = dict(
        existing_trips=100.0,
        trains_in_period=2,
        trains_per_day=2,
        added_in_period=1,
        added_per_day=1,
    )
    diluted, _, _ = calibrated_leg(
        _gravity(), None, None, "Monday", "am_peak", apply_dilution=True, **args
    )
    held, _, _ = calibrated_leg(
        _gravity(), None, None, "Monday", "am_peak", apply_dilution=False, **args
    )
    assert diluted == pytest.approx(100 * (3 / 2) ** 0.5 / 3)
    assert held == pytest.approx(100 / 2)
    assert diluted < held


def test_new_direct_link_captures_only_its_logit_share() -> None:
    # No existing direct service, so the added train is the whole
    # denominator. Without a route split it would be credited with every
    # trip on the pair.
    args = dict(
        existing_trips=1000.0,
        trains_in_period=0,
        trains_per_day=0,
        added_in_period=1,
        added_per_day=1,
    )
    whole, _, label = calibrated_leg(_gravity(), None, None, "Monday", "am_peak", **args)
    assert (whole, label) == (1000.0, "gravity_new_link")

    choice = demand.ChoiceTerms(direct_share=0.3)
    share, generated, label = calibrated_leg(
        _gravity(), None, None, "Monday", "am_peak", choice=choice, **args
    )
    assert share == pytest.approx(300.0)
    # Diverting existing trips off another route generates nothing new
    # to rail.
    assert generated == pytest.approx(0.0)


def test_a_first_direct_link_grows_the_pair_by_the_measured_step() -> None:
    # A pair gaining >= 4 direct trains a day grows by
    # exp(uplift), scaled by the share the direct route wins; the growth
    # is generated.
    choice = demand.ChoiceTerms(direct_share=0.5)
    args = dict(
        existing_trips=1000.0,
        trains_in_period=0,
        added_in_period=1,
        choice=choice,
        new_link_uplift=0.28,
    )
    captured, generated, _ = calibrated_leg(
        _gravity(), None, None, "Monday", "am_peak", trains_per_day=0, added_per_day=4, **args
    )
    grown = 1000.0 * (np.exp(0.28 * 0.5) - 1)
    assert (captured, generated) == (pytest.approx(500.0 + grown), pytest.approx(grown))
    # A token service, or a pair already direct in other periods, gets
    # no step.
    for per_day, added in ((0, 2), (3, 4)):
        captured, generated, _ = calibrated_leg(
            _gravity(),
            None,
            None,
            "Monday",
            "am_peak",
            trains_per_day=per_day,
            added_per_day=added,
            **args,
        )
        assert (captured, generated) == (pytest.approx(500.0), pytest.approx(0.0))
    # Journeys routed through the leg on the way elsewhere gain no
    # direct link.
    captured, generated, _ = calibrated_leg(
        _gravity(),
        None,
        None,
        "Monday",
        "am_peak",
        trains_per_day=0,
        added_per_day=4,
        step_share=0.25,
        **args,
    )
    assert generated == pytest.approx(0.25 * grown)


def test_the_scan_ranks_corridors_on_their_observed_market() -> None:
    # An unserved pair already has a market by other routes.
    # Each corridor once, ranked on it; a pair with no flow, or off the
    # matrices, is out.
    from ortla.config import RouteTypeConfig
    from ortla.discovery import _score_observed_market

    codes = ["AAA", "BBB", "CCC"]
    market = pd.DataFrame([[0, 10, 0], [10, 0, 300], [0, 300, 0]], index=codes, columns=codes)
    pairs = [
        ("AAA", "BBB", 20.0, 0.0),
        ("BBB", "AAA", 20.0, 0.0),
        ("CCC", "BBB", 40.0, 1.0),
        ("AAA", "CCC", 50.0, 0.0),
        ("AAA", "ZZZ", 30.0, 0.0),
    ]
    ranked = _score_observed_market(pairs, market, {}, 0.28, RouteTypeConfig(), 10)
    assert [(c.origin, c.destination) for c in ranked] == [("CCC", "BBB"), ("AAA", "BBB")]
    assert ranked[0].observed_annual_trips == 600.0
    assert ranked[0].predicted_annual_trips == pytest.approx(600.0 * (np.exp(0.28) - 1))
    # A station's second code is the station: its trains count, it is
    # not a new pair.
    from ortla.discovery import _one_code_per_station

    one = {"XRO": "RET", "SPX": "STP"}
    kept, served = _one_code_per_station(
        [("LCN", "XRO", 18.0), ("STP", "SPX", 1.0)],
        {("LCN", "XRO"): 16.0, ("LCN", "RET"): 2.0},
        lambda c: one.get(c, c),
    )
    assert (kept, served) == ([("LCN", "RET", 18.0)], {("LCN", "RET"): 18.0})


def test_a_frequency_option_is_priced_as_that_many_trains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The sweep prices one train; f a day take `_frequency_scaling`'s
    # multiplier, and one a day is unchanged.
    from ortla.scenario import ScenarioAssumptions, _options_for_departure

    assumptions = ScenarioAssumptions(tag=None, frequency_options=[1, 4])
    row = pd.Series(
        {
            "predicted_annual_trips_captured": 1000.0,
            "predicted_annual_trips_generated": 100.0,
            "predicted_annual_revenue_gbp": float("nan"),
            "stop_0_departure": "08:00",
            "shift_minutes": 0.0,
            "worst_gap_minutes": float("nan"),
        }
    )
    monkeypatch.setattr("ortla.scenario._quick_metrics", lambda _: (1.0, 0.0, 0.0, 0.0))
    options = _options_for_departure(
        row, assumptions, "x", 0.0, scaling={1: (1.0, 0.1), 4: (4.4, 1.2)}
    )
    got = [(o.trains_per_day, o.annual_passengers, o.annual_passengers_generated) for o in options]
    assert got == [(1, 1000, 100), (4, 4400, 1200)]


def test_a_candidate_inserted_in_the_timetable_gives_its_own_legs() -> None:
    # Transfers route on the timetable after the change, so the
    # candidate's calls must read as one train of public legs.
    from ortla.config import CandidateService, ServiceStop
    from ortla.demand import candidate_calls
    from ortla.network_assignment import build_legs, day_calls

    candidate = CandidateService(
        name="c",
        traction_code="150",
        days_of_week=["Wednesday"],
        stops=[
            ServiceStop(station="BHM", departure="08:00"),
            ServiceStop(station="COV", arrival="08:20", departure="08:22"),
            ServiceStop(station="GTW", arrival="10:30"),
        ],
    )
    date = pd.Timestamp("2026-11-18")
    calls = candidate_calls(candidate, date)
    legs = build_legs(
        day_calls(calls, date, "Wednesday"), {"am_peak": "7:00", "interpeak": "9:30"}
    )
    got = {
        (o, d): m for o, d, m in zip(legs["o"], legs["d"], legs["in_vehicle_minutes"], strict=True)
    }
    assert got == {("BHM", "COV"): 20, ("BHM", "GTW"): 150, ("COV", "GTW"): 128}
    assert set(calls["days_runs"]) == {"0010000"}


def test_a_return_working_mirrors_the_outbound_times() -> None:
    from ortla.config import CandidateService, ServiceStop

    out = CandidateService(
        name="c",
        traction_code="150",
        days_of_week=["Monday"],
        return_after_minutes=15,
        stops=[
            ServiceStop(station="BHM", departure="08:00"),
            ServiceStop(station="COV", arrival="08:20", departure="08:22"),
            ServiceStop(station="GTW", arrival="10:30"),
        ],
    )
    back = out.return_working()
    assert back.name == "c_return" and back.return_after_minutes is None
    assert [(s.station, s.arrival, s.departure) for s in back.stops] == [
        ("GTW", None, "10:45"),
        ("COV", "12:53", "12:55"),
        ("BHM", "13:15", None),
    ]


def test_the_median_fare_resolves_clusters_and_drops_token_fares() -> None:
    # GTW is priced only through cluster Q001; the pool keeps adult
    # singles and halved returns, and drops the GBP 999 placeholder, the
    # 50p token, the child fare, first class and seasons.
    from ortla.fares import FaresTable, median_fare

    rows = [  # ticket description, class, type, pence, non-retail
        ("ADVANCE", "2", "S", 3000, False),
        ("ADVANCE", "2", "S", 5000, False),
        ("OFF-PEAK R", "2", "R", 8000, False),  # 40.00 a journey
        ("ANYTIME S", "2", "S", 12000, False),
        ("ADVANCE", "2", "S", 50, False),  # token
        ("SMALL GROUP 5", "2", "S", 99900, True),
        ("CHILD FLTFARE S", "2", "S", 400, False),
        ("ANYTIME S", "1", "S", 30000, False),
        ("SEASON", "2", "N", 900000, False),
    ]
    fares = pd.DataFrame(
        rows,
        columns=["ticket_description", "tkt_class", "tkt_type", "fare_pence", "non_retail_fare"],
    )
    fares = fares.assign(
        origin_code="Q001",
        destination_code="1127",
        direction="R",
        toc="XC",
        ticket_code=[f"T{i}" for i in range(len(fares))],
        restriction_code="",
        route_code="00000",
        fare_gbp=fares["fare_pence"] / 100,
    )
    table = FaresTable(fares, {"GTW": frozenset({"5416", "Q001"}), "BHM": frozenset({"1127"})}, {})
    # 30, 40, 50, 120 -> 45; the reversible flow prices the way back
    # too.
    assert median_fare(table, "GTW", "BHM") == 45.0
    assert median_fare(table, "BHM", "GTW") == 45.0
    assert median_fare(table, "GTW", "XXX") is None


def test_leave_one_out_on_an_in_sample_cell_returns_the_national_share() -> None:
    # The matrix saw all 4 trains in the slot. Strip one and put it
    # back: it should carry 1/4 of the cell, as the national assignment
    # gives it, not the 1/3 the 3 trains left behind each carry (case
    # 20, fix A), whatever the frequency response.
    args = dict(
        existing_trips=1200.0,
        trains_in_period=3,
        trains_per_day=10,
        added_in_period=1,
        added_per_day=1,
    )
    for dilution in (True, False):
        captured, generated, label = calibrated_leg(
            _gravity(),
            None,
            None,
            "Monday",
            "am_peak",
            apply_dilution=dilution,
            cell_is_after=True,
            **args,
        )
        assert (captured, generated, label) == (pytest.approx(300.0), 0.0, "observed_share")
    stale, _, _ = calibrated_leg(
        _gravity(), None, None, "Monday", "am_peak", apply_dilution=False, **args
    )
    assert stale == pytest.approx(400.0)  # what the strip did before the fix
    # The slot's only train: the whole cell, never re-priced as a new
    # link.
    only, _, label = calibrated_leg(
        _gravity(),
        None,
        None,
        "Monday",
        "am_peak",
        cell_is_after=True,
        existing_trips=55.0,
        trains_in_period=0,
        trains_per_day=0,
        added_in_period=1,
        added_per_day=1,
    )
    assert (only, label) == (55.0, "observed_share")


def test_new_link_floor_options() -> None:
    # Cell 55, the link wins all of it; the gravity prediction is 16,000
    # (VIC -> LBG early in the day) with a smearing factor of 2.9.
    model = dataclasses.replace(_gravity().model, smearing=2.9)
    gravity = types.SimpleNamespace(
        model=model, synthetic_slot_trips=lambda *a, **k: (16000.0, "x")
    )
    args = dict(
        existing_trips=55.0,
        trains_in_period=0,
        trains_per_day=0,
        added_in_period=1,
        added_per_day=1,
    )
    got = {
        floor: calibrated_leg(
            gravity, None, None, "Monday", "early", new_link_floor=floor, **args
        )[0]
        for floor in ("synthetic", "median", "none")
    }
    assert got == {"synthetic": 16000.0, "median": pytest.approx(16000 / 2.9), "none": 55.0}


def test_destination_substitution_conserves_the_group_and_shifts_its_share(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "odm_Monday_am_peak.csv"
    # Maidstone East in miniature: almost nothing to Blackfriars, the
    # real London demand sitting on Victoria and London Bridge.
    path.write_text("origin_tlc,BFR,VIC,LBG\nMDE,10,6000,4000\n")
    context = demand.SubstitutionContext(
        aliases={},
        groups={"london": ["BFR", "VIC", "LBG"]},
        theta=0.05,
        interchange_penalty=20.0,
    )
    calls = _stripped_calls()

    # 55 minutes direct against 40 + 20 penalty + 25 via Victoria.
    terms = context.terms(path, calls, "MDE", "BFR", 10.0, 55.0)
    assert terms.group_trips == pytest.approx(10010.0)
    assert terms.minutes_before == pytest.approx(85.0)
    assert terms.substituted_trips is not None
    # The new cell is worth far more than its own 10 trips, and still
    # only a slice of the group, because substitution moves trips
    # without inventing them.
    assert 10.0 < terms.substituted_trips < terms.group_trips
    assert terms.substituted_trips == pytest.approx(
        10010.0
        * (10 / 10010 * np.exp(-0.05 * (55 - 85)))
        / (10 / 10010 * np.exp(-0.05 * (55 - 85)) + 1 - 10 / 10010)
    )


def test_interchange_journey_prices_the_change(tmp_path: pathlib.Path) -> None:
    calls = _stripped_calls()
    # MDE -> VIC is direct in 40 minutes; MDE -> BFR needs the change.
    assert best_existing_minutes(calls, "MDE", "VIC", 20.0) == 40.0
    assert best_existing_minutes(calls, "MDE", "BFR", 20.0) == 85.0
    assert best_existing_minutes(calls, "MDE", "ZZZ", 20.0) is None


def _stripped_calls() -> pd.DataFrame:
    """A two-service day: MDE->VIC->BFR with a change at Victoria."""
    rows = [
        ("S1", 0, "MDE", "", "0800"),
        ("S1", 1, "VIC", "0840", "0842"),
        ("S2", 0, "VIC", "", "0900"),
        ("S2", 1, "BFR", "0925", ""),
    ]
    return pd.DataFrame(
        [
            {
                "train_uid": uid,
                "seq": seq,
                "crs": crs,
                "public_arrival": arr,
                "public_departure": dep,
                "is_public_call": "True",
            }
            for uid, seq, crs, arr, dep in rows
        ]
    )


def test_the_time_saving_is_each_legs_saving_on_todays_best_journey(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Today MDE->VIC is 40 min direct and MDE->BFR 85 with the change;
    # VIC->BFR is 25 direct. The candidate runs MDE 08:00, VIC 08:45,
    # BFR 09:00.
    from ortla import scenario
    from ortla.config import CandidateService, ServiceStop

    monkeypatch.setattr(scenario, "day_services", lambda calls, *_: calls)
    candidate = CandidateService(
        name="c",
        traction_code="150",
        days_of_week=["Monday"],
        stops=[
            ServiceStop(station="MDE", departure="08:00"),
            ServiceStop(station="VIC", arrival="08:45", departure="08:46"),
            ServiceStop(station="BFR", arrival="09:00"),
        ],
    )
    demand = pd.DataFrame(
        {
            "leg_stop_i": [0, 0, 1],
            "leg_stop_j": [1, 2, 2],
            "predicted_annual_trips_captured": [100.0, 300.0, 100.0],
        }
    )
    config = types.SimpleNamespace(interchange_penalty_minutes=20.0)
    # Slower to VIC saves 0; 60 against 85 saves 25; 14 against 25 saves
    # 11.
    got = scenario._leg_time_saving(
        config, candidate, _stripped_calls(), pd.Timestamp("2026-11-16"), demand
    )
    assert got == pytest.approx((0 * 100 + 25 * 300 + 11 * 100) / 500)


def test_a_period_with_no_direct_train_does_not_take_the_whole_cell() -> None:
    # The pair has a train elsewhere in the day but none in this period,
    # so the period's demand is reaching the destination some other way.
    # Branching on trains/day rather than trains in the period put all
    # of it (and, with a linear frequency ratio, twice all of it) on the
    # one added train.
    args = dict(
        existing_trips=27842.0,
        trains_in_period=0,
        trains_per_day=1,
        added_in_period=1,
        added_per_day=1,
    )
    captured, _, label = calibrated_leg(
        _gravity(),
        None,
        None,
        "Friday",
        "interpeak",
        choice=demand.ChoiceTerms(direct_share=0.47),
        apply_dilution=False,
        **args,
    )
    assert label == "gravity_new_link"
    assert captured == pytest.approx(27842.0 * 0.47)


def test_market_share_rescues_a_destination_no_one_can_reach_yet() -> None:
    path = tmp_matrix()
    context = demand.SubstitutionContext(
        aliases={},
        groups={"london": ["BFR", "VIC", "LBG"]},
        theta=0.05,
        interchange_penalty=20.0,
    )
    calls = _stripped_calls()
    terms = context.terms(path, calls, "MDE", "BFR", 10.0, 55.0)
    # BFR's observed share is ~0 only because there was no direct
    # service, and a pivot-point logit preserves ~0 however good the new
    # journey is. The market's own size-and-time split has to carry it
    # instead.
    pivoted = 10 / 10010 * np.exp(-0.05 * (55 - 85))
    pivoted = 10010 * pivoted / (pivoted + 1 - 10 / 10010)
    assert terms.substituted_trips > 20 * pivoted


def tmp_matrix() -> pathlib.Path:
    """A matrix where Blackfriars is big, but not from Maidstone."""
    import tempfile

    path = pathlib.Path(tempfile.mkdtemp()) / "odm_Monday_am_peak.csv"
    path.write_text(
        "origin_tlc,BFR,VIC,LBG\n"
        "MDE,10,6000,4000\n"
        # Everyone else already floods into Blackfriars, which is what
        # says it is worth serving even though Maidstone cannot reach
        # it.
        "OTH,900000,400000,500000\n"
    )
    return path


# Directional demand split


def _flat_split(intercept: float = 0.0, slope: float = 0.0):
    """A DirectionSplitModel with the given pooled coefficients."""
    return direction_split.DirectionSplitModel(
        coefficients={
            direction_split.POOLED: {
                "intercept": intercept,
                "ln_footfall_ratio": slope,
            }
        },
        n_cells=0,
        r_squared=0.0,
        source="test",
    )


def test_direction_split_cancels_the_frequency_divisor():
    """With zero coefficients, load per train is equal both ways.

    That is the whole point of the fixed unit coefficient on
    ``ln(trains_fwd / trains_rev)``: it undoes the frequency divisor,
    which otherwise predicts the higher load per train in the direction
    with *fewer* trains.
    """
    model = _flat_split()
    forward = model.split(100.0, 100.0, "am_peak", None, None, 10, 5)
    reverse = 200.0 - forward
    assert forward / 10 == pytest.approx(reverse / 5)


def test_direction_split_conserves_the_pair_total():
    model = _flat_split(intercept=0.3, slope=0.4)
    forward = model.split(120.0, 120.0, "am_peak", 2e6, 5e5, 7, 4)
    reverse = 240.0 - forward
    assert forward + reverse == pytest.approx(240.0)
    assert 0 < forward < 240.0


def test_direction_split_leaves_one_way_slots_alone():
    """No train one way => share 0.5, so a symmetric cell is unchanged.

    The fit only ever saw pairs running both ways; giving a one-way slot
    the pair's whole combined demand would double it in an untested
    regime.
    """
    model = _flat_split(intercept=1.0, slope=1.0)
    assert model.share("am_peak", 1e6, 1e5, 8, 0) == 0.5
    assert model.split(50.0, 50.0, "am_peak", 1e6, 1e5, 8, 0) == pytest.approx(50.0)


def test_direction_split_missing_footfall_drops_only_that_term():
    model = _flat_split(intercept=0.0, slope=2.0)
    assert model.share("am_peak", None, None, 5, 5) == pytest.approx(0.5)
    assert model.share("am_peak", 1e6, 0, 5, 5) == pytest.approx(0.5)
    assert model.share("am_peak", 1e6, 1e5, 5, 5) > 0.5


def test_direction_split_period_falls_back_to_pooled():
    model = direction_split.DirectionSplitModel(
        coefficients={
            direction_split.POOLED: {"intercept": 0.0, "ln_footfall_ratio": 0.0},
            "am_peak": {"intercept": 1.0, "ln_footfall_ratio": 0.0},
        },
        n_cells=1,
        r_squared=0.0,
        source="test",
    )
    assert model.share("am_peak", None, None, 5, 5) > 0.5
    assert model.share("evening", None, None, 5, 5) == pytest.approx(0.5)
    assert model.share(None, None, None, 5, 5) == pytest.approx(0.5)


def test_direction_split_calibrate_recovers_a_planted_footfall_slope():
    """Cells with a known asymmetry fit back to roughly that slope."""
    rng = np.random.default_rng(0)
    footfall = {f"S{i:03d}": int(10 ** rng.uniform(4, 7)) for i in range(60)}
    codes = sorted(footfall)
    rows = []
    for i in range(0, len(codes) - 1, 2):
        origin, destination = codes[i], codes[i + 1]
        if origin > destination:
            origin, destination = destination, origin
        x = np.log(footfall[origin] / footfall[destination])
        for period in ("am_peak", "pm_peak"):
            ratio = float(np.exp(0.6 * x))  # planted slope of 0.6
            rows.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "period": period,
                    "flow": 100.0,
                    "trains": 5,
                    "load_per_train": 20.0 * ratio,
                    "flow_reverse": 100.0,
                    "trains_reverse": 5,
                    "load_per_train_reverse": 20.0,
                }
            )
    model = direction_split.calibrate(pd.DataFrame(rows), footfall, "test")
    assert model.coefficients[direction_split.POOLED]["ln_footfall_ratio"] == pytest.approx(
        0.6, abs=0.05
    )
    assert model.n_cells == len(rows)


def test_split_multiplier_matches_the_scalar_share():
    model = _flat_split(intercept=0.0, slope=-0.45)
    got = direction_split.split_multiplier(
        np.array([-0.45]),
        np.array([0, 0]),
        np.array([np.log(2e5 / 3e7), 0.7]),
        np.array([np.log(6 / 2), np.nan]),
    )
    assert got[0] == pytest.approx(2 * model.share("am_peak", 2e5, 3e7, 6, 2))
    assert got[1] == 1.0  # one-way slot: the split does not fire


def test_calibrate_to_segments_recovers_planted_tidal_slopes():
    """Segment loads with known per-period slopes fit back to them."""
    from scipy import sparse

    rng = np.random.default_rng(1)
    planted = {"am_peak": -0.5, "pm_peak": 0.4}
    legs, rows, cols, n_segments = [], [], [], 0
    for _train in range(120):
        period = str(rng.choice(list(planted)))
        stops = int(rng.integers(3, 7))
        footfall = 10 ** rng.uniform(4, 7, stops)
        for i in range(stops):
            for j in range(i + 1, stops):
                legs.append(
                    {
                        "captured": float(rng.uniform(50, 500)),
                        "period": period,
                        "ln_footfall_ratio": float(np.log(footfall[i] / footfall[j])),
                        "ln_train_ratio": float(rng.normal(0, 0.3)),
                    }
                )
                rows += range(n_segments + i, n_segments + j)
                cols += [len(legs) - 1] * (j - i)
        n_segments += stops - 1
    legs = pd.DataFrame(legs)
    incidence = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_segments, len(legs))
    )
    index = legs["period"].map({"am_peak": 0, "pm_peak": 1}).to_numpy()
    truth = direction_split.split_multiplier(
        np.array([planted["am_peak"], planted["pm_peak"]]),
        index,
        legs["ln_footfall_ratio"].to_numpy(),
        legs["ln_train_ratio"].to_numpy(),
    )
    observed = incidence @ (legs["captured"].to_numpy() * truth)

    model = direction_split.calibrate_to_segments(legs, incidence, observed, "test")

    for period, slope in planted.items():
        assert model.coefficients[period]["ln_footfall_ratio"] == pytest.approx(slope, abs=0.02)
        assert model.coefficients[period]["intercept"] == 0.0
    assert model.r_squared > 0.99


def _function_split(footfall_slope: float, function_slope: float):
    """A zero-intercept split carrying a station-function term."""
    return direction_split.DirectionSplitModel(
        coefficients={
            direction_split.POOLED: {
                "intercept": 0.0,
                "ln_footfall_ratio": footfall_slope,
                direction_split.FUNCTION_TERM: function_slope,
            }
        },
        n_cells=0,
        r_squared=0.0,
        source="test",
        station_function={"HOM": 1.5, "JOB": -2.0, "MID": 0.2},
    )


def test_station_function_term_moves_demand_without_creating_it():
    """The term flips sign with direction, so the shares sum to one."""
    model = _function_split(-0.3, 0.6)
    forward = model.share("am_peak", 2e5, 3e6, 6, 4, origin="HOM", destination="JOB")
    reverse = model.share("am_peak", 3e6, 2e5, 4, 6, origin="JOB", destination="HOM")
    assert forward + reverse == pytest.approx(1.0)
    # Home end to jobs end in the morning: more than footfall alone
    # would send.
    footfall_only = model.share("am_peak", 2e5, 3e6, 6, 4)
    assert forward > footfall_only


def test_station_function_missing_station_drops_only_that_term():
    model = _function_split(-0.3, 0.6)
    without = model.share("am_peak", 2e5, 3e6, 6, 4)
    assert model.share("am_peak", 2e5, 3e6, 6, 4, origin="HOM", destination="XXX") == without
    assert model.share("am_peak", 2e5, 3e6, 6, 4, origin=None, destination="JOB") == without


def test_a_model_without_the_function_term_ignores_station_codes(tmp_path):
    """A split fitted before the term existed behaves as it did."""
    old = _flat_split(intercept=0.0, slope=-0.45)
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {  # written without the station_function field
                "coefficients": old.coefficients,
                "n_cells": 0,
                "r_squared": 0.0,
                "source": "test",
            }
        )
    )
    loaded = direction_split.DirectionSplitModel.load(path)
    assert loaded.station_function is None
    assert loaded.share("am_peak", 2e5, 3e6, 6, 4, origin="HOM", destination="JOB") == (
        old.share("am_peak", 2e5, 3e6, 6, 4)
    )


def test_split_multiplier_with_function_matches_the_scalar_share():
    model = _function_split(-0.3, 0.6)
    got = direction_split.split_multiplier(
        np.array([-0.3]),
        np.array([0]),
        np.array([np.log(2e5 / 3e6)]),
        np.array([np.log(6 / 4)]),
        np.array([0.6]),
        np.array([1.5 - (-2.0)]),
    )
    assert got[0] == pytest.approx(
        2 * model.share("am_peak", 2e5, 3e6, 6, 4, origin="HOM", destination="JOB")
    )


def test_calibrate_to_segments_recovers_planted_function_slopes():
    """Loads with footfall and function slopes fit back to both."""
    from scipy import sparse

    rng = np.random.default_rng(2)
    planted = {"am_peak": (-0.3, 0.6), "pm_peak": (0.2, -0.4)}
    legs, rows, cols, n_segments = [], [], [], 0
    for _train in range(160):
        period = str(rng.choice(list(planted)))
        stops = int(rng.integers(3, 7))
        footfall = 10 ** rng.uniform(4, 7, stops)
        function = rng.normal(0, 1.2, stops)
        for i in range(stops):
            for j in range(i + 1, stops):
                legs.append(
                    {
                        "captured": float(rng.uniform(50, 500)),
                        "period": period,
                        "ln_footfall_ratio": float(np.log(footfall[i] / footfall[j])),
                        "ln_train_ratio": float(rng.normal(0, 0.3)),
                        "function_difference": float(function[i] - function[j]),
                    }
                )
                rows += range(n_segments + i, n_segments + j)
                cols += [len(legs) - 1] * (j - i)
        n_segments += stops - 1
    legs = pd.DataFrame(legs)
    incidence = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_segments, len(legs))
    )
    index = legs["period"].map({"am_peak": 0, "pm_peak": 1}).to_numpy()
    truth = direction_split.split_multiplier(
        np.array([planted["am_peak"][0], planted["pm_peak"][0]]),
        index,
        legs["ln_footfall_ratio"].to_numpy(),
        legs["ln_train_ratio"].to_numpy(),
        np.array([planted["am_peak"][1], planted["pm_peak"][1]]),
        legs["function_difference"].to_numpy(),
    )
    observed = incidence @ (legs["captured"].to_numpy() * truth)

    model = direction_split.calibrate_to_segments(
        legs, incidence, observed, "test", station_function={"A": 0.0}
    )

    for period, (footfall_slope, function_slope) in planted.items():
        terms = model.coefficients[period]
        assert terms["ln_footfall_ratio"] == pytest.approx(footfall_slope, abs=0.03)
        assert terms[direction_split.FUNCTION_TERM] == pytest.approx(function_slope, abs=0.03)
    assert model.station_function == {"A": 0.0}
    assert model.r_squared > 0.99


def test_forecast_footfall_fills_only_stations_missing_from_table_1410(tmp_path):
    from ortla.config import CandidateService, ServiceStop
    from ortla.hindcast import apply_forecast_footfall

    pd.DataFrame(
        {"Three Letter Code\n(TLC)": ["OLD"], "Entries and exits - All tickets": ["12,000"]}
    ).to_csv(tmp_path / "table_1410.csv", index=False)
    pd.DataFrame(
        {
            "crs": ["NEW", "OLD", "NEW"],
            "footfall_year": ["2023-24", "2023-24", "2022-23"],
            "forecast_entries_exits": [50_000, 999_999, 7],
        }
    ).to_csv(tmp_path / "forecasts.csv", index=False)
    config = types.SimpleNamespace(
        forecast_footfall_path=tmp_path / "forecasts.csv",
        forecast_footfall_year="2023-24",
        station_aliases=None,
        table_1410=tmp_path / "table_1410.csv",
    )
    candidate = CandidateService(
        name="T1",
        traction_code="158",
        days_of_week=["Monday"],
        stops=[
            ServiceStop(station="OLD", departure="07:00"),
            ServiceStop(station="NEW", arrival="07:10"),
        ],
    )

    applied = apply_forecast_footfall(config, [candidate])

    # The observed station keeps table 1410's figure; only the other
    # year's row is ignored.
    assert applied == {"NEW": 50_000}
    assert [stop.estimated_annual_footfall for stop in candidate.stops] == [None, 50_000]


def test_direction_split_round_trips_through_json(tmp_path):
    model = _flat_split(intercept=0.25, slope=-0.4)
    path = tmp_path / direction_split.MODEL_FILENAME
    model.save(path)
    assert direction_split.DirectionSplitModel.load(path) == model


class TestTable1410Rebalance:
    """`rebalance_to_station_totals` (ACCURACY_IMPROVEMENTS.md §2)."""

    @staticmethod
    def _symmetric_odm() -> pd.DataFrame:
        rows = []
        for origin, dest, journeys in [
            ("AAA", "BBB", 100.0),
            ("AAA", "CCC", 50.0),
            ("BBB", "CCC", 20.0),
        ]:
            rows.append({"origin_tlc": origin, "destination_tlc": dest, "journeys": journeys})
            rows.append({"origin_tlc": dest, "destination_tlc": origin, "journeys": journeys})
        return pd.DataFrame(rows)

    def test_station_marginals_are_moved_onto_their_table_1410_shares(self) -> None:
        odm = self._symmetric_odm()
        # Marginals are 300/240/140; AAA is over-stated against its
        # counted total and BBB under-stated.
        entries = {"AAA": 250.0, "BBB": 280.0, "CCC": 150.0}

        balanced, report = rebalance_to_station_totals(
            odm, entries, tolerance=1e-10, max_iterations=5000
        )

        marginal = (
            balanced.groupby("origin_tlc")["journeys"].sum()
            + balanced.groupby("destination_tlc")["journeys"].sum()
        )
        shares = marginal / marginal.sum()
        expected = pd.Series(entries) / sum(entries.values())
        pd.testing.assert_series_equal(
            shares.sort_index(), expected.sort_index(), check_names=False, atol=1e-9
        )
        assert report.converged

    def test_the_grand_total_is_preserved_so_the_definitional_gap_is_not_propagated(
        self,
    ) -> None:
        odm = self._symmetric_odm()
        # Targets summing to a tenth of the matrix: shares differ, level
        # must not move.
        entries = {"AAA": 25.0, "BBB": 28.0, "CCC": 15.0}

        balanced, _ = rebalance_to_station_totals(odm, entries)

        assert balanced["journeys"].sum() == pytest.approx(odm["journeys"].sum())

    def test_symmetry_survives_so_the_direction_split_still_applies(self) -> None:
        odm = self._symmetric_odm()

        balanced, _ = rebalance_to_station_totals(odm, {"AAA": 250.0, "BBB": 280.0, "CCC": 150.0})

        forward = balanced.set_index(["origin_tlc", "destination_tlc"])["journeys"]
        reverse = forward.swaplevel().reindex(forward.index)
        pd.testing.assert_series_equal(forward, reverse, check_names=False)

    def test_a_station_absent_from_table_1410_floats_rather_than_being_dropped(self) -> None:
        odm = self._symmetric_odm()

        balanced, report = rebalance_to_station_totals(odm, {"AAA": 250.0, "BBB": 280.0})

        assert report.unconstrained == 1
        assert set(balanced["origin_tlc"]) == {"AAA", "BBB", "CCC"}
        assert balanced.loc[balanced["origin_tlc"] == "CCC", "journeys"].sum() > 0

    def test_no_overlap_with_table_1410_is_an_error_rather_than_a_silent_no_op(self) -> None:
        odm = self._symmetric_odm()

        with pytest.raises(ValueError, match="nothing to constrain"):
            rebalance_to_station_totals(odm, {"ZZZ": 10.0})


def _direct_calls(minutes: int) -> pd.DataFrame:
    """One direct COL -> LST train taking ``minutes``."""
    arrive = f"{8 + (minutes // 60):02d}{minutes % 60:02d}"
    return pd.DataFrame(
        [
            {
                "train_uid": "D1",
                "seq": 0,
                "crs": "COL",
                "public_arrival": "",
                "public_departure": "0800",
                "is_public_call": "True",
            },
            {
                "train_uid": "D1",
                "seq": 1,
                "crs": "LST",
                "public_arrival": arrive,
                "public_departure": "",
                "is_public_call": "True",
            },
        ]
    )


def test_substitution_share_of_its_own_group_cannot_exceed_one(tmp_path: pathlib.Path) -> None:
    """A figure above the whole group must not reach the pivot logit.

    Handed the direction-split cell (1.75 x the raw one) for a terminal
    that holds 95% of its group, the old code computed a share of 1.66,
    and a slower candidate (pull < 1) put the logit's denominator near
    zero: Colchester -> Liverpool Street came out 52 times too big in
    the Beaulieu Park hindcast.
    """
    path = tmp_path / "odm_Monday_am_peak.csv"
    path.write_text("origin_tlc,LST,SRA\nCOL,28500,1500\n")
    context = demand.SubstitutionContext(
        aliases={}, groups={"london": ["LST", "SRA"]}, theta=0.05, interchange_penalty=20.0
    )
    calls = _direct_calls(50)
    raw = context.terms(path, calls, "COL", "LST", 28500.0, 60.0)
    assert raw.substituted_trips is not None
    assert raw.substituted_trips < raw.group_trips
    misused = context.terms(path, calls, "COL", "LST", 28500.0 * 1.75, 60.0)
    assert misused.substituted_trips <= misused.group_trips + 1e-9


def test_split_direction_is_applied_after_substitution() -> None:
    """The substituted base follows the split's factor on the cell."""
    choice = demand.ChoiceTerms(substituted_trips=27604.0, group_trips=30000.0)
    moved = demand._on_split_direction(choice, 28500.0, 28500.0 * 1.75)
    assert moved.substituted_trips == pytest.approx(27604.0 * 1.75)
    assert moved.group_trips == 30000.0
    # No split, no substitution, or no raw cell: left exactly as it was.
    assert demand._on_split_direction(choice, 28500.0, 28500.0) is choice
    empty = demand.ChoiceTerms()
    assert demand._on_split_direction(empty, 28500.0, 40000.0) is empty
    assert demand._on_split_direction(choice, 0.0, 40000.0) is choice


def test_a_slower_train_does_not_make_its_destination_less_attractive(
    tmp_path: pathlib.Path,
) -> None:
    """Only a journey better than today's moves demand.

    The quicker train still runs, so a slower candidate leaves the
    destination's journey as it was and its demand base on its raw cell.
    A faster one pulls.
    """
    path = tmp_path / "odm_Monday_am_peak.csv"
    path.write_text("origin_tlc,LST,SRA\nCOL,6000,4000\n")
    context = demand.SubstitutionContext(
        aliases={}, groups={"london": ["LST", "SRA"]}, theta=0.05, interchange_penalty=20.0
    )
    calls = _direct_calls(50)
    slower = context.terms(path, calls, "COL", "LST", 6000.0, 65.0)
    assert slower.substituted_trips == pytest.approx(6000.0)
    faster = context.terms(path, calls, "COL", "LST", 6000.0, 40.0)
    assert 6000.0 < faster.substituted_trips < 10000.0


def test_split_counts_a_candidate_still_in_the_timetable_once() -> None:
    """Calls-only strip: a candidate still running is counted once.

    Counting it again tilted both directions towards themselves, and
    with equal footfall the two shares summed to more than one.
    """
    in_timetable = pd.DataFrame({"train_uid": ["C1", "C2", "X9"]})
    assert demand._trains_after(in_timetable, "C1", 3, 3, 2, 2) == (3, 3)
    model = _flat_split()
    forward, reverse = demand._trains_after(in_timetable, "C1", 3, 3, 2, 2)
    there = model.share("am_peak", None, None, forward, reverse)
    back = model.share("am_peak", None, None, reverse, forward)
    assert there + back == pytest.approx(1.0)


def test_split_counts_added_trains_both_ways_when_they_are_new() -> None:
    """A proposed or wholly stripped train adds candidates both ways."""
    elsewhere = pd.DataFrame({"train_uid": ["X9"]})
    assert demand._trains_after(elsewhere, "C1", 3, 2, 2, 1) == (5, 3)
    empty = pd.DataFrame(columns=["train_uid"])
    assert demand._trains_after(empty, "C1", 0, 0, 1, 0) == (1, 0)


def test_the_round_trip_sums_both_directions_and_picks_the_best_bcr():
    from ortla.discovery import _round_trip
    from ortla.scenario import ScenarioOption, ScenarioResult

    def option(freq, time, pvb, pvc):
        return ScenarioOption(
            label="",
            shift_minutes=0.0,
            departure_time=time,
            trains_per_day=freq,
            feasible=True,
            worst_gap_minutes=None,
            annual_passengers=100 * freq,
            annual_revenue=0.0,
            avg_fare=0.0,
            time_saving_minutes=0.0,
            bcr=pvb / pvc,
            npv=pvb - pvc,
            pvb=pvb,
            pvc=pvc,
        )

    def result(*options):
        return ScenarioResult(
            template_name="x",
            options=list(options),
            time_saving_minutes=0.0,
            n_departures_evaluated=0,
            n_departures_feasible=0,
            running_times=None,
        )

    out = result(option(1, "07:00", 2, 10), option(1, "12:00", 3, 10), option(4, "12:00", 8, 40))
    back = result(option(1, "16:00", 4, 10), option(4, "16:00", 30, 40))
    trip = _round_trip(out, back)
    # 1/day: (3 + 4) / 20 = 0.35 at the better outbound slot; 4/day: 38
    # / 80 = 0.475.
    assert (trip.trains_per_day, trip.departure_time, trip.return_departure_time) == (
        4,
        "12:00",
        "16:00",
    )
    assert trip.bcr == 38 / 80
    assert trip.annual_passengers == 800
    # A frequency the return cannot run is dropped, and no return means
    # one way.
    assert _round_trip(out, result(option(1, "16:00", 4, 10))).trains_per_day == 1
    assert _round_trip(out, None).bcr == 0.3
    assert _round_trip(out, result()) is None


def test_a_long_one_change_journey_and_a_second_code_still_count_as_served():
    import pandas as pd

    from ortla.service_lookup import existing_journey

    def train(uid, *calls):
        return [
            {
                "train_uid": uid,
                "seq": i,
                "crs": crs,
                "public_arrival": arr,
                "public_departure": dep,
                "is_public_call": "true",
            }
            for i, (crs, arr, dep) in enumerate(calls)
        ]

    filtered = pd.DataFrame(
        train("A", ("EUS", None, "0800"), ("CRE", "0930", None))
        + train("B", ("CRE", None, "0940"), ("SHR", "1020", None))
        + train("C", ("RYE", None, "0800"), ("AFK", "0825", None))
        + train("D", ("AFK", None, "0830"), ("SPX", "0907", None))
        + train("E", ("BHM", None, "0800"), ("EUS", "0922", None))
        + train("F", ("VIC", None, "1000"), ("BTN", "1055", None))
    )
    # 90 + 40 minutes on board, past the old 120-minute cap; plus the
    # change penalty.
    assert existing_journey(filtered, "EUS", "SHR", 20.0) == (150.0, False)
    # The Kent trains call at SPX; the demand data holds St Pancras as
    # STP.
    assert existing_journey(filtered, "RYE", "STP", 20.0) == (82.0, False)
    # Across London: 82 + 55 on board, 30 to cross, 20 for the change.
    assert existing_journey(filtered, "BHM", "BTN", 20.0) == (187.0, False)


def test_discovery_calls_at_the_busiest_stations_on_the_route(tmp_path):
    from ortla.config import DiscoveryConfig
    from ortla.discovery import DiscoveryCandidate, _route_calls, _synthesize_candidate

    # A-X-Y-B, with a stop-to-stop timing on X-Y the route itself never
    # uses.
    rows = [
        {
            "from_tiploc": u,
            "to_tiploc": v,
            "traction_code": "150",
            "entry_speed": -1,
            "exit_speed": -1,
            "sectional_running_time_minutes": 10.0,
        }
        for a, b in [("A", "X"), ("X", "Y"), ("Y", "B")]
        for u, v in [(a, b), (b, a)]
    ]
    rows.append(
        {
            "from_tiploc": "X",
            "to_tiploc": "Y",
            "traction_code": "150",
            "entry_speed": 0,
            "exit_speed": 0,
            "sectional_running_time_minutes": 12.0,
        }
    )
    pd.DataFrame(rows).to_csv(tmp_path / "bplan_timing_links.csv", index=False)
    graph = BplanTimingGraph(tmp_path)
    tiplocs = {s: s for s in "AXYB"}
    config = DiscoveryConfig.model_construct(
        default_traction_code="150",
        max_intermediate_stops=1,
        intermediate_stop_min_footfall=1_000_000.0,
        dwell_minutes=1.0,
        anchor_departure_time="08:00",
        running_time_margin_minutes=5.0,
        default_days_of_week=["Monday"],
        search_window=None,
        return_after_minutes=None,
    )
    dc = DiscoveryCandidate("A", "B", 20.0, 0.0)
    _route_calls(dc, tiplocs, tiplocs, graph, {"X": 2_000_000, "Y": 3_000_000}, {}, config)
    # Y is the busier of the two stations on the way.
    assert dc.calls == ["A", "Y", "B"]
    assert dc.leg_minutes == [20.0, 10.0]
    stops = _synthesize_candidate(dc, config).stops
    assert [(s.station, s.arrival, s.departure) for s in stops] == [
        ("A", None, "08:00"),
        ("Y", "08:20", "08:21"),
        ("B", "08:36", None),
    ]
    # No station above the footfall floor: the pair stays non-stop.
    dc = DiscoveryCandidate("A", "B", 20.0, 0.0)
    _route_calls(dc, tiplocs, tiplocs, graph, {"X": 10, "Y": 10}, {}, config)
    assert dc.calls is None


def test_od_profile_moves_the_split_and_keeps_each_pair_day_total():
    from ortla import od_profile

    shares = od_profile.hour_shares({"am_peak": "7:00", "interpeak": "9:30"})
    assert shares.loc[9, "am_peak"] == shares.loc[9, "interpeak"] == 0.5
    assert np.allclose(shares.sum(axis=1), 1.0)
    # A-B: ours even over two periods and both directions; the profile
    # is all A->B in the am peak.
    cells = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B", "C"],
            "destination": ["B", "B", "A", "A", "D"],
            "period": ["am_peak", "interpeak", "am_peak", "interpeak", "am_peak"],
            "trips": [10.0, 10.0, 10.0, 10.0, 7.0],
        }
    )
    profile = pd.Series({("A", "B", "am_peak"): 30.0, ("B", "A", "interpeak"): 10.0})
    profile.index.names = ["origin", "destination", "period"]
    assert np.allclose(od_profile.reweight(cells, profile, 0.0, 0.0), 1.0)
    f = od_profile.reweight(cells, profile, 1.0, 0.0)
    new = cells["trips"] * f
    assert np.isclose(new[:4].sum(), 40.0)  # the pair's day total is kept
    assert np.allclose(new[:4], [30.0, 0.0, 0.0, 10.0], atol=1e-6)  # and takes the profile's split
    assert f[4] == 1.0  # C-D has no profile
    half = cells["trips"] * od_profile.reweight(cells, profile, 0.5, 0.0)
    assert half[0] > half[3] > half[1] and np.isclose(half[:4].sum(), 40.0)
