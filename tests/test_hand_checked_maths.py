"""Every stage of the demand chain, on inputs small enough to check.

Each expected number is written out from the inputs in the comment
beside it, not read back from the code, so a test failing here means the
code does something other than the stated formula. Stages, in pipeline
order:

ODM build -> count readers (boardings, averaging) -> temporal factors ->
period matrices -> table 1410 rebalance -> legs and slot counts -> loads
-> change routes -> gravity and leg pricing -> accuracy metrics ->
appraisal.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ortla import appraisal as ea
from ortla import commercial_case as cc
from ortla import loading_data as ol
from ortla import network_assignment as na
from ortla import period_matrices as la
from ortla.clock import (
    WEEKDAYS_PER_YEAR,
    classify_minutes,
    classify_time_period,
    period_minutes,
    to_per_day,
)
from ortla.demand import (
    ChoiceTerms as _ChoiceTerms,
)
from ortla.demand import (
    _logit_share,
    calibrated_leg,
)
from ortla.gravity_model import GravityModel
from ortla.hindcast import accuracy_metrics
from ortla.od_matrix import build_odm_matrix, read_odm, rebalance_to_station_totals

CUTOFFS = {"am_peak": "07:00", "interpeak": "09:30", "pm_peak": "16:00", "evening": "19:00"}


# ODM build
def test_odm_sums_repeated_pairs_strips_commas_and_maps_nlcs(tmp_path):
    (tmp_path / "odm.csv").write_text(
        "origin_tlc,destination_tlc,journeys\n"
        ' aaa ,BBB,"1,000"\n'
        "AAA,BBB,250\n"  # same pair again: 1,000 + 250 = 1,250
        "BBB,AAA,40\n"
        "AAA,ZZZ,7\n"  # ZZZ is not a valid station
    )
    cfg = SimpleNamespace(
        fill_value="!",
        valid_stations=tmp_path / "st.csv",
        source_odm=tmp_path / "odm.csv",
        nlc_reference_paths=None,
        chunk_size=2,  # chunks split the repeated pair
    )
    (tmp_path / "st.csv").write_text("crsCode\nAAA\nBBB\n")
    matrix = build_odm_matrix(cfg)
    assert matrix.loc["AAA", "BBB"] == 1250
    assert matrix.loc["BBB", "AAA"] == 40
    assert matrix.loc["AAA", "AAA"] == 0
    assert matrix.loc["AAA", "ZZZ"] == "!" and matrix.loc["ZZZ", "AAA"] == "!"

    # An NLC-only release takes its TLCs from a release carrying both.
    (tmp_path / "ref.csv").write_text(
        "origin_nlc,origin_tlc,destination_nlc,destination_tlc,journeys\n1234,AAA,567,BBB,1\n"
    )
    (tmp_path / "old.csv").write_text(
        "origin_nlc,destination_nlc,journeys\n1234,0567,30\n1234,9999,5\n"
    )
    odm = read_odm(tmp_path / "old.csv", [tmp_path / "ref.csv"])
    # 0567 and 567 are one NLC; 9999 has no TLC and is dropped.
    assert odm.to_dict("records") == [
        {"origin_tlc": "AAA", "destination_tlc": "BBB", "journeys": 30.0}
    ]


# Count readers: boardings from loads, averaging dated runs
def _calls(loads, stations=("AAA", "BBB", "CCC", "DDD")):
    rows = []
    for seq, (crs, load) in enumerate(zip(stations, loads, strict=True)):
        rows.append(
            {
                "Train Operator": "op",
                "Unique ID": "U1",
                "Headcode": "1A01",
                "Service Week Day": "Monday",
                "Origin CRS": stations[0],
                "Destination CRS": stations[-1],
                "Origin Departure Time": "08:00:00",
                "Period Start Date": pd.Timestamp("2025-05-01"),
                "Departure from CRS": crs,
                "Call Sequence": seq,
                "Mean Load on Departure": load,
                "Mean Passengers On": np.nan,
                "Mean Passengers Off": np.nan,
            }
        )
    return pd.DataFrame(rows)


def test_boardings_are_the_rise_in_load_and_alightings_the_fall():
    # Loads 50 -> 80 -> 30 -> (terminus) : on 50, 30, 0, 0 ; off 0, 0,
    # 50, 30.
    out = ol.derive_on_off(_calls([50, 80, 30, 99])).sort_values("Call Sequence")
    assert out["Mean Passengers On"].tolist() == [50, 30, 0, 0]
    assert out["Mean Passengers Off"].tolist() == [0, 0, 50, 30]


def test_an_uncounted_call_is_skipped_and_the_next_differs_from_the_last_count():
    # 50, missing, 20: the missing call has no figure; the third is 20 -
    # 50 = -30 (30 off).
    out = ol.derive_on_off(_calls([50, np.nan, 20], ("AAA", "BBB", "CCC"))).sort_values(
        "Call Sequence"
    )
    on, off = out["Mean Passengers On"].tolist(), out["Mean Passengers Off"].tolist()
    assert on[0] == 50 and np.isnan(on[1]) and on[2] == 0
    assert off[2] == 50  # terminus: everyone off from the last count of 50


def test_dated_runs_average_to_one_row_with_their_sample_size():
    rows = []
    for load, on in ((40, 10), (60, 30)):
        rows.append(
            {
                "Train Operator": "op",
                "Unique ID": "U1",
                "Headcode": "1A01",
                "Service Week Day": "Monday",
                "Origin CRS": "AAA",
                "Destination CRS": "CCC",
                "Origin Departure Time": "08:00:00",
                "Departure from CRS": "BBB",
                "RSP Period": "P1",
                "load_units": "passengers",
                "Departure Time": "08:10:00",
                "Call Sequence": 1,
                "Period Start Date": pd.Timestamp("2025-05-01"),
                "Period End Date": pd.Timestamp("2025-05-28"),
                "Mean Capacity": 200,
                "Mean Load on Departure": load,
                "Mean Passengers On": on,
                "Mean Passengers Off": 0,
            }
        )
    out = ol._average_dated_runs(pd.DataFrame(rows))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Mean Load on Departure"] == 50  # (40 + 60) / 2
    assert row["Mean Passengers On"] == 20  # (10 + 30) / 2
    assert row["Sample Size"] == 2


# Time periods and distance bands
@pytest.mark.parametrize(
    ("clock", "period"),
    [
        ("06:59", "early"),
        ("07:00", "am_peak"),
        ("09:29:59", "am_peak"),
        ("09:30", "interpeak"),
        ("15:59", "interpeak"),
        ("16:00", "pm_peak"),
        ("19:00", "evening"),
        ("23:59", "evening"),
        ("00:00", "early"),
        ("junk", "Unknown"),
    ],
)
def test_period_boundaries_belong_to_the_later_period(clock, period):
    assert classify_time_period(clock, CUTOFFS) == period
    hh, mm = [*clock.split(":"), "0"][:2] if ":" in clock else (None, None)
    minutes = np.array([np.nan if hh is None else int(hh) * 60 + int(mm)], dtype=float)
    assert classify_minutes(minutes, CUTOFFS)[0] == period


def test_distance_band_upper_limits_are_inclusive():
    cfg = SimpleNamespace(bands={"commute": 30.0, "regional": 100.0}, fallback="long_distance")
    got = [la.classify_route_type(m, cfg) for m in (0.0, 30.0, 30.01, 100.0, 100.5, None)]
    assert got == ["commute", "commute", "regional", "regional", "long_distance", "Unknown"]
    vector = la._classify_distances(pd.Series([0.0, 30.0, 30.01, 100.0, 100.5, np.nan]), cfg)
    assert vector.tolist() == got


# Temporal factors and period matrices
def _loading_config(tmp_path, mode="distance", min_rows=200):
    return SimpleNamespace(
        demand_column="Mean Passengers On",
        journey_type_column="route_type_pax_off",
        journey_type_mode=mode,
        journey_type_min_rows=min_rows,
        output_path=tmp_path,
        analysis=SimpleNamespace(day_order=["Monday", "Saturday"], time_order=list(CUTOFFS)),
    )


def _analysed():
    #   type      day       period     boardings  origin
    rows = [
        ("commute", "Monday", "am_peak", "30", "AAA"),
        ("commute", "Monday", "am_peak", "10", "AAA"),
        ("commute", "Monday", "interpeak", "20", "BBB"),
        ("commute", "Saturday", "interpeak", "1,040", "BBB"),  # comma read as 1040
        ("commute", "Monday", "Unknown", "999", "BBB"),  # no period: dropped
        ("Unknown", "Monday", "am_peak", "999", "BBB"),  # no type: dropped
        ("long_distance", "Monday", "am_peak", "5", "CCC"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "route_type_pax_off",
            "Service Week Day",
            "time_period",
            "Mean Passengers On",
            "o",
        ],
    ).assign(route_type_origin=lambda f: "O:" + f["o"])


def test_factors_are_each_cells_share_of_its_types_boardings(tmp_path):
    f = la.compute_temporal_factors(_analysed(), _loading_config(tmp_path))
    got = {(r.journey_type, r._2, r.time_period): r.factor for r in f.itertuples()}
    # commute: 40 + 20 + 1040 = 1100 boardings.
    assert got == pytest.approx(
        {
            ("commute", "Monday", "am_peak"): 40 / 1100,
            ("commute", "Monday", "interpeak"): 20 / 1100,
            ("commute", "Saturday", "interpeak"): 1040 / 1100,
            ("long_distance", "Monday", "am_peak"): 1.0,
        }
    )


def test_period_matrices_split_annual_journeys_by_factor_and_sum_back(tmp_path):
    cfg = _loading_config(tmp_path)
    factors = la.compute_temporal_factors(_analysed(), cfg)
    odm = pd.DataFrame(
        {
            "origin_tlc": ["AAA", "BBB", "AAA", "AAA"],
            "destination_tlc": ["BBB", "AAA", "CCC", "DDD"],
            "journeys": [1100.0, 550.0, 200.0, 300.0],
            "journey_type": ["commute", "commute", "long_distance", "Unknown"],
        }
    )
    paths = la.generate_period_matrices(factors, cfg, typed_odm=odm, output_dir=tmp_path / "m")
    cells = {p.stem: pd.read_csv(p, index_col=0) for p in paths}
    assert cells["odm_Monday_am_peak"].loc["AAA", "BBB"] == pytest.approx(40)  # 1100 x 40/1100
    assert cells["odm_Monday_am_peak"].loc["BBB", "AAA"] == pytest.approx(20)  # 550 x 40/1100
    assert cells["odm_Monday_am_peak"].loc["AAA", "CCC"] == pytest.approx(200)  # 200 x 1
    assert cells["odm_Saturday_interpeak"].loc["AAA", "BBB"] == pytest.approx(1040)
    assert cells["odm_Saturday_interpeak"].loc["AAA", "CCC"] == 0
    # An unclassified pair gets nothing anywhere (the warned zero-demand
    # case).
    assert all(c.loc["AAA", "DDD"] == 0 for c in cells.values())
    # Conservation: a classified pair's cells sum back to its annual
    # journeys.
    assert sum(c.loc["AAA", "BBB"] for c in cells.values()) == pytest.approx(1100)
    # And a cell on a typical day is the annual cell over 365.25 / 7.
    assert to_per_day(pd.DataFrame([[WEEKDAYS_PER_YEAR * 3]])).iloc[0, 0] == pytest.approx(3)
    assert pytest.approx(52.17857, abs=1e-5) == WEEKDAYS_PER_YEAR


def test_origin_profiles_are_added_beside_bands_fitted_to_every_row(tmp_path):
    cfg = _loading_config(tmp_path, mode="origin_station", min_rows=2)
    f = la.compute_temporal_factors(_analysed(), cfg)
    by_type = f.groupby("journey_type")["factor"]
    # O:AAA has 2 rows (30, 10, both Monday am) -> 1.0; O:BBB has 2
    # usable rows (20, 1040) -> 20/1060 and 1040/1060; O:CCC has 1 row,
    # below the minimum.
    assert set(f["journey_type"]) == {"commute", "long_distance", "O:AAA", "O:BBB"}
    assert by_type.sum().to_dict() == pytest.approx({t: 1.0 for t in by_type.groups})
    # The commute band is still fitted to all three of its origins.
    assert f.loc[f["journey_type"].eq("commute"), "boardings"].sum() == 1100


# Table 1410 rebalance
def test_rebalance_matches_station_shares_symmetrically_and_keeps_the_total():
    # Four stations, every ordered pair 10 trips: 120 trips, each
    # marginal 60. Targets in shares 1/3, 1/3, 1/6, 1/6 of 240 -> 80,
    # 80, 40, 40. By symmetry r_A = r_B = a, r_C = r_D = c with
    #   20 (a^2 + 2ac) = 80  and  20 (c^2 + 2ac) = 40,
    # so a^2 = 4 / sqrt(3).
    codes = ["A", "B", "C", "D"]
    odm = pd.DataFrame(
        [(o, d, 10.0) for o in codes for d in codes if o != d],
        columns=["origin_tlc", "destination_tlc", "journeys"],
    )
    out, report = rebalance_to_station_totals(
        odm, {"A": 2000, "B": 2000, "C": 1000, "D": 1000}, tolerance=1e-10, max_iterations=500
    )
    a2 = 4 / math.sqrt(3)
    a = math.sqrt(a2)
    c = (4 - a2) / (2 * a)
    cell = out.set_index(["origin_tlc", "destination_tlc"])["journeys"]
    assert cell["A", "B"] == pytest.approx(10 * a2, rel=1e-6)  # 23.094
    assert cell["C", "D"] == pytest.approx(10 * c * c, rel=1e-6)  # 3.094
    assert cell["A", "C"] == pytest.approx(10 * a * c, rel=1e-6)  # 8.453
    assert cell["A", "C"] == pytest.approx(cell["C", "A"])
    assert out["journeys"].sum() == pytest.approx(120)
    assert report.converged


def test_size_exponent_targets_shrink_the_gap_between_large_and_small_stations():
    # 10,000 and 100 entries with exponent -0.5: targets 10,000^0.5 =
    # 100 and 100^0.5 = 10, so the big station's share falls from 100x
    # to 10x the small one's. Zero entries drop out.
    from ortla.od_matrix import station_targets

    assert station_targets({"A": 10_000, "B": 100, "C": 0}, -0.5) == pytest.approx(
        {"A": 100, "B": 10}
    )
    assert station_targets({"A": 10_000, "B": 100}) == {"A": 10_000.0, "B": 100.0}


# Legs, slot counts and loads
def _day(trains):
    rows = []
    for service, (uid, stops) in enumerate(trains):
        for crs, arr, dep in stops:
            rows.append(
                {
                    "service": service,
                    "crs": crs,
                    "public_arrival": arr,
                    "public_departure": dep,
                    "train_uid": uid,
                    "atoc_code": "XX",
                    "signalling_id": uid,
                    "svc_origin_crs": stops[0][0],
                    "svc_dest_crs": stops[-1][0],
                }
            )
    return pd.DataFrame(rows)


def test_legs_are_every_ordered_pair_with_times_across_midnight():
    calls = _day([("T1", [("A", None, "2350"), ("B", "2355", "2356"), ("C", "0010", None)])])
    legs = na.build_legs(calls, CUTOFFS)
    got = {(r.o, r.d): (r.in_vehicle_minutes, r.period) for r in legs.itertuples()}
    assert got == {
        ("A", "B"): (5, "evening"),  # 23:50 -> 23:55
        ("A", "C"): (20, "evening"),  # 23:50 -> 00:10, over midnight
        ("B", "C"): (14, "evening"),  # 23:56 -> 00:10
    }


def test_a_slot_shares_its_cell_over_its_trains_and_loads_accumulate():
    # T1 A 08:00 - B 08:10 - C 08:20 ; T2 A 08:05 - C 08:25 (both
    # am_peak).
    calls = _day(
        [
            ("T1", [("A", None, "0800"), ("B", "0810", "0811"), ("C", "0820", None)]),
            ("T2", [("A", None, "0805"), ("C", "0825", None)]),
        ]
    )
    legs = na.build_legs(calls, CUTOFFS)
    n = na.trains_per_leg(legs)
    assert dict(zip(zip(legs["o"], legs["d"], strict=True), n, strict=True)) == {
        ("A", "B"): 1,
        ("A", "C"): 2,
        ("B", "C"): 1,  # A-C is one slot, two trains
    }
    # Daily cells: A-B 6, A-C 10, B-C 4 (stored annual, x 365.25/7).
    daily_cell = {("A", "B"): 6, ("A", "C"): 10, ("B", "C"): 4}
    annual = np.array(
        [daily_cell[k] * WEEKDAYS_PER_YEAR for k in zip(legs["o"], legs["d"], strict=True)]
    )
    trips = na.leg_daily_trips(annual, n)
    seg = na.segment_loads(calls, legs, trips).set_index(["train_uid", "from_station"])
    # T1 A->B carries A-B 6 + half of A-C 5 = 11; B->C carries 5 + B-C 4
    # = 9.
    assert seg.loc[("T1", "A"), "load"] == pytest.approx(11)
    assert seg.loc[("T1", "A"), "boardings"] == pytest.approx(11)
    assert seg.loc[("T1", "B"), "boardings"] == pytest.approx(4)
    assert seg.loc[("T1", "B"), "alightings"] == pytest.approx(6)
    assert seg.loc[("T1", "B"), "load"] == pytest.approx(9)
    assert seg.loc[("T2", "A"), "load"] == pytest.approx(5)  # its half of A-C
    # Everything boarded is the day's matrix total: 6 + 10 + 4 = 20.
    assert seg["boardings"].sum() == pytest.approx(20)


def test_slot_with_no_counted_train_falls_back_to_one():
    assert na.leg_daily_trips(np.array([WEEKDAYS_PER_YEAR * 8]), np.array([0.0]))[0] == 8


def test_period_lengths_and_haversine():
    assert period_minutes(CUTOFFS) == {
        "early": 180.0,
        "am_peak": 150.0,
        "interpeak": 390.0,
        "pm_peak": 180.0,
        "evening": 300.0,
    }
    coords = pd.DataFrame(
        {"lat": [0.0, 0.0, 51.5308], "lon": [0.0, 1.0, -0.1238]}, index=["X", "Y", "KGX"]
    )
    km = na.station_km(coords, np.array(["X", "Z"]), np.array(["Y", "X"]))
    assert km[0] == pytest.approx(2 * math.pi * 6371 / 360)  # one degree of the equator
    assert np.isnan(km[1])


def test_change_route_picks_the_fastest_via_and_loads_both_legs():
    # A-B 10, B-D 10 (20 via B); A-C 5, C-D 30 (35 via C). No A-D train.
    o = np.array(["A", "B", "A", "C"])
    d = np.array(["B", "D", "C", "D"])
    routes = na.ChangeRoutes(o, d, np.array([10.0, 10.0, 5.0, 30.0]), interchange_penalty=5)
    routes.route(np.array(["A", "A"]), np.array(["D", "B"]))  # A-B is direct: not routed
    s = routes.stations
    assert list(zip(s[routes.origin], s[routes.via], s[routes.destination], strict=True)) == [
        ("A", "B", "D")
    ]
    assert routes.minutes.tolist() == [25.0]  # 10 + 10 + 5 penalty
    demand = routes.leg_demand(np.array([7.0]))
    assert demand[s.get_loc("A"), s.get_loc("B")] == 7
    assert demand[s.get_loc("B"), s.get_loc("D")] == 7
    assert demand.sum() == 14


# Gravity model and leg pricing
GRAVITY = GravityModel(
    coefficients={
        "intercept": 1.0,
        "ln_footfall_origin": 0.5,
        "ln_footfall_destination": 0.5,
        "ln_distance": -1.0,
        "ln_trains": 0.6,
    },
    n_pairs=1,
    r_squared=1.0,
    residual_std=0.0,
    source_odm="",
    smearing=1.5,
)


def test_gravity_prediction_and_frequency_ratio():
    # e^1 x 10000^0.5 x 40000^0.5 / 20 x 4^0.6 x 1.5 = e x 1000 x 2.2974
    # x 1.5
    expected = math.e * 1000 * 4**0.6 * 1.5
    assert GRAVITY.predict(10_000, 40_000, 20, 4) == pytest.approx(expected)
    assert expected == pytest.approx(9367.5, abs=0.1)
    # Distances under a mile and fewer than one train are floored at 1.
    assert GRAVITY.predict(1, 1, 0.2, 0) == pytest.approx(math.e * 1.5)
    assert GRAVITY.frequency_ratio(4, 8) == pytest.approx(2**0.6)  # 1.5157
    assert GRAVITY.frequency_ratio(4, 8, 0.1) == pytest.approx(2**0.1)  # 1.0718


class _Gravity:
    model = GRAVITY

    def __init__(self, synthetic=50.0):
        self.synthetic = synthetic

    def synthetic_slot_trips(self, *_):
        return self.synthetic, None


def _leg(**kw):
    defaults = {
        "existing_trips": 100.0,
        "trains_in_period": 2,
        "trains_per_day": 10,
        "added_in_period": 1,
        "added_per_day": 1,
    }
    return calibrated_leg(
        _Gravity(kw.pop("synthetic", 50.0)), None, None, "Monday", "am_peak", **{**defaults, **kw}
    )


def test_pivot_grows_the_slot_by_the_frequency_ratio_and_shares_it_over_all_trains():
    # new = 100 x (11/10)^0.6 = 105.885; over 3 trains -> 35.295
    # captured, 1.962 generated.
    captured, generated, label = _leg()
    assert label == "gravity_pivot"
    assert captured == pytest.approx(100 * 1.1**0.6 / 3)
    assert generated == pytest.approx(100 * (1.1**0.6 - 1) / 3)
    # Without dilution demand grows with the period's trains: 100 x 3/2
    # = 150 -> 50 each.
    captured, generated, _ = _leg(apply_dilution=False)
    assert (captured, generated) == pytest.approx((50, 50 / 3))


def test_new_link_takes_its_share_plus_the_step_or_the_median_floor():
    first = {
        "trains_in_period": 0,
        "trains_per_day": 0,
        "added_per_day": 4,
        "choice": _ChoiceTerms(direct_share=0.3),
        "new_link_uplift": 0.28,
    }
    # old = 100 x 0.3 = 30; step 100 x (e^(0.28 x 0.3) - 1) = 8.763;
    # floor 50 / 1.5 = 33.3.
    captured, generated, label = _leg(**first)
    assert label == "gravity_new_link"
    assert captured == pytest.approx(30 + 100 * (math.exp(0.084) - 1))
    assert generated == pytest.approx(100 * (math.exp(0.084) - 1))
    # Only 3 trains a day: no step, the median floor (33.3) beats the
    # share (30). The floor's 3.3 extra come off the pair's 100 existing
    # trips, so nothing is new to rail.
    captured, generated, _ = _leg(**{**first, "added_per_day": 3})
    assert (captured, generated) == pytest.approx((50 / 1.5, 0.0))
    # With only 10 existing trips the floor (33.3) takes all 10 and
    # generates the other 23.3.
    captured, generated, _ = _leg(**{**first, "added_per_day": 3, "existing_trips": 10.0})
    assert (captured, generated) == pytest.approx((50 / 1.5, 50 / 1.5 - 10))


def test_a_slower_new_link_never_generates_more():
    # Falling direct share as the new train slows.
    out = [
        _leg(
            trains_in_period=0,
            trains_per_day=0,
            added_per_day=4,
            choice=_ChoiceTerms(direct_share=s),
            new_link_uplift=0.28,
        )[1]
        for s in (0.9, 0.5, 0.2, 0.05)
    ]
    assert all(a >= b for a, b in itertools.pairwise(out))


def test_no_observed_demand_uses_the_synthetic_prediction_and_after_cells_are_shared():
    captured, generated, label = _leg(existing_trips=0.0, synthetic=60.0, trains_in_period=0)
    assert (captured, generated, label) == (60.0, 60.0, "gravity_synthetic")
    # A matrix observed with the train running: cell over trains after,
    # nothing generated.
    assert _leg(existing_trips=90.0, cell_is_after=True) == (30.0, 0.0, "observed_share")


def test_logit_share():
    assert _logit_share(10, 20, 0.1) == pytest.approx(1 / (1 + math.exp(-1)))  # 0.7311
    assert _logit_share(20, 20, 0.1) == 0.5


# Accuracy metrics
def test_accuracy_metrics_by_hand():
    # Predictions 2, 4 and 6 against actuals 1, 4 and 8: errors 1, 0
    # and -2.
    m = accuracy_metrics(pd.Series([2.0, 4.0, 6.0, np.nan]), pd.Series([1.0, 4.0, 8.0, 5.0]))
    assert m["n"] == 3
    assert m["ratio_total_pred_to_actual"] == pytest.approx(12 / 13)
    assert m["mean_error"] == pytest.approx(-1 / 3)
    assert m["mae"] == pytest.approx(1)
    assert m["rmse"] == pytest.approx(math.sqrt(5 / 3))
    assert m["mape_pct"] == pytest.approx((1 + 0 + 0.25) / 3 * 100)
    # GEH = sqrt(2 e^2 / (p + a)): sqrt(2/3), 0, sqrt(8/14)
    assert m["geh_mean"] == pytest.approx((math.sqrt(2 / 3) + math.sqrt(8 / 14)) / 3)
    # r = 14 / sqrt(8 x 222/9)
    assert m["pearson_r"] == pytest.approx(14 / math.sqrt(8 * 222 / 9))


# Appraisal
def test_commercial_npv_irr_and_payback_by_hand():
    # Build 2027 (-£1m), run 2028-29 at £1m revenue - £0.2m opex =
    # +£0.8m, 10%.
    params = cc.CommercialCaseParameters(
        proposed_services=[cc.CommercialServiceInput("s", 100_000, 10.0)],
        appraisal_year=2028,
        appraisal_period=2,
        construction_period=1,
        demand_ramp_up_years=0,
        demand_ramp_up_profile=[1.0],
        capital_cost_m=1.0,
        annual_operating_cost_m=0.2,
        discount_rate=0.10,
    )
    r = cc.run_commercial_case(params)
    assert r.npv_m == pytest.approx(-1 + 0.8 / 1.1 + 0.8 / 1.21)  # 0.38843
    # -1 + 0.8x + 0.8x^2 = 0 with x = 1 / (1 + irr)
    x = (-0.8 + math.sqrt(0.64 + 3.2)) / 1.6
    assert r.irr == pytest.approx(1 / x - 1, abs=1e-5)  # 0.3798
    assert r.payback_year == 2029 and r.discounted_payback_year == 2029


def test_ramp_up_profile_by_operating_year():
    p = cc.CommercialCaseParameters(demand_growth_rate=0.1)
    # profile 0.5, 0.75, 0.9 then 1.0, times 1.1^year
    got = [cc._demand_factor(p, y) for y in range(5)]
    assert got == pytest.approx([0.5, 0.75 * 1.1, 0.9 * 1.21, 1.331, 1.4641])


def test_crowding_multiplier_is_continuous_at_its_knots():
    knots = {0.85: 1.0, 1.0: 1.05, 1.5: 1.40, 2.0: 1.90, 3.0: 1.90}
    for load, value in knots.items():
        assert ea._crowding_multiplier(load) == pytest.approx(value)
        assert ea._crowding_multiplier(load - 1e-9) == pytest.approx(value, abs=1e-6)


def test_a_pair_whose_journeys_all_change_by_choice_is_not_treated_as_empty():
    # The base nets to 0 (everyone changes) but the pair holds 100 trips:
    # new-link branch, the median floor 33.3 is drawn from those 100.
    captured, generated, label = _leg(
        existing_trips=0.0,
        pair_trips=100.0,
        trains_in_period=0,
        trains_per_day=0,
        added_per_day=3,
        choice=_ChoiceTerms(direct_share=0.01),
    )
    assert label == "gravity_new_link"
    assert (captured, generated) == pytest.approx((50 / 1.5, 0.0))
