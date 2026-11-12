"""Tests for the whole-network base-year assignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ortla import direction_split
from ortla import network_assignment as na
from ortla.clock import (
    UNKNOWN_PERIOD,
    WEEKDAYS_PER_YEAR,
    cif_minutes,
    classify_minutes,
    classify_time_period,
)
from ortla.config import CandidateService, ProposeConfig, ServiceStop
from ortla.demand import predict_demand
from ortla.service_lookup import extract_od_services
from ortla.service_lookup import filter_services as _filter_services
from ortla.stations import DEFAULT_STATION_RECODES, recode_columns, resolve_recodes

CUTOFFS = {"am_peak": "7:00", "interpeak": "9:30", "pm_peak": "16:00", "evening": "19:00"}
WEDNESDAY = pd.Timestamp("2026-06-17")


def _train(uid, headcode, stops, *, atoc="XX", days="1111100"):
    """Calls rows for one train from ``(crs, arrival, departure)``."""
    rows = []
    for seq, (crs, arr, dep) in enumerate(stops):
        rows.append(
            {
                "train_uid": uid,
                "stp_indicator": "P",
                "atoc_code": atoc,
                "days_runs": days,
                "start_date": "2026-05-17",
                "end_date": "2026-12-12",
                "train_category": "OO",
                "signalling_id": headcode,
                "svc_origin_crs": stops[0][0],
                "svc_origin_dep": stops[0][2],
                "svc_dest_crs": stops[-1][0],
                "seq": str(seq),
                "tiploc": crs,
                "crs": crs,
                "public_arrival": arr,
                "public_departure": dep,
                "is_public_call": "True",
            }
        )
    return rows


@pytest.fixture
def calls():
    return pd.DataFrame(
        _train(
            "T1", "1A01", [("AAA", None, "0800"), ("BBB", "0810", "0811"), ("CCC", "0820", None)]
        )
        + _train("T2", "1A02", [("AAA", None, "0805"), ("CCC", "0825", None)])
        + _train(
            "T3", "1A03", [("AAA", None, "1000"), ("BBB", "1010", "1011"), ("CCC", "1020", None)]
        )
        # A second listing of T1 under another UID: one train, not two.
        + _train(
            "T4", "1A01", [("AAA", None, "0800"), ("BBB", "0810", "0811"), ("CCC", "0820", None)]
        )
        # Runs at weekends only: not in Wednesday's timetable.
        + _train("T5", "1A05", [("AAA", None, "0802"), ("CCC", "0822", None)], days="0000011")
    )


@pytest.fixture
def matrices(tmp_path):
    """Symmetric cells of 10 (A-C), 4 (A-B) and 6 (B-C) trips a day."""
    codes = ["AAA", "BBB", "CCC"]
    daily = pd.DataFrame(0.0, index=codes, columns=codes)
    for o, d, trips in (("AAA", "CCC", 10), ("AAA", "BBB", 4), ("BBB", "CCC", 6)):
        daily.loc[o, d] = daily.loc[d, o] = trips
    for period in ("am_peak", "interpeak"):
        (daily * WEEKDAYS_PER_YEAR).rename_axis("origin_tlc").to_csv(
            tmp_path / f"odm_Wednesday_{period}.csv"
        )
    return na.PeriodMatrices(tmp_path)


def _assign(calls, matrices):
    day = na.day_calls(calls, WEDNESDAY, "Wednesday")
    legs = na.build_legs(day, CUTOFFS)
    trains = na.trains_per_leg(legs)
    annual = matrices.cells(
        "Wednesday", legs["period"].to_numpy(), legs["o"].to_numpy(), legs["d"].to_numpy()
    )
    return day, legs, na.segment_loads(day, legs, na.leg_daily_trips(annual, trains))


def test_loads_share_each_slot_over_every_train_serving_it(calls, matrices):
    _, _, segments = _assign(calls, matrices)
    load = segments.set_index(["train_uid", "from_station"])["load"]
    # A -> C in the morning has two trains (T1 and T2), so 5 each; T1
    # alone serves A -> B and B -> C then.
    assert load[("T1", "AAA")] == pytest.approx(4 + 5)
    assert load[("T1", "BBB")] == pytest.approx(5 + 6)
    assert load[("T2", "AAA")] == pytest.approx(5)
    # T3 is alone in the interpeak.
    assert load[("T3", "AAA")] == pytest.approx(4 + 10)
    assert load[("T3", "BBB")] == pytest.approx(10 + 6)


def test_a_second_listing_of_a_train_is_loaded_once_and_counted_once(calls, matrices):
    day, legs, _ = _assign(calls, matrices)
    assert "T4" not in set(day["train_uid"])
    assert "T5" not in set(day["train_uid"])
    slots = na.slot_trains(legs)
    assert slots[("AAA", "CCC", "am_peak")] == 2


def test_every_trip_that_boards_alights(calls, matrices):
    _, _, segments = _assign(calls, matrices)
    last = segments.groupby("train_uid").tail(1)
    totals = segments.groupby("train_uid")[["boardings", "alightings"]].sum()
    # The final segment's load is what alights at the terminus.
    arriving = totals["boardings"] - totals["alightings"]
    assert np.allclose(arriving.loc[last["train_uid"]].to_numpy(), last["load"].to_numpy())


def test_slot_counts_match_the_service_lookup(calls):
    """The vectorised count matches `extract_od_services` everywhere."""
    day = na.day_calls(calls, WEDNESDAY, "Wednesday")
    slots = na.slot_trains(na.build_legs(day, CUTOFFS))
    filtered = _filter_services(calls, WEDNESDAY, [2]).assign(
        seq=lambda d: pd.to_numeric(d["seq"])
    )
    for o, d in (("AAA", "BBB"), ("AAA", "CCC"), ("BBB", "CCC")):
        listed = extract_od_services(filtered, o, d, CUTOFFS, 1)
        expected = listed.groupby("time_period").size()
        for period, n in expected.items():
            assert slots[(o, d, period)] == n


def test_a_leg_with_no_matrix_cell_marks_its_segments_incomplete(calls, tmp_path):
    empty = na.PeriodMatrices(tmp_path)  # no files at all
    day = na.day_calls(calls, WEDNESDAY, "Wednesday")
    legs = na.build_legs(day, CUTOFFS)
    annual = empty.cells(
        "Wednesday", legs["period"].to_numpy(), legs["o"].to_numpy(), legs["d"].to_numpy()
    )
    segments = na.segment_loads(day, legs, na.leg_daily_trips(annual, na.trains_per_leg(legs)))
    assert segments["incomplete"].all()
    assert (segments["load"] == 0).all()


def test_a_pair_with_no_direct_train_rides_both_legs_of_its_fastest_change():
    # A -> B 10 min, B -> D 15 min, A -> C 5 min, C -> D 40 min: A -> D
    # goes via B.
    routes = na.ChangeRoutes(
        np.array(["AAA", "BBB", "AAA", "CCC"]),
        np.array(["BBB", "DDD", "CCC", "DDD"]),
        np.array([10.0, 15.0, 5.0, 40.0]),
        interchange_penalty=20.0,
    ).route(np.array(["AAA", "AAA", "BBB"]), np.array(["DDD", "BBB", "CCC"]))
    stations = routes.stations
    # A -> B has a direct train and B -> C no route at all: only A -> D
    # is routed.
    assert len(routes.origin) == 1
    assert stations[routes.via[0]] == "BBB"
    assert routes.minutes[0] == pytest.approx(10 + 15 + 20)
    legs = routes.leg_demand(np.array([7.0]))
    index = {code: k for k, code in enumerate(stations)}
    assert legs[index["AAA"], index["BBB"]] == pytest.approx(7.0)
    assert legs[index["BBB"], index["DDD"]] == pytest.approx(7.0)
    assert legs.sum() == pytest.approx(14.0)


def test_period_classification_matches_the_scalar_rule():
    times = ["0000", "0659", "0700", "0929", "0930", "1559", "1600", "1859", "1900", "2359"]
    minutes = cif_minutes(pd.Series(times))
    vectorised = classify_minutes(minutes, CUTOFFS)
    scalar = [classify_time_period(f"{t[:2]}:{t[2:]}:00", CUTOFFS) for t in times]
    assert list(vectorised) == scalar
    assert classify_minutes(np.array([np.nan]), CUTOFFS)[0] == UNKNOWN_PERIOD


def test_cif_times_read_like_the_scalar_parser():
    minutes = cif_minutes(pd.Series(["0831H", "0831", "", None, "12"]))
    assert minutes[0] == minutes[1] == 8 * 60 + 31
    assert np.isnan(minutes[2:]).all()


# --- journeys that change trains, and second station codes
# ------------------


@pytest.fixture
def change_calls():
    """A -> B and B -> C in the morning peak; a train back B -> A."""
    return pd.DataFrame(
        _train("U1", "2A01", [("AAA", None, "0800"), ("BBB", "0810", None)])
        + _train("U2", "2A02", [("BBB", None, "0820"), ("CCC", "0835", None)])
        + _train("U3", "2A03", [("BBB", None, "0800"), ("AAA", "0810", None)])
    )


def _transfers(calls, directory, coordinates=None, routing="day"):
    return na.TransferDemand(
        calls,
        WEDNESDAY,
        CUTOFFS,
        {},
        directory,
        interchange_penalty=20.0,
        detour_base_minutes=30.0,
        detour_minutes_per_km=2.0,
        coordinates=coordinates,
        routing=routing,
    )


def test_a_pair_needing_a_change_adds_its_trips_to_both_legs(change_calls, matrices):
    transfers = _transfers(change_calls, matrices.directory)
    a_to_c = 10 * WEEKDAYS_PER_YEAR
    assert transfers.trips("Wednesday", "am_peak", "AAA", "BBB") == pytest.approx(a_to_c)
    assert transfers.trips("Wednesday", "am_peak", "BBB", "CCC") == pytest.approx(a_to_c)
    # A pair a train serves is not routed, C has no train out of it, and
    # no train runs in the interpeak.
    assert transfers.trips("Wednesday", "am_peak", "BBB", "AAA") == 0.0
    assert transfers.trips("Wednesday", "am_peak", "CCC", "BBB") == 0.0
    assert transfers.trips("Wednesday", "interpeak", "AAA", "BBB") == 0.0
    assert transfers.summary == [
        {
            "day": "Wednesday",
            "pairs_with_demand": 6,
            "routed_one_change": 1,
            "dropped_as_detours": 0,
        }
    ]


def test_scale_multiplies_routed_trips(change_calls, matrices):
    transfers = _transfers(change_calls, matrices.directory)
    transfers.scale, transfers.london_scale = 1.2, 2.3
    a_to_b = 10 * WEEKDAYS_PER_YEAR
    assert transfers.trips("Wednesday", "am_peak", "AAA", "BBB") == pytest.approx(1.2 * a_to_b)
    assert transfers.trips("Wednesday", "am_peak", "AAA", "BBB", london=True) == pytest.approx(
        2.3 * a_to_b
    )


def test_period_routing_changes_trains_where_the_direct_train_does_not_run(matrices):
    # A -> C direct in the morning peak only; in the interpeak A -> B
    # then B -> C.
    calls = pd.DataFrame(
        _train("P1", "2C01", [("AAA", None, "0800"), ("CCC", "0820", None)])
        + _train("P2", "2C02", [("AAA", None, "1000"), ("BBB", "1010", None)])
        + _train("P3", "2C03", [("BBB", None, "1030"), ("CCC", "1045", None)])
    )
    a_to_c = 10 * WEEKDAYS_PER_YEAR
    by_day = _transfers(calls, matrices.directory, routing="day")
    by_period = _transfers(calls, matrices.directory, routing="period")
    # The day rule sees A -> C served and routes nothing: the
    # interpeak's A -> C trips are lost.
    assert by_day.trips("Wednesday", "interpeak", "AAA", "BBB") == 0.0
    assert by_period.trips("Wednesday", "interpeak", "AAA", "BBB") == pytest.approx(a_to_c)
    assert by_period.trips("Wednesday", "interpeak", "BBB", "CCC") == pytest.approx(a_to_c)
    # In the morning the direct train serves it, so nothing is routed
    # then.
    assert by_period.trips("Wednesday", "am_peak", "AAA", "BBB") == 0.0


def test_period_routing_prefers_the_via_with_the_shorter_wait():
    # Two vias, equal riding time; B's second leg runs four times as
    # often as C's.
    legs = pd.DataFrame(
        {
            "service": [0, 1, 2, 3, 4, 5, 6],
            "o": ["AAA", "AAA", "BBB", "BBB", "BBB", "BBB", "CCC"],
            "d": ["BBB", "CCC", "DDD", "DDD", "DDD", "DDD", "DDD"],
            "period": ["interpeak"] * 7,
            "count_period": ["interpeak"] * 7,
            "in_vehicle_minutes": [10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            "count_dep": [600.0, 600.0, 620.0, 680.0, 740.0, 800.0, 620.0],
            "count_arr": np.nan,
            "atoc_code": "XX",
            "signalling_id": [f"2D0{k}" for k in range(7)],
            "svc_origin_crs": "AAA",
            "svc_dest_crs": "DDD",
        }
    )
    routes = na.PeriodChangeRoutes(legs, legs["o"].to_numpy(), legs["d"].to_numpy(), 20.0)
    routes.route(np.array(["AAA"]), np.array(["DDD"]))
    _, via, _ = routes.routes["interpeak"]
    assert routes.stations[via[0]] == "BBB"
    demand = routes.leg_demand("interpeak", np.array([3.0]))
    assert demand.sum() == pytest.approx(6.0)


def test_a_change_far_slower_than_the_straight_line_is_dropped(matrices):
    # An hour in the train for A -> C.
    slow = pd.DataFrame(
        _train("S1", "2B01", [("AAA", None, "0800"), ("BBB", "0830", None)])
        + _train("S2", "2B02", [("BBB", None, "0840"), ("CCC", "0910", None)])
    )
    close = pd.DataFrame(
        {"lat": [51.0] * 3, "lon": [0.0, 0.001, 0.002]}, index=["AAA", "BBB", "CCC"]
    )
    far = close.assign(lon=[0.0, 0.5, 1.0])  # about 70 km end to end
    for routing in ("day", "period"):
        trips = [
            _transfers(slow, matrices.directory, where, routing).trips(
                "Wednesday", "am_peak", "AAA", "BBB"
            )
            for where in (close, far, None)
        ]
        # Held to the limit when close; within it when far; no limit
        # without coordinates.
        assert trips[0] == 0.0
        assert trips[1] > 0
        assert trips[2] > 0


def test_second_codes_are_rewritten_and_can_be_turned_off():
    frame = pd.DataFrame({"crs": ["RDZ", "RDG", "PDX", "WAT"], "other": ["RDZ"] * 4})
    recoded = recode_columns(frame, resolve_recodes(None), ["crs"])
    assert list(recoded["crs"]) == ["RDG", "RDG", "PAD", "WAT"]
    assert list(recoded["other"]) == ["RDZ"] * 4
    assert recode_columns(frame, resolve_recodes({}), ["crs"]) is frame
    assert resolve_recodes(None) == DEFAULT_STATION_RECODES


def test_trains_at_a_second_code_share_the_slot_with_the_main_code():
    calls = pd.DataFrame(
        _train("G1", "1G01", [("SLO", None, "0800"), ("PAD", "0820", None)])
        + _train("X1", "9X01", [("SLO", None, "0805"), ("PDX", "0830", None)])
        + _train("T1", "9T01", [("BDM", None, "0800"), ("SPL", "0840", None)])
        + _train("E1", "1E01", [("BDM", None, "0810"), ("STP", "0845", None)])
    )
    recoded = recode_columns(calls, resolve_recodes(None), ["crs"])
    slots = na.slot_trains(na.build_legs(na.day_calls(recoded, WEDNESDAY, "Wednesday"), CUTOFFS))
    assert slots[("SLO", "PAD", "am_peak")] == 2
    assert slots[("BDM", "STP", "am_peak")] == 2


def test_predict_demand_adds_routed_trips_before_the_split(change_calls, matrices, tmp_path):
    model = direction_split.DirectionSplitModel(
        coefficients={direction_split.POOLED: {"intercept": 0.4, "ln_footfall_ratio": 0.0}},
        n_cells=0,
        r_squared=0.0,
        source="test",
    )
    model.save(tmp_path / "split.json")
    config = ProposeConfig(
        output_path=tmp_path,
        services=[],
        timetable_calls_path=tmp_path / "unused.csv",
        bplan_output_path=tmp_path,
        period_matrices_dir=matrices.directory,
        demand_model="legacy",
        direction_split_model=tmp_path / "split.json",
    )
    candidate = CandidateService(
        name="U1",
        traction_code="150",
        days_of_week=["Wednesday"],
        stops=[
            ServiceStop(station="AAA", departure="08:00"),
            ServiceStop(station="BBB", arrival="08:10"),
        ],
    )
    with_transfers = predict_demand(
        config,
        candidate,
        change_calls,
        WEDNESDAY,
        transfers=_transfers(change_calls, matrices.directory),
    ).iloc[0]
    without = predict_demand(config, candidate, change_calls, WEDNESDAY).iloc[0]

    cell, routed = 4 * WEEKDAYS_PER_YEAR, 10 * WEEKDAYS_PER_YEAR
    # U1 runs A -> B and U3 B -> A in the slot, so the split fires; its
    # factor comes from the pair's own cells and scales the routed trips
    # as well.
    factor = model.split(cell, cell, "am_peak", None, None, 1, 1) / cell
    assert factor > 1.1
    assert with_transfers["existing_transfer_trips"] == pytest.approx(routed)
    assert with_transfers["existing_annual_trips"] == pytest.approx((cell + routed) * factor)
    assert without["existing_transfer_trips"] == 0.0
    assert without["existing_annual_trips"] == pytest.approx(cell * factor)


# --- Case 19: trains calling at a station twice, and the detour limit
# ----------------------------
def _loop_calls() -> pd.DataFrame:
    """A loop from LLL through BBB and back, and a train BBB -> LLL."""
    return pd.DataFrame(
        _train(
            "L1",
            "2L01",
            [
                ("LLL", None, "0800"),
                ("BBB", "0805", "0806"),
                ("FAR", "0830", "0831"),
                ("BBB", "0855", "0856"),
                ("LLL", "0901", None),
            ],
        )
        + _train("D1", "2D01", [("BBB", None, "0810"), ("LLL", "0815", None)])
    )


def _coordinates() -> pd.DataFrame:
    # LLL and BBB 3 km apart; FAR 20 km out.
    return pd.DataFrame(
        {"lat": [51.50, 51.527, 51.68], "lon": [0.0, 0.0, 0.0]}, index=["LLL", "BBB", "FAR"]
    )


def test_one_leg_per_pair_keeps_the_shortest_leg() -> None:
    day = na.day_calls(_loop_calls(), WEDNESDAY, "Wednesday")
    legs = na.build_legs(day, CUTOFFS)
    loop = legs[legs["train_uid"] == "L1"]
    assert not loop.duplicated(["service", "o", "d"]).any()
    # BBB -> LLL rides from the second BBB call, not round the loop from
    # the first.
    bl = loop[(loop["o"] == "BBB") & (loop["d"] == "LLL")]
    assert bl[["i", "j"]].to_numpy().tolist() == [[3, 4]]
    # LLL -> BBB is the first call along, not the one after the loop.
    lb = loop[(loop["o"] == "LLL") & (loop["d"] == "BBB")]
    assert lb[["i", "j"]].to_numpy().tolist() == [[0, 1]]
    # The slot count and the loads agree: the loop is one train for BBB
    # -> LLL.
    slots = na.slot_trains(legs)
    assert slots.loc[("BBB", "LLL", "am_peak")] == 2


def test_drop_detour_legs_removes_the_long_way_round_only_where_a_faster_leg_exists() -> None:
    day = na.day_calls(_loop_calls(), WEDNESDAY, "Wednesday")
    legs = na.build_legs(day, CUTOFFS, coordinates=_coordinates())
    loop = legs[legs["train_uid"] == "L1"]
    # BBB(first) -> ... is the loop's way to FAR: 24 min for ~20 km,
    # within the limit, kept.
    assert ((loop["o"] == "BBB") & (loop["d"] == "FAR")).any()
    # FAR -> LLL: 30 min for ~20 km is within 30 + 2 x 20; kept.
    assert ((loop["o"] == "FAR") & (loop["d"] == "LLL")).any()
    # LLL -> BBB exists only as the 5-minute leg; a pair whose only leg
    # is slow keeps it.
    kept = na.drop_detour_legs(
        legs.assign(
            in_vehicle_minutes=np.where(legs["o"] == "LLL", 500.0, legs["in_vehicle_minutes"])
        ),
        _coordinates(),
    )
    assert ((kept["o"] == "LLL") & (kept["d"] == "BBB")).any()


def test_drop_detour_legs_drops_a_slow_leg_beside_a_fast_one() -> None:
    legs = pd.DataFrame(
        {
            "o": ["BBB", "BBB"],
            "d": ["LLL", "LLL"],
            "period": ["am_peak", "am_peak"],
            "in_vehicle_minutes": [5.0, 55.0],
        }
    )
    kept = na.drop_detour_legs(legs, _coordinates())
    assert kept["in_vehicle_minutes"].tolist() == [5.0]


def test_per_train_path_serves_a_pair_once_and_obeys_the_detour_limit() -> None:
    from ortla import demand as sp

    stops = [
        ServiceStop(station="LLL", departure="08:00"),
        ServiceStop(station="BBB", arrival="08:05", departure="08:06"),
        ServiceStop(station="FAR", arrival="08:30", departure="08:31"),
        ServiceStop(station="BBB", arrival="08:55", departure="08:56"),
        ServiceStop(station="LLL", arrival="09:01"),
    ]
    shortest = sp.shortest_legs(stops)
    assert shortest[("BBB", "LLL")] == (3, 4)
    assert shortest[("LLL", "BBB")] == (0, 1)
    listings = pd.DataFrame(
        {
            "dep_from_origin": ["08:10", "08:06"],
            "arr_at_destination": ["08:15", "09:01"],
            "time_period": ["am_peak", "am_peak"],
        }
    )
    kept, fast = sp._drop_slow_listings(
        listings, limit=36.0, period="am_peak", candidate_fast=False
    )
    assert fast and kept["dep_from_origin"].tolist() == ["08:10"]
    # Every listing slow and the candidate slow too: all kept.
    kept, fast = sp._drop_slow_listings(
        listings, limit=1.0, period="am_peak", candidate_fast=False
    )
    assert not fast and len(kept) == 2


def test_a_loop_never_prices_a_station_to_itself() -> None:
    from ortla import demand as sp

    day = na.day_calls(_loop_calls(), WEDNESDAY, "Wednesday")
    legs = na.build_legs(day, CUTOFFS)
    assert not (legs["o"] == legs["d"]).any()
    stops = [
        ServiceStop(station="LLL", departure="08:00"),
        ServiceStop(station="BBB", arrival="08:05", departure="08:06"),
        ServiceStop(station="BBB", arrival="08:55", departure="08:56"),
        ServiceStop(station="LLL", arrival="09:01"),
    ]
    assert ("BBB", "BBB") not in sp.shortest_legs(stops)
    assert ("LLL", "LLL") not in sp.shortest_legs(stops)


# --- Case 23: slot shares by departure time and journey time
# ------------------------------------
def test_slot_shares_without_weights_are_the_equal_share(calls):
    legs = na.build_legs(na.day_calls(calls, WEDNESDAY, "Wednesday"), CUTOFFS)
    flat = na.SlotShareModel(timing={}, theta=0.0)
    assert na.slot_shares(legs, flat) == pytest.approx(na.slot_shares(legs, None))
    assert na.slot_shares(legs, None) == pytest.approx(
        1.0 / np.maximum(na.trains_per_leg(legs), 1.0)
    )


def test_a_faster_train_takes_more_of_its_slot_and_shares_still_sum_to_one(tmp_path):
    two = pd.DataFrame(
        _train(
            "S1", "2S01", [("AAA", None, "0800"), ("CCC", "0820", None)]
        )  # half-hour 16, 20 min
        + _train(
            "S2", "2S02", [("AAA", None, "0840"), ("CCC", "0850", None)]
        )  # half-hour 17, 10 min
    )
    legs = na.build_legs(na.day_calls(two, WEDNESDAY, "Wednesday"), CUTOFFS)
    na.SlotShareModel(timing={"am_peak|16|0": 0.2}, theta=0.05).save(tmp_path / "share.json")
    shares = na.slot_shares(legs, na.SlotShareModel.load(tmp_path / "share.json"))
    by_train = dict(zip(legs["train_uid"], shares, strict=True))
    assert by_train["S1"] + by_train["S2"] == pytest.approx(1.0)
    # log w: S1 -0.2 - 0.05 x 20, S2 -0.05 x 10.
    assert by_train["S2"] / by_train["S1"] == pytest.approx(np.exp(0.2 + 0.05 * 10))


def test_the_hourly_profile_moves_a_slot_towards_the_busier_hour(tmp_path):
    two = pd.DataFrame(
        _train("S1", "2S01", [("AAA", None, "0710"), ("CCC", "0730", None)])
        + _train("S2", "2S02", [("AAA", None, "0840"), ("CCC", "0900", None)])
    )
    legs = na.build_legs(na.day_calls(two, WEDNESDAY, "Wednesday"), CUTOFFS)
    pd.DataFrame(
        {  # AAA->CCC: 10 start at 7, 30 at 8; the 0-23 row has no profile and is dropped
            "OrigCRS": ["AAA", "AAA", "AAA"],
            "DestCRS": ["CCC", "CCC", "CCC"],
            "StartHour": [7, 8, 0],
            "EndHour": [7, 8, 23],
            "Demand": [10.0, 30.0, 500.0],
        }
    ).to_csv(tmp_path / "o03.csv", index=False)
    flat = na.SlotShareModel(timing={}, theta=0.0)
    full = flat.with_profile(tmp_path / "o03.csv", CUTOFFS, strength=1.0, smoothing=0.0)
    by_train = dict(zip(legs["train_uid"], na.slot_shares(legs, full), strict=True))
    assert by_train["S2"] / by_train["S1"] == pytest.approx(3.0)  # 30 / 10
    assert by_train["S1"] + by_train["S2"] == pytest.approx(1.0)
    # Smoothing pulls towards equal shares; strength 0 or an unprofiled
    # pair is the plain model.
    smooth = dict(
        zip(
            legs["train_uid"],
            na.slot_shares(
                legs, flat.with_profile(tmp_path / "o03.csv", CUTOFFS, strength=1.0, smoothing=2.0)
            ),
            strict=True,
        )
    )
    assert 1.0 < smooth["S2"] / smooth["S1"] < 3.0
    off = flat.with_profile(tmp_path / "o03.csv", CUTOFFS, strength=0.0)
    assert na.slot_shares(legs, off) == pytest.approx(na.slot_shares(legs, flat))
    assert (
        full.log_weight(
            np.array(["am_peak"]),
            np.array([490.0]),
            np.array([False]),
            np.array([0.0]),
            np.array(["XXX"]),
            np.array(["YYY"]),
        )[0]
        == 0.0
    )


def test_predict_demand_weights_the_candidate_against_the_trains_in_its_slot(
    calls, matrices, tmp_path
):
    base = dict(
        output_path=tmp_path,
        services=[],
        timetable_calls_path=tmp_path / "unused.csv",
        bplan_output_path=tmp_path,
        period_matrices_dir=matrices.directory,
        demand_model="legacy",
        frequency_elasticity=1.0,
    )
    # A fast candidate A -> C (10 min) beside T1 and T2 (20 min each) in
    # the morning peak.
    candidate = CandidateService(
        name="FAST",
        traction_code="150",
        days_of_week=["Wednesday"],
        stops=[
            ServiceStop(station="AAA", departure="08:10"),
            ServiceStop(station="CCC", arrival="08:20"),
        ],
    )
    equal = predict_demand(ProposeConfig(**base), candidate, calls, WEDNESDAY).iloc[0]
    na.SlotShareModel(timing={}, theta=0.0).save(tmp_path / "flat.json")
    flat = predict_demand(
        ProposeConfig(**base, slot_share_model=tmp_path / "flat.json"), candidate, calls, WEDNESDAY
    ).iloc[0]
    na.SlotShareModel(timing={}, theta=0.05).save(tmp_path / "fast.json")
    fast = predict_demand(
        ProposeConfig(**base, slot_share_model=tmp_path / "fast.json"), candidate, calls, WEDNESDAY
    ).iloc[0]
    assert flat["predicted_annual_trips_captured"] == pytest.approx(
        equal["predicted_annual_trips_captured"]
    )
    # Elasticity 1: the slot's 10 trips a day grow to 15 over three
    # trains.
    total = 15 * WEEKDAYS_PER_YEAR
    assert equal["predicted_annual_trips_captured"] == pytest.approx(total / 3)
    weight = np.exp(-0.05 * 20) / np.exp(-0.05 * 10)
    assert fast["predicted_annual_trips_captured"] == pytest.approx(total / (2 * weight + 1))


def test_a_strip_rule_naming_a_second_code_still_finds_its_trains():
    from ortla.config import StripRule
    from ortla.hindcast import recode_strip_rule, select_schedules

    calls = recode_columns(
        pd.DataFrame(
            _train("C1", "1C01", [("CBW", None, "0700"), ("SPX", "0800", None)], atoc="SE")
        ),
        resolve_recodes(None),
        ["crs"],
    )
    rule = StripRule(atoc_code="SE", calls_at=["CBW", "SPX"], origin="CBW", destination="SPX")
    assert select_schedules(calls, rule).empty
    found = select_schedules(calls, recode_strip_rule(rule, resolve_recodes(None)))
    assert list(found["train_uid"]) == ["C1"]


def test_choice_changes_move_a_served_pair_to_a_quicker_change(matrices):
    # A -> C has one slow direct train in the morning; A -> B and B -> C
    # are quick.
    calls = pd.DataFrame(
        _train("D1", "2E01", [("AAA", None, "0700"), ("CCC", "0840", None)])
        + _train("Q1", "2E02", [("AAA", None, "0700"), ("BBB", "0710", None)])
        + _train("Q2", "2E03", [("BBB", None, "0712"), ("CCC", "0722", None)])
    )
    a_to_c = 10 * WEEKDAYS_PER_YEAR
    plain = na.TransferDemand(
        calls,
        WEDNESDAY,
        CUTOFFS,
        {},
        matrices.directory,
        interchange_penalty=20.0,
        detour_base_minutes=30.0,
        detour_minutes_per_km=2.0,
        routing="period",
    )
    chosen = na.TransferDemand(
        calls,
        WEDNESDAY,
        CUTOFFS,
        {},
        matrices.directory,
        interchange_penalty=20.0,
        detour_base_minutes=30.0,
        detour_minutes_per_km=2.0,
        routing="period",
        choice_theta=0.2,
    )
    # Without a choice step a served pair is never routed.
    assert plain.trips("Wednesday", "am_peak", "AAA", "BBB") == 0.0
    assert plain.direct_share("Wednesday", "am_peak", "AAA", "CCC") == 0.0
    # Direct: 100 minutes + half the slot's headway; change: 10 + 10 +
    # half its headway + 20.
    share = chosen.direct_share("Wednesday", "am_peak", "AAA", "CCC")
    assert 0.5 < share < 1.0
    assert chosen.trips("Wednesday", "am_peak", "AAA", "BBB") == pytest.approx(share * a_to_c)
    assert chosen.trips("Wednesday", "am_peak", "BBB", "CCC") == pytest.approx(share * a_to_c)
    # A pair with no quicker change keeps its trips on the direct train.
    assert chosen.direct_share("Wednesday", "am_peak", "AAA", "BBB") == 0.0


def test_a_call_repeated_under_one_sequence_number_is_dropped_with_a_warning(calls, caplog):
    clean = na.day_calls(calls, WEDNESDAY, "Wednesday")
    t1 = calls[calls["train_uid"] == "T1"]
    doubled = pd.concat([calls, t1.iloc[[1]]], ignore_index=True)
    day = na.day_calls(doubled, WEDNESDAY, "Wednesday")
    assert len(day) == len(clean)
    assert "repeat another call" in caplog.text


def test_times_running_backwards_are_warned_but_midnight_is_not(caplog):
    back = pd.DataFrame(
        _train(
            "B1", "9Z01", [("AAA", None, "0900"), ("BBB", "0850", "0851"), ("CCC", "0840", None)]
        )
    )
    na.day_calls(back, WEDNESDAY, "Wednesday")
    assert "running backwards" in caplog.text and "B1" in caplog.text
    caplog.clear()
    night = pd.DataFrame(
        _train(
            "N1", "9Z02", [("AAA", None, "2350"), ("BBB", "0005", "0006"), ("CCC", "0020", None)]
        )
    )
    na.day_calls(night, WEDNESDAY, "Wednesday")
    assert "running backwards" not in caplog.text
