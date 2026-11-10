"""Checks on the new-station balance, outturn screen and day index."""

from __future__ import annotations

import itertools
import types

import pandas as pd
import pytest

from ortla import (
    demand,
    hindcast,
    period_matrices,
    track_distance,
)
from ortla.config import ServiceStop
from ortla.service_lookup import _DayIndex, extract_od_services

CUTOFFS = {"am_peak": "7:00", "interpeak": "9:30", "pm_peak": "16:00", "evening": "19:00"}


def _day_calls() -> pd.DataFrame:
    """A small day: through trains, a loop, a reversal, variants."""
    rows = []

    def train(uid: str, start: str, stops: list[tuple[str, str]], fraction: float = 1.0) -> None:
        for seq, (crs, time) in enumerate(stops):
            rows.append(
                {
                    "train_uid": uid,
                    "start_date": start,
                    "end_date": "2026-12-01",
                    "svc_origin_dep": stops[0][1],
                    "signalling_id": f"1{uid[-2:]}",
                    "atoc_code": "XX",
                    "seq": float(seq),
                    "crs": crs,
                    "public_departure": time,
                    "public_arrival": time,
                    "weekday_fraction": fraction,
                    "svc_origin_crs": stops[0][0],
                    "svc_dest_crs": stops[-1][0],
                }
            )

    train("A01", "2026-01-01", [("AAA", "0700"), ("BBB", "0710"), ("CCC", "0720")])
    train("A02", "2026-01-01", [("CCC", "0800"), ("BBB", "0810"), ("AAA", "0820")])
    # Calls at BBB twice: the train serves BBB -> CCC once, by the
    # shorter leg from its second BBB call (case 19, L1).
    train(
        "A03",
        "2026-01-01",
        [("BBB", "0900"), ("DDD", "0905"), ("BBB", "0910"), ("CCC", "0915")],
    )
    # Two schedule records of one train that together make up the week.
    train("A04", "2026-01-01", [("AAA", "1700"), ("CCC", "1730")], fraction=0.4)
    train("A04", "2026-02-01", [("AAA", "1700"), ("CCC", "1730")], fraction=0.6)
    train("A05", "2026-01-01", [("DDD", "1830"), ("AAA", "1845")])
    return pd.DataFrame(rows)


def test_day_index_matches_extract_od_services() -> None:
    calls = _day_calls()
    index = _DayIndex(calls)
    codes = ["AAA", "BBB", "CCC", "DDD", "ZZZ"]
    for origin, destination in itertools.product(codes, codes):
        expected = extract_od_services(calls, origin, destination, CUTOFFS, n_weekdays=1)
        got = index.od_services(origin, destination, CUTOFFS)
        assert len(got) == len(expected), (origin, destination)
        if len(expected):
            pd.testing.assert_frame_equal(
                got.reset_index(drop=True).astype(str),
                expected.reset_index(drop=True).astype(str),
            )


def test_fit_station_factors_hits_targets_jointly() -> None:
    # Two new stations N1, N2 and a hub H; N1-N2 is shared.
    pairs = {("N1", "H"): 100.0, ("H", "N1"): 100.0, ("N2", "H"): 50.0, ("N1", "N2"): 20.0}
    targets = {"N1": 440.0, "N2": 30.0}
    factors = hindcast.fit_station_factors(pairs, targets)
    totals = hindcast._station_totals(factors, pairs, set(targets))
    assert totals["N1"] == pytest.approx(440.0, rel=1e-6)
    assert totals["N2"] == pytest.approx(30.0, rel=1e-6)


def test_apply_station_balance_scales_only_synthetic_legs() -> None:
    demand = pd.DataFrame(
        {
            "from_station": ["N1", "N1", "H", "N1", "H"],
            "to_station": ["H", "N2", "N1", "H", "X"],
            "demand_model_used": [
                "gravity_synthetic",
                "gravity_synthetic",
                "gravity_synthetic",
                "gravity_pivot",
                "gravity_synthetic",
            ],
            "predicted_annual_trips_captured": [10.0, 10.0, 10.0, 10.0, 10.0],
            "predicted_annual_trips_generated": [5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    out = hindcast.apply_station_balance(demand, {"N1": 2.0, "N2": 3.0}, {})
    assert out["predicted_annual_trips_captured"].tolist() == [20.0, 60.0, 20.0, 10.0, 10.0]
    assert out["predicted_annual_trips_generated"].tolist() == [10.0, 30.0, 10.0, 5.0, 5.0]


def test_screen_count_resets_drops_the_whole_train_day() -> None:
    segments = pd.DataFrame(
        {
            "train_uid": ["T1"] * 4 + ["T2"] * 4,
            "day": "Tuesday",
            "from_station": ["CBG", "CMS", "FXN", "RYS"] * 2,
            # T1: 25 on at CBG, all "alight" at CMS with nobody
            # boarding, then the counter restarts. T2: a real station
            # where people get on.
            "actual_load": [25.0, 0.0, 17.0, 26.0, 25.0, 3.0, 20.0, 26.0],
            "actual_on": [25.0, 0.0, 17.0, 9.0, 25.0, 1.0, 17.0, 6.0],
            "actual_off": [0.0, 25.0, 0.0, 0.0, 0.0, 23.0, 0.0, 0.0],
        }
    )
    screened, flagged = hindcast.screen_count_resets(segments, min_load=10.0)
    assert flagged[["train_uid", "reset_at"]].to_numpy().tolist() == [["T1", "CMS"]]
    assert screened.loc[screened.train_uid == "T1", "actual_load"].isna().all()
    assert screened.loc[screened.train_uid == "T2", "actual_load"].notna().all()


def test_station_beside_a_fragment_snaps_to_main_network_within_a_kilometre(tmp_path) -> None:
    import networkx as nx
    import numpy as np
    from pyproj import Transformer
    from scipy.spatial import KDTree

    # British National Grid metres. A 20-node fragment hugs both
    # stations, so all 12 nearest candidates are on it; the main network
    # is 700 m from S (Spondon) and 5 km from W (an Isle of Wight
    # station, a network of its own).
    graph = nx.Graph()
    for x0 in (450_000.0, 470_000.0):
        frag = [(x0 + 5.0 * k, 335_000.0) for k in range(20)]
        nx.add_path(graph, frag, weight=5.0)
    main = [(450_000.0 + 50.0 * k, 335_700.0) for k in range(60)]
    main += [(470_000.0 + 50.0 * k, 340_000.0) for k in range(60)]
    nx.add_path(graph, main, weight=50.0)
    to_wgs = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    rows = []
    for code, (x, y) in {"S": (450_040.0, 335_010.0), "W": (470_040.0, 335_010.0)}.items():
        lon, lat = to_wgs.transform(x, y)
        rows.append({"crs": code, "lat": lat, "lon": lon})
    path = tmp_path / "stations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    calc = object.__new__(track_distance.TrackDistanceCalculator)
    calc.config = types.SimpleNamespace(
        stations=types.SimpleNamespace(path=path, crs="crs", lat="lat", lon="lon"),
        coordinate_systems=types.SimpleNamespace(input_crs="EPSG:4326", working_crs="EPSG:27700"),
    )
    calc.graph = graph
    calc._nodes = list(graph.nodes())
    calc._kdtree = KDTree(np.array(calc._nodes))
    calc._load_stations()
    main_component = max(nx.connected_components(graph), key=len)
    assert calc._station_nodes["S"] in main_component
    assert calc._station_nodes["W"] not in main_component


def test_plausible_distance_replaces_impossible_routes() -> None:
    calc = types.SimpleNamespace(
        station_locations={"A": (0.0, 0.0), "B": (0.0, 10_000.0), "C": (0.0, 100.0)},
        _reported_implausible=set(),
    )
    plausible = track_distance.TrackDistanceCalculator._plausible_metres
    # A believable route is left alone.
    assert plausible(calc, "A", "B", 12_500.0) == 12_500.0
    # Shorter than the straight line, or over three times it: straight x
    # circuity.
    assert plausible(calc, "A", "B", 5_000.0) == pytest.approx(12_000.0)
    assert plausible(calc, "A", "B", 63_000.0) == pytest.approx(12_000.0)
    # Stations too close together for the ratio to mean anything.
    assert plausible(calc, "A", "C", 5_000.0) == 5_000.0


def test_date_filter_refuses_an_empty_window() -> None:
    frame = pd.DataFrame({"Period Start Date": ["2026-05-31", "2026-06-28"]})
    with pytest.raises(ValueError, match="none of its 2 rows"):
        period_matrices._filter_dates(frame, "gwr", None, "2026-05-30")
    kept = period_matrices._filter_dates(frame, "gwr", None, "2026-06-27")
    assert len(kept) == 1


def test_synthetic_split_conserves_the_pair_and_points_the_peak() -> None:
    model = demand.direction_split.DirectionSplitModel(
        coefficients={
            "_pooled": {"intercept": 0.0, "ln_footfall_ratio": 0.0},
            "am_peak": {"intercept": 0.0, "ln_footfall_ratio": -0.46},
        },
        n_cells=1,
        r_squared=0.0,
        source="test",
    )
    calls = _day_calls()
    gravity = types.SimpleNamespace(synthetic_slot_trips=lambda *_: (100.0, "commute"))
    small, big = ServiceStop(station="AAA"), ServiceStop(station="CCC")
    forward = demand._synthetic_split_factor(
        model,
        gravity,
        calls,
        small,
        big,
        "Monday",
        "am_peak",
        CUTOFFS,
        trains_forward=1,
        trains_per_day=2,
        added_trains=None,
        footfall_origin=1e5,
        footfall_destination=1e7,
    )
    backward = demand._synthetic_split_factor(
        model,
        gravity,
        calls,
        big,
        small,
        "Monday",
        "am_peak",
        CUTOFFS,
        trains_forward=1,
        trains_per_day=2,
        added_trains=None,
        footfall_origin=1e7,
        footfall_destination=1e5,
    )
    # Small station towards the big one in the morning peak carries
    # more...
    assert forward > 1.0 > backward
    # ... and the pair's combined demand is unchanged (equal trains
    # each way).
    assert forward + backward == pytest.approx(2.0)


def test_gtr_count_cache_ignores_the_tiploc_map(tmp_path, monkeypatch) -> None:
    """GTR's cache must serve both the loading and hindcast maps."""
    from ortla import loading_data

    calls = []

    def reader(path, **_):
        calls.append(path)
        return pd.DataFrame(
            {
                "Departure from CRS": ["AAA"],
                "Service Week Day": ["Monday"],
                "Mean Load on Departure": [10.0],
                "load_units": ["passengers"],
            }
        )

    monkeypatch.setitem(loading_data.READERS, "gtr", reader)
    monkeypatch.setattr(loading_data, "_dedupe", lambda df: df)
    monkeypatch.setattr(loading_data, "derive_on_off", lambda df: df)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "p01.xlsx").write_text("x")
    for mapping in ({"A": "AAA"}, {"B": "BBB"}):
        loading_data.load_operator("gtr", tmp_path / "data", mapping, cache_dir=tmp_path / "cache")
    assert len(calls) == 1
