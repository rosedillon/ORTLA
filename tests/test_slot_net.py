"""Tests for the neural slot share (case 117's network)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from test_network_assignment import CUTOFFS, WEDNESDAY, _train

from orta import demand
from orta import network_assignment as na
from orta.config import ServiceStop
from orta.slot_net import FEATURES, SlotShareNet
from orta.stations import resolve_aliases


@pytest.fixture(scope="module")
def neural() -> na.SlotShareModel:
    return na.SlotShareModel.load("neural")


def _three_trains() -> pd.DataFrame:
    """AAA -> CCC, am peak: a stopper, a fast train, a late one."""
    return pd.DataFrame(
        _train(
            "S1", "2S01", [("AAA", None, "0700"), ("BBB", "0712", "0713"), ("CCC", "0730", None)]
        )
        + _train("F1", "1F01", [("AAA", None, "0715"), ("CCC", "0735", None)])
        + _train(
            "S2", "2S02", [("AAA", None, "0850"), ("BBB", "0902", "0903"), ("CCC", "0920", None)]
        )
    )


def test_the_packaged_models_load_by_name(neural):
    m3 = na.SlotShareModel.load("m3")
    assert m3.net is None and m3.theta == pytest.approx(0.06)
    assert neural.net is not None and neural.timing == m3.timing
    assert len(neural.net.fits) == 3
    assert neural.net.fits[0][0].shape == (16, len(FEATURES))
    # Stations it was trained on carry their own distance and footfall.
    assert neural.net.km_london["LBG"] < 1 < neural.net.km_london["MAN"]


def test_saving_and_loading_keeps_the_network(neural, tmp_path):
    neural.save(tmp_path / "copy.json")
    again = na.SlotShareModel.load(tmp_path / "copy.json")
    x = np.random.default_rng(0).normal(size=(50, len(FEATURES))).astype(np.float32)
    assert again.net(x) == pytest.approx(neural.net(x), abs=0)


def test_features_read_the_slot_from_the_timetable(neural):
    day = na.day_calls(_three_trains(), WEDNESDAY, "Wednesday")
    legs = na.build_legs(day, CUTOFFS)
    codes = resolve_aliases(None)
    cols = na._net_columns(legs, day, codes, na._filled_minutes(legs, "period"))
    first = legs.drop_duplicates(["service", "o", "d"])
    listed = cols.loc[first.index[first["count_period"] != "Unknown"]]
    x = pd.DataFrame(neural.net.features(cols, "period", listed, None), columns=FEATURES)
    x["uid"], x["o"], x["d"] = legs["train_uid"].to_numpy(), legs["o"], legs["d"]
    ac = x[(x["o"] == "AAA") & (x["d"] == "CCC")].set_index("uid")
    # S1 07:00, F1 07:15, S2 08:50 share the am peak AAA -> CCC slot.
    assert ac.loc["S1", "gap_prev"] == pytest.approx(np.log1p(120))
    assert ac.loc["S1", "gap_next"] == pytest.approx(np.log1p(15))
    assert ac.loc["F1", "gap_next"] == pytest.approx(np.log1p(95))
    assert ac.loc["S2", "gap_next"] == pytest.approx(np.log1p(120))
    assert ac["ln_slot_trains"].tolist() == pytest.approx([np.log(3)] * 3)
    assert ac.loc["S1", "ln_calls_od"] == pytest.approx(np.log(2))
    assert ac.loc["F1", "ln_calls_od"] == 0
    assert ac.loc["F1", "rel_ln_minutes"] < 0 < ac.loc["S1", "rel_ln_minutes"]
    assert ac["o_is_origin"].all() and ac["d_is_terminus"].all()
    assert (ac["p_am_peak"] == 1).all()
    # Unknown stations: neither towards nor away, the median footfall.
    assert ac["neither"].all()
    assert ac["ln_ee_o"].tolist() == pytest.approx([neural.net.ln_footfall.median()] * 3)
    bc = x[(x["o"] == "BBB") & (x["d"] == "CCC")].set_index("uid")
    assert not bc.loc["S1", "o_is_origin"]


def test_a_zero_network_is_the_base_model(neural):
    flat = dataclasses.replace(
        neural.net,
        fits=tuple((w, b, np.zeros_like(v), np.float32(0)) for w, b, v, _ in neural.net.fits),
    )
    day = na.day_calls(_three_trains(), WEDNESDAY, "Wednesday")
    legs = na.build_legs(day, CUTOFFS)
    base = dataclasses.replace(neural, net=None)
    assert na.slot_shares(legs, dataclasses.replace(neural, net=flat), calls=day) == pytest.approx(
        na.slot_shares(legs, base), rel=1e-12
    )


def test_neural_shares_sum_to_one_and_need_the_calls(neural):
    day = na.day_calls(_three_trains(), WEDNESDAY, "Wednesday")
    legs = na.build_legs(day, CUTOFFS)
    shares = na.slot_shares(legs, neural, calls=day)
    ac = (legs["o"] == "AAA") & (legs["d"] == "CCC") & (legs["period"] == "am_peak")
    assert shares[ac.to_numpy()].sum() == pytest.approx(1.0)
    assert not np.allclose(shares, na.slot_shares(legs, dataclasses.replace(neural, net=None)))
    with pytest.raises(ValueError, match="calls"):
        na.slot_shares(legs, neural)


def test_a_candidate_takes_the_share_the_whole_network_gives_it(neural):
    """The per-candidate path (propose, hindcast) matches `assign`."""
    calls = _three_trains()
    day = na.day_calls(calls, WEDNESDAY, "Wednesday")
    legs = na.build_legs(day, CUTOFFS)
    shares = dict(
        zip(
            legs["train_uid"] + legs["o"] + legs["d"],
            na.slot_shares(legs, neural, calls=day),
            strict=True,
        )
    )
    # F1 re-added to the timetable without it, as a hindcast does.
    without = demand.prepare_calls(calls[calls["train_uid"] != "F1"])
    filtered = demand.day_services(without, WEDNESDAY, "Wednesday")
    listings = demand.od_services(filtered, "AAA", "CCC", CUTOFFS)
    stops = [
        ServiceStop(station="AAA", departure="07:15"),
        ServiceStop(station="CCC", arrival="07:35"),
    ]
    weighted = demand._share_trains(
        neural, listings, "am_peak", stops, 0, 1, False, 1, None, filtered, None
    )
    assert 1.0 / weighted == pytest.approx(shares["F1AAACCC"], rel=1e-5)


def test_a_network_with_other_features_is_refused(neural):
    data = neural.net.to_dict()
    data["features"] = data["features"][::-1]
    with pytest.raises(ValueError, match="features"):
        SlotShareNet.from_dict(data)
