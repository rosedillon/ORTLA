"""Checks on the station ensemble's maths that hold on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ortla.station_ensemble import ensemble, frame, gravity, members, terms


def _pairs(n: int, rng: np.random.Generator) -> pd.DataFrame:
    i, j = np.triu_indices(n, 1)
    lnkm = rng.uniform(0, 4, len(i))
    p = pd.DataFrame({"ii": i, "jj": j, "lnkm": lnkm, "lnkm2": lnkm**2})
    p["lntrains"] = rng.uniform(0, 3, len(i)) * (lnkm < 2)
    p["minutes"] = np.exp(lnkm) / 60 + rng.uniform(0, 0.3, len(i))
    p["change1"] = ((lnkm >= 2) & (rng.uniform(size=len(i)) < 0.5)).astype(float)
    p["change2"] = ((lnkm >= 2) & (p["change1"] == 0)).astype(float)
    p["london"] = (rng.uniform(size=len(i)) < 0.1).astype(float)
    return p


def test_stage1_recovers_masses_and_impedance() -> None:
    rng = np.random.default_rng(1)
    n = 60
    p = _pairs(n, rng)
    a = rng.normal(5, 1, n)
    b = np.array([-1.0, 0.1, 0.4, -0.8, -0.3, -0.9, 0.5])
    p["J"] = np.exp(a[p["ii"]] + a[p["jj"]] + p[gravity.IMPEDANCE].to_numpy() @ b)
    a_hat, b_hat = gravity.stage1(p, np.ones(len(p), bool), n)
    assert b_hat == pytest.approx(b, abs=1e-5)
    assert a_hat - a_hat.mean() == pytest.approx(a - a.mean(), abs=1e-5)


def test_stage1_leaves_stations_without_flows_unfitted() -> None:
    rng = np.random.default_rng(2)
    n = 30
    p = _pairs(n, rng)
    p["J"] = rng.uniform(1, 100, len(p))
    rows = ~((p["ii"] == 0) | (p["jj"] == 0)).to_numpy()
    a, _ = gravity.stage1(p, rows, n)
    assert np.isnan(a[0])
    assert np.isfinite(a[1:]).all()


def test_access_without_rivals_is_the_sum_of_reach() -> None:
    rng = np.random.default_rng(3)
    n = 8
    p = _pairs(n, rng)
    xy = np.column_stack([np.arange(n) * 100_000.0, np.zeros(n)])  # 100 km apart: no rivals
    net = gravity.Network(p, xy)
    a, b = rng.normal(size=n), rng.normal(scale=0.1, size=len(gravity.IMPEDANCE))
    u = np.zeros((n, n))
    imp = np.exp(p[gravity.IMPEDANCE].to_numpy() @ b)
    u[p["ii"], p["jj"]] = imp * np.exp(a[p["jj"]])
    u[p["jj"], p["ii"]] = imp * np.exp(a[p["ii"]])
    assert net.log_access(a, b) == pytest.approx(np.log(u.sum(1)))


def test_borrowing_leaves_a_station_out_of_its_own_adjustment() -> None:
    xy = np.array([[0.0, 0.0], [1000.0, 0.0], [50_000.0, 0.0]])
    r = np.array([1.0, -1.0, 0.0])
    own = ensemble.borrow(r, xy, xy, same=True)
    w = np.exp(-1.0 / ensemble.BORROW_KM)
    assert own[0] == pytest.approx(
        -w / (ensemble.BORROW_KAPPA + w + np.exp(-50 / ensemble.BORROW_KM))
    )
    far = ensemble.borrow(r, xy, np.array([[500_000.0, 0.0]]))
    assert far == pytest.approx([0.0], abs=1e-12)


def test_car_catchment_shares_every_car_once() -> None:
    xy = np.array([[0.0, 0.0], [3000.0, 0.0], [6000.0, 0.0]])
    cars = pd.DataFrame({"e": [1000.0, 5000.0], "n": [0.0, 0.0], "cars": [100.0, 40.0]})
    shares = terms.car_catchment(xy, cars, np.array([1.0, 2.0, 0.0]), 5.0)
    assert shares.sum() == pytest.approx(140.0)
    assert shares[2] == 0.0  # no car park, no share


def test_service_catchment_shares_every_resident_once() -> None:
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 20_000, (12, 2))
    people = rng.uniform(0, 20_000, (300, 2))
    pop = rng.uniform(1, 50, 300)
    out = members.service_catchment(xy, people, pop, rng.uniform(5, 100, 12))
    assert out.sum() == pytest.approx(pop.sum())


def test_folds_are_fixed_by_position() -> None:
    xy = np.array([[10.0, 10.0], [24_999.0, 24_999.0], [25_000.0, 0.0]])
    fold = frame.cells(xy)
    assert fold[0] == fold[1]  # same 25 km cell
    assert set(fold) <= set(range(frame.FOLDS))
    assert (frame.cells(xy) == fold).all()
