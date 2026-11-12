"""How much of the network a station reaches: a structural gravity model on the ODM.

Stage 1 fits journeys between every pair of stations by Poisson pseudo-maximum likelihood
with a mass for each station and an impedance from the timetable::

    J_ij = exp(a_i + a_j + x_ij b)

Openings' pairs are left out, so their masses are not fitted. Accessibility is then the
reach of the other stations' masses, each destination counted in proportion to how much
better this station serves it than its neighbours within about 5 km::

    U_ij = exp(x_ij b + a_j),    A_i = sum_j U_ij^2 / (U_ij + sum_k w_ik U_kj)

with ``w_ik = exp(-km / 5)`` over stations within 30 km. A station without a fitted mass
gets one from the trip end, iterated three times (:func:`access_stage`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ortla.station_ensemble import sources

if TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)

IMPEDANCE = ["lnkm", "lnkm2", "lntrains", "minutes", "change1", "change2", "london"]
CHANGE_MINUTES = 10.0
WALK_KM = 1.0  # stations this close are linked on foot at 4.8 km/h
CROSS_LONDON_MINUTES = 30.0
# fmt: off
LONDON_TERMINALS = frozenset({
    "BFR", "CHX", "CST", "CTK", "EPH", "EUS", "FST", "KGX", "LBG", "LST", "MYB", "MOG", "OLD",
    "PAD", "STP", "VIC", "VXH", "WAT", "WAE", "ZFD",
})
# fmt: on
COMPETITION_KM = 5.0
COMPETITION_RADIUS_M = 30_000
STAGE1_ITERATIONS = 2000
TRIP_END_ITERATIONS = 3
# fmt: off
_SKIM_COLUMNS = [
    "train_uid", "stp_indicator", "days_runs", "start_date", "end_date", "train_category",
    "svc_origin_dep", "signalling_id", "seq", "crs", "is_public_call", "public_arrival",
    "public_departure",
]
# fmt: on


# ---------------------------------------------------------------- the timetable skim
def _clock_minutes(times: pd.Series) -> np.ndarray:
    s = times.fillna("").astype(str)
    hh, mm = pd.to_numeric(s.str[:2], errors="coerce"), pd.to_numeric(s.str[2:4], errors="coerce")
    return (hh * 60 + mm + np.where(s.str[4:5] == "H", 0.5, 0.0)).to_numpy(float)


def _day_calls(calls_path: pathlib.Path, date: str) -> pd.DataFrame:
    """Read the public calls of passenger trains on ``date``, in order along each train.

    Times past midnight run on past 1440, so they rise along every train.
    """
    ref = pd.Timestamp(date)
    keep = []
    for ch in pd.read_csv(calls_path, dtype=str, usecols=_SKIM_COLUMNS, chunksize=1_000_000):
        ok = (
            ch["stp_indicator"].eq("P")
            & ch["is_public_call"].eq("True")
            & ch["crs"].notna()
            & ch["train_category"].fillna("").str[:1].isin(["O", "X"])
            & (pd.to_datetime(ch["start_date"]) <= ref)
            & (pd.to_datetime(ch["end_date"]) >= ref)
            & (ch["days_runs"].fillna("").str[ref.weekday()] == "1")
        )
        keep.append(ch[ok])
    c = pd.concat(keep, ignore_index=True)
    c["crs"] = c["crs"].str.strip().str.upper().replace(sources.station_aliases())
    train = (
        c["train_uid"]
        + c["start_date"]
        + c["svc_origin_dep"].fillna("")
        + c["signalling_id"].fillna("")
    )
    c["svc"] = pd.factorize(train)[0]
    c["seq"] = pd.to_numeric(c["seq"])
    c = c.sort_values(["svc", "seq"]).reset_index(drop=True)
    dep, arr = _clock_minutes(c["public_departure"]), _clock_minutes(c["public_arrival"])
    c["dep"] = np.where(np.isnan(dep), arr, dep)
    c["arr"] = np.where(np.isnan(arr), dep, arr)
    start = c.groupby("svc")["dep"].transform("first").to_numpy()
    for col in ("dep", "arr"):  # past midnight
        v = c[col].to_numpy()
        c[col] = np.where(v < start - 60, v + 1440, v)
    return c


def skim(calls_path: pathlib.Path, date: str, stations_csv: pathlib.Path) -> pd.DataFrame:
    """Find the direct trains and fastest journeys between every pair of stations.

    The fastest journey is found direct, with one change and with two.

    Changes may be made on foot between stations within a kilometre, and across London between
    terminals (30 minutes). Each change costs 10 minutes. Returns one row per unordered pair
    (``a`` < ``b``) that is reachable: ``trains`` (direct trains a day, mean of the two
    directions), ``t1``, ``t2``, ``t3`` (minutes).
    """
    c = _day_calls(calls_path, date)
    st = pd.Index(sorted(c["crs"].unique()))
    n = len(st)
    code = st.get_indexer(c["crs"])
    t1 = np.full((n, n), np.inf, dtype=np.float32)
    count = np.zeros((n, n), dtype=np.float32)
    svc = c["svc"].to_numpy()
    starts = np.flatnonzero(np.r_[True, svc[1:] != svc[:-1]])
    ends = np.r_[starts[1:], len(c)]
    dep, arr = c["dep"].to_numpy(), c["arr"].to_numpy()
    for s, e in zip(starts, ends, strict=True):
        if e - s < 2:
            continue
        i, j = np.triu_indices(e - s, 1)
        o, d = code[s + i], code[s + j]
        m = (arr[s + j] - dep[s + i]).astype(np.float32)
        ok = (o != d) & np.isfinite(m) & (m > 0)
        pair = o[ok].astype(np.int64) * n + d[ok]
        count.flat[np.unique(pair)] += 1  # a train counts once per pair, even on a loop
        np.minimum.at(t1.reshape(-1), pair, m[ok])
    logger.info(
        "Skim: %s trains, %s stations, %s direct pairs",
        f"{svc.max() + 1:,}",
        f"{n:,}",
        f"{(count > 0).sum():,}",
    )

    xy = pd.read_csv(stations_csv).drop_duplicates("crsCode").set_index("crsCode")[["lat", "long"]]
    lat, lon = xy["lat"].reindex(st).to_numpy(float), xy["long"].reindex(st).to_numpy(float)
    ex, ny = np.nan_to_num(lon * 111.2 * np.cos(np.radians(54))), np.nan_to_num(lat * 111.2)
    links: dict[int, list[tuple[int, float]]] = {k: [] for k in range(n)}
    for k in range(n):
        km = np.hypot(ex - ex[k], ny - ny[k])
        for other in np.flatnonzero((km <= WALK_KM) & (np.arange(n) != k) & np.isfinite(lat)):
            links[k].append((int(other), float(km[other] / 4.8 * 60)))
    terminals = [st.get_loc(t) for t in LONDON_TERMINALS if t in st]
    for k in terminals:
        linked = {other for other, _ in links[k]}
        links[k] += [(t, CROSS_LONDON_MINUTES) for t in terminals if t != k and t not in linked]
    # ride to k, then walk or cross London to l
    t1_link = t1.copy()
    for k in range(n):
        for other, minutes in links[k]:
            np.minimum(t1_link[:, other], t1[:, k] + minutes, out=t1_link[:, other])
    # one change: ride, (link), change, ride; or a ride then a link alone
    t2 = np.minimum(
        _one_more_change(t1_link, t1), np.where(t1_link < t1, t1_link + CHANGE_MINUTES, np.inf)
    )
    np.fill_diagonal(t2, np.inf)
    t3 = _one_more_change(t1_link, t2)
    np.fill_diagonal(t3, np.inf)
    a, b = np.triu_indices(n, 1)
    out = pd.DataFrame(
        {
            "a": st[a],
            "b": st[b],
            "trains": (count[a, b] + count[b, a]) / 2,
            "t1": np.minimum(t1[a, b], t1[b, a]),
            "t2": np.minimum(t2[a, b], t2[b, a]),
            "t3": np.minimum(t3[a, b], t3[b, a]),
        }
    )
    return out[np.isfinite(out["t3"]) | np.isfinite(out["t1"])].reset_index(drop=True)


def _one_more_change(first_leg: np.ndarray, rest: np.ndarray) -> np.ndarray:
    out = np.full(first_leg.shape, np.inf, dtype=np.float32)
    for i in range(len(first_leg)):
        k = np.flatnonzero(np.isfinite(first_leg[i]))
        if len(k):
            out[i] = (first_leg[i, k][:, None] + rest[k]).min(axis=0) + CHANGE_MINUTES
    return out


# ---------------------------------------------------------------- pairs
def pairs(
    sk: pd.DataFrame,
    journeys: pd.Series,
    footfall: pd.DataFrame,
    stations_csv: pathlib.Path,
    index: pd.Index,
) -> pd.DataFrame:
    """Build every pair of used stations in ``index`` with its ODM journeys and impedance.

    ``journeys`` is indexed by (a, b) with a < b. Pairs out of reach within two changes take
    their minutes from straight-line km, fitted on two-change pairs, plus 30.
    """
    xy = pd.read_csv(stations_csv).drop_duplicates("crsCode").set_index("crsCode")[["lat", "long"]]
    codes = set(sk["a"]) | set(sk["b"])
    st = pd.Index(
        sorted(
            c
            for c in codes
            if c in footfall.index
            and footfall.loc[c, "footfall"] > 0
            and c in xy.index
            and np.isfinite(xy.loc[c, "lat"])
        )
    )
    a, b = np.triu_indices(len(st), 1)
    p = pd.DataFrame({"i": st[a], "j": st[b]}).merge(
        sk.rename(columns={"a": "i", "b": "j"}), on=["i", "j"], how="left"
    )
    lat, lon = xy["lat"].reindex(st).to_numpy(float), xy["long"].reindex(st).to_numpy(float)
    p["km"] = np.hypot(
        (lat[a] - lat[b]) * 111.2,
        (lon[a] - lon[b]) * 111.2 * np.cos(np.radians((lat[a] + lat[b]) / 2)),
    )
    p["J"] = journeys.reindex(pd.MultiIndex.from_arrays([p["i"], p["j"]])).fillna(0.0).to_numpy()
    p["trains"] = p["trains"].fillna(0.0)
    served = p["trains"] >= 1
    best = np.where(served, p["t1"], np.where(np.isfinite(p["t2"]), p["t2"], p["t3"]))
    two = np.isfinite(p["t3"]) & ~np.isfinite(p["t2"])
    slope = np.polyfit(p.loc[two, "km"], p.loc[two, "t3"], 1)
    p["minutes"] = np.where(np.isfinite(best), best, np.polyval(slope, p["km"]) + 30) / 60.0
    p["served"] = served.astype(float)
    p["change1"] = (~served & np.isfinite(p["t2"])).astype(float)
    p["change2"] = (~served & ~np.isfinite(p["t2"])).astype(float)
    p["lntrains"] = np.log(np.maximum(p["trains"], 1.0)) * p["served"]

    p = p[p["i"].isin(index) & p["j"].isin(index)].reset_index(drop=True)
    london = footfall["region"] == "London"
    either_end = london.reindex(p["i"]).to_numpy() | london.reindex(p["j"]).to_numpy()
    p["london"] = either_end.astype(float)
    p["lnkm"] = np.log(np.maximum(p["km"], 1.0))
    p["lnkm2"] = p["lnkm"] ** 2
    p["ii"], p["jj"] = index.get_indexer(p["i"]), index.get_indexer(p["j"])
    return p[["i", "j", "ii", "jj", "J", "km", "served", *IMPEDANCE]]


def odm_journeys(odm_csv: pathlib.Path) -> pd.Series:
    """Sum journeys between each pair of stations, both directions, indexed (a, b) with a < b."""
    o = pd.read_csv(odm_csv, usecols=["origin_tlc", "destination_tlc", "journeys"])
    o = o[o["origin_tlc"] != o["destination_tlc"]].dropna()
    a = np.minimum(o["origin_tlc"], o["destination_tlc"])
    b = np.maximum(o["origin_tlc"], o["destination_tlc"])
    return o.groupby([a, b])["journeys"].sum()


# ---------------------------------------------------------------- stage 1
def stage1(pairs_: pd.DataFrame, rows: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit station masses ``a`` and impedance ``b`` on the pairs in ``rows``.

    A station with no flows in ``rows`` gets a nan mass. Each iteration takes the masses'
    fixed point, then a Newton step on ``b``.
    """
    x = pairs_[IMPEDANCE].to_numpy(float)[rows]
    ii, jj = pairs_["ii"].to_numpy()[rows], pairs_["jj"].to_numpy()[rows]
    y = pairs_["J"].to_numpy(float)[rows]
    total = np.bincount(ii, y, n) + np.bincount(jj, y, n)
    has = total > 0
    a = np.where(has, np.log(np.where(has, total, 1.0)) - 0.5 * np.log(total.sum()), -np.inf)
    keep = has[ii] & has[jj]
    ii, jj, y, x = ii[keep], jj[keep], y[keep], x[keep]
    b = np.zeros(x.shape[1])
    for _ in range(STAGE1_ITERATIONS):
        mu = np.exp(a[ii] + a[jj] + x @ b)
        fitted = np.bincount(ii, mu, n) + np.bincount(jj, mu, n)
        step_a = np.zeros(n)
        step_a[has] = np.log(total[has] / fitted[has])
        a = a + step_a
        mu *= np.exp(step_a[ii] + step_a[jj])
        step_b = np.linalg.solve((x * mu[:, None]).T @ x, x.T @ (y - mu))
        step_b /= max(1.0, np.abs(step_b).max())
        b = b + step_b
        if np.abs(step_a).max() < 1e-6 and np.abs(step_b).max() < 1e-7:  # noqa: PLR2004
            break
    else:
        logger.warning("Stage 1 did not converge in %d iterations", STAGE1_ITERATIONS)
    a[~has] = np.nan
    return a, b


# ---------------------------------------------------------------- accessibility
class Network:
    """Pairs, impedance and the competition kernel over a fixed set of stations."""

    def __init__(self, pairs_: pd.DataFrame, xy: np.ndarray) -> None:
        self.n = len(xy)
        self.ii, self.jj = pairs_["ii"].to_numpy(), pairs_["jj"].to_numpy()
        self.x = pairs_[IMPEDANCE].to_numpy(float)
        tree = cKDTree(xy)
        w = tree.sparse_distance_matrix(
            tree, COMPETITION_RADIUS_M, output_type="coo_matrix"
        ).tocsr()
        w.setdiag(0.0)
        w.eliminate_zeros()
        w.data = np.exp(-w.data / 1000 / COMPETITION_KM)
        self.competition = w

    def log_access(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return ln A for every station, given every station's mass."""
        imp = np.exp(self.x @ b)
        u = np.zeros((self.n, self.n))
        u[self.ii, self.jj] = imp * np.exp(a[self.jj])
        u[self.jj, self.ii] = imp * np.exp(a[self.ii])
        rival = np.asarray(self.competition @ u)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(u > 0, u / (u + rival), 0.0)
        return np.log((u * share).sum(1))


def ols(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of ``y`` on a constant and ``z``."""
    return np.linalg.lstsq(np.column_stack([np.ones(len(z)), z]), y, rcond=None)[0]


def ols_predict(z: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(z)), z]) @ coef


def access_stage(
    net: Network,
    terms_x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    train: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit accessibility, filling in every missing mass from the trip end.

    A missing mass starts from its trip-end terms ``terms_x``, regressed on the fitted masses.
    Each pass then refits ln y on [ln A, terms_x] over ``train`` and takes the mass that makes
    the trip end consistent with the fitted ones. Returns ``ln_access``, the trip end's
    ``coef``, the filled masses ``a``, the constant ``c0`` and the mass regression
    ``mass_coef``.
    """
    missing = np.isnan(a)
    known = ~missing & train & np.isfinite(y)
    fit = train & np.isfinite(y)
    mass_coef = ols(terms_x[known], a[known])
    full = a.copy()
    full[missing] = ols_predict(terms_x[missing], mass_coef)
    for _ in range(TRIP_END_ITERATIONS):
        ln_access = net.log_access(full, b)
        coef = ols(np.column_stack([ln_access, terms_x])[fit], y[fit])
        fitted = ols_predict(np.column_stack([ln_access, terms_x]), coef)
        c0 = np.median((y - coef[1] * ln_access - a)[known])
        full[missing] = (fitted - coef[1] * ln_access - c0)[missing]
    return {"ln_access": ln_access, "coef": coef, "a": full, "c0": c0, "mass_coef": mass_coef}
