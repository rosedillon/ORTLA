"""Fit the members, average them, borrow neighbours' residuals and score the result.

Every member is fitted five times with one spatial fold held out (accessibility refitted
without that fold's stations too), giving an out-of-fold forecast for every station in the
fitted set, then once on all of it. The ensemble is the plain mean of the thirteen members.

Borrowing adds the in-sample residuals of established stations nearby, weighted
exp(-km / 4) and shrunk by kappa 0.5. The 80% interval is the 80th percentile of the absolute
cross-validated error after borrowing, taken separately for each fifth of forecast size.
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pandas as pd

from ortla import new_station as ns
from ortla.station_ensemble import frame, gravity, members, sources, terms

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from ortla.station_ensemble.frame import StationFrame
    from ortla.station_ensemble.members import Member
    from ortla.station_ensemble.sources import DataPaths

logger = logging.getLogger(__name__)
T = TypeVar("T")

ENSEMBLE = "ensemble"
BORROW_KM, BORROW_KAPPA = 4.0, 0.5
INTERVAL_LEVEL = 0.8
SIZE_BANDS = 5


class Cache:
    """Pickles slow intermediate results in a folder; with no folder, nothing is kept.

    Entries are keyed by name only, so empty the folder when the inputs change.
    """

    def __init__(self, folder: pathlib.Path | None) -> None:
        self.folder = folder

    def __call__(self, name: str, build: Callable[[], T]) -> T:
        if self.folder is None:
            return build()
        path = self.folder / f"{name}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                return pickle.load(fh)  # noqa: S301  (written below)
        out = build()
        self.folder.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(out, fh)
        return out


def borrow(
    residual: np.ndarray, pool_xy: np.ndarray, xy: np.ndarray, same: bool = False
) -> np.ndarray:
    """Average neighbours' residuals, weighted exp(-km / 4) and shrunk by kappa.

    With ``same``, the pool is the stations themselves and each one's own residual is left out.
    """
    km = np.sqrt(((xy[:, None, :] - pool_xy[None, :, :]) ** 2).sum(-1)) / 1000
    w = np.exp(-km / BORROW_KM)
    if same:
        np.fill_diagonal(w, 0.0)
    return ns.borrow(w, residual, BORROW_KAPPA)


@dataclasses.dataclass
class Model:
    """Everything fitted, kept so that new stations can be forecast against it."""

    sf: StationFrame
    terms: pd.DataFrame
    access_columns: list[str]
    members: list[Member]
    pairs: pd.DataFrame
    stage1_b: np.ndarray
    access: dict[str, np.ndarray]  # the full-data accessibility stage
    access_by_fold: list[np.ndarray]
    y3: np.ndarray
    people: pd.DataFrame
    places: terms.Places
    cv: dict[str, np.ndarray]  # out-of-fold forecast, by member and for the ensemble
    full: dict[str, np.ndarray]  # fitted on every wide station, by member and for the ensemble

    @property
    def wide(self) -> np.ndarray:
        return np.flatnonzero(self.sf.wide)

    def residual(self, key: str = ENSEMBLE) -> np.ndarray:
        """In-sample residuals on the fitted set, which neighbours borrow."""
        w = self.wide
        return self.sf.y[w] - self.full[key][w]

    def borrowed(
        self, key: str, positions: np.ndarray, xy: np.ndarray | None = None
    ) -> np.ndarray:
        """``key``'s forecast after borrowing, for stations not in the fitted set."""
        xy = self.sf.xy[positions] if xy is None else xy
        return self.full[key][positions] + borrow(self.residual(key), self.sf.xy[self.wide], xy)

    def cv_after_borrowing(self, key: str = ENSEMBLE) -> np.ndarray:
        """Out-of-fold error after borrowing, on the fitted set (forecast minus actual)."""
        w = self.wide
        error = self.cv[key][w] - self.sf.y[w]
        return error + borrow(self.residual(key), self.sf.xy[w], self.sf.xy[w], same=True)

    def interval_half_width(self, forecast: np.ndarray) -> np.ndarray:
        """Return the 80% interval's half-width in log terms, by the forecast's size fifth."""
        scored = self.sf.scored[self.wide]
        error = self.cv_after_borrowing()[scored]
        predicted = self.sf.y[self.wide][scored] + error
        edges = np.quantile(predicted, np.linspace(0, 1, SIZE_BANDS + 1))[1:-1]
        band = np.digitize(predicted, edges)
        q = np.array(
            [np.quantile(np.abs(error[band == k]), INTERVAL_LEVEL) for k in range(SIZE_BANDS)]
        )
        return q[np.digitize(forecast, edges)]


def design(table: pd.DataFrame, columns: list[str], ln_access: np.ndarray | None) -> np.ndarray:
    x = table[columns].to_numpy(float)
    return x if ln_access is None else np.column_stack([ln_access, x])


def fit(paths: DataPaths, cache: Cache) -> Model:
    """Build every input and fit the ensemble, cross-validated and on all stations."""
    people = cache(
        "postcodes",
        lambda: ns.postcodes(
            paths.sdft / "census2021_postcode_population.csv",
            paths.sdft / "codepoint_open" / "Data" / "CSV",
        ),
    )
    service = cache(
        "service_calibration",
        lambda: ns.service(paths.calibration_calls, sources.CALIBRATION_DATE),
    )
    sf = frame.build(paths, people, service)
    places = terms.Places.load(paths)
    place = terms.place_terms(sf.xy, places)
    timetable = cache(
        "timetable_terms", lambda: terms.timetable_terms(sf.index, sf.xy, paths, places)
    )
    kb = terms.knowledgebase_terms(paths, sf.index)
    table, _, operators = terms.build(sf, place, timetable, kb, places)
    all_terms = [c for c in table.columns if table[c].notna().all() and table[c].std() > 0]
    table["ln_pop_service"] = np.where(
        sf.scotland,
        table["ln_pop"],
        np.log1p(
            members.service_catchment(
                sf.xy,
                people[["e", "n"]].to_numpy(float),
                people["pop"].to_numpy(),
                sf.f["freq"].to_numpy(float),
            )
        ),
    )
    # Accessibility is fitted with every service SRAM lists; the members with ticketed trains.
    access_terms = table.assign(
        ln_trains=np.log(sf.f["freq_all"]),
        car_catchment_trains=terms.car_catchment_by_trains(
            sf.xy, places.cars, table["ln_car"].to_numpy(), sf.f["freq_all"].to_numpy(float)
        ),
    )

    sk = cache(
        "skim",
        lambda: gravity.skim(
            paths.calibration_calls, sources.CALIBRATION_DATE, paths / "stations.csv"
        ),
    )
    pairs = gravity.pairs(
        sk,
        gravity.odm_journeys(paths / "ODM_for_RDM_2024-25.csv"),
        sources.footfall(paths, frame.FOOTFALL_YEAR),
        paths / "stations.csv",
        sf.index,
    )
    stage1 = cache("stage1", lambda: _stage1(sf, pairs))
    net = gravity.Network(pairs, sf.xy)
    access_columns = members.trip_end_terms(operators, extra=False)
    terms_x = access_terms[access_columns].to_numpy(float)
    by_fold = [
        gravity.access_stage(net, terms_x, sf.y, *stage1[k], sf.wide & (sf.fold != k))["ln_access"]
        for k in range(frame.FOLDS)
    ]
    model = Model(
        sf=sf,
        terms=table,
        access_columns=access_columns,
        members=members.fitted_members(
            members.trip_end_terms(operators), all_terms, terms.SDFT_TERMS
        ),
        pairs=pairs,
        stage1_b=stage1[None][1],
        access=gravity.access_stage(net, terms_x, sf.y, *stage1[None], sf.wide),
        access_by_fold=by_fold,
        y3=_three_year_mean(paths, sf),
        people=people,
        places=places,
        cv={},
        full={},
    )
    for m in model.members:
        logger.info("Fitting %s", m.key)
        model.cv[m.key], model.full[m.key] = _cross_validate(model, m)
    model.cv[members.KRIGING], model.full[members.KRIGING] = _kriging(model)
    model.cv[ENSEMBLE] = np.mean([model.cv[k] for k in members.ORDER], axis=0)
    model.full[ENSEMBLE] = np.mean([model.full[k] for k in members.ORDER], axis=0)
    return model


def _stage1(sf: StationFrame, pairs: pd.DataFrame) -> dict:
    """Fit stage 1 on every pair clear of the held-out stations, then once without each fold.

    Each fold's run also leaves out that fold's fitted stations, so their masses are not fitted.
    """
    held_out = pairs["i"].isin(sf.excluded) | pairs["j"].isin(sf.excluded)
    rows = ~held_out.to_numpy()
    n = len(sf.index)
    ii, jj = pairs["ii"].to_numpy(), pairs["jj"].to_numpy()
    out = {None: gravity.stage1(pairs, rows, n)}
    for k in range(frame.FOLDS):
        out_k = sf.wide & (sf.fold == k)
        logger.info("Stage 1 without fold %d", k)
        out[k] = gravity.stage1(pairs, rows & ~out_k[ii] & ~out_k[jj], n)
    return out


def _three_year_mean(paths: DataPaths, sf: StationFrame) -> np.ndarray:
    """Average ln footfall over 2022-23 to 2024-25; 2024-25 alone where a year is missing."""
    years = []
    for year in (2022, 2023, 2024):
        t = pd.read_csv(paths.footfall(year)).iloc[:, [2, 3]]
        t.columns = ["tlc", "ff"]
        t["ff"] = pd.to_numeric(t["ff"].astype(str).str.replace(",", ""), errors="coerce")
        ff = t.dropna().drop_duplicates("tlc").set_index("tlc")["ff"]
        years.append(np.log(ff.where(ff > 0)).reindex(sf.index))
    mean = pd.concat(years, axis=1).mean(axis=1, skipna=False).to_numpy(float)
    return np.where(np.isfinite(mean), mean, sf.y)


def member_design(m: Member, table: pd.DataFrame, ln_access: np.ndarray) -> np.ndarray:
    """Build member ``m``'s design from a term table and ln A."""
    if m.population != "ln_pop":
        table = table.assign(ln_pop=table[m.population])
    return design(table, m.columns, ln_access if m.with_access else None)


def target(model: Model, m: Member) -> np.ndarray:
    return model.y3 if m.target == "y3" else model.sf.y


def _cross_validate(model: Model, m: Member) -> tuple[np.ndarray, np.ndarray]:
    sf, y = model.sf, target(model, m)
    cv = np.full(len(sf.index), np.nan)
    for k in range(frame.FOLDS):
        x = member_design(m, model.terms, model.access_by_fold[k])
        train = sf.wide & (sf.fold != k) & np.isfinite(y)
        test = sf.wide & (sf.fold == k)
        cv[test] = m.learner(x[train], y[train], x[test])
    x = member_design(m, model.terms, model.access["ln_access"])
    train = sf.wide & np.isfinite(y)
    return cv, m.learner(x[train], y[train], x)


def _kriging(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Fit footfall per catchment resident, with borrowing as its only spatial model."""
    sf = model.sf
    lp = model.terms["ln_pop"].to_numpy(float)
    cv = np.full(len(sf.index), np.nan)
    for k in range(frame.FOLDS):
        train = np.flatnonzero(sf.wide & (sf.fold != k) & np.isfinite(sf.y))
        test = sf.wide & (sf.fold == k)
        mu = np.mean(sf.y[train] - lp[train])
        r = sf.y[train] - lp[train] - mu
        cv[test] = lp[test] + mu + borrow(r, sf.xy[train], sf.xy[test])
    w = np.flatnonzero(sf.wide)
    return cv, lp + np.mean((sf.y - lp)[w])


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error**2)))


def scores(model: Model) -> pd.DataFrame:
    """Score each member and the ensemble: cross-validated, on openings and in Scotland."""
    sf = model.sf
    w = model.wide
    scored = sf.scored[w]
    test = sf.positions(sf.openings)
    actual = sf.y[test]
    scot_est, scot_open = sf.scotland_established, sf.positions(sf.scotland_openings)
    rows = []
    for key in [*members.ORDER, ENSEMBLE]:
        error = model.borrowed(key, test) - actual
        row = {
            "member": key,
            "cv_rmse": _rmse((model.cv[key][w] - sf.y[w])[scored]),
            "cv_rmse_after_borrowing": _rmse(model.cv_after_borrowing(key)[scored]),
            "openings_rmse": _rmse(error),
            "openings_median_ratio": np.exp(np.median(error)),
            "openings_within_x2": np.mean(np.abs(error) <= np.log(2)),
        }
        # The service catchment needs residents by postcode, which Scotland's census lacks.
        if key != "service_catchment":
            row["scotland_established_rmse"] = _rmse(
                model.borrowed(key, scot_est) - sf.y[scot_est]
            )
            row["scotland_openings_rmse"] = _rmse(model.borrowed(key, scot_open) - sf.y[scot_open])
        rows.append(row)
    out = pd.DataFrame(rows).set_index("member")
    forecast = model.borrowed(ENSEMBLE, test)
    half = model.interval_half_width(forecast)
    out.loc[ENSEMBLE, "interval_80_typical_factor"] = np.exp(np.mean(half))
    out.loc[ENSEMBLE, "interval_80_coverage"] = np.mean(np.abs(forecast - actual) <= half)
    return out


def openings_table(model: Model) -> pd.DataFrame:
    """Tabulate the ensemble's forecast for each opening, with its interval and outturn."""
    sf = model.sf
    test = sf.positions(sf.openings)
    forecast = model.borrowed(ENSEMBLE, test)
    half = model.interval_half_width(forecast)
    return pd.DataFrame(
        {
            "name": sf.f["name"].iloc[test].to_numpy(),
            "first_year": sf.first_year.reindex(sf.openings).to_numpy(),
            "actual_2024_25": np.exp(sf.y[test]).round(),
            "forecast": np.exp(forecast).round(),
            "low_80": np.exp(forecast - half).round(),
            "high_80": np.exp(forecast + half).round(),
            "ratio": np.exp(forecast - sf.y[test]).round(3),
        },
        index=pd.Index(sf.openings, name="crs"),
    )
