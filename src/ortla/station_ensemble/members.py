"""The ensemble's thirteen members: one trip end, fitted thirteen different ways.

Each learner is ``fit_predict(x_train, y_train, x_new) -> ln forecast``. Most members read the
same design, ln accessibility followed by the terms of :func:`trip_end_terms`. They differ in
the learner (least squares, trees, analogues, Poisson, median, ridge, splines), and three in
what they are fitted to: one shares residents among stations by service, one is kriging
alone, and one is fitted to a three-year mean of footfall.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, QuantileRegressor, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler

from ortla.station_ensemble.gravity import ols, ols_predict

Learner = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

ANALOGUES = 15
SUNDAY_AND_STAFFING = ["sunday_ratio", "staffed_full_time", "staffed_part_time"]
ANALOGUE_TERMS = [
    "ln_pop",
    "ln_jobs_zones",
    "ln_trains",
    "ln_car",
    "travelcard",
    "terminus",
    "request_share",
    "ln_venues",
    "ln_beds",
]
SPLINE_TERMS = ["ln_pop", "ln_trains", "ln_jobs_zones", "ln_car"]  # after ln A
INCREASING = (
    "ln_pop",
    "ln_trains",
    "ln_jobs_zones",
)  # constrained in the boosted trees, with ln A
# The service catchment shares residents among their ten nearest stations by
# exp(-km x 1.3 / 1) x trains.
CHOICE_KM, CHOICE_TRAINS = 1.0, 1.0


def least_squares(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return ols_predict(x_new, ols(x_train, y_train))


def _standardised(make: Callable[[], object]) -> Learner:
    def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        sc = StandardScaler().fit(x_train)
        return make().fit(sc.transform(x_train), y_train).predict(sc.transform(x_new))

    return fit_predict


def poisson(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Poisson regression on footfall itself (in 100,000s), so big stations weigh more."""
    sc = StandardScaler().fit(x_train)
    model = PoissonRegressor(alpha=1e-4, max_iter=3000).fit(
        sc.transform(x_train), np.exp(y_train) / 1e5
    )
    return np.log(model.predict(sc.transform(x_new)) * 1e5)


def median_regression(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    sc = StandardScaler().fit(x_train)
    model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
    return model.fit(sc.transform(x_train), y_train).predict(sc.transform(x_new))


def splines(n_smooth: int) -> Learner:
    """Cubic splines on the first ``n_smooth`` columns, the rest linear, ridge-penalised."""

    def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        st = SplineTransformer(n_knots=5, degree=3).fit(x_train[:, :n_smooth])
        sc = StandardScaler().fit(x_train[:, n_smooth:])

        def basis(x: np.ndarray) -> np.ndarray:
            return np.column_stack([st.transform(x[:, :n_smooth]), sc.transform(x[:, n_smooth:])])

        return Ridge(alpha=1.0).fit(basis(x_train), y_train).predict(basis(x_new))

    return fit_predict


def boosted_trees(monotone: list[int]) -> Learner:
    def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        model = HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.05, monotonic_cst=monotone, random_state=0
        )
        return model.fit(x_train, y_train).predict(x_new)

    return fit_predict


def forest(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    model = RandomForestRegressor(500, min_samples_leaf=5, n_jobs=-1, random_state=0)
    return model.fit(x_train, y_train).predict(x_new)


def analogues(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Distance-weighted mean of the 15 most similar stations (standardised inputs)."""
    sc = StandardScaler().fit(x_train)
    model = KNeighborsRegressor(ANALOGUES, weights="distance").fit(sc.transform(x_train), y_train)
    return model.predict(sc.transform(x_new))


def least_squares_plus_analogues(
    x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    """Least squares plus the mean residual of the 15 most similar training stations."""
    coef = ols(x_train, y_train)
    residual = y_train - ols_predict(x_train, coef)
    sc = StandardScaler().fit(x_train)
    near = KNeighborsRegressor(ANALOGUES).fit(sc.transform(x_train), residual)
    return ols_predict(x_new, coef) + near.predict(sc.transform(x_new))


def ridge(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return _standardised(lambda: RidgeCV(np.logspace(-2, 3, 21)))(x_train, y_train, x_new)


def service_catchment(
    xy: np.ndarray, people_xy: np.ndarray, pop: np.ndarray, freq: np.ndarray
) -> np.ndarray:
    """Share each resident among the ten nearest stations by distance and trains."""
    dist, near = cKDTree(xy).query(people_xy, k=10)
    w = np.exp(-dist * 1.3 / 1000 / CHOICE_KM) * freq[near] ** CHOICE_TRAINS
    w /= np.maximum(w.sum(1, keepdims=True), 1e-300)
    return np.bincount(near.ravel(), weights=(pop[:, None] * w).ravel(), minlength=len(xy))


@dataclasses.dataclass(frozen=True)
class Member:
    """One member: its learner, its columns, and whether ln A leads the design."""

    key: str
    learner: Learner
    columns: list[str]
    with_access: bool = True
    target: str = "y"  # "y", or "y3" for the three-year mean
    population: str = "ln_pop"  # the column standing for residents; one member swaps in its own


KRIGING = "kriging"  # the member with no learner: ln footfall per resident, borrowed
# The thirteen members in the order the ensemble averages them.
ORDER = [
    "least_squares", "sdft_form", "analogues", "least_squares_plus_analogues", "boosted_trees",
    "random_forest", "poisson", "median", "ridge", "splines", "service_catchment", KRIGING,
    "three_year",
]  # fmt: skip


def trip_end_terms(operators: list[str], extra: bool = True) -> list[str]:
    """Return the trip end's columns; ``extra`` adds Sunday service and staffing.

    The accessibility stage reads them without ``extra``. The trees depend on the order.
    """
    columns = [
        "ln_pop", "ln_trains", "ln_jobs_zones", "ln_car", "electric", "travelcard", "terminus",
        "ln_beds", "car_catchment_trains", *operators, "request_share", "ln_venues",
    ]  # fmt: skip
    return columns + SUNDAY_AND_STAFFING if extra else columns


def fitted_members(
    trip_end: list[str], all_terms: list[str], sdft_terms: list[str]
) -> list[Member]:
    """List the twelve members with a learner (all but kriging)."""
    smooth = list(SPLINE_TERMS)
    rest = [c for c in trip_end if c not in smooth]
    monotone = [1] + [1 if c in INCREASING else 0 for c in trip_end]
    return [
        Member("least_squares", least_squares, trip_end),
        Member("sdft_form", least_squares, sdft_terms, with_access=False),
        Member("analogues", analogues, ANALOGUE_TERMS),
        Member("least_squares_plus_analogues", least_squares_plus_analogues, trip_end),
        Member("boosted_trees", boosted_trees(monotone), trip_end),
        Member("random_forest", forest, trip_end),
        Member("poisson", poisson, trip_end),
        Member("median", median_regression, trip_end),
        Member("ridge", ridge, all_terms),
        Member("splines", splines(1 + len(smooth)), smooth + rest),
        Member("service_catchment", least_squares, trip_end, population="ln_pop_service"),
        Member("three_year", least_squares, trip_end, target="y3"),
    ]
