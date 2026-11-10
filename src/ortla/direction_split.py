r"""Direction split of a symmetric OD matrix, fitted on loading counts.

The ODM gives every pair the same flow both ways, and the period matrices keep that symmetry.
``demand.predict_demand`` divides a slot's demand by the trains in each direction, so on its own
the model puts the higher load per train in the direction with fewer trains. Counts show the
opposite: operators run more trains where the demand is, and load per train comes out roughly
level. The frequency divisor alone makes the direction error worse than predicting no asymmetry.

For a pair {o, d} in a slot, the forward direction takes share ``s`` of the pair's combined demand:

.. math::

    \operatorname{logit} s = \ln\frac{n_\text{fwd}}{n_\text{rev}}
        + a_p + b_p \ln\frac{F_o}{F_d} + c_p (g_o - g_d)

``n`` is trains in the slot each way, ``F`` table 1410 entries and exits, and ``g`` an optional
station function, ln(catchment population / jobs within a mile). The train term is fixed at 1 so
it cancels the divisor; with the other terms at zero, load per train is level both ways. Both
inputs exist for a service that does not run yet.

Fit with :func:`calibrate_to_segments`, on the on-board segment loads a hindcast is scored on.
:func:`calibrate` fits boardings at a train's first stop, which mostly measure the origin's size
(an Ely to London morning train fills up after Ely) and gives an am-peak slope near zero. On case
06's segments, with corridors held out, the segment fit takes rmse(log) from 0.858 to 0.727. The
slopes run towards the bigger station in the morning (am peak -0.46) and away from it in the
evening (+0.47). The function term helps where footfall misreads a station: Cambridge South's
morning boarding share goes from 0.61 to 0.42 (counted 0.16).

Splitting in proportion to trains assumes supply follows demand; that suits forecasting a planned
service but is not a causal claim. The evidence is in ``docs/notes/DIRECTIONAL_DEMAND_SPLIT.md``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import optimize, sparse

from ortla.clock import classify_times

if TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)

MODEL_FILENAME = "direction_split_model.json"

#: Key used for the pooled (all-periods) coefficients and as the
#: fallback for a period the fit never saw.
POOLED = "_pooled"

#: Logit values beyond this are clamped, so an extreme frequency or
#: footfall ratio cannot drive a share to exactly 0 or 1 (which would
#: zero one direction's demand outright).
_MAX_LOGIT = 6.0

_TERMS = ["intercept", "ln_footfall_ratio"]

#: Coefficient key for the optional station-function term (module
#: docstring).
FUNCTION_TERM = "station_function"


@dataclasses.dataclass
class DirectionSplitModel:
    """Fitted directional split coefficients per period.

    ``coefficients`` maps period -> ``{"intercept", "ln_footfall_ratio"}`` (plus
    :data:`FUNCTION_TERM` with a station-function term) and always has a :data:`POOLED` entry for
    periods not fitted. ``r_squared`` is in sample, for the record; the held-out figure in the
    module docstring is the one that matters. ``station_function`` is CRS -> ln(catchment
    population / jobs within a mile), or None for models fitted before the term existed.
    """

    coefficients: dict[str, dict[str, float]]
    n_cells: int
    r_squared: float
    source: str
    station_function: dict[str, float] | None = None

    def _terms(self, period: str | None) -> dict[str, float]:
        """Coefficients for ``period``, else the pooled fit."""
        if period is not None and period in self.coefficients:
            return self.coefficients[period]
        return self.coefficients[POOLED]

    def footfall_slope(self, period: str | None) -> float:
        """Return the footfall slope for ``period`` (else pooled)."""
        return self._terms(period)["ln_footfall_ratio"]

    def function_difference(self, origin: str | None, destination: str | None) -> float | None:
        """``g_o - g_d``, or ``None`` when either station lacks one."""
        if not self.station_function or origin is None or destination is None:
            return None
        g_origin = self.station_function.get(origin)
        g_destination = self.station_function.get(destination)
        if g_origin is None or g_destination is None:
            return None
        return g_origin - g_destination

    def share(
        self,
        period: str | None,
        footfall_origin: float | None,
        footfall_destination: float | None,
        trains_forward: float,
        trains_reverse: float,
        *,
        origin: str | None = None,
        destination: str | None = None,
    ) -> float:
        """Share of the slot's demand going ``origin -> destination``, in (0, 1).

        A missing or non-positive footfall drops the footfall term, an unknown period takes the
        pooled coefficients, and an origin or destination without a station-function value drops
        that term.

        Returns exactly 0.5 when either direction has no train in the slot. The split was fitted
        only on pairs running both ways, and handing a one-way slot its whole demand would double
        that direction in a regime the fit never saw.
        """
        if trains_forward <= 0 or trains_reverse <= 0:
            return 0.5
        terms = self._terms(period)
        logit = math.log(trains_forward / trains_reverse) + terms["intercept"]
        if (
            footfall_origin
            and footfall_destination
            and footfall_origin > 0
            and footfall_destination > 0
        ):
            logit += terms["ln_footfall_ratio"] * math.log(footfall_origin / footfall_destination)
        difference = self.function_difference(origin, destination)
        if difference is not None:
            logit += terms.get(FUNCTION_TERM, 0.0) * difference
        logit = max(-_MAX_LOGIT, min(_MAX_LOGIT, logit))
        return 1.0 / (1.0 + math.exp(-logit))

    def split(
        self,
        trips_forward: float,
        trips_reverse: float,
        period: str | None,
        footfall_origin: float | None,
        footfall_destination: float | None,
        trains_forward: float,
        trains_reverse: float,
        *,
        origin: str | None = None,
        destination: str | None = None,
    ) -> float:
        """Forward slot demand after splitting the pair's total.

        The pair's combined demand ``trips_forward + trips_reverse`` is
        conserved, so this moves trips between directions without
        creating or destroying any. Feeding it a symmetric matrix (the
        usual case) means the combined total is simply twice the cell.
        """
        combined = trips_forward + trips_reverse
        if combined <= 0:
            return 0.0
        return combined * self.share(
            period,
            footfall_origin,
            footfall_destination,
            trains_forward,
            trains_reverse,
            origin=origin,
            destination=destination,
        )

    def save(self, path: pathlib.Path) -> None:
        """Write the model as JSON."""
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: pathlib.Path) -> DirectionSplitModel:
        """Read a model written by :meth:`save`."""
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def build_cells(
    frame: pd.DataFrame,
    cutoffs: dict[str, str],
    *,
    min_trains: int = 3,
    min_flow: float = 20.0,
) -> pd.DataFrame:
    """Bidirectional OD x period cells from a canonical loading frame.

    One row per unordered pair (``origin < destination``) and period that the feed sees running
    both ways, with each direction's boardings, trains and load per train. Only rows leaving the
    train's own origin are used, so ``Mean Passengers On`` is that service's origin loading. A
    direction needs ``min_trains`` services and ``min_flow`` boardings.
    """
    df = frame[frame["Departure from CRS"] == frame["Origin CRS"]].copy()
    times = df["Departure Time"].where(df["Departure Time"].notna(), df["Origin Departure Time"])
    df["period"] = classify_times(times, cutoffs)
    # The canonical frame carries pandas nullable/object dtypes (a
    # reader that never saw a column fills it with pd.NA), which would
    # make the grouped sums object-dtype and blow up in np.log
    # downstream.
    df["Mean Passengers On"] = pd.to_numeric(df["Mean Passengers On"], errors="coerce")
    df = df.dropna(subset=["period", "Mean Passengers On", "Origin CRS", "Destination CRS"])
    df = df[df["period"] != "Unknown"]

    grouped = (
        df.groupby(["Origin CRS", "Destination CRS", "period"], observed=True)
        .agg(flow=("Mean Passengers On", "sum"), trains=("Mean Passengers On", "size"))
        .reset_index()
        .rename(columns={"Origin CRS": "origin", "Destination CRS": "destination"})
    )
    grouped["load_per_train"] = grouped["flow"] / grouped["trains"]
    reverse = grouped.rename(
        columns={
            "origin": "destination",
            "destination": "origin",
            "flow": "flow_reverse",
            "trains": "trains_reverse",
            "load_per_train": "load_per_train_reverse",
        }
    )
    cells = grouped.merge(reverse, on=["origin", "destination", "period"])
    cells = cells[cells["origin"] < cells["destination"]]
    keep = (
        (cells["flow"] >= min_flow)
        & (cells["flow_reverse"] >= min_flow)
        & (cells["trains"] >= min_trains)
        & (cells["trains_reverse"] >= min_trains)
    )
    return cells[keep].reset_index(drop=True)


def calibrate(
    cells: pd.DataFrame,
    footfall: dict[str, int],
    source: str,
    *,
    min_rows_per_period: int = 25,
) -> DirectionSplitModel:
    """Fit the split per period on the asymmetry in load per train.

    The target is ``ln(load_per_train / load_per_train_reverse)``, what a loading hindcast is
    scored on, against ``ln(F_origin / F_destination)``. The frequency term is not fitted: its
    coefficient is fixed at 1 so it cancels the frequency divisor exactly. A period with fewer than
    ``min_rows_per_period`` cells keeps the pooled coefficients.
    """
    df = cells.copy()
    for column in ("load_per_train", "load_per_train_reverse"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["f_o"] = pd.to_numeric(df["origin"].map(footfall), errors="coerce")
    df["f_d"] = pd.to_numeric(df["destination"].map(footfall), errors="coerce")
    df = df[(df["f_o"] > 0) & (df["f_d"] > 0)]
    df = df[(df["load_per_train"] > 0) & (df["load_per_train_reverse"] > 0)]
    df["y"] = np.log(df["load_per_train"] / df["load_per_train_reverse"])
    df["x"] = np.log(df["f_o"] / df["f_d"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "x"])
    if df.empty:
        raise ValueError("No usable bidirectional cells to fit a direction split.")

    def _fit(sub: pd.DataFrame) -> tuple[dict[str, float], float]:
        x = np.column_stack([np.ones(len(sub)), sub["x"].to_numpy(float)])
        y = sub["y"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ beta
        r2 = float(1 - residuals.var() / y.var()) if y.var() > 0 else 0.0
        return dict(zip(_TERMS, map(float, beta), strict=True)), r2

    pooled, r_squared = _fit(df)
    coefficients = {POOLED: pooled}
    for period, sub in df.groupby("period", observed=True):
        if len(sub) < min_rows_per_period:
            continue
        coefficients[str(period)], _ = _fit(sub)

    model = DirectionSplitModel(
        coefficients=coefficients,
        n_cells=len(df),
        r_squared=r_squared,
        source=source,
    )
    logger.info(
        "Direction split fitted on %d bidirectional cells (pooled R^2 %.3f); "
        "periods with their own slope: %s",
        model.n_cells,
        model.r_squared,
        ", ".join(sorted(k for k in coefficients if k != POOLED)) or "none",
    )
    return model


def split_multiplier(
    slopes: np.ndarray,
    period_index: np.ndarray,
    ln_footfall_ratio: np.ndarray,
    ln_train_ratio: np.ndarray,
    function_slopes: np.ndarray | None = None,
    function_difference: np.ndarray | None = None,
) -> np.ndarray:
    """Return the factor the split applies to a leg's demand.

    Vectorised :meth:`DirectionSplitModel.share` with zero intercepts:
    ``2 * s`` where the split fires, ``1`` where it does not
    (``ln_train_ratio`` is NaN for a one-way slot, which
    :meth:`~DirectionSplitModel.share` leaves at exactly 0.5).
    ``function_slopes`` and ``function_difference`` (``g_o - g_d``, 0
    where either end has no function) add the station-function term.
    """
    fires = np.isfinite(ln_train_ratio)
    logit = np.where(fires, ln_train_ratio, 0.0) + slopes[period_index] * ln_footfall_ratio
    if function_slopes is not None and function_difference is not None:
        logit = logit + function_slopes[period_index] * function_difference
    logit = np.clip(logit, -_MAX_LOGIT, _MAX_LOGIT)
    return np.where(fires, 2.0 / (1.0 + np.exp(-logit)), 1.0)


def calibrate_to_segments(
    legs: pd.DataFrame,
    incidence: sparse.spmatrix,
    observed: np.ndarray,
    source: str,
    *,
    scale: float = 1.0,
    station_function: dict[str, float] | None = None,
) -> DirectionSplitModel:
    """Fit footfall slopes per period on a hindcast's on-board loads.

    :func:`calibrate` regresses on boardings at a train's first stop, which is the wrong target for
    a tidal flow: an Ely to London morning train picks up most of its load after Ely, so first-stop
    boardings mostly measure the origin's size and the am-peak slope comes out near zero. This fits
    the same model through each segment's on-board load, the sum of every leg spanning it, as
    :func:`ortla.demand.predict_demand` predicts it.

    Only the slopes are fitted. A nonzero intercept enters both directions with the same sign and
    so creates or destroys demand instead of moving it.

    ``legs`` comes from a hindcast with the split off: ``captured`` (annual trips per train),
    ``period``, ``ln_footfall_ratio`` (0 where unknown) and ``ln_train_ratio`` (NaN where the split
    does not fire). ``incidence`` is segments x legs; segments without a positive observation are
    ignored. ``scale`` turns annual trips into the observed unit
    (:data:`~ortla.clock.WEEKDAYS_PER_YEAR` for a typical day). With ``station_function``, ``legs``
    also needs ``function_difference`` and a function slope is fitted per period too. ``r_squared``
    is the share of log-error variance removed in sample.
    """
    periods = sorted(legs["period"].astype(str).unique())
    period_index = legs["period"].astype(str).map({p: i for i, p in enumerate(periods)})
    period_index = period_index.to_numpy(int)
    captured = legs["captured"].fillna(0.0).to_numpy(float)
    x = legs["ln_footfall_ratio"].fillna(0.0).to_numpy(float)
    train_ratio = legs["ln_train_ratio"].to_numpy(float)
    with_function = station_function is not None
    z = (
        pd.to_numeric(legs["function_difference"], errors="coerce").fillna(0.0).to_numpy(float)
        if with_function
        else None
    )
    incidence = sparse.csr_matrix(incidence)
    use = np.asarray(observed, float) > 0
    target = np.log(np.where(use, observed, 1.0))

    def predict(theta: np.ndarray, index: np.ndarray, width: int) -> np.ndarray:
        function_slopes = theta[width:] if with_function else None
        multiplier = split_multiplier(theta[:width], index, x, train_ratio, function_slopes, z)
        return incidence @ (captured * multiplier) / scale

    def loss(theta: np.ndarray, index: np.ndarray, width: int) -> float:
        predicted = predict(theta, index, width)
        keep = use & (predicted > 0)
        return float(np.mean((np.log(predicted[keep]) - target[keep]) ** 2))

    terms = 2 if with_function else 1
    base = loss(np.zeros(terms * len(periods)), period_index, len(periods))
    theta = optimize.minimize(
        loss,
        np.zeros(terms * len(periods)),
        args=(period_index, len(periods)),
        method="L-BFGS-B",
    ).x
    pooled = optimize.minimize(
        loss, np.zeros(terms), args=(np.zeros_like(period_index), 1), method="L-BFGS-B"
    ).x
    fitted = loss(theta, period_index, len(periods))

    def entry(values: np.ndarray, i: int, width: int) -> dict[str, float]:
        out = {"intercept": 0.0, "ln_footfall_ratio": float(values[i])}
        if with_function:
            out[FUNCTION_TERM] = float(values[width + i])
        return out

    coefficients = {
        POOLED: entry(pooled, 0, 1),
        **{p: entry(theta, i, len(periods)) for i, p in enumerate(periods)},
    }
    model = DirectionSplitModel(
        coefficients=coefficients,
        n_cells=int(use.sum()),
        r_squared=float(1 - fitted / base) if base > 0 else 0.0,
        source=source,
        station_function=dict(station_function) if with_function else None,
    )
    logger.info(
        "Direction split fitted on %d observed segments: mean squared log error "
        "%.3f -> %.3f; slopes %s",
        model.n_cells,
        base,
        fitted,
        ", ".join(
            f"{p} {c['ln_footfall_ratio']:+.3f}"
            + (f" / function {c[FUNCTION_TERM]:+.3f}" if with_function else "")
            for p, c in coefficients.items()
            if p != POOLED
        ),
    )
    return model
