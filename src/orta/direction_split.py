r"""Direction split of a symmetric OD matrix, fitted on loading counts.

Why this exists
---------------
``ODM_for_RDM_2024-25.csv`` carries no directional information. Every
one of its 1,597,996 directed pairs is exactly equal to its reverse: the
annual flow between A and B is reported as the same number in both
directions. (It is not a halved total: 54.5% of cells hold odd integers,
which cannot be halves of integers, and the directed total of 1.53bn
journeys sits at or above the national journey count. The value is one
figure duplicated into both directions.)

:func:`orta.period_matrices.generate_period_matrices` then
expands each cell with ``trips(o, d) = annual_journeys(o, d) *
factor(journey_type, day, period)``, and under the default
``journey_type_mode="distance"`` the journey type is a distance band,
identical for A->B and B->A, so the period matrices are
direction-symmetric by construction.

On its own that would only be a missing effect. But
``demand.predict_demand`` divides a slot's demand by
``trains_in_period``, which is directional, so the model's implied
per-train directional asymmetry is exactly
``-log(trains_fwd / trains_rev)``: it predicts the higher load per train
in the direction with fewer trains.

The data show the opposite. Measured over 419 bidirectional OD x period
cells across four operator feeds (Northern, Greater Anglia, Southeastern
and GWR, the operators publishing CSV; GTR and SWR publish XLSX and were
not in this measurement, though the shipped fit reads them):

===================================================  ======
sd(log load-per-train asymmetry)                      0.969
sd(log total-flow asymmetry)                          1.177
sd(log service-count asymmetry)                       0.748
corr(total-flow asymmetry, service-count asymmetry)  +0.572
corr(per-train asymmetry, service-count asymmetry)   -0.077
===================================================  ======

Operators absorb directional demand by running more trains, so load per
train comes out roughly equal in both directions while total flow does
not. The frequency divisor leaves an error of sd 1.177, where predicting
no directional asymmetry at all would leave 0.969: the divisor makes the
direction error 48% worse in variance than doing nothing.

What this module does
---------------------
It replaces the implicit 50/50 split with a fitted one. For an unordered
pair {o, d} in a (day, period) slot, the two directions share the pair's
combined slot demand in proportion to

.. math::

    \operatorname{logit} s = \ln\frac{n_\text{fwd}}{n_\text{rev}}
                            + a_p + b_p \ln\frac{F_o}{F_d}

where ``n`` are trains in the slot each way, ``F`` is table-1410 annual
entries/exits at each end, and ``a_p``/``b_p`` are fitted per time
period. The forward direction takes share ``s`` of
``trips(o,d) + trips(d,o)``.

The ``ln(n_fwd / n_rev)`` term is not fitted; it is there to cancel the
frequency divisor. With ``a_p = b_p = 0`` the split is exactly
proportional to train counts, which makes predicted load per train
identical in both directions: the pair's combined slot demand divided by
the combined train count. The fitted terms then tilt away from that
baseline.

How well it works
-----------------
Fitted on half the corridors and scored on 61 corridors never seen in
training (target: log per-train directional asymmetry; the last column
is the share of its variance removed):

============================================  ===========  =======
rule                                          sd(log err)  removed
============================================  ===========  =======
current: symmetric demand / trains in period  1.187        none
split in proportion to trains (a = b = 0)     0.968        34%
+ footfall ratio, pooled                      0.742        61%
+ footfall ratio, per-period slopes           0.631        72%
============================================  ===========  =======

Both inputs are available for a service that does not yet exist: the
proposed timetable supplies trains by direction and period, and footfall
is known for existing stations and forecast for new ones (the gravity
model's origin elasticity is 0.508, so a +/-50% footfall error moves
demand only +23%/-30%). It is a rule with two coefficients per period,
validated on corridors outside the fit, not a per-corridor correction
factor (which would memorise route identity and transfer to nothing).

Caveats
-------
* Splitting demand in proportion to trains assumes supply follows
  demand. For forecasting a proposed service that is what you want,
  since it encodes the planner's directional intent, but it is not a
  causal claim and must not be written up as one.
* The fit is modest in size: ~410 cells on ~123 corridors.
* Counted flow is taken as boardings at the train's origin, which
  conflates service pattern with demand.

Which fit to use: :func:`calibrate_to_segments`
-----------------------------------------------
The last caveat decides it. Boardings at a train's first stop measure
the size of the origin station at least as much as the direction of
flow: an Ely -> London morning train picks up most of its load after
Ely, while a London -> Ely train boards everyone at the terminal. Fitted
on that target, :func:`calibrate` returns an am-peak slope of +0.08 and
an early one of +0.04, about zero and the wrong sign for tidal
commuting, so the split does almost nothing in the peaks, where the
directional error is largest.

:func:`calibrate_to_segments` fits the same model (frequency term fixed
at 1, intercepts at zero so pair totals are conserved) on the on-board
segment loads a hindcast is scored on, through the model's own leg
structure. On case 06's 9,318 segments, scored on corridors held out of
the fit:

==================================  =========  =====  ==========
split                               rmse(log)  r      RMSE (pax)
==================================  =========  =====  ==========
off                                 0.858      0.439  115.8
first-stop fit (:func:`calibrate`)  0.853      0.507  108.7
segment fit, routes held out        0.727      0.672  89.7
segment fit, operators held out     0.726      0.676  89.3
==================================  =========  =====  ==========

Slopes on ``ln(F_o / F_d)``: am_peak -0.46, early -0.63, interpeak
-0.09, pm_peak +0.25, evening +0.47: towards the bigger station in the
morning, away from it in the evening. Six numbers, near-identical in and
out of sample (0.718 in sample), so this is a rule rather than a
memorised corridor effect.

Station function (optional)
---------------------------
Footfall says how big a station is, not what it is for. It points a
small station's morning demand at the big one, which is right for a
commuter station and wrong for a destination: Cambridge South's
biomedical campus is arrived at in the morning, and the DfT new-stations
study names misreading a station's function as a leading cause of
forecast error. A model may therefore carry a per-station ``g =
ln(catchment population / jobs within a mile)`` (production over
attraction) and a per-period coefficient ``c_p`` on ``g_o - g_d``:

.. math::

    \operatorname{logit} s = \ln\frac{n_\text{fwd}}{n_\text{rev}}
        + a_p + b_p \ln\frac{F_o}{F_d} + c_p (g_o - g_d)

The term flips sign with direction, so with zero intercepts pair totals
are still conserved. A station missing from the table drops the term, as
a missing footfall drops its own. On case 06, scored the same way as the
table above (routes held out, two fold seeds; operators held out),
footfall plus function gives rmse(log) 0.718-0.723 against 0.726-0.729
for footfall alone; residents over jobs within 1 km instead of the
catchment makes it worse (0.731-0.734), and either function measure
without footfall is worse still (0.75). Its main value is at stations
whose function footfall misreads: replayed on the new-station cases it
moves Cambridge South's morning boarding share from 0.61 to 0.42
(counted 0.16).
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

from orta.clock import classify_times

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
    """Fitted per-period directional split coefficients.

    Attributes
    ----------
    coefficients : dict[str, dict[str, float]]
        Time period -> ``{"intercept", "ln_footfall_ratio"}``, plus
        :data:`FUNCTION_TERM` when the model carries a station-function
        term. Always contains a :data:`POOLED` entry used for periods
        that were not fitted.
    n_cells : int
        Bidirectional OD x period cells the fit used.
    r_squared : float
        In-sample R^2 of the pooled fit on log per-train directional
        asymmetry. Reported for the record; the number that matters is
        the held-out-corridor figure in the module docstring.
    source : str
        What the fit was built from, for provenance.
    station_function : dict[str, float] | None
        Canonical CRS -> ``ln(catchment population / jobs within a
        mile)``, the production-over-attraction measure the optional
        function term reads. ``None`` (every model fitted before the
        term existed) leaves it out.
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
        """Share of the slot demand going ``origin -> destination``.

        Parameters
        ----------
        period : str | None
            Time-period label. An unknown period uses the pooled
            coefficients.
        footfall_origin, footfall_destination : float | None
            Annual entries/exits at each end. When either is missing or
            non-positive the footfall term drops out (treated as a ratio
            of 1), leaving the frequency term and the period intercept.
        trains_forward, trains_reverse : float
            Trains serving the pair in this slot in each direction.
        origin, destination : str | None, optional
            Canonical CRS codes, for the station-function term. Left
            out, or either absent from :attr:`station_function`, the
            term drops out.

        Returns
        -------
        float
            A share in ``(0, 1)``. Returns exactly 0.5, leaving a
            symmetric matrix untouched, when either direction has no
            train in the slot. The split was fitted and validated only
            on pairs the feed observes running both ways, and handing a
            one-way slot its whole combined demand would double that
            direction's figure in a regime the fit never saw. Keeping
            the change inside the validated regime is deliberate; see
            the module docstring.
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

    One row per unordered pair and period that the feed observes running
    both ways, carrying each direction's boardings, train count and
    implied load per train.

    Parameters
    ----------
    frame : pd.DataFrame
        Canonical loading frame
        (:data:`orta.loading_data.CANONICAL_COLUMNS`). Only
        rows departing the train's own origin are used, so
        ``Mean Passengers On`` is that service's origin loading rather
        than a mid-journey top-up.
    cutoffs : dict[str, str]
        Period name -> start time, as
        :class:`~orta.config.LoadingConfig` holds them.
    min_trains : int, optional
        Minimum services observed in a direction for the cell to be
        used.
    min_flow : float, optional
        Minimum summed boardings in a direction for the cell to be used.

    Returns
    -------
    pd.DataFrame
        Columns ``origin``, ``destination``, ``period``, ``flow``,
        ``trains``, ``load_per_train`` and the ``_reverse`` counterparts
        of the last three, restricted to ``origin < destination`` so
        each pair appears once.
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

    The regression target is ``y = ln(load_per_train /
    load_per_train_reverse)``, the quantity a loading hindcast is scored
    on, against ``ln(F_origin / F_destination)``. The frequency term is
    not fitted: it enters :meth:`DirectionSplitModel.share` with a fixed
    coefficient of 1 so that it exactly cancels the frequency divisor,
    and ``y`` is what remains after that cancellation.

    Parameters
    ----------
    cells : pd.DataFrame
        Output of :func:`build_cells`.
    footfall : dict[str, int]
        CRS -> annual entries/exits (table 1410).
    source : str
        Provenance string stored on the model.
    min_rows_per_period : int, optional
        A period fitted on fewer cells than this keeps the pooled
        coefficients rather than its own, so a thin period cannot
        contribute a wild slope.

    Returns
    -------
    DirectionSplitModel
        Fitted model, always carrying a :data:`POOLED` entry.
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

    :func:`calibrate` regresses on boardings at a train's first stop.
    That is the wrong target for a tidal flow: an Ely -> London morning
    train picks up most of its load after Ely, while a London -> Ely one
    boards everyone at the terminal, so first-stop boardings mostly
    measure the size of the origin station and the fit returns an
    am-peak slope of about zero. This fits the same model through the
    quantity case 06 is scored on instead: each segment's on-board load,
    the sum of every leg spanning it, predicted exactly as
    :func:`orta.demand.predict_demand` predicts it.

    Only the slopes ``b_p`` are fitted. The intercepts stay at zero,
    because a nonzero ``a_p`` enters both directions' logits with the
    same sign and so creates or destroys demand instead of moving it;
    the frequency term keeps its fixed coefficient of 1.

    Parameters
    ----------
    legs : pd.DataFrame
        One row per predicted leg, from a hindcast run with the split
        off: ``captured`` (annual trips per train), ``period``,
        ``ln_footfall_ratio`` (``ln(F_o / F_d)``, 0 where either is
        unknown) and ``ln_train_ratio`` (``ln(n_fwd / n_rev)`` with the
        candidate counted forward, NaN for a slot where the split does
        not fire).
    incidence : scipy.sparse.spmatrix
        Segments x legs, 1 where the leg is on board over the segment.
    observed : np.ndarray
        Observed on-board load per segment; segments with no positive
        observation are ignored.
    source : str
        Provenance string stored on the model.
    scale : float, optional
        Divides the summed annual trips down to the observed unit; for
        a typical day it is
        :data:`~orta.clock.WEEKDAYS_PER_YEAR`.
    station_function : dict[str, float] | None, optional
        Canonical CRS -> ``g``. When given, ``legs`` must also carry
        ``function_difference`` (``g_o - g_d``, NaN where either end has
        no ``g``), and a function slope is fitted per period beside the
        footfall slope; the table is stored on the model so it can be
        applied.

    Returns
    -------
    DirectionSplitModel
        Zero intercepts, a slope per period plus a :data:`POOLED` slope
        (and a :data:`FUNCTION_TERM` slope of each with
        ``station_function``); ``r_squared`` is the share of log-error
        variance removed in sample.
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
