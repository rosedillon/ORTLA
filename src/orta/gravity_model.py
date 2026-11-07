"""Calibrated gravity model of annual station-to-station rail journeys.

Fitted by ordinary least squares on every origin-destination pair in an
annual ODM that has journeys, footfall at both ends, a track distance
and at least one direct weekday train::

    ln J_ij = b0 + b1 ln F_i + b2 ln F_j + b3 ln d_ij + b4 ln f_ij

where ``J`` is annual journeys, ``F`` table-1410 entries/exits, ``d``
track miles and ``f`` direct weekday trains per day (from the
``timetable`` command's route summary).

Only directly served pairs are used. Fitted across all pairs, a
``ln(1 + f)`` term also absorbs the difference between pairs with and
without a direct train (mostly awkward, indirect journeys) and came out
at ~1.3, implying a 40-fold rise for a new 16-trains-a-day link. Among
served pairs the coefficient is ~0.6, at the upper end of published
service-frequency elasticities; it is still a cross-sectional
association (busy pairs attract trains as well as the reverse), reported
with the fit.

The model is used incrementally (see ``demand.calibrated_leg``):

* a pair already served: observed demand x
  ``(f_after / f_before) ** e``, where ``e`` is ``b4`` unless the caller
  passes a within-pair elasticity;
* a pair gaining its first direct train but with observed (indirect)
  demand: the share of that demand the new service wins on
  generalised journey time, floored by the served-pair prediction
  (by default its median rather than its mean), plus any market
  uplift for a new direct link;
* a pair with no observed demand (e.g. a new station): the prediction
  itself.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from orta.stations import recode_columns, resolve_recodes

if TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)

MODEL_FILENAME = "gravity_model.json"
_TERMS = ["intercept", "ln_footfall_origin", "ln_footfall_destination", "ln_distance", "ln_trains"]
_MIN_MILES = 1.0


@dataclasses.dataclass
class GravityModel:
    """Fitted coefficients and fit statistics."""

    coefficients: dict[str, float]
    n_pairs: int
    r_squared: float
    residual_std: float
    source_odm: str
    # Duan smearing factor, mean(exp(residual)): converts the log-model
    # median to a mean without assuming normal, constant-variance
    # residuals.
    smearing: float = 1.0

    @property
    def frequency_coefficient(self) -> float:
        """``b4``, the response to ``ln(direct trains/day)``."""
        return self.coefficients["ln_trains"]

    def predict(
        self, footfall_origin: float, footfall_destination: float, miles: float, trains: float
    ) -> float:
        """Annual journeys for a pair with ``trains`` trains a day.

        Log-linear OLS predicts the median, so the mean is recovered
        with Duan's smearing factor.
        """
        b = self.coefficients
        x = (
            b["intercept"]
            + b["ln_footfall_origin"] * np.log(footfall_origin)
            + b["ln_footfall_destination"] * np.log(footfall_destination)
            + b["ln_distance"] * np.log(max(miles, _MIN_MILES))
            + b["ln_trains"] * np.log(max(trains, 1.0))
        )
        return float(np.exp(x) * self.smearing)

    def with_frequency_coefficient(self, b4: float) -> GravityModel:
        """Copy the model with ``ln_trains`` set to ``b4``.

        Substituting one coefficient is not a refit: the intercept and
        the smearing factor were estimated with the fitted ``b4`` in
        place, so a synthetic prediction moves by
        ``(f ** (b4 - fitted))`` and carries that mis-specification. The
        pivot on an already-served pair (:meth:`frequency_ratio`, which
        is what the hindcasts exercise) is unaffected, because the
        intercept cancels in the ratio.
        """
        return dataclasses.replace(self, coefficients={**self.coefficients, "ln_trains": b4})

    def frequency_ratio(
        self, trains_before: float, trains_after: float, elasticity: float | None = None
    ) -> float:
        """Multiplier on a pair's journeys when its trains change.

        ``elasticity`` replaces ``b4`` as the exponent; None uses
        ``b4``.
        """
        if trains_before <= 0:
            raise ValueError("frequency_ratio needs an existing direct service.")
        e = self.frequency_coefficient if elasticity is None else elasticity
        return float((trains_after / trains_before) ** e)

    def save(self, path: pathlib.Path) -> None:
        """Write the model as JSON."""
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: pathlib.Path) -> GravityModel:
        """Read a model written by :meth:`save`."""
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def load_direct_frequency(route_summary_path: pathlib.Path | None) -> dict[tuple[str, str], float]:
    """Direct weekday trains a day per pair, from the route summary.

    Read from ``route_frequency_summary.csv``. Second codes are folded
    into their station's code first, with their trains added: the ODM
    holds the Elizabeth line's Paddington trains (PDX) and Reading's bay
    platforms (RDZ) under PAD and RDG, so read as their own codes those
    pairs lose the trains (see :mod:`orta.stations`).
    """
    if route_summary_path is None or not route_summary_path.exists():
        return {}
    df = pd.read_csv(route_summary_path, usecols=["origin_crs", "destination_crs", "daily_trains"])
    df = recode_columns(df, resolve_recodes(None), ["origin_crs", "destination_crs"])
    df = df[df["origin_crs"] != df["destination_crs"]]
    trains = df.groupby(["origin_crs", "destination_crs"])["daily_trains"].sum()
    return {pair: float(n) for pair, n in trains.items()}


def calibrate(
    typed_odm: pd.DataFrame,
    footfall: dict[str, int],
    frequency: dict[tuple[str, str], float],
    source_odm: str,
) -> GravityModel:
    """Fit the model on an ODM carrying a ``distance_miles`` column.

    Pairs are used when they have journeys, footfall at both ends, a
    positive track distance, distinct stations and at least one direct
    train a day.
    """
    df = typed_odm[["origin_tlc", "destination_tlc", "journeys", "distance_miles"]].copy()
    df["f_o"] = df["origin_tlc"].map(footfall)
    df["f_d"] = df["destination_tlc"].map(footfall)
    df["trains"] = [
        frequency.get((o, d), 0.0)
        for o, d in zip(df["origin_tlc"], df["destination_tlc"], strict=True)
    ]
    usable = (
        (df["journeys"] > 0)
        & (df["f_o"] > 0)
        & (df["f_d"] > 0)
        & (df["distance_miles"] >= _MIN_MILES)
        & (df["origin_tlc"] != df["destination_tlc"])
        & (df["trains"] > 0)
    )
    df = df[usable]
    x = np.column_stack(
        [
            np.ones(len(df)),
            np.log(df["f_o"].to_numpy(float)),
            np.log(df["f_d"].to_numpy(float)),
            np.log(df["distance_miles"].to_numpy(float)),
            np.log(df["trains"].to_numpy(float)),
        ]
    )
    y = np.log(df["journeys"].to_numpy(float))
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    r_squared = 1 - residuals.var() / y.var()
    model = GravityModel(
        coefficients=dict(zip(_TERMS, map(float, beta), strict=True)),
        n_pairs=len(df),
        r_squared=float(r_squared),
        residual_std=float(residuals.std(ddof=len(beta))),
        source_odm=source_odm,
        smearing=float(np.exp(residuals).mean()),
    )
    logger.info(
        "Gravity model fitted on %d pairs (R^2 %.3f, smearing %.2f): %s",
        model.n_pairs,
        model.r_squared,
        model.smearing,
        ", ".join(f"{k}={v:.3f}" for k, v in model.coefficients.items()),
    )
    return model
