"""A learned term in the slot share: a small network, in numpy.

:class:`ortla.network_assignment.SlotShareModel` shares a slot's demand
between its trains by ``exp(log w)``. With a network attached, each
train's log weight gains ``g(x)``, a small neural network (24 inputs, 16
tanh units, one output; three fits averaged) of the train's place in its
slot and its run, all read from the timetable:

* departure time (sine and cosine of the clock), position within the
  period and the period itself;
* whether the pair heads towards London, away from it or neither
  (within 150 km of London Bridge at either end);
* the leg's in-train minutes against the mean of its slot's trains, its
  calls, and the gaps to the slot's trains before and after it;
* the slot's train count, whether the leg starts at the train's origin
  or ends at its terminus, whether the train serves a London terminal,
  the leg's and the whole train's straight-line km, the train's calls,
  and the two stations' entries and exits.

Features that are the same for every train in a slot shift every weight
in it alike and cancel in the share; they matter only through the
network's interactions. It was trained through the whole model on the
counts: held out by corridor 0.4727 to 0.4610 and by operator 0.4816
to 0.4732, every operator better. Its base was the ``m3`` timing table
and theta 0.06 with the hourly profile (``slot_share_profile``), so use
it on that base.

Stations the model's table does not hold (one opened after 2024-25, say)
take the median entries and exits and count as neither towards nor away
from London, as in training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ortla.stations import LONDON_TERMINALS

PERIODS = ("early", "am_peak", "interpeak", "pm_peak", "evening")
# Minutes after midnight each period spans, for the position within it;
# the default cutoffs, which the network was trained on.
_SPAN = {
    "early": (240, 420),
    "am_peak": (420, 570),
    "interpeak": (570, 960),
    "pm_peak": (960, 1140),
    "evening": (1140, 1560),
}
FEATURES = (
    "dep_sin",
    "dep_cos",
    "in_period",
    *(f"p_{p}" for p in PERIODS),
    "towards",
    "away",
    "neither",
    "rel_ln_minutes",
    "ln_calls_od",
    "gap_prev",
    "gap_next",
    "ln_slot_trains",
    "o_is_origin",
    "d_is_terminus",
    "london_train",
    "ln_leg_km",
    "ln_train_km",
    "ln_train_calls",
    "ln_ee_o",
    "ln_ee_d",
)
NEAR_KM = 150.0  # of London Bridge, for towards/away
GAP_CAP = 120.0  # minutes
_LEG_KM, _TRAIN_KM = 20.0, 50.0  # stand-ins without coordinates


@dataclass(frozen=True)
class SlotShareNet:
    """The fitted network and the station table it reads."""

    mean: np.ndarray  # input standardisation
    scale: np.ndarray
    # (hidden weight, hidden bias, output weight, output bias) per fit.
    fits: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, float], ...]
    km_london: pd.Series  # by CRS
    ln_footfall: pd.Series  # log(1 + entries and exits), by CRS

    @classmethod
    def from_dict(cls, data: dict) -> SlotShareNet:
        """Build from the ``"net"`` entry of a slot share model file."""
        if list(data["features"]) != list(FEATURES):
            raise ValueError("slot share network: its features are not this version's")
        stations = pd.DataFrame(data["stations"], index=["km_london", "ln_footfall"]).T
        return cls(
            mean=np.asarray(data["mean"], dtype=float),
            scale=np.asarray(data["scale"], dtype=float),
            fits=tuple(
                (
                    np.asarray(f["hidden_weight"], dtype=np.float32),
                    np.asarray(f["hidden_bias"], dtype=np.float32),
                    np.asarray(f["out_weight"], dtype=np.float32),
                    np.float32(f["out_bias"]),
                )
                for f in data["fits"]
            ),
            km_london=stations["km_london"].astype(float),
            ln_footfall=stations["ln_footfall"].astype(float),
        )

    def to_dict(self) -> dict:
        """Return the dict :meth:`from_dict` reads."""
        return {
            "features": list(FEATURES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fits": [
                {
                    "hidden_weight": w.tolist(),
                    "hidden_bias": b.tolist(),
                    "out_weight": v.tolist(),
                    "out_bias": float(c),
                }
                for w, b, v, c in self.fits
            ],
            "stations": {
                k: [float(self.km_london[k]), float(self.ln_footfall[k])]
                for k in self.km_london.index
            },
        }

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """``g`` for each row of raw features (float32, as trained)."""
        z = ((np.asarray(x, dtype=np.float32).astype(float) - self.mean) / self.scale).astype(
            np.float32
        )
        total = np.zeros(len(z))
        for w, b, v, c in self.fits:
            total += (np.tanh(z @ w.T + b) @ v + c).astype(float)
        return total / len(self.fits)

    def features(
        self,
        frame: pd.DataFrame,
        period: str,
        listed: pd.DataFrame,
        coordinates: pd.DataFrame | None,
    ) -> np.ndarray:
        """Return the inputs for each row of ``frame``.

        ``frame`` and ``listed`` (the slots' trains) carry ``o``, ``d``
        (timetable codes, which define a slot), ``o_key``, ``d_key``,
        ``svc_o_key``, ``svc_d_key`` (the codes the station table and
        ``coordinates`` hold), ``svc_origin_crs``, ``svc_dest_crs``,
        ``dep_minutes`` and gap-filled ``minutes``; ``frame`` also has
        ``o_is_origin``, ``d_is_terminus``, ``calls_od`` and
        ``train_calls``. ``frame[period]`` is each row's slot period,
        ``listed["count_period"]`` each listed train's.
        """
        x = np.zeros((len(frame), len(FEATURES)), dtype=np.float32)
        col = {f: i for i, f in enumerate(FEATURES)}
        dep = frame["dep_minutes"].to_numpy(dtype=float)
        dep0 = np.nan_to_num(dep, nan=720.0)
        x[:, col["dep_sin"]] = np.sin(2 * np.pi * dep0 / 1440)
        x[:, col["dep_cos"]] = np.cos(2 * np.pi * dep0 / 1440)
        per = frame[period].to_numpy(dtype=object)
        for p in PERIODS:
            at = per == p
            x[at, col[f"p_{p}"]] = 1.0
            lo, hi = _SPAN[p]
            x[at, col["in_period"]] = np.clip((dep0[at] - lo) / (hi - lo), 0.0, 1.0)
        ok, dk = frame["o_key"].to_numpy(dtype=object), frame["d_key"].to_numpy(dtype=object)
        ko = self.km_london.reindex(ok).to_numpy()
        kd = self.km_london.reindex(dk).to_numpy()
        far = ~((ko <= NEAR_KM) | (kd <= NEAR_KM)) | np.isnan(ko) | np.isnan(kd)
        x[:, col["neither"]] = far
        x[:, col["towards"]] = ~far & (kd < ko)
        x[:, col["away"]] = ~far & ~(kd < ko)

        # The slot's listed trains: their count, mean log minutes and
        # departures, for the gaps either side of each row.
        slot_of = pd.MultiIndex.from_arrays(
            [listed["o"], listed["d"], listed["count_period"]]
        ).unique()
        slot = slot_of.get_indexer(pd.MultiIndex.from_arrays([frame["o"], frame["d"], per]))
        lslot = slot_of.get_indexer(
            pd.MultiIndex.from_arrays([listed["o"], listed["d"], listed["count_period"]])
        )
        n = len(slot_of)
        lnm = np.log1p(listed["minutes"].to_numpy(dtype=float))
        count = np.bincount(lslot, minlength=n).astype(float)
        mean_lnm = np.bincount(lslot, lnm, n) / np.maximum(count, 1)
        has = slot >= 0
        s0 = np.maximum(slot, 0)
        x[:, col["rel_ln_minutes"]] = np.where(
            has, np.log1p(frame["minutes"].to_numpy(dtype=float)) - mean_lnm[s0], 0.0
        )
        x[:, col["ln_slot_trains"]] = np.log(np.where(has, count[s0], 1.0))
        # Slot and departure in one sortable key; a missing departure
        # sits far before the day.
        ldep = np.nan_to_num(listed["dep_minutes"].to_numpy(dtype=float), nan=-1e4)
        keys = np.sort(lslot * 1e5 + ldep)
        k = s0 * 1e5 + np.nan_to_num(dep, nan=-1e4)
        left, right = np.searchsorted(keys, k, "left"), np.searchsorted(keys, k, "right")
        prev = np.where(left > 0, keys[np.maximum(left - 1, 0)], -np.inf)
        nxt = np.where(right < len(keys), keys[np.minimum(right, len(keys) - 1)], np.inf)
        same_prev = (prev // 1e5 == s0) & has
        same_next = (nxt // 1e5 == s0) & has
        x[:, col["gap_prev"]] = np.log1p(
            np.where(same_prev, np.minimum(k - prev, GAP_CAP), GAP_CAP)
        )
        x[:, col["gap_next"]] = np.log1p(
            np.where(same_next, np.minimum(nxt - k, GAP_CAP), GAP_CAP)
        )

        x[:, col["o_is_origin"]] = frame["o_is_origin"].to_numpy(dtype=bool)
        x[:, col["d_is_terminus"]] = frame["d_is_terminus"].to_numpy(dtype=bool)
        x[:, col["london_train"]] = (
            frame[["svc_origin_crs", "svc_dest_crs"]].isin(LONDON_TERMINALS).any(axis=1)
        )
        x[:, col["ln_calls_od"]] = np.log(np.maximum(frame["calls_od"].to_numpy(dtype=float), 1))
        x[:, col["ln_train_calls"]] = np.log(frame["train_calls"].to_numpy(dtype=float))
        leg_km = train_km = np.full(len(frame), np.nan)
        if coordinates is not None:
            from ortla.network_assignment import station_km  # noqa: PLC0415 (cycle)

            leg_km = station_km(coordinates, ok, dk)
            train_km = station_km(
                coordinates,
                frame["svc_o_key"].to_numpy(dtype=object),
                frame["svc_d_key"].to_numpy(dtype=object),
            )
        x[:, col["ln_leg_km"]] = np.log1p(np.nan_to_num(leg_km, nan=_LEG_KM))
        x[:, col["ln_train_km"]] = np.log1p(np.nan_to_num(train_km, nan=_TRAIN_KM))
        median = float(self.ln_footfall.median())
        x[:, col["ln_ee_o"]] = np.nan_to_num(self.ln_footfall.reindex(ok).to_numpy(), nan=median)
        x[:, col["ln_ee_d"]] = np.nan_to_num(self.ln_footfall.reindex(dk).to_numpy(), nan=median)
        return x
