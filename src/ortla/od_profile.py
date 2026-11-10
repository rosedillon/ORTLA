"""Reweight the period matrices towards an observed hourly OD profile.

The temporal factors give every pair of one journey type the same split
of its day over the periods, and the same split in both directions.
Network Rail's SRAM publishes, for every OD pair, demand by start hour
from mobile network data (output O03, ``OrigCRS``, ``DestCRS``,
``StartHour``, ``EndHour``, ``Demand``). This module moves each pair's
split across its ten cells of a day (two directions by five periods)
part of the way towards that profile and keeps the pair's day total, so
only when and which way people travel changes, not how many.

For a pair with our shares ``m`` and the profile's shares ``s`` over its
cells,

    s' = (S + kappa * m) / sum(S + kappa * m)
    f  = (s' / m) ** strength

where ``s'`` is the profile smoothed by ``kappa`` trips and ``f`` is
rescaled so ``sum(trips * f)`` is unchanged.

Strength 0 leaves the matrix as it is and 1 takes the profile's split.
Case 37 found 0.5 best, both against counted loads (rmse(log) 0.562 to
0.532, held out by corridor 0.558 to 0.519) and against AWC's hourly
gateline counts, where the profile on its own over-weights the evening
and ours the early and morning peak. Pairs the profile does not cover
keep their split. The profile describes a weekday, so weekends are left
alone.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")


def hour_shares(cutoffs: dict[str, str]) -> pd.DataFrame:
    """Fraction of each clock hour (rows) in each period (columns)."""
    bounds = [int(v.split(":")[0]) * 60 + int(v.split(":")[1]) for v in cutoffs.values()]
    labels = np.array(["early", *cutoffs], dtype=object)
    minutes = np.arange(24 * 60)
    period = labels[
        np.searchsorted(bounds, minutes, side="right")
    ]  # as network_assignment.classify_minutes
    t = pd.crosstab(minutes // 60, period) / 60.0
    t.index = t.index.astype(int)
    return t


def _by_hour(path: pathlib.Path) -> pd.DataFrame:
    """O03 demand by origin, destination and start hour.

    The unprofiled all-day rows (hours 0 to 23) are left out.
    """
    cols = ["OrigCRS", "DestCRS", "StartHour", "EndHour", "Demand"]
    path = pathlib.Path(path)
    o = (
        pd.read_parquet(path, columns=cols)
        if path.suffix == ".parquet"
        else pd.read_csv(path, usecols=cols)
    )
    o = o[~((o["StartHour"] == 0) & (o["EndHour"] == 23))]
    return o.groupby(["OrigCRS", "DestCRS", "StartHour"], as_index=False)["Demand"].sum()


def _by_period(o: pd.DataFrame, cutoffs: dict[str, str]) -> pd.DataFrame:
    """Hourly rows spread over the periods by their share ``w``."""
    long = hour_shares(cutoffs).stack().rename("w").reset_index()  # noqa: PD013
    long.columns = ["StartHour", "period", "w"]
    o = o.merge(long[long["w"] > 0], on="StartHour")
    o["Demand"] *= o["w"]
    return o


def read_profile(path: pathlib.Path, cutoffs: dict[str, str]) -> pd.Series:
    """Profile demand by pair and period from an O03-style file.

    Rows starting at hour 0 and ending at 23 are SRAM's all-day demand
    with no profile, and are dropped. An hour a period boundary cuts
    (9:30, say) is shared by its minutes.
    """
    out = (
        _by_period(_by_hour(path), cutoffs)
        .groupby(["OrigCRS", "DestCRS", "period"])["Demand"]
        .sum()
    )
    out.index.names = ["origin", "destination", "period"]
    return out


def read_hourly(path: pathlib.Path, cutoffs: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    """Profile demand by pair and hour, and its mean per period hour.

    The mean is over the period's minutes, so an hour a boundary cuts
    counts by its share. These feed the slot share's hourly term
    (:class:`ortla.network_assignment.SlotShareModel`).
    """
    o = _by_hour(path)
    by_hour = o.set_index(["OrigCRS", "DestCRS", "StartHour"])["Demand"]
    by_hour.index.names = ["origin", "destination", "hour"]
    hours = hour_shares(cutoffs).sum()  # hours in each period
    t = _by_period(o, cutoffs).groupby(["OrigCRS", "DestCRS", "period"])["Demand"].sum()
    mean = t / hours.reindex(t.index.get_level_values("period")).to_numpy()
    mean.index.names = ["origin", "destination", "period"]
    return by_hour, mean


def reweight(
    cells: pd.DataFrame, profile: pd.Series, strength: float = 0.5, kappa: float = 10.0
) -> np.ndarray:
    """Multiplier for each row of one day's ``cells``.

    ``cells`` has origin, destination, period and trips columns.

    Each undirected pair's rows are moved together and keep their summed
    trips.
    """
    o = cells["origin"].to_numpy(object)
    d = cells["destination"].to_numpy(object)
    pair = np.where(o < d, o + ">" + d, d + ">" + o)
    trips = np.nan_to_num(cells["trips"].to_numpy(float))
    s = profile.reindex(
        pd.MultiIndex.from_frame(cells[["origin", "destination", "period"]])
    ).fillna(0.0)
    s = s.to_numpy(float)

    def total(v: np.ndarray) -> np.ndarray:
        return pd.Series(v).groupby(pair).transform("sum").to_numpy()

    t_pair, s_pair = total(trips), total(s)
    m = np.divide(trips, t_pair, out=np.zeros_like(trips), where=t_pair > 0)
    s = s + kappa * m
    s_sum = total(s)
    target = np.divide(s, s_sum, out=m.copy(), where=s_sum > 0)
    f = np.ones_like(trips)
    move = (m > 0) & (s_pair > 0)
    f[move] = (np.maximum(target[move], 1e-12) / m[move]) ** strength
    after = total(trips * f)
    return f * np.divide(t_pair, after, out=np.ones_like(trips), where=after > 0)
