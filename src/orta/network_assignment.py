"""Load every train in a day's timetable from the period matrices.

:func:`orta.demand.predict_demand` prices one candidate
train, leg by leg, against the timetable it would join; a hindcast
strips trains and re-adds them that way. This module is the base-year
version for the whole network in one pass. Every train stays in the
timetable, and a train's leg from ``o`` to ``d`` departing ``o`` in slot
``p`` carries

    D[o, d, p] / n[o, d, p]

where ``D`` is the period-matrix cell (annual trips for that weekday and
period) and ``n`` is the number of trains serving ``o -> d`` from ``o``
in ``p``: the count :func:`orta.service_lookup.od_services`
returns, with variants of one train merged as
:func:`orta.service_lookup._collapse_variants` merges them.
Nothing is stripped, so there is no ``(n + 1) / n`` term: these are the
loads the matrices imply for the timetable itself, which is where an
error in the matrices or in the assignment shows, free of anything a
strip adds.

The leg rules follow the per-train path so that the two agree leg for
leg:

* a leg's slot is classified from the departure at its first stop,
  falling back to the arrival
  (:func:`orta.demand.stop_time`);
* ``n`` counts a train in the slot of its public departure from the
  origin, as the service lookup does, so a train with no public
  departure there is in no slot's count;
* matrix lookups use the canonical CRS codes
  (:mod:`orta.stations`), train counts the timetable's own
  codes;
* a day's load is the annual figure over
  :data:`orta.clock.WEEKDAYS_PER_YEAR`, as in
  :func:`orta.demand.estimate_loading`.

Everything is vectorised over the day's legs (about a million on a
weekday), so the whole network loads in seconds where the per-train path
takes hours.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import pickle
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse

from orta import clock, models, od_profile, stations, timetable
from orta.clock import (
    PERIOD_MINUTES,
    UNKNOWN_PERIOD,
    WEEKDAYS,
    WEEKDAYS_PER_YEAR,
    cif_minutes,
    classify_minutes,
    period_matrix_path,
    period_minutes,
)
from orta.slot_net import SlotShareNet
from orta.stations import LONDON_TERMINALS, canonical_crs
from orta.timetable import SERVICE_KEY, filter_services

logger = logging.getLogger(__name__)
# The code that decides the one-change routes, part of TransferDemand's
# cache key.
_ROUTING_CODE = (__file__, clock.__file__, stations.__file__, timetable.__file__)

# What makes two listings of a train one train when counting a slot: the
# keys `service_lookup._collapse_variants` groups on (its `dep_hour`
# follows from the departure).
_VARIANT_KEYS = [
    "o",
    "d",
    "atoc_code",
    "signalling_id",
    "count_dep",
    "count_arr",
    "count_period",
    "svc_origin_crs",
    "svc_dest_crs",
]


def day_calls(calls: pd.DataFrame, reference_date: pd.Timestamp, weekday: str) -> pd.DataFrame:
    """One weekday's public passenger calls, in calling order.

    The schedules :func:`orta.service_lookup.day_services`
    selects (permanent, public, passenger, active on ``reference_date``
    and running on ``weekday``), each numbered as a ``service`` by the
    timetable's service key. A second listing of the same physical train
    (same operator, headcode, ends and calling pattern to the minute) is
    dropped here, so the train is loaded once; the slot counts merge
    such listings anyway.
    """
    day = filter_services(calls, reference_date, [WEEKDAYS.index(weekday)])
    day = day.assign(
        crs=day["crs"].astype(str).str.strip().str.upper(),
        seq=pd.to_numeric(day["seq"], errors="coerce"),
    ).dropna(subset=["seq"])
    key = day[SERVICE_KEY].astype("string").fillna("<NA>")
    day["service"] = pd.factorize(pd.MultiIndex.from_frame(key))[0]
    day = day.sort_values(["service", "seq"], kind="stable").reset_index(drop=True)

    pattern = (
        day["crs"]
        + "@"
        + day["public_arrival"].astype("string").fillna("")
        + "/"
        + day["public_departure"].astype("string").fillna("")
    )
    signature = (
        pattern.groupby(day["service"])
        .agg("|".join)
        .rename("pattern")
        .to_frame()
        .join(
            day.groupby("service")[
                ["atoc_code", "signalling_id", "svc_origin_crs", "svc_dest_crs"]
            ].first()
        )
    )
    repeat = signature.astype("string").fillna("").duplicated().to_numpy()
    if repeat.any():
        logger.info(
            "%s: %d schedules repeat another train's calling pattern exactly; loaded once.",
            weekday,
            int(repeat.sum()),
        )
        day = day[~day["service"].isin(signature.index[repeat])]
        day["service"] = pd.factorize(day["service"])[0]
        day = day.reset_index(drop=True)
    return day


def build_legs(
    calls: pd.DataFrame,
    cutoffs: dict[str, str],
    coordinates: pd.DataFrame | None = None,
    detour_base_minutes: float = 30.0,
    detour_minutes_per_km: float = 2.0,
) -> pd.DataFrame:
    """Every ordered pair of calls on every train: the rideable legs.

    ``calls`` is :func:`day_calls` output. One row per (train, i < j),
    with the row positions of both calls in ``calls``
    (``origin_row``/``dest_row``), the leg's slot (``period``), its
    in-vehicle minutes, and the fields the slot count uses
    (``count_*``). A leg whose first stop has no public time at all gets
    no slot, as the per-train path skips it.

    A train serves a pair once (:func:`one_leg_per_pair`): one that
    calls at a station twice (a loop, a reversal) would otherwise take
    two shares of a slot that counts it once. With ``coordinates``, legs
    slower than the detour limit are dropped where their slot has a
    faster option (:func:`drop_detour_legs`).
    """
    service = calls["service"].to_numpy()
    n = len(service)
    starts = np.flatnonzero(np.r_[True, service[1:] != service[:-1]])
    sizes = np.diff(np.r_[starts, n])
    position = np.arange(n) - np.repeat(starts, sizes)
    outgoing = np.repeat(sizes, sizes) - 1 - position
    origin = np.repeat(np.arange(n), outgoing)
    offsets = np.repeat(np.cumsum(outgoing) - outgoing, outgoing)
    dest = origin + 1 + (np.arange(len(origin)) - offsets)

    public_dep = cif_minutes(calls["public_departure"])
    public_arr = cif_minutes(calls["public_arrival"])
    first = position == 0
    # A candidate stop's time is its departure, else its arrival, except
    # at the first stop, which has no arrival
    # (`hindcast.build_candidates`).
    stop_time = np.where(np.isnan(public_dep) & ~first, public_arr, public_dep)
    arrive = np.where(np.isnan(public_arr), public_dep, public_arr)

    dep_min = stop_time[origin]
    arr_min = arrive[dest]
    minutes = arr_min - dep_min
    minutes = np.where(minutes < 0, minutes + 24 * 60, minutes)

    crs = calls["crs"].to_numpy(dtype=object)
    legs = pd.DataFrame(
        {
            "service": service[origin],
            "origin_row": origin,
            "dest_row": dest,
            "i": position[origin],
            "j": position[dest],
            "o": crs[origin],
            "d": crs[dest],
            "period": classify_minutes(dep_min, cutoffs),
            "dep_minutes": dep_min,
            "in_vehicle_minutes": np.where(minutes > 0, minutes, np.nan),
            # What the slot count reads: the public departure alone, and
            # the public arrival alone, as `service_lookup` lists a
            # train.
            "count_dep": public_dep[origin],
            "count_arr": public_arr[dest],
            "count_period": classify_minutes(public_dep[origin], cutoffs),
        }
    )
    for column in ("train_uid", "atoc_code", "signalling_id", "svc_origin_crs", "svc_dest_crs"):
        legs[column] = calls[column].to_numpy(dtype=object)[origin]
    legs = one_leg_per_pair(legs)
    if coordinates is not None:
        legs = drop_detour_legs(legs, coordinates, detour_base_minutes, detour_minutes_per_km)
    return legs


def one_leg_per_pair(legs: pd.DataFrame) -> pd.DataFrame:
    """Keep each train's leg with the fewest calls for each pair.

    Legs from a station back to itself are dropped first.

    A train calling at a station twice (a Cannon Street loop passes
    London Bridge going out and coming back; a reversal at Hampden Park)
    has two legs for some pairs. :func:`slot_trains` counts the train
    once for the pair, but each leg took the slot's D / n, so the train
    carried two shares (one of them the long way round) and the pair
    more trips than its cell holds. Case 19 (PLAN.md addendum L1):
    held-out rmse(log) 0.649 -> 0.646, 5/5 and 4/5 folds; loop trains'
    level 2.22 -> 1.96. Ties keep the earliest leg.
    """
    if legs.empty:
        return legs
    # A loop's two calls at one station are no journey (the matrix has
    # no such cell); the per-train path would otherwise price them with
    # the gravity model.
    legs = legs[legs["o"].to_numpy() != legs["d"].to_numpy()].reset_index(drop=True)
    span = (legs["j"] - legs["i"]).to_numpy()
    order = np.lexsort(
        (
            span,
            legs["d"].to_numpy(dtype=object).astype(str),
            legs["o"].to_numpy(dtype=object).astype(str),
            legs["service"].to_numpy(),
        )
    )
    repeat = legs.iloc[order][["service", "o", "d"]].duplicated().to_numpy()
    keep = np.zeros(len(legs), dtype=bool)
    keep[order[~repeat]] = True
    if keep.all():
        return legs
    logger.debug("one_leg_per_pair: %d legs of trains calling twice dropped.", int((~keep).sum()))
    return legs[keep].reset_index(drop=True)


def detour_limit(
    km: np.ndarray, base_minutes: float = 30.0, minutes_per_km: float = 2.0
) -> np.ndarray:
    """Most in-train minutes accepted for ``km`` straight-line km."""
    return base_minutes + minutes_per_km * km


def drop_detour_legs(
    legs: pd.DataFrame,
    coordinates: pd.DataFrame,
    base_minutes: float = 30.0,
    minutes_per_km: float = 2.0,
) -> pd.DataFrame:
    """Drop legs slower than :func:`detour_limit` if a slot has others.

    The limit is the one transfer routes already obey
    (:class:`TransferDemand`). It removes the long way round a loop (out
    on one side, back on the other), which took an equal share of the
    slot beside the direct run the other way. A slot whose every leg is
    slow keeps them all, so no pair loses its trains. Codes are the
    timetable's; a station without coordinates is never dropped. Case 19
    (addendum L2): held-out 0.646 -> 0.633, r 0.72 -> 0.76, loop trains'
    level 1.96 -> 1.10.
    """
    if legs.empty:
        return legs
    km = station_km(
        coordinates, legs["o"].to_numpy(dtype=object), legs["d"].to_numpy(dtype=object)
    )
    minutes = legs["in_vehicle_minutes"].to_numpy(dtype=float)
    slow = (
        np.isfinite(km)
        & np.isfinite(minutes)
        & (minutes > detour_limit(km, base_minutes, minutes_per_km))
    )
    if not slow.any():
        return legs
    slot = [
        legs["o"].to_numpy(dtype=object),
        legs["d"].to_numpy(dtype=object),
        legs["period"].to_numpy(dtype=object),
    ]
    has_fast = pd.Series(~slow).groupby(slot).transform("any").to_numpy()
    drop = slow & has_fast
    logger.info(
        "drop_detour_legs: %d legs slower than the detour limit dropped; %d kept as their"
        " slot's only option.",
        int(drop.sum()),
        int((slow & ~has_fast).sum()),
    )
    return legs[~drop].reset_index(drop=True)


def slot_trains(legs: pd.DataFrame) -> pd.Series:
    """Trains serving each ``(o, d, period)`` slot.

    A train is listed once per pair (its earliest call at ``o`` with the
    first later call at ``d``), in the slot of its public departure from
    ``o``, and listings that agree on :data:`_VARIANT_KEYS` are one
    train.
    """
    first = legs.drop_duplicates(["service", "o", "d"])  # legs run in (service, i, j) order
    listed = first[first["count_period"] != UNKNOWN_PERIOD]
    merged = listed.drop_duplicates(_VARIANT_KEYS)
    return merged.groupby(["o", "d", "count_period"]).size().rename("trains")


def trains_per_leg(legs: pd.DataFrame, slots: pd.Series | None = None) -> np.ndarray:
    """Return the ``n`` each leg's slot demand is shared over, or 0."""
    slots = slot_trains(legs) if slots is None else slots
    index = pd.MultiIndex.from_arrays([legs["o"], legs["d"], legs["period"]])
    return slots.reindex(index).fillna(0).to_numpy(dtype=float)


@dataclass
class SlotShareModel:
    """How a slot's demand divides between its trains, when not equally.

    A train's share is ``w / (sum of w over the slot's trains)``, with

        log w = -timing[period, departure half-hour, London-bound] -
        theta x minutes

    * ``timing`` (case 17's E5): the mean within-group log error of each
      departure half-hour, by period and whether the train is
      London-bound, fitted on counted train-segments. Keys are
      ``"period|half_hour|0 or 1"``; a missing key is 0.
    * ``theta`` (case 23's S3): a logit on the leg's in-train minutes,
      so a fast train takes more of a slot it shares with a stopper.
      Only differences within a slot matter; a train with no time takes
      the slot's mean.

    * an hourly term (case 39), when :meth:`with_profile` has attached
      an hourly OD profile: ``+ strength x log((S_h + k S) / ((1 + k)
      S))``, with ``S_h`` the leg's pair's profile demand in the hour
      the train leaves and ``S`` its mean hourly demand over the period.
      Pairs the profile does not cover get 0.

    * a learned term ``g(x)`` (case 117), when the model file carries a
      network (:mod:`orta.slot_net`): it reads the train's place in its
      slot and its run, so :func:`slot_shares` needs the day's calls.

    All weights 1 is the equal share, ``1 / n``. Fitted on held-out
    corridor folds in case 23 (``direction_split``-style JSON,
    :meth:`load`/:meth:`save`). The models in :mod:`orta.models` ship
    with the package: ``m3`` (cases 23 and 33) and ``neural`` (``m3``
    with case 117's network).
    """

    timing: dict[str, float]
    theta: float = 0.0
    source: str = ""
    hourly: pd.Series | None = None  # profile demand by (origin, destination, hour)
    hourly_mean: pd.Series | None = None  # mean hourly demand by (origin, destination, period)
    hourly_strength: float = 0.0
    hourly_smoothing: float = 2.0
    net: SlotShareNet | None = None

    @classmethod
    def load(cls, path: pathlib.Path | str) -> SlotShareModel:
        """Read a model written by :meth:`save`, or a built-in."""
        path = models.resolve(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            timing={k: float(v) for k, v in data["timing"].items()},
            theta=float(data["theta"]),
            source=data.get("source", ""),
            net=SlotShareNet.from_dict(data["net"]) if "net" in data else None,
        )

    def save(self, path: pathlib.Path) -> None:
        """Write the model as JSON."""
        data: dict[str, object] = {
            "timing": self.timing,
            "theta": self.theta,
            "source": self.source,
        }
        if self.net is not None:
            data["net"] = self.net.to_dict()
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def with_profile(
        self,
        path: pathlib.Path,
        cutoffs: dict[str, str],
        strength: float = 0.75,
        smoothing: float = 2.0,
    ) -> SlotShareModel:
        """Add the hourly term of an O03-style profile to a copy."""
        hourly, mean = od_profile.read_hourly(path, cutoffs)
        return replace(
            self,
            hourly=hourly,
            hourly_mean=mean,
            hourly_strength=strength,
            hourly_smoothing=smoothing,
        )

    def log_weight(
        self,
        periods: np.ndarray,
        dep_minutes: np.ndarray,
        london: np.ndarray,
        minutes: np.ndarray,
        origin: np.ndarray | None = None,
        destination: np.ndarray | None = None,
    ) -> np.ndarray:
        """``log w`` for each train, from gap-filled ``minutes``.

        ``origin`` and ``destination`` are the legs' stations, needed
        only for the hourly term.
        """
        dep = np.nan_to_num(np.asarray(dep_minutes, dtype=float), nan=-30)
        half_hour = (dep // 30).astype(int) % 48
        flags = np.asarray(london, dtype=bool).astype(int)
        # Plain dict lookups: a hindcast calls this for a handful of
        # trains at a time.
        timing = np.array(
            [
                self.timing.get(f"{p}|{h}|{f}", 0.0)
                for p, h, f in zip(periods, half_hour, flags, strict=True)
            ],
            dtype=float,
        )
        timing[np.isnan(timing)] = 0.0
        out = -timing - self.theta * np.asarray(minutes, dtype=float)
        if self.hourly is None or origin is None or not self.hourly_strength:
            return out
        s_h = (
            self.hourly.reindex(pd.MultiIndex.from_arrays([origin, destination, half_hour // 2]))
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        s_bar = (
            self.hourly_mean.reindex(pd.MultiIndex.from_arrays([origin, destination, periods]))
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        k = self.hourly_smoothing
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.log((s_h + k * s_bar) / ((1 + k) * s_bar))
        return out + self.hourly_strength * np.where((s_bar > 0) & np.isfinite(term), term, 0.0)


def _filled_minutes(frame: pd.DataFrame, period: str) -> pd.Series:
    """In-train minutes, a missing one the slot's mean, else 0."""
    slot = [
        frame["o"].to_numpy(dtype=object),
        frame["d"].to_numpy(dtype=object),
        frame[period].to_numpy(dtype=object),
    ]
    minutes = frame["in_vehicle_minutes"]
    return minutes.fillna(minutes.groupby(slot).transform("mean")).fillna(0.0)


def _net_columns(
    frame: pd.DataFrame, calls: pd.DataFrame, codes: dict[str, str], minutes: pd.Series
) -> pd.DataFrame:
    """Return the columns :meth:`SlotShareNet.features` reads.

    ``calls`` is the day the legs were built from, in service order;
    ``origin_row`` and ``dest_row`` index it.
    """
    service = calls["service"].to_numpy()
    first = np.searchsorted(service, service, "left")
    last = np.searchsorted(service, service, "right") - 1
    o_row, d_row = frame["origin_row"].to_numpy(), frame["dest_row"].to_numpy()
    return pd.DataFrame(
        {
            "o": frame["o"].to_numpy(dtype=object),
            "d": frame["d"].to_numpy(dtype=object),
            "o_key": canonical_codes(frame["o"], codes),
            "d_key": canonical_codes(frame["d"], codes),
            "svc_o_key": canonical_codes(frame["svc_origin_crs"], codes),
            "svc_d_key": canonical_codes(frame["svc_dest_crs"], codes),
            "svc_origin_crs": frame["svc_origin_crs"].to_numpy(dtype=object),
            "svc_dest_crs": frame["svc_dest_crs"].to_numpy(dtype=object),
            "period": frame["period"].to_numpy(dtype=object),
            "count_period": frame["count_period"].to_numpy(dtype=object),
            "dep_minutes": frame["dep_minutes"].to_numpy(dtype=float),
            "minutes": minutes.to_numpy(dtype=float),
            "o_is_origin": o_row == first[o_row],
            "d_is_terminus": d_row == last[d_row],
            "calls_od": d_row - o_row,
            "train_calls": last[o_row] - first[o_row] + 1,
        }
    )


def slot_shares(
    legs: pd.DataFrame,
    model: SlotShareModel | None = None,
    aliases: dict[str, str] | None = None,
    calls: pd.DataFrame | None = None,
    coordinates: pd.DataFrame | None = None,
) -> np.ndarray:
    """Each leg's share of its slot: ``1 / n`` or the weighted share.

    The slot's trains are those :func:`slot_trains` lists (every
    operator, variants merged), each weighted at its own departure and
    in-train minutes. A leg whose slot lists no train carries the slot
    alone (share 1), as :func:`leg_daily_trips` does. The hourly term
    looks the profile up by canonical codes (``aliases``, default the
    built-ins), as the matrices are. A model with a network needs the
    day's ``calls`` the legs were built from (:func:`day_calls`) and
    takes the stations' ``coordinates`` for its distances.
    """
    if model is None:
        return 1.0 / np.maximum(trains_per_leg(legs), 1.0)
    first = legs.drop_duplicates(["service", "o", "d"])
    listed = first[first["count_period"] != UNKNOWN_PERIOD].drop_duplicates(_VARIANT_KEYS)
    codes = stations.resolve_aliases(aliases)

    def log_weights(frame: pd.DataFrame, period: str) -> tuple[np.ndarray, pd.Series]:
        minutes = _filled_minutes(frame, period)
        lw = model.log_weight(
            frame[period].to_numpy(dtype=object),
            frame["dep_minutes"].to_numpy(dtype=float),
            frame["svc_dest_crs"].isin(LONDON_TERMINALS).to_numpy(),
            minutes.to_numpy(dtype=float),
            canonical_codes(frame["o"], codes),
            canonical_codes(frame["d"], codes),
        )
        return lw, minutes

    lw_listed, min_listed = log_weights(listed, "count_period")
    lw_legs, min_legs = log_weights(legs, "period")
    if model.net is not None:
        if calls is None:
            raise ValueError("a slot share model with a network needs the day's calls")
        net_listed = _net_columns(listed, calls, codes, min_listed)
        lw_listed = lw_listed + model.net(
            model.net.features(net_listed, "count_period", net_listed, coordinates)
        )
        lw_legs = lw_legs + model.net(
            model.net.features(
                _net_columns(legs, calls, codes, min_legs), "period", net_listed, coordinates
            )
        )
    total = (
        pd.Series(np.exp(lw_listed))
        .groupby(
            [
                listed["o"].to_numpy(dtype=object),
                listed["d"].to_numpy(dtype=object),
                listed["count_period"].to_numpy(dtype=object),
            ]
        )
        .sum()
    )
    index = pd.MultiIndex.from_arrays([legs["o"], legs["d"], legs["period"]])
    slot = total.reindex(index).to_numpy(dtype=float)
    own = np.exp(lw_legs)
    return np.where(slot > 0, own / np.where(slot > 0, slot, 1.0), 1.0)


class PeriodMatrices:
    """A folder of ``odm_{day}_{period}.csv`` matrices, as arrays.

    Each file is read once, on first use, into a float array with its
    origin and destination codes; cells that are not numbers (the
    suppressed ``!``) are NaN, as
    :func:`orta.demand.matrix_cell` treats them.
    """

    def __init__(self, directory: pathlib.Path) -> None:
        self.directory = directory
        self._arrays: dict[tuple[str, str], tuple[pd.Index, pd.Index, np.ndarray] | None] = {}

    def _load(self, day: str, period: str) -> tuple[pd.Index, pd.Index, np.ndarray] | None:
        key = (day, period)
        if key not in self._arrays:
            path = period_matrix_path(self.directory, day, period)
            if not path.exists():
                self._arrays[key] = None
            else:
                frame = pd.read_csv(path, index_col=0)
                text = frame.columns[frame.dtypes == "object"]
                if len(text):
                    frame[text] = frame[text].apply(pd.to_numeric, errors="coerce")
                values = frame.to_numpy(dtype=float)
                self._arrays[key] = (
                    pd.Index(frame.index.astype(str)),
                    pd.Index(frame.columns.astype(str)),
                    values,
                )
        return self._arrays[key]

    def cells(
        self, day: str, periods: np.ndarray, origins: np.ndarray, destinations: np.ndarray
    ) -> np.ndarray:
        """Annual trips per period and pair; NaN where absent."""
        out = np.full(len(origins), np.nan)
        for period in pd.unique(periods):
            loaded = self._load(day, str(period))
            if loaded is None:
                continue
            rows, cols, values = loaded
            mask = periods == period
            r = rows.get_indexer(origins[mask])
            c = cols.get_indexer(destinations[mask])
            found = (r >= 0) & (c >= 0)
            picked = np.full(mask.sum(), np.nan)
            picked[found] = values[r[found], c[found]]
            out[mask] = picked
        return out


class ChangeRoutes:
    """The best single change for each pair no train serves directly.

    Built from one day's legs: the fastest direct in-vehicle minutes
    between every two stations one train links, then for each routed
    pair the station ``via`` minimising
    ``minutes(o, via) + minutes(via, d)``. The change itself costs the
    same at every station, so it does not move the choice; it is added
    to ``minutes`` for reporting. Stations are the codes the legs are
    keyed by (``stations``); ``origin``/``via``/``destination`` index
    into them.
    """

    def __init__(
        self,
        leg_origins: np.ndarray,
        leg_destinations: np.ndarray,
        leg_minutes: np.ndarray,
        interchange_penalty: float,
    ) -> None:
        self.stations = pd.Index(np.unique(np.concatenate([leg_origins, leg_destinations])))
        n = len(self.stations)
        o = self.stations.get_indexer(leg_origins)
        d = self.stations.get_indexer(leg_destinations)
        ok = ~np.isnan(leg_minutes)
        fastest = pd.Series(leg_minutes[ok]).groupby(o[ok].astype(np.int64) * n + d[ok]).min()
        self.direct = np.full((n, n), np.inf, dtype=np.float32)
        self.direct.flat[fastest.index.to_numpy()] = fastest.to_numpy(dtype=np.float32)
        np.fill_diagonal(self.direct, np.inf)
        self.interchange_penalty = interchange_penalty
        self.origin = self.via = self.destination = np.empty(0, dtype=np.int64)
        self.minutes = np.empty(0)

    def route(self, origins: np.ndarray, destinations: np.ndarray) -> ChangeRoutes:
        """Find the best change for each pair with no direct train.

        Pairs a train does link, pairs with a station no train calls at,
        and pairs no single change joins are left out.
        """
        o = self.stations.get_indexer(origins)
        d = self.stations.get_indexer(destinations)
        known = (o >= 0) & (d >= 0) & (o != d)
        o, d = o[known], d[known]
        unserved = ~np.isfinite(self.direct[o, d])
        o, d = o[unserved], d[unserved]
        via = np.full(len(o), -1, dtype=np.int64)
        minutes = np.full(len(o), np.inf)
        order = np.argsort(o, kind="stable")
        starts = np.flatnonzero(np.r_[True, o[order][1:] != o[order][:-1]])
        for start, stop in zip(starts, np.r_[starts[1:], len(order)], strict=True):
            rows = order[start:stop]
            # Only stations one train reaches from the origin can be the
            # change; the rest are inf. They stay in index order, so
            # ties pick as before.
            reach = np.flatnonzero(np.isfinite(self.direct[o[rows[0]]]))
            if not len(reach):
                continue
            candidates = self.direct[o[rows[0]], reach][:, None] + self.direct[reach][:, d[rows]]
            best = candidates.argmin(axis=0)
            via[rows] = reach[best]
            minutes[rows] = candidates[best, np.arange(len(rows))]
        found = np.isfinite(minutes)
        self.origin, self.via, self.destination = o[found], via[found], d[found]
        self.minutes = minutes[found] + self.interchange_penalty
        return self

    def keep(self, mask: np.ndarray) -> ChangeRoutes:
        """Drop the routes where ``mask`` is False."""
        self.origin, self.via, self.destination = (
            self.origin[mask],
            self.via[mask],
            self.destination[mask],
        )
        self.minutes = self.minutes[mask]
        return self

    def leg_demand(self, trips: np.ndarray) -> np.ndarray:
        """Demand each station-to-station leg carries for routed pairs.

        ``trips`` is each route's demand; it rides both legs of its
        route. The result is a stations x stations array, added to the
        matrix cell of every leg it names before the slot's demand is
        shared over its trains.
        """
        n = len(self.stations)
        weights = np.nan_to_num(trips)
        return (
            np.bincount(self.origin * n + self.via, weights=weights, minlength=n * n)
            + np.bincount(self.via * n + self.destination, weights=weights, minlength=n * n)
        ).reshape(n, n)


class PeriodChangeRoutes:
    """The best single change, by period, for pairs with no direct.

    Case 17's T1. :class:`ChangeRoutes` routes a pair only if no train
    links it all day, so a pair with direct trains in some periods loses
    its demand in the others (1-5% of most operators' pairs, 10% of
    GWR's): a branch whose through train runs once in the evening gets
    none of the shuttle-and-change trips the rest of the day. Here each
    period has its own fastest direct minutes ``T_p`` and headways
    ``H_p`` (period length over the slot's trains), every pair with
    demand and no direct train in ``p`` is routed through the via
    minimising ``T_p(o, v) + T_p(v, d) + H_p(v, d) / 2`` plus the change
    penalty, among vias whose riding time is within the detour limit,
    and its trips load both legs in ``p``. Ties keep the lowest station
    index. Held out on corridors (case 17), rmse(log) 0.680 -> 0.654;
    case 23 on the rebuilt matrices, see that case's RESULTS.
    """

    def __init__(
        self,
        legs: pd.DataFrame,
        o_key: np.ndarray,
        d_key: np.ndarray,
        interchange_penalty: float,
        period_minutes: dict[str, float] | None = None,
    ) -> None:
        self.stations = pd.Index(np.unique(np.concatenate([o_key, d_key])))
        self.interchange_penalty = interchange_penalty
        n = len(self.stations)
        oi, di = self.stations.get_indexer(o_key), self.stations.get_indexer(d_key)
        slots = slot_trains(legs.assign(o=o_key, d=d_key))
        periods = legs["period"].to_numpy(dtype=object)
        minutes = legs["in_vehicle_minutes"].to_numpy(dtype=float)
        self._times: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for period, length in (period_minutes or PERIOD_MINUTES).items():
            at = (periods == period) & ~np.isnan(minutes)
            pair = oi[at].astype(np.int64) * n + di[at]
            fastest = pd.Series(minutes[at]).groupby(pair).min()
            t = np.full((n, n), np.inf, dtype=np.float32)
            t.flat[fastest.index.to_numpy()] = fastest.to_numpy(dtype=np.float32)
            np.fill_diagonal(t, np.inf)
            h = np.full((n, n), np.inf, dtype=np.float32)
            if period in slots.index.get_level_values(2):
                trains = slots.xs(period, level="count_period")
                ti = self.stations.get_indexer(trains.index.get_level_values(0))
                tj = self.stations.get_indexer(trains.index.get_level_values(1))
                h[ti, tj] = length / trains.to_numpy(dtype=float)
            # a direct train the slot count misses (first stop has only
            # an arrival): one train
            h = np.where(np.isfinite(t) & ~np.isfinite(h), length, h).astype(np.float32)
            self._times[period] = (t, h)
        self.routes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def route(
        self,
        origins: np.ndarray,
        destinations: np.ndarray,
        km: np.ndarray | None = None,
        detour_base_minutes: float = 30.0,
        detour_minutes_per_km: float = 2.0,
    ) -> PeriodChangeRoutes:
        """Route every pair with no direct train, period by period.

        ``km`` is each pair's straight-line distance (NaN: no limit).
        """
        o = self.stations.get_indexer(origins)
        d = self.stations.get_indexer(destinations)
        known = (o >= 0) & (d >= 0) & (o != d)
        o, d = o[known], d[known]
        limit = np.full(len(o), np.inf)
        if km is not None:
            k = np.asarray(km, dtype=float)[known]
            within = detour_limit(k, detour_base_minutes, detour_minutes_per_km)
            limit = np.where(np.isnan(k), np.inf, within)
        for period, (t, h) in self._times.items():
            unserved = ~np.isfinite(t[o, d])
            po, pd_, plimit = o[unserved], d[unserved], limit[unserved]
            boarding = t + h / 2  # the second train: its ride plus half its headway
            via = np.full(len(po), -1, dtype=np.int64)
            order = np.argsort(po, kind="stable")
            starts = np.flatnonzero(np.r_[True, po[order][1:] != po[order][:-1]])
            for start, stop in zip(starts, np.r_[starts[1:], len(order)], strict=True):
                rows = order[start:stop]
                reach = np.flatnonzero(np.isfinite(t[po[rows[0]]]))
                if not len(reach):
                    continue
                first = t[po[rows[0]], reach][:, None]
                ride = first + t[reach][:, pd_[rows]]
                cost = first + boarding[reach][:, pd_[rows]]
                cost[~np.isfinite(ride) | (ride > plimit[rows][None, :])] = np.inf
                best = cost.argmin(axis=0)
                found = np.isfinite(cost[best, np.arange(len(rows))])
                via[rows[found]] = reach[best[found]]
            ok = via >= 0
            self.routes[period] = (po[ok], via[ok], pd_[ok])
        return self

    def pairs(self, period: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the routed pairs' station codes in ``period``."""
        o, _, d = self.routes.get(period, (np.empty(0, int),) * 3)
        stations = self.stations.to_numpy(dtype=object)
        return stations[o], stations[d]

    def leg_demand(self, period: str, trips: np.ndarray) -> np.ndarray:
        """Station by station leg demand in ``period`` for ``trips``."""
        n = len(self.stations)
        o, via, d = self.routes.get(period, (np.empty(0, int),) * 3)
        weights = np.nan_to_num(trips)
        return (
            np.bincount(o * n + via, weights=weights, minlength=n * n)
            + np.bincount(via * n + d, weights=weights, minlength=n * n)
        ).reshape(n, n)

    def choose(
        self,
        origins: np.ndarray,
        destinations: np.ndarray,
        theta: float,
        km: np.ndarray | None = None,
        detour_base_minutes: float = 30.0,
        detour_minutes_per_km: float = 2.0,
    ) -> PeriodChangeRoutes:
        """Case 88: some riders with a direct train change by choice.

        For each period and each pair with a direct train, the best
        change (picked as :meth:`route` picks it) costs
        ``T(o, v) + T(v, d) + H(v, d) / 2`` plus the change penalty, the
        direct train ``T(o, d) + H(o, d) / 2``, and the pair's share
        changing is ``1 / (1 + exp(theta * (change - direct)))``. Stored
        per period in ``choices`` as (origin, via, destination, share).
        Held out by corridor (case 88), theta 0.2 took rmse(log) from
        0.492 to 0.481, better in every fold and for 7 of 9 operators.
        """
        o = self.stations.get_indexer(origins)
        d = self.stations.get_indexer(destinations)
        known = (o >= 0) & (d >= 0) & (o != d)
        o, d = o[known], d[known]
        limit = np.full(len(o), np.inf)
        if km is not None:
            k = np.asarray(km, dtype=float)[known]
            within = detour_limit(k, detour_base_minutes, detour_minutes_per_km)
            limit = np.where(np.isnan(k), np.inf, within)
        self.choices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for period, (t, h) in self._times.items():
            served = np.isfinite(t[o, d])
            po, pd_, plimit = o[served], d[served], limit[served]
            if not len(po):
                continue
            boarding = t + h / 2
            via = np.full(len(po), -1, dtype=np.int64)
            change = np.full(len(po), np.inf)
            order = np.argsort(po, kind="stable")
            starts = np.flatnonzero(np.r_[True, po[order][1:] != po[order][:-1]])
            for start, stop in zip(starts, np.r_[starts[1:], len(order)], strict=True):
                rows = order[start:stop]
                reach = np.flatnonzero(np.isfinite(t[po[rows[0]]]))
                if not len(reach):
                    continue
                first = t[po[rows[0]], reach][:, None]
                ride = first + t[reach][:, pd_[rows]]
                cost = first + boarding[reach][:, pd_[rows]]
                cost[~np.isfinite(ride) | (ride > plimit[rows][None, :])] = np.inf
                cost[reach[:, None] == pd_[rows][None, :]] = np.inf  # the destination is no via
                best = cost.argmin(axis=0)
                lowest = cost[best, np.arange(len(rows))]
                found = np.isfinite(lowest)
                via[rows[found]] = reach[best[found]]
                change[rows[found]] = lowest[found]
            ok = via >= 0
            direct = t[po, pd_] + h[po, pd_] / 2
            gap = change[ok] + self.interchange_penalty - direct[ok]
            share = 1.0 / (1.0 + np.exp(np.clip(theta * gap, -50.0, 50.0)))
            self.choices[period] = (po[ok], via[ok], pd_[ok], share)
        return self

    def choice_demand(
        self, period: str, trips: np.ndarray
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Leg trips gained from choice changes, and each pair's share.

        ``trips`` are the chosen pairs' cells, in the order of
        ``choices[period]``.
        """
        n = len(self.stations)
        o, via, d, share = self.choices.get(period, (np.empty(0, int),) * 3 + (np.empty(0),))
        moved = share * np.nan_to_num(trips)
        extra = sparse.csr_matrix(
            (np.r_[moved, moved], (np.r_[o, via], np.r_[via, d])), shape=(n, n)
        )
        return extra, sparse.csr_matrix((share, (o, d)), shape=(n, n))


def canonical_codes(codes: pd.Series, aliases: dict[str, str]) -> np.ndarray:
    """:func:`canonical_crs` over a column, one lookup per code."""
    mapping = {code: canonical_crs(code, aliases) for code in pd.unique(codes)}
    return codes.map(mapping).to_numpy(dtype=object)


def leg_daily_trips(annual: np.ndarray, trains: np.ndarray) -> np.ndarray:
    """Return a leg's trips on one day: its slot's cell per train.

    A slot with no counted train (a leg whose first stop has only an
    arrival time) is shared over the leg's own train alone.
    """
    return annual / np.maximum(trains, 1.0) / WEEKDAYS_PER_YEAR


def segment_loads(calls: pd.DataFrame, legs: pd.DataFrame, daily: np.ndarray) -> pd.DataFrame:
    """Boardings, alightings and departing load at every call.

    ``daily`` is each leg's trips on the day (:func:`leg_daily_trips`);
    a leg with NaN demand adds nothing but marks every segment it spans
    ``incomplete``, as :func:`orta.demand.estimate_loading`
    flags them. Returns one row per call that has a next call (a
    segment), for the trains that have legs in ``legs``.
    """
    n = len(calls)
    missing = np.isnan(daily)
    trips = np.where(missing, 0.0, daily)
    board = np.bincount(legs["origin_row"], weights=trips, minlength=n)
    alight = np.bincount(legs["dest_row"], weights=trips, minlength=n)
    gaps = np.bincount(legs["origin_row"], weights=missing, minlength=n) - np.bincount(
        legs["dest_row"], weights=missing, minlength=n
    )
    service = calls["service"].to_numpy()
    load = pd.Series(board - alight).groupby(service).cumsum().to_numpy()
    open_gaps = pd.Series(gaps).groupby(service).cumsum().to_numpy()

    nxt = np.r_[service[1:] == service[:-1], False]
    keep = nxt & np.isin(service, pd.unique(legs["service"]))
    rows = np.flatnonzero(keep)
    crs = calls["crs"].to_numpy(dtype=object)
    return pd.DataFrame(
        {
            "service": service[rows],
            "train_uid": calls["train_uid"].to_numpy(dtype=object)[rows],
            "atoc_code": calls["atoc_code"].to_numpy(dtype=object)[rows],
            "position": rows - np.searchsorted(service, service[rows]),
            "from_station": crs[rows],
            "to_station": crs[rows + 1],
            "boardings": board[rows],
            "alightings": alight[rows],
            "load": np.clip(load[rows], 0.0, None),
            "incomplete": open_gaps[rows] > 0.5,  # noqa: PLR2004 (counts held as floats)
        }
    )


def demand_pairs(
    matrices: PeriodMatrices, day: str, periods: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Every ordered pair with demand in any of ``day``'s periods."""
    rows = cols = total = None
    for period in periods:
        loaded = matrices._load(day, period)
        if loaded is None:
            continue
        r, c, values = loaded
        positive = np.nan_to_num(values) > 0
        if total is None:
            rows, cols, total = r, c, positive
        elif r.equals(rows) and c.equals(cols):
            total |= positive
        else:
            total |= (
                pd.DataFrame(positive, index=r, columns=c)
                .reindex(index=rows, columns=cols, fill_value=False)
                .to_numpy(dtype=bool)
            )
    if total is None:
        return np.empty(0, dtype=object), np.empty(0, dtype=object)
    r, c = np.nonzero(total)
    return rows.to_numpy(dtype=object)[r], cols.to_numpy(dtype=object)[c]


def station_km(
    coordinates: pd.DataFrame, origins: np.ndarray, destinations: np.ndarray
) -> np.ndarray:
    """Straight-line km between stations; NaN without coordinates.

    ``coordinates`` is indexed by CRS code with ``lat`` and ``lon``
    columns.
    """
    a = coordinates.reindex(origins)[["lat", "lon"]].to_numpy(dtype=float)
    b = coordinates.reindex(destinations)[["lat", "lon"]].to_numpy(dtype=float)
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


class TransferDemand:
    """Annual trips each leg carries for journeys with one change.

    A leg is priced from its own matrix cell, so a pair no single train
    serves reaches no train at all. This routes every such pair with
    demand through its best single change on one day's timetable
    (:class:`ChangeRoutes`), drops routes slower in the train than
    ``detour_base_minutes + detour_minutes_per_km`` x their
    straight-line km, and adds each route's period demand to both of its
    legs. :meth:`trips` is what a leg adds to its own cell before the
    slot's demand is shared over its trains. It is case 16's fix 2,
    which took held-out rmse(log) from 0.898 to 0.754 and the level from
    0.69 to 0.84. ``routing="period"`` routes each period on its own
    trains instead (:class:`PeriodChangeRoutes`, case 17's T1);
    ``"day"`` is the original rule. ``scale`` multiplies every routed
    trip, and ``london_scale`` (when set) replaces it on trains that
    start or end at a London terminal: case 33 found the routed volume
    too low, by 2.30 on London trains and 1.20 on the rest (fitted with
    a journey-time slot share of theta 0.06). 1.0 and None reproduce
    earlier runs.

    Route on the timetable *after* the change: a hindcast passes the
    full timetable, not the stripped one. Routed on the stripped one, a
    stripped train's own pairs would go round by a change and be counted
    twice once the train is re-added.
    """

    def __init__(
        self,
        calls: pd.DataFrame,
        reference_date: pd.Timestamp,
        cutoffs: dict[str, str],
        aliases: dict[str, str],
        matrices_dir: pathlib.Path,
        *,
        interchange_penalty: float,
        detour_base_minutes: float,
        detour_minutes_per_km: float,
        coordinates: pd.DataFrame | None = None,
        direct_detour_limit: bool = True,
        routing: str = "period",
        cache_dir: pathlib.Path | None = None,
        scale: float = 1.0,
        london_scale: float | None = None,
        choice_theta: float | None = None,
    ) -> None:
        if routing not in ("period", "day"):
            raise ValueError(f"routing must be 'period' or 'day', not {routing!r}")
        self.routing = routing
        self.scale = scale
        self.london_scale = scale if london_scale is None else london_scale
        # Case 88: pairs with a direct train also change by choice
        # (PeriodChangeRoutes.choose); their moved trips ride the change
        # legs unscaled. None (the default) routes as before.
        self.choice_theta = choice_theta if routing == "period" else None
        self.calls = calls
        self.direct_detour_limit = direct_detour_limit
        self.reference_date = reference_date
        self.cutoffs = cutoffs
        self.aliases = aliases
        self.matrices_dir = matrices_dir
        self.interchange_penalty = interchange_penalty
        self.detour_base_minutes = detour_base_minutes
        self.detour_minutes_per_km = detour_minutes_per_km
        self.coordinates = coordinates
        self.cache_dir = cache_dir
        self._calls_hash: str | None = None
        # Per day: station code -> row/column, and each period's leg
        # demand as a sparse stations x stations matrix (a few MB, where
        # a dense one or a dictionary of pairs runs to hundreds).
        self._days: dict[str, tuple[dict[str, int], dict[str, sparse.csr_matrix]]] = {}
        self.summary: list[dict[str, object]] = []

    def trips(
        self, day: str, period: str, origin: str, destination: str, london: bool = False
    ) -> float:
        """Routed trips riding ``origin -> destination`` in this slot.

        Codes are the matrices' (canonical) codes; 0 where nothing is
        routed. ``london``: the train starts or ends at a London
        terminal.
        """
        scale = self.london_scale if london else self.scale
        return scale * self._lookup(day, period, origin, destination) + self._lookup(
            day, f"choice:{period}", origin, destination
        )

    def direct_share(self, day: str, period: str, origin: str, destination: str) -> float:
        """Return the share of a pair's cell changing trains by choice.

        Zero when ``choice_theta`` is not set.
        """
        return self._lookup(day, f"share:{period}", origin, destination)

    def _lookup(self, day: str, key: str, origin: str, destination: str) -> float:
        if day not in self._days:
            self._days[day] = self._cached_build(day)
        index, by_period = self._days[day]
        matrix = by_period.get(key)
        r, c = index.get(origin), index.get(destination)
        if matrix is None or r is None or c is None:
            return 0.0
        # matrix[r, c] costs ~0.3 ms; a search of row r's sorted columns
        # is what it does.
        if not matrix.has_canonical_format:
            matrix.sum_duplicates()
        start, end = matrix.indptr[r], matrix.indptr[r + 1]
        k = start + np.searchsorted(matrix.indices[start:end], c)
        return float(matrix.data[k]) if k < end and matrix.indices[k] == c else 0.0

    def _cached_build(self, day: str) -> tuple[dict[str, int], dict[str, sparse.csr_matrix]]:
        """:meth:`_build`, read from ``cache_dir`` when already routed.

        A leave-one-out run starts a process per batch of trains, each
        routing the same full timetable on the same matrices; the key
        covers every input and the code.
        """
        if self.cache_dir is None:
            return self._build(day)
        path = self.cache_dir / f"transfers_{day}_{self._key(day)}.pkl"
        if path.exists():
            index, by_period, summary = pickle.loads(path.read_bytes())  # noqa: S301 (our own cache)
            self.summary.extend(summary)
            logger.info("%s: one-change routes read from %s.", day, path)
            return index, by_period
        n = len(self.summary)
        index, by_period = self._build(day)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(pickle.dumps((index, by_period, self.summary[n:])))
        tmp.replace(path)
        return index, by_period

    def _key(self, day: str) -> str:
        if self._calls_hash is None:
            hashed = pd.util.hash_pandas_object(self.calls, index=False).to_numpy()
            self._calls_hash = hashlib.sha256(
                hashed.tobytes() + repr(list(self.calls.columns)).encode()
            ).hexdigest()
        code = [pathlib.Path(f).read_bytes() for f in _ROUTING_CODE]
        files = sorted(pathlib.Path(self.matrices_dir).glob(f"odm_{day}_*.csv"))
        coordinates = (
            None
            if self.coordinates is None
            else pd.util.hash_pandas_object(self.coordinates).to_numpy().tobytes()
        )
        parts = [
            day,
            self._calls_hash,
            str(self.reference_date),
            sorted(self.cutoffs.items()),
            sorted(self.aliases.items()),
            [(str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files],
            self.interchange_penalty,
            self.detour_base_minutes,
            self.detour_minutes_per_km,
            self.direct_detour_limit,
            self.routing,
            coordinates,
            code,
            self.choice_theta,
        ]
        return hashlib.sha256(repr(parts).encode()).hexdigest()[:24]

    def _build(self, day: str) -> tuple[dict[str, int], dict[str, sparse.csr_matrix]]:
        legs = build_legs(
            day_calls(self.calls, self.reference_date, day),
            self.cutoffs,
            coordinates=self.coordinates if self.direct_detour_limit else None,
            detour_base_minutes=self.detour_base_minutes,
            detour_minutes_per_km=self.detour_minutes_per_km,
        )
        o_key = canonical_codes(legs["o"], self.aliases)
        d_key = canonical_codes(legs["d"], self.aliases)
        if self.routing == "period":
            return self._build_by_period(day, legs, o_key, d_key)
        routes = ChangeRoutes(
            o_key, d_key, legs["in_vehicle_minutes"].to_numpy(), self.interchange_penalty
        )
        periods = [p for p in pd.unique(legs["period"]) if p != UNKNOWN_PERIOD]
        matrices = PeriodMatrices(self.matrices_dir)
        pair_o, pair_d = demand_pairs(matrices, day, periods)
        routes.route(pair_o, pair_d)
        routed = len(routes.origin)
        stations = routes.stations.to_numpy(dtype=object)
        if self.coordinates is not None:
            km = station_km(
                self.coordinates, stations[routes.origin], stations[routes.destination]
            )
            riding = routes.minutes - self.interchange_penalty
            limit = self.detour_base_minutes + self.detour_minutes_per_km * km
            routes.keep(np.isnan(km) | (riding <= limit))
        self.summary.append(
            {
                "day": day,
                "pairs_with_demand": len(pair_o),
                "routed_one_change": routed,
                "dropped_as_detours": routed - len(routes.origin),
            }
        )
        route_o, route_d = stations[routes.origin], stations[routes.destination]
        by_period: dict[str, sparse.csr_matrix] = {}
        for period in periods:
            trips = matrices.cells(
                day, np.full(len(route_o), period, dtype=object), route_o, route_d
            )
            by_period[period] = sparse.csr_matrix(routes.leg_demand(trips))
        logger.info(
            "%s: %d pairs with demand need one change; %d kept within the detour limit.",
            day,
            routed,
            len(routes.origin),
        )
        return {code: k for k, code in enumerate(stations)}, by_period

    def _build_by_period(
        self, day: str, legs: pd.DataFrame, o_key: np.ndarray, d_key: np.ndarray
    ) -> tuple[dict[str, int], dict[str, sparse.csr_matrix]]:
        routes = PeriodChangeRoutes(
            legs, o_key, d_key, self.interchange_penalty, period_minutes(self.cutoffs)
        )
        periods = [p for p in pd.unique(legs["period"]) if p != UNKNOWN_PERIOD]
        matrices = PeriodMatrices(self.matrices_dir)
        pair_o, pair_d = demand_pairs(matrices, day, periods)
        km = None
        if self.coordinates is not None:
            km = station_km(self.coordinates, pair_o, pair_d)
        routes.route(pair_o, pair_d, km, self.detour_base_minutes, self.detour_minutes_per_km)
        by_period: dict[str, sparse.csr_matrix] = {}
        for period in periods:
            route_o, route_d = routes.pairs(period)
            periods_of = np.full(len(route_o), period, dtype=object)
            trips = matrices.cells(day, periods_of, route_o, route_d)
            by_period[period] = sparse.csr_matrix(routes.leg_demand(period, trips))
        if self.choice_theta is not None:
            routes.choose(
                pair_o,
                pair_d,
                self.choice_theta,
                km,
                self.detour_base_minutes,
                self.detour_minutes_per_km,
            )
            stations = routes.stations.to_numpy(dtype=object)
            for period in periods:
                o, _, d, _ = routes.choices.get(period, (np.empty(0, int),) * 4)
                trips = matrices.cells(
                    day, np.full(len(o), period, dtype=object), stations[o], stations[d]
                )
                extra, share = routes.choice_demand(period, trips)
                by_period[f"choice:{period}"], by_period[f"share:{period}"] = extra, share
        self.summary.append(
            {
                "day": day,
                "pairs_with_demand": len(pair_o),
                **{f"routed_{p}": len(routes.routes.get(p, ((),))[0]) for p in periods},
            }
        )
        logger.info(
            "%s: routed by period: %s.",
            day,
            {p: len(routes.routes.get(p, ((),))[0]) for p in periods},
        )
        stations = routes.stations.to_numpy(dtype=object)
        return {code: k for k, code in enumerate(stations)}, by_period
