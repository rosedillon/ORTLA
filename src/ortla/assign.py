"""Load every train in the timetable: the national assignment.

For each weekday, every leg of every train (each ordered pair of its
calls) carries

    (cell x (1 - c) + chosen + lambda x routed) x split x share / days

* ``cell``: the period matrix's annual trips for the leg's pair and
  slot, from the matrices of the train's operator;
* ``routed``: trips of pairs with no direct train in the slot, routed
  through their best change (:class:`~ortla.network_assignment.
  PeriodChangeRoutes`), scaled by ``lambda``, the London or other
  transfer scale by whether the train serves a London terminal;
* ``c`` and ``chosen``: the share of a served pair that changes trains
  by choice when that is quicker, and the trips other pairs move onto
  the leg that way (``transfer_choice_theta``);
* ``split``: the direction split on the slot's trains each way and the
  footfall of the two ends (``direction_split_model``);
* ``share``: the leg's share of its slot's trains
  (``slot_share_model``; equal shares without one);
* ``days``: :data:`ortla.clock.WEEKDAYS_PER_YEAR`.

A leg's trips board at its first call and alight at its last, so a
train's load at each call is the running sum. With the adopted settings
(``examples/assign.yml``; ``reference/configs/reference.yml`` in full) this is the reference
model, held out by corridor at rmse(log) 0.461 on nine operators' counts.

Trains of operators named in ``operators`` load from their own
matrices; any other train loads from ``other_period_matrices_dir`` when
set, and is left out otherwise. Operators with ``counts`` are scored
against them, train by train and call by call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ortla import loading_data, models
from ortla.clock import WEEKDAYS, WEEKDAYS_PER_YEAR, cif_hhmm, period_minutes
from ortla.direction_split import DirectionSplitModel, split_multiplier
from ortla.network_assignment import (
    PeriodChangeRoutes,
    PeriodMatrices,
    SlotShareModel,
    build_legs,
    canonical_codes,
    day_calls,
    demand_pairs,
    drop_joint_legs,
    fold_joint_segments,
    merge_portions,
    segment_loads,
    slot_shares,
    slot_trains,
    station_km,
)
from ortla.stations import (
    LONDON_TERMINALS,
    load_table_1410,
    recode_columns,
    resolve_aliases,
    resolve_recodes,
    tiploc_to_crs_map,
)
from ortla.timetable import read_public_calls

if TYPE_CHECKING:
    from ortla.config import AssignConfig

logger = logging.getLogger(__name__)
OTHER = "other"  # the group of trains no listed operator runs


def split_factors(
    legs: pd.DataFrame,
    o_key: np.ndarray,
    d_key: np.ndarray,
    model: DirectionSplitModel,
    footfall: dict[str, int],
    periods: list[str],
) -> np.ndarray:
    """Each leg's direction split factor on its slot's demand.

    :func:`ortla.direction_split.split_multiplier` on the trains each way
    in the slot and the footfall ratio of the two ends (canonical
    codes); 1 where either direction has no train. A leg with no period
    takes the pooled slope.
    """
    slots = slot_trains(legs)
    n_fwd = slots.reindex(pd.MultiIndex.from_arrays([legs["o"], legs["d"], legs["period"]]))
    n_rev = slots.reindex(pd.MultiIndex.from_arrays([legs["d"], legs["o"], legs["period"]]))
    n_fwd, n_rev = n_fwd.fillna(0).to_numpy(float), n_rev.fillna(0).to_numpy(float)
    f_o = pd.Series(o_key).map(footfall).to_numpy(float)
    f_d = pd.Series(d_key).map(footfall).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_trains = np.where((n_fwd > 0) & (n_rev > 0), np.log(n_fwd / n_rev), np.nan)
        ln_footfall = np.where((f_o > 0) & (f_d > 0), np.log(f_o / f_d), 0.0)
    slopes = np.array([model.footfall_slope(p) for p in periods] + [model.footfall_slope(None)])
    index = legs["period"].map({p: i for i, p in enumerate(periods)}).fillna(len(periods))
    return split_multiplier(slopes, index.to_numpy(int), ln_footfall, ln_trains)


def _associations(config: AssignConfig, weekday: str) -> pd.DataFrame:
    """Return the joins and divides running on ``weekday``, by the schedules' rule."""
    associations = pd.read_csv(config.associations_path, dtype=str)
    if "start_date" in associations:  # as `timetable` writes it
        on = config.reference_timestamp
        associations = associations[
            associations["CIF_stp_indicator"].eq("P")
            & (pd.to_datetime(associations["start_date"]) <= on)
            & (pd.to_datetime(associations["end_date"]) >= on)
        ]
    if "days" in associations:  # CIF days-run mask, Monday first
        associations = associations[associations["days"].str[WEEKDAYS.index(weekday)] == "1"]
    return recode_columns(associations, resolve_recodes(config.station_recodes), ["crs"])


def assign_day(  # noqa: C901, PLR0912
    calls: pd.DataFrame,
    weekday: str,
    config: AssignConfig,
    *,
    share_model: SlotShareModel | None,
    split: DirectionSplitModel | None,
    footfall: dict[str, int],
    coordinates: pd.DataFrame | None,
) -> pd.DataFrame:
    """Every loaded train's segments on ``weekday``, with operator.

    Demand is the fixed base-year matrix, so this loads a timetable; it does not forecast a
    change to it. Cut trains and their passengers move to the trains left, and a pair that
    loses its direct train changes instead, adding a boarding. The direct and
    changing journeys are logged and kept in ``attrs`` (``journeys_direct``,
    ``journeys_changing``); for a service change use ``propose`` or ``scenario``.
    """
    cutoffs = config.time_periods_cutoffs
    periods = ["early", *cutoffs]
    aliases = resolve_aliases(config.station_aliases)
    day = day_calls(calls, config.reference_timestamp, weekday)
    portions = None
    if config.associations_path is not None:
        day, portions = merge_portions(day, _associations(config, weekday))
    legs = build_legs(
        day,
        cutoffs,
        coordinates=coordinates if config.direct_detour_limit else None,
        detour_base_minutes=config.transfer_detour_base_minutes,
        detour_minutes_per_km=config.transfer_detour_minutes_per_km,
    )
    if config.portion_counts == "joined":
        legs = drop_joint_legs(day, legs)
    o_key, d_key = canonical_codes(legs["o"], aliases), canonical_codes(legs["d"], aliases)
    leg_period = legs["period"].to_numpy(dtype=object)
    factors = (
        split_factors(legs, o_key, d_key, split, footfall, periods)
        if split is not None
        else np.ones(len(legs))
    )
    groups = config.operator_groups()
    if not groups:
        raise ValueError("assign: name at least one operator or other_period_matrices_dir")
    atoc = legs["atoc_code"].to_numpy(dtype=object)
    listed = {c for _, codes, _ in groups for c in codes}

    routes = None
    if config.transfer_assignment:
        pair_o, pair_d = demand_pairs(PeriodMatrices(groups[0][2]), weekday, periods)
        km = station_km(coordinates, pair_o, pair_d) if coordinates is not None else None
        routes = PeriodChangeRoutes(
            legs, o_key, d_key, config.interchange_penalty_minutes, period_minutes(cutoffs)
        )
        detour = (config.transfer_detour_base_minutes, config.transfer_detour_minutes_per_km)
        routes.route(pair_o, pair_d, km, *detour)
        if config.transfer_choice_theta is not None:
            routes.choose(pair_o, pair_d, config.transfer_choice_theta, km, *detour)
    shares = slot_shares(legs, share_model, aliases, calls=day, coordinates=coordinates)
    london = legs[["svc_origin_crs", "svc_dest_crs"]].isin(LONDON_TERMINALS).any(axis=1)
    scale_london = (
        config.transfer_scale
        if config.transfer_scale_london is None
        else config.transfer_scale_london
    )
    lam = np.where(london.to_numpy(), scale_london, config.transfer_scale)

    daily = np.full(len(legs), np.nan)
    direct = np.zeros(len(legs))
    operator = np.full(len(legs), "", dtype=object)
    for name, codes, directory in groups:
        k = np.flatnonzero(~np.isin(atoc, list(listed)) if name == OTHER else np.isin(atoc, codes))
        if not len(k):
            continue
        operator[k] = name
        matrices = PeriodMatrices(directory, config.odm_open_share)
        raw = matrices.cells(weekday, leg_period[k], o_key[k], d_key[k])
        cell, has = np.nan_to_num(raw), ~np.isnan(raw)
        routed, changing, chosen = np.zeros(len(k)), np.zeros(len(k)), np.zeros(len(k))
        if routes is not None:
            oi = routes.stations.get_indexer(o_key[k])
            di = routes.stations.get_indexer(d_key[k])
            names = routes.stations.to_numpy(dtype=object)
            for p in periods:
                at = (leg_period[k] == p) & (oi >= 0) & (di >= 0)
                if not at.any():
                    continue
                if p in routes.routes:
                    ro, rd = routes.pairs(p)
                    trips = matrices.cells(weekday, np.full(len(ro), p, dtype=object), ro, rd)
                    routed[at] = routes.leg_demand(p, trips)[oi[at], di[at]]
                if p in getattr(routes, "choices", {}):
                    co, _, cd, _ = routes.choices[p]
                    trips = matrices.cells(
                        weekday, np.full(len(co), p, dtype=object), names[co], names[cd]
                    )
                    extra, share = routes.choice_demand(p, trips)
                    chosen[at] = np.asarray(extra[oi[at], di[at]]).ravel()
                    changing[at] = np.asarray(share[oi[at], di[at]]).ravel()
        scale = factors[k] * shares[k] / WEEKDAYS_PER_YEAR
        keep = has | (chosen > 0)
        value = (cell * (1 - changing) + chosen + lam[k] * routed) * scale
        daily[k] = np.where(keep, value, np.nan)
        direct[k] = np.where(keep, cell * (1 - changing) * scale, 0.0)
    loaded = operator != ""
    segments = segment_loads(day, legs[loaded], np.nan_to_num(daily)[loaded])
    trains = pd.Series(operator[loaded]).groupby(legs["service"][loaded].to_numpy()).first()
    segments["operator"] = trains.reindex(segments["service"]).to_numpy()
    # The train's ends and first departure, as the count feeds give
    # them, for matching counts to trains.
    first = day.groupby("service").head(1).set_index("service")
    last = day.groupby("service").tail(1).set_index("service")
    service = segments["service"]
    departures = day["public_departure"].astype("string").fillna("").to_numpy()
    rows = np.searchsorted(day["service"].to_numpy(), service.to_numpy())
    segments["departure"] = [
        cif_hhmm(t) for t in departures[rows + segments["position"].to_numpy()]
    ]
    segments["headcode"] = first["signalling_id"].reindex(service).to_numpy()
    segments["train_origin"] = first["crs"].reindex(service).to_numpy()
    segments["train_destination"] = last["crs"].reindex(service).to_numpy()
    start = first["public_departure"].astype("string").fillna("").reindex(service)
    segments["train_departure"] = [cif_hhmm(t) for t in start]
    if portions is not None and not portions.empty:
        # A rebuilt portion keeps its own ends, as the count feeds give them.
        own = segments["service"].isin(portions.index)
        ends = portions.reindex(segments.loc[own, "service"])
        segments.loc[own, "train_origin"] = ends["origin"].to_numpy()
        segments.loc[own, "train_destination"] = ends["destination"].to_numpy()
        segments.loc[own, "train_departure"] = [
            cif_hhmm(t)
            for t in pd.Series(ends["departure"].to_numpy()).astype("string").fillna("")
        ]
        if config.portion_counts == "joined":
            segments = fold_joint_segments(day, segments, portions)
    on_direct = float(direct[loaded].sum())
    on_change = float(np.nan_to_num(daily)[loaded].sum()) - on_direct
    logger.info(
        "%s: %d trains loaded, %d legs, %.0f boardings, about %.0f journeys (%.0f direct,"
        " %.0f changing once). Demand is the fixed matrix: a service cut moves journeys"
        " to other trains and changes, it does not lose them.",
        weekday,
        segments["service"].nunique(),
        int(loaded.sum()),
        segments["boardings"].sum(),
        on_direct + on_change / 2,
        on_direct,
        on_change / 2,
    )
    out = segments.assign(day=weekday)
    out.attrs["journeys_direct"], out.attrs["journeys_changing"] = on_direct, on_change / 2
    return out


def match_counts(
    counts: pd.DataFrame, segments: pd.DataFrame, atoc_codes: list[str]
) -> pd.DataFrame:
    """One operator's counts averaged per train, weekday and call.

    Matched by CIF UID where the feed carries one the timetable knows,
    else by headcode, ends and origin departure; a key two trains share
    is not used. Columns ``train_uid``, ``day``, ``from_station``,
    ``actual_load``, ``actual_on``, ``actual_off``, ``n_periods``.
    """
    trains = (
        segments[segments["atoc_code"].isin(atoc_codes)]
        .drop_duplicates(["day", "service"])
        .assign(
            key=lambda f: (
                f["headcode"].astype(str)
                + "|"
                + f["train_origin"].astype(str)
                + "|"
                + f["train_destination"].astype(str)
                + "|"
                + f["train_departure"].astype(str)
            )
        )
    )
    frames = []
    for weekday, own in trains.groupby("day"):
        rows = counts[counts["Service Week Day"] == weekday]
        by_uid = rows["Unique ID"].isin(set(own["train_uid"]))
        unique = own[~own["key"].duplicated(keep=False)].set_index("key")["train_uid"]
        rest = rows[~by_uid]
        rest_key = (
            rest["Headcode"].astype(str)
            + "|"
            + rest["Origin CRS"].astype(str)
            + "|"
            + rest["Destination CRS"].astype(str)
            + "|"
            + rest["Origin Departure Time"].astype("string").fillna("").str[:5]
        )
        by_key = rest_key.map(unique)
        matched = pd.concat(
            [
                rows[by_uid].assign(train_uid=rows.loc[by_uid, "Unique ID"]),
                rest[by_key.notna()].assign(train_uid=by_key[by_key.notna()]),
            ]
        )
        # A train calling at a station twice (a loop, a reversal) has a
        # count for each call: number the visits by call order so each
        # call keeps its own count rather than the mean of both.
        sequence = pd.to_numeric(matched["Call Sequence"], errors="coerce")
        matched = matched.assign(
            visit=sequence.groupby(
                [matched["train_uid"], matched["RSP Period"], matched["Departure from CRS"]]
            )
            .rank(method="dense")
            .fillna(1)
            .astype(int)
            - 1
        )
        frames.append(
            matched.groupby(["train_uid", "Departure from CRS", "visit"])
            .agg(
                actual_load=("Mean Load on Departure", "mean"),
                actual_on=("Mean Passengers On", "mean"),
                actual_off=("Mean Passengers Off", "mean"),
                n_periods=("Mean Load on Departure", "count"),
            )
            .reset_index()
            .rename(columns={"Departure from CRS": "from_station"})
            .assign(day=weekday)
        )
    columns = [
        "train_uid",
        "from_station",
        "visit",
        "actual_load",
        "actual_on",
        "actual_off",
        "n_periods",
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def score(frame: pd.DataFrame) -> dict[str, float]:
    """Accuracy on segments with a count above zero.

    ``ratio`` is total predicted over total counted; ``rmse_log`` the
    root mean square of log(predicted / counted) where both are above
    zero; ``geh5_pct`` the share with GEH below 5.
    """
    f = frame[(frame["actual_load"] > 0) & frame["load"].notna()]
    p, a = f["load"].to_numpy(float), f["actual_load"].to_numpy(float)
    positive = p > 0
    geh = np.sqrt(2 * (p - a) ** 2 / np.maximum(p + a, 1e-9))
    return {
        "segments": len(f),
        "trains": f.groupby(["train_uid", "day"]).ngroups,
        "ratio": float(p.sum() / a.sum()) if len(f) else np.nan,
        "rmse_log": float(np.sqrt(np.mean(np.log(p[positive] / a[positive]) ** 2)))
        if positive.any()
        else np.nan,
        "zero_predictions": int((~positive).sum()),
        "r": float(np.corrcoef(p, a)[0, 1]) if len(f) > 1 else np.nan,
        "geh5_pct": float(np.mean(geh < 5) * 100) if len(f) else np.nan,
    }


def _coordinates(config: AssignConfig) -> pd.DataFrame | None:
    if config.stations is None:
        return None
    frame = pd.read_csv(config.stations.path)
    coordinates = pd.DataFrame(
        {
            "lat": pd.to_numeric(frame[config.stations.lat], errors="coerce").to_numpy(),
            "lon": pd.to_numeric(frame[config.stations.lon], errors="coerce").to_numpy(),
        },
        index=frame[config.stations.crs].astype(str).str.strip().str.upper(),
    )
    return coordinates[~coordinates.index.duplicated()]


def _share_model(config: AssignConfig) -> SlotShareModel | None:
    if config.slot_share_model is None:
        return None
    model = SlotShareModel.load(config.slot_share_model)
    if config.slot_share_profile is not None:
        model = model.with_profile(
            config.slot_share_profile,
            config.time_periods_cutoffs,
            config.slot_share_profile_strength,
            config.slot_share_profile_smoothing,
        )
    elif model.net is not None:
        logger.warning(
            "The neural slot share was fitted with slot_share_profile on; without it the"
            " network sits on a different base."
        )
    return model


def run_assign(config: AssignConfig) -> None:
    """Load every train, write the tables and score against counts.

    Writes to ``config.output_path``:

    * ``assign_segments.csv``: load, boardings and alightings at every
      call of every loaded train, each weekday (with the count where
      matched)
    * ``assign_trains.csv``: each train's peak and mean load and
      boardings per weekday
    * ``assign_stations.csv``: boardings and alightings by station and
      weekday
    * ``assign_links.csv``: passengers and trains on each link between
      consecutive calls, by weekday
    * ``assign_scores.csv``: accuracy against counts, by operator and in
      all (when any operator has ``counts``)
    """
    out = config.output_path
    out.mkdir(parents=True, exist_ok=True)
    recodes = resolve_recodes(config.station_recodes)
    calls = recode_columns(read_public_calls(config.timetable_calls_path), recodes, ["crs"])
    coordinates = _coordinates(config)
    split = (
        DirectionSplitModel.load(models.resolve(config.direction_split_model))
        if config.direction_split_model is not None
        else None
    )
    footfall = load_table_1410(config.table_1410) if config.table_1410 is not None else {}
    for crs, share in (config.odm_open_share or {}).items():
        if crs in footfall:
            footfall[crs] = int(footfall[crs] / share)
    if split is not None and not footfall:
        raise ValueError("assign: direction_split_model needs table_1410 for the footfall")
    share_model = _share_model(config)
    segments = pd.concat(
        [
            assign_day(
                calls,
                day,
                config,
                share_model=share_model,
                split=split,
                footfall=footfall,
                coordinates=coordinates,
            )
            for day in config.days
        ],
        ignore_index=True,
    )

    scores = []
    for name, settings in config.operators.items():
        if settings.counts is None:
            continue
        counts = loading_data.load_operator(
            name,
            settings.counts,
            tiploc_to_crs_map(
                config.stations.path if config.stations else None, config.timetable_calls_path
            ),
            cache_dir=settings.counts_cache_dir,
        )
        start = pd.to_datetime(counts["Period Start Date"], errors="coerce")
        if settings.counts_date_from:
            counts = counts[start >= pd.Timestamp(settings.counts_date_from)]
        if settings.counts_date_to:
            counts = counts[start <= pd.Timestamp(settings.counts_date_to)]
        counts = recode_columns(
            counts, recodes, ["Departure from CRS", "Origin CRS", "Destination CRS"]
        )
        matched = match_counts(counts, segments, settings.atoc_codes).assign(operator=name)
        logger.info(
            "%s: counts matched on %d calls of %d trains.",
            name,
            len(matched),
            matched["train_uid"].nunique(),
        )
        scores.append(matched)
    if scores:
        segments["visit"] = segments.groupby(["train_uid", "day", "from_station"]).cumcount()
        segments = segments.merge(
            pd.concat(scores, ignore_index=True).drop(columns="operator"),
            on=["train_uid", "day", "from_station", "visit"],
            how="left",
        )
        table = [
            {"operator": name, **score(group)}
            for name, group in segments[segments["actual_load"].notna()].groupby("operator")
        ]
        table.append({"operator": "all", **score(segments[segments["actual_load"].notna()])})
        pd.DataFrame(table).to_csv(out / "assign_scores.csv", index=False)
    segments.to_csv(out / "assign_segments.csv", index=False)

    keys = ["day", "operator", "train_uid", "headcode", "train_origin", "train_destination"]
    segments.groupby([*keys, "train_departure"], dropna=False).agg(
        peak_load=("load", "max"),
        mean_load=("load", "mean"),
        boardings=("boardings", "sum"),
        calls=("position", "size"),
    ).reset_index().to_csv(out / "assign_trains.csv", index=False)
    on = segments.groupby(["day", "from_station"])["boardings"].sum()
    on.index.names = ["day", "crs"]
    # A segment's alightings are at its first call: the train's last
    # call alights what is left on board.
    last = segments.groupby(["day", "service"]).tail(1)
    final = last.groupby(["day", "to_station"])["load"].sum()
    final.index.names = ["day", "crs"]
    first_off = segments.groupby(["day", "from_station"])["alightings"].sum()
    first_off.index.names = ["day", "crs"]
    pd.DataFrame({"boardings": on, "alightings": first_off.add(final, fill_value=0)}).fillna(
        0.0
    ).reset_index().to_csv(out / "assign_stations.csv", index=False)
    segments.groupby(["day", "from_station", "to_station"]).agg(
        passengers=("load", "sum"), trains=("load", "size"), peak_train=("load", "max")
    ).reset_index().to_csv(out / "assign_links.csv", index=False)
    logger.info("assign: wrote %s", out)
