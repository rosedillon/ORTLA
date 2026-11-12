"""Forecast stations that are not open yet, and what they take from their neighbours.

A proposed station is added to the network as described in the config: its position, its
weekday and Sunday trains, the stations its trains call at (with minutes), car spaces,
staffing and operator. Everything the members read is then rebuilt with it open: residents
are shared afresh between it and its neighbours, car catchments are reshared, and its pairs
join the accessibility model (direct where its trains call, otherwise one change at its best
stop). Its mass is filled in from the trip end, as for an opening. The members, fitted on
established stations, then forecast every station with the proposals open.

Two timetable terms cannot be read for a station without a timetable, the shares of
departures in the morning peak and after 19:00, so they take the scored stations' mean.
The forecast covers England and Wales, where the members are fitted.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ortla import new_station as ns
from ortla.station_ensemble import frame, gravity, members, sources, terms
from ortla.station_ensemble.ensemble import ENSEMBLE, Model, borrow, member_design, target

if TYPE_CHECKING:
    from ortla.config import EnsembleProposedStation
    from ortla.station_ensemble.sources import DataPaths

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _Network:
    """Existing and proposed stations together; proposals come after the existing ones."""

    codes: pd.Index
    xy: np.ndarray
    region: np.ndarray
    n_existing: int

    def position(self, code: str) -> int | None:
        code = code.upper()
        return self.codes.get_loc(code) if code in self.codes else None


def _attributes(proposed: list[EnsembleProposedStation]) -> pd.DataFrame:
    xy = sources.to_grid([p.latitude for p in proposed], [p.longitude for p in proposed])
    codes = pd.Index([p.code.upper() for p in proposed], name="crs")
    if codes.has_duplicates:
        raise ValueError(f"Proposed codes repeat: {', '.join(codes[codes.duplicated()])}")
    return pd.DataFrame(
        {
            "e": xy[:, 0],
            "n": xy[:, 1],
            "name": [p.name or p.code for p in proposed],
            "car_spaces": [float(p.car_spaces) for p in proposed],
            "cctv": [float(p.cctv) for p in proposed],
            "ticket_machine": [float(p.ticket_machine) for p in proposed],
            "bus": [float(p.bus_interchange) for p in proposed],
            "travelcard": [float(p.travelcard_boundary) for p in proposed],
            "cat_f": [float(not p.staffed) for p in proposed],
            "freq": [float(p.trains_per_day) for p in proposed],
            "electric": [float(p.electric) for p in proposed],
            "terminus": [float(p.terminus) for p in proposed],
        },
        index=codes,
    )


def _network(model: Model, proposed: list[EnsembleProposedStation], new: pd.DataFrame) -> _Network:
    """Place the proposals beside the existing stations and check their stops exist."""
    sf = model.sf
    clash = sorted(set(new.index) & set(sf.index))
    if clash:
        raise ValueError(f"Proposed codes clash with existing stations: {', '.join(clash)}")
    xy = new[["e", "n"]].to_numpy(float)
    nearest = np.argmin(((sf.xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1), axis=0)
    region = sf.f["region"].to_numpy()[nearest]
    if (region == "Scotland").any():
        logger.warning(
            "Proposals in Scotland are forecast with England and Wales's residents only"
        )
    net = _Network(
        codes=sf.index.append(new.index),
        xy=np.vstack([sf.xy, xy]),
        region=np.r_[sf.f["region"].to_numpy(), region],
        n_existing=len(sf.index),
    )
    for station in proposed:
        unknown = sorted(c for c in station.stops if net.position(c) is None)
        if unknown:
            raise ValueError(f"{station.code}: stops not in the network: {', '.join(unknown)}")
    return net


def _new_pairs(
    model: Model, proposed: list[EnsembleProposedStation], net: _Network
) -> pd.DataFrame:
    """Build each proposal's pairs: direct where its trains call, else changing at a stop."""
    p, n = model.pairs, net.n_existing
    hours = np.full((n, n), np.inf)
    direct = np.zeros((n, n), bool)
    for matrix, values in ((hours, p["minutes"]), (direct, p["served"] > 0)):
        matrix[p["ii"], p["jj"]] = values.to_numpy()
        matrix[p["jj"], p["ii"]] = values.to_numpy()
    london = net.region == "London"
    rows = []
    for j, station in enumerate(proposed):
        here = n + j
        stops = {c.upper(): minutes / 60 for c, minutes in station.stops.items()}
        each_way = np.log(max(station.trains_per_day / 2, 1.0))
        via = np.array([net.position(c) for c in stops if net.position(c) < n], dtype=int)
        via_hours = np.array([stops[net.codes[k]] for k in via])
        for t, code in enumerate(net.codes):
            if code in stops:
                hrs, served, lntrains, change1 = stops[code], 1.0, each_way, 0.0
            elif t >= n or not len(via):
                continue  # itself, or a proposal it doesn't call at
            else:
                total = via_hours + gravity.CHANGE_MINUTES / 60 + hours[via, t]
                best = int(np.argmin(total))
                if not np.isfinite(total[best]):
                    continue
                hrs, served, lntrains = total[best], 0.0, 0.0
                change1 = float(direct[via[best], t])
            rows.append({
                "ii": here, "jj": t, "km": np.hypot(*(net.xy[t] - net.xy[here])) / 1000,
                "served": served, "lntrains": lntrains, "minutes": hrs, "change1": change1,
                "change2": 1.0 - served - change1, "london": float(london[t] or london[here]),
            })  # fmt: skip
    out = pd.DataFrame(rows)
    # Two proposals that call at each other give the pair twice; keep it once.
    pair = pd.DataFrame(np.sort(out[["ii", "jj"]].to_numpy(), axis=1))
    out = out[~pair.duplicated().to_numpy()]
    out["lnkm"] = np.log(np.maximum(out["km"], 1.0))
    out["lnkm2"] = out["lnkm"] ** 2
    return out


def _terms(
    model: Model,
    paths: DataPaths,
    proposed: list[EnsembleProposedStation],
    new: pd.DataFrame,
    net: _Network,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the term table with the proposals open; also return the access stage's version."""
    sf, table, places = model.sf, model.terms, model.places
    n, scored = net.n_existing, sf.scored
    choice = ["e", "n", "cat_f", "cctv", "car_spaces", "ticket_machine", "bus"]
    open_ = sf.f.loc[~sf.scotland, choice].assign(freq=sf.f["freq_all"])
    before = ns.catchments(open_, model.people)
    after = ns.catchments(pd.concat([open_, new[[*choice, "freq"]]]), model.people)
    jobs = frame.jobs_within_mile(
        new,
        ns.workplaces(
            paths.sdft / "census2021_wp001" / "WP001_oa.csv", paths.sdft / "oa2021_pwc_osgb.csv"
        ),
    )
    place = terms.place_terms(net.xy[n:], places)
    reach = [set() for _ in range(n)] + [{net.position(c) for c in p.stops} for p in proposed]
    served, share = terms.servable_commuting(net.xy, reach, places)

    added = pd.DataFrame(index=new.index, columns=table.columns, dtype=float)
    added["ln_pop"] = np.log1p(after.loc[new.index].to_numpy())
    added["ln_jobs"] = np.log1p(jobs)
    added["ln_car"] = np.log1p(new["car_spaces"])
    added["ln_trains"] = np.log(new["freq"])
    for c in ("electric", "terminus", "travelcard"):
        added[c] = new[c]
    for c in place.columns.intersection(added.columns):
        added[c] = place[c].to_numpy()
    added["ln_jobs_zones"] = np.log1p(place["jobs_zones"].to_numpy())
    added["ln_jobs_2011"] = np.log1p(place["jobs_2011"].to_numpy())
    named = new["name"].str.contains("Parkway|Airport|International")
    added["parkway_or_airport"] = (named | (new["car_spaces"] >= 500)).astype(float)
    added["request_share"] = [float(p.request_stop) for p in proposed]
    for c in ("am_peak_share", "evening_share"):
        added[c] = table[c].to_numpy()[scored].mean()
    added["ln_direct_destinations"] = np.log1p([len(p.stops) + 1 for p in proposed])
    added["ln_servable_commuters"] = np.log1p(served[n:])
    added["servable_share"] = share[n:]
    added["underground_managed"] = 0.0
    for c in table.columns:
        if c.startswith("region_"):
            added[c] = (net.region[n:] == c.removeprefix("region_")).astype(float)
        elif c.startswith("operator_"):
            added[c] = [float(p.operator.upper() == c.removeprefix("operator_")) for p in proposed]
    mean_sunday = table["sunday_ratio"].to_numpy()[scored].mean()
    added["sunday_ratio"] = [
        mean_sunday
        if p.sunday_trains_per_day is None
        else np.log(p.sunday_trains_per_day + 1) - np.log(p.trains_per_day + 1)
        for p in proposed
    ]
    added["staffed_full_time"] = [float(p.staffed and not p.part_time) for p in proposed]
    added["staffed_part_time"] = [float(p.staffed and p.part_time) for p in proposed]

    both = pd.concat([table, added])
    # Existing stations' residents shift by the share the proposals take.
    shift = np.log1p(after.loc[before.index]) - np.log1p(before)
    both.loc[shift.index, "ln_pop"] = table.loc[shift.index, "ln_pop"] + shift
    ln_car = both["ln_car"].to_numpy()
    for term, km in (("car_catchment_5km", 5.0), ("car_catchment_10km", 10.0)):
        both[term] = np.log1p(terms.car_catchment(net.xy, places.cars, ln_car, km))
    freq = np.r_[sf.f["freq"].to_numpy(float), new["freq"].to_numpy()]
    freq_all = np.r_[sf.f["freq_all"].to_numpy(float), new["freq"].to_numpy()]
    both["car_catchment_trains"] = terms.car_catchment_by_trains(
        net.xy, places.cars, ln_car, freq, reference=freq[:n]
    )
    people = model.people
    service = members.service_catchment(
        net.xy, people[["e", "n"]].to_numpy(float), people["pop"].to_numpy(), freq
    )
    scotland = np.r_[sf.scotland, np.zeros(len(new), bool)]
    both["ln_pop_service"] = np.where(scotland, both["ln_pop_service"], np.log1p(service))
    access_terms = both.assign(
        ln_trains=np.log(freq_all),
        car_catchment_trains=terms.car_catchment_by_trains(
            net.xy, places.cars, ln_car, freq_all, reference=freq_all[:n]
        ),
    )
    return both, access_terms


def _accessibility(
    model: Model,
    proposed: list[EnsembleProposedStation],
    net: _Network,
    access_terms: pd.DataFrame,
) -> np.ndarray:
    """Return ln A for every station with the proposals open, their masses from the trip end."""
    n, stage = net.n_existing, model.access
    pairs = pd.concat([model.pairs, _new_pairs(model, proposed, net)], ignore_index=True)
    network = gravity.Network(pairs, net.xy)
    x = access_terms[model.access_columns].to_numpy(float)
    a = np.r_[stage["a"], gravity.ols_predict(x[n:], stage["mass_coef"])]
    for _ in range(gravity.TRIP_END_ITERATIONS):
        ln_access = network.log_access(a, model.stage1_b)
        fitted = gravity.ols_predict(np.column_stack([ln_access, x]), stage["coef"])
        a[n:] = (fitted - stage["coef"][1] * ln_access - stage["c0"])[n:]
    return ln_access


def forecast(
    model: Model, paths: DataPaths, proposed: list[EnsembleProposedStation], neighbour_km: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast each proposal, and the change at existing stations within ``neighbour_km``."""
    sf = model.sf
    new = _attributes(proposed)
    net = _network(model, proposed, new)
    n = net.n_existing
    table, access_terms = _terms(model, paths, proposed, new, net)
    ln_access = _accessibility(model, proposed, net, access_terms)

    predictions = {}
    for m in model.members:
        y = target(model, m)
        train = sf.wide & np.isfinite(y)
        fitted_on = member_design(m, model.terms, model.access["ln_access"])
        predictions[m.key] = m.learner(
            fitted_on[train], y[train], member_design(m, table, ln_access)
        )
    wide = np.flatnonzero(sf.wide)
    predictions[members.KRIGING] = table["ln_pop"].to_numpy() + np.mean(
        (sf.y - model.terms["ln_pop"].to_numpy())[wide]
    )
    log_forecast = np.mean([predictions[k] for k in members.ORDER], axis=0)
    lift = borrow(model.residual(), sf.xy[wide], net.xy)
    log_forecast += lift
    half = model.interval_half_width(log_forecast[n:])
    forecasts = pd.DataFrame(
        {
            "name": new["name"],
            "forecast_entries_exits": np.exp(log_forecast[n:]).round(),
            "low_80": np.exp(log_forecast[n:] - half).round(),
            "high_80": np.exp(log_forecast[n:] + half).round(),
            "member_spread_log": np.std([v[n:] for v in predictions.values()], axis=0).round(3),
            "neighbour_adjustment": lift[n:].round(3),
            "catchment_residents": np.expm1(table["ln_pop"].to_numpy()[n:]).round(),
            "jobs_within_mile": np.expm1(table["ln_jobs"].to_numpy()[n:]).round(),
            "trains_per_day": new["freq"],
            "car_spaces": new["car_spaces"],
            "ln_access": ln_access[n:].round(3),
            "region": net.region[n:],
        },
        index=new.index,
    ).join(ns.lat_lon(new))

    km = np.sqrt(((sf.xy[:, None, :] - net.xy[None, n:, :]) ** 2).sum(-1)) / 1000
    near = np.flatnonzero((km <= neighbour_km).any(axis=1))
    without = model.full[ENSEMBLE][near] + borrow(model.residual(), sf.xy[wide], sf.xy[near])
    neighbours = pd.DataFrame(
        {
            "name": sf.f["name"].iloc[near].to_numpy(),
            "forecast_without": np.exp(without).round(),
            "forecast_with": np.exp(log_forecast[near]).round(),
            "actual_2024_25": np.exp(sf.y[near]).round(),
        },
        index=pd.Index(sf.index[near], name="crs"),
    )
    neighbours["change"] = neighbours["forecast_with"] - neighbours["forecast_without"]
    return forecasts, neighbours.sort_values("change")
