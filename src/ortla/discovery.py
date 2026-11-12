"""Scenario Discovery: scan the network for new direct services.

Scenario Studio (:mod:`ortla.scenario`) optimises departure
time and frequency for a candidate service someone has already picked.
This module asks the question before that: which station pairs have no
decent direct service today but plausibly should, given the demand data
we already have?

The scan runs in stages, cheapest first, so the expensive stages only
see a small shortlist:

1. Distance pre-filter (:func:`_pairs_within_distance_band`): a
   vectorised straight-line distance scan over
   :class:`~ortla.track_distance.TrackDistanceCalculator`'s
   station locations, capped per origin so the candidate count stays
   bounded however dense the network is.
2. Existing-service filter (:func:`_drop_well_served_pairs`): drops
   pairs already served in either direction, using the whole-network
   ``route_frequency_summary.csv`` a ``timetable`` run writes
   (:func:`ortla.gravity_model.load_direct_frequency`).
3. Demand scoring (:func:`_score_observed_market`): each corridor's
   observed market, the trips the period matrices record on it by any
   route, times the growth a first direct link brings.
4. Feasibility: a running-time check on the physical network
   (:func:`_running_time_feasible`, via
   :meth:`ortla.bplan_geography.BplanTimingGraph.minimum_running_time`),
   then candidate synthesis (:func:`_synthesize_candidate`) for the
   survivors.
5. Full sweep (:func:`_promote_and_appraise`): each synthesized
   candidate goes to :func:`ortla.scenario.generate_scenarios`
   unmodified, as Scenario Studio sends a hand-picked one.

Stage 5 is most of a run's cost. Every promoted candidate reads the same
timetable calls, BPLAN graphs and gravity and fares models, differing
only in ``services``, so this module builds one
:class:`~ortla.scenario.ScenarioContext`
(:func:`~ortla.scenario.build_scenario_context`) from
``propose_template`` before the sweep and passes it to every
``generate_scenarios`` call. :data:`DiscoveryConfig.promote_to_sweep_n`
still bounds the sweep, since each candidate's departure-window search
and appraisal are its own work.

With the cost defaults left at 0.0, the ranking is a demand-driven guide
to which pairs are worth a closer look, not a bankable appraisal. Real
costs belong in a full Scenario Studio run once a candidate is promoted.
"""

from __future__ import annotations

import collections
import itertools
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ortla.appraisal import TagDataBook
from ortla.bplan_geography import BplanTimingGraph
from ortla.clock import format_hhmm, hhmm_minutes
from ortla.gravity_model import load_direct_frequency
from ortla.period_matrices import classify_route_type
from ortla.scenario import (
    ScenarioAssumptions,
    build_scenario_context,
    generate_scenarios,
    with_route_cost,
)
from ortla.service_lookup import clear_timetable_caches, crs_to_tiploc_map
from ortla.stations import (
    canonical_crs,
    load_table_1410,
    read_station_list,
    resolve_aliases,
    resolve_recodes,
)
from ortla.timetable import read_public_calls
from ortla.track_distance import TrackDistanceCalculator, convert_metres

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from ortla.config import (
        CandidateService,
        DiscoveryConfig,
        ProposeConfig,
        RouteTypeConfig,
    )
    from ortla.scenario import ScenarioContext, ScenarioOption, ScenarioResult


logger = logging.getLogger(__name__)


# Inputs / outputs
@dataclass
class DiscoveryCandidate:
    """One station pair as it moves through the discovery funnel."""

    origin: str
    destination: str
    distance_miles: float
    existing_daily_trains: float
    # Both directions, any route, from the period matrices.
    observed_annual_trips: float | None = None
    # The growth a first direct link adds.
    predicted_annual_trips: float | None = None
    route_type_band: str | None = None
    minimum_running_minutes: float | None = None
    # Calling pattern, ends included, and each leg's minimum running
    # time (slower direction); unset for a non-stop candidate.
    calls: list[str] | None = None
    leg_minutes: list[float] | None = None
    candidate: CandidateService | None = None
    scenario: ScenarioResult | None = None
    return_scenario: ScenarioResult | None = None
    round_trip: RoundTrip | None = None
    # "candidate" | "no_bplan_path" | "promoted_no_option" |
    # "promoted_feasible"
    status: str = "candidate"


@dataclass
class RoundTrip:
    """The best frequency for a pair, both directions summed."""

    trains_per_day: int
    departure_time: str | None
    return_departure_time: str | None
    annual_passengers: int
    annual_passengers_generated: int
    pvb: float
    pvc: float
    bcr: float
    npv: float


@dataclass
class DiscoveryResult:
    """Full output of a discovery scan."""

    candidates: list[DiscoveryCandidate]  # promoted only, ranked best-first
    shortlist: list[DiscoveryCandidate]  # every demand-ranked pair, market order
    n_pairs_considered: int
    n_within_distance_band: int
    n_no_or_poor_existing_service: int
    n_demand_shortlisted: int
    n_running_time_feasible: int
    n_promoted_to_sweep: int

    @property
    def best(self) -> DiscoveryCandidate | None:
        """The top promoted candidate, or ``None`` if there is none."""
        return self.candidates[0] if self.candidates else None


# Stage 1: distance pre-filter
def _pairs_within_distance_band(
    calc: TrackDistanceCalculator,
    stations: set[str],
    min_miles: float,
    max_miles: float,
    max_per_origin: int,
) -> list[tuple[str, str, float]]:
    """Ordered station pairs within a straight-line distance band.

    One pairwise distance-matrix call rather than a Python loop over
    every pair, which with ~2,500 stations would be millions of
    iterations. The per-origin cap bounds the result to roughly
    ``len(codes) * max_per_origin`` however dense that part of the
    network is; a band filter alone doesn't guarantee that on GB's dense
    geography. Track routing comes in a later, more expensive stage.
    """
    codes = sorted(c for c in calc.station_locations if c in stations)
    if len(codes) < 2:
        return []

    coords = np.array([calc.station_locations[c] for c in codes])
    distances_miles = convert_metres(cdist(coords, coords), unit="miles")

    pairs: list[tuple[str, str, float]] = []
    for i, origin in enumerate(codes):
        row = distances_miles[i]
        in_band = [
            (j, float(row[j]))
            for j in range(len(codes))
            if j != i and min_miles <= row[j] <= max_miles
        ]
        in_band.sort(key=lambda item: item[1])
        for j, distance_miles in in_band[:max_per_origin]:
            pairs.append((origin, codes[j], distance_miles))
    return pairs


# Stage 2: existing-service filter
def _one_code_per_station(
    pairs: list[tuple[str, str, float]],
    frequency: dict[tuple[str, str], float],
    station: Callable[[str], str],
) -> tuple[list[tuple[str, str, float]], dict[tuple[str, str], float]]:
    """Pairs and daily trains with every station under one code.

    The timetable books some calls under a station's second code
    (Retford's low-level platforms as XRO, Worcestershire Parkway's high
    level as WPH), so read code by code Lincoln -> Retford and
    Paddington -> Worcestershire Parkway looked unserved, and St
    Pancras's three codes took three places on the shortlist. A pair's
    trains at all its codes are summed.
    """
    merged: dict[tuple[str, str], float] = collections.defaultdict(float)
    for (origin, destination), trains in frequency.items():
        merged[(station(origin), station(destination))] += trains
    recoded = [(station(o), station(d), miles) for o, d, miles in pairs]
    return [pair for pair in recoded if pair[0] != pair[1]], dict(merged)


def _drop_well_served_pairs(
    pairs: list[tuple[str, str, float]],
    frequency: dict[tuple[str, str], float],
    max_existing_daily_trains: float,
    both_directions: bool = True,
) -> list[tuple[str, str, float, float]]:
    """Keep only pairs with no/poor existing direct service.

    A pair missing from ``frequency`` is treated as having zero existing
    daily trains, not dropped: the whole-network summary only lists
    routes the CIF timetable schedules a direct service on.

    With ``both_directions`` (the default) a pair is judged on the
    busier of its two directions, because a direct link is a property of
    a corridor rather than of a direction. The route summary counts
    ordered pairs, and real timetables are not symmetric: Chichester ->
    London Victoria has one through working on a typical Wednesday
    against 32 the other way (Southern's Coastway services divide and
    join, so the northbound portion is not a single through schedule).
    Judged one direction at a time, that corridor scores as an unserved
    market and ranked first in a national scan.
    """
    kept: list[tuple[str, str, float, float]] = []
    for origin, destination, distance_miles in pairs:
        daily_trains = frequency.get((origin, destination), 0.0)
        if both_directions:
            daily_trains = max(daily_trains, frequency.get((destination, origin), 0.0))
        if daily_trains <= max_existing_daily_trains:
            kept.append((origin, destination, distance_miles, daily_trains))
    return kept


# Stage 3: demand scoring
def _annual_market(matrices_dir: pathlib.Path) -> pd.DataFrame:
    """Annual trips between every two stations, all matrices summed."""
    total: pd.DataFrame | None = None
    for path in sorted(matrices_dir.glob("odm_*.csv")):
        frame = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce")
        total = frame if total is None else total.add(frame, fill_value=0.0)
    if total is None:
        raise FileNotFoundError(f"No odm_*.csv period matrices in {matrices_dir}")
    total.index = total.index.astype(str)
    total.columns = total.columns.astype(str)
    return total


def _score_observed_market(
    pairs: list[tuple[str, str, float, float]],
    market: pd.DataFrame,
    aliases: dict[str, str],
    new_link_uplift: float,
    route_type: RouteTypeConfig,
    top_n: int,
) -> list[DiscoveryCandidate]:
    """Rank unserved pairs on the trips they make by changing trains.

    An unserved pair is not an empty market: the ODM records the
    journeys made on it by any route. A first direct link grows that
    market by ``exp(new_link_uplift) - 1`` (at a direct share of one), so the ranking is the observed market's, and
    ``predicted_annual_trips`` is the growth. Ranking on the gravity
    model's served-pair prediction instead ranked pairs on station size.
    A corridor is scored once, both directions together; a
    pair with no ODM flow, or a station missing from the matrices,
    scores nothing.
    """
    frame = pd.DataFrame(pairs, columns=["origin", "destination", "miles", "trains"])
    corridor = np.where(
        frame["origin"] < frame["destination"],
        frame["origin"] + "|" + frame["destination"],
        frame["destination"] + "|" + frame["origin"],
    )
    frame = frame[~pd.Series(corridor).duplicated().to_numpy()]
    o = market.index.get_indexer([canonical_crs(c, aliases) for c in frame["origin"]])
    d = market.columns.get_indexer([canonical_crs(c, aliases) for c in frame["destination"]])
    found = (o >= 0) & (d >= 0)
    values = market.to_numpy()
    observed = np.zeros(len(frame))
    observed[found] = np.nan_to_num(values[o[found], d[found]]) + np.nan_to_num(
        values[d[found], o[found]]
    )
    frame = frame.assign(observed=observed)
    frame = frame[frame["observed"] > 0].nlargest(top_n, "observed")
    step = float(np.exp(new_link_uplift) - 1.0)
    return [
        DiscoveryCandidate(
            origin=row.origin,
            destination=row.destination,
            distance_miles=row.miles,
            existing_daily_trains=row.trains,
            observed_annual_trips=row.observed,
            predicted_annual_trips=row.observed * step,
            route_type_band=classify_route_type(row.miles, route_type),
        )
        for row in frame.itertuples()
    ]


# Stage 4: feasibility + candidate synthesis
def _running_time_feasible(
    candidates: list[DiscoveryCandidate],
    crs_tiploc: dict[str, str],
    graph: BplanTimingGraph,
    traction_code: str,
) -> list[DiscoveryCandidate]:
    """Drop pairs with no running-time route on the BPLAN network.

    A pair is timed on its slower direction, so a mirrored return
    working (``return_after_minutes``) also clears (Liverpool ->
    Edinburgh needs 205 minutes, the other way 148).
    """
    feasible: list[DiscoveryCandidate] = []
    for dc in candidates:
        origin_tiploc = crs_tiploc.get(dc.origin)
        destination_tiploc = crs_tiploc.get(dc.destination)
        minutes = (
            graph.minimum_running_time(origin_tiploc, destination_tiploc, traction_code)
            if origin_tiploc and destination_tiploc
            else None
        )
        if minutes is None:
            dc.status = "no_bplan_path"
            continue
        back = graph.minimum_running_time(destination_tiploc, origin_tiploc, traction_code)
        dc.minimum_running_minutes = max(minutes, back or 0.0)
        feasible.append(dc)
    return feasible


def _route_calls(
    dc: DiscoveryCandidate,
    crs_tiploc: dict[str, str],
    tiploc_crs: dict[str, str],
    graph: BplanTimingGraph,
    footfall: dict[str, int],
    aliases: dict[str, str],
    config: DiscoveryConfig,
    known: set[str] | None = None,
) -> None:
    """Set ``dc.calls`` and ``dc.leg_minutes`` from the busiest stops.

    The route is the fastest one BPLAN finds; a station on it is a call
    if its entries and exits reach ``intermediate_stop_min_footfall``,
    keeping the busiest ``max_intermediate_stops``. Each leg is timed on
    its slower direction, as the pair is, so the mirrored return clears.
    Leaves the pair non-stop when no station qualifies or a leg has no
    timing. With ``known``, only those stations can be calls (the scan's
    station list, which the distance model locates; Willesden Junction's
    low-level WJL is on routes but not in it).
    """
    # Simplification: a station on the route may have no platform on the line
    # the route takes (fast lines through a slow-line-only station).
    traction = config.default_traction_code
    path = graph.fastest_path(crs_tiploc[dc.origin], crs_tiploc[dc.destination], traction)
    ends = {canonical_crs(dc.origin, aliases), canonical_crs(dc.destination, aliases)}

    def entries(crs: str) -> int:
        return footfall.get(canonical_crs(crs, aliases), 0)

    on_route = [
        crs
        for crs in dict.fromkeys(tiploc_crs[t] for t in (path or [])[1:-1] if t in tiploc_crs)
        if canonical_crs(crs, aliases) not in ends
        and (known is None or crs in known)
        and entries(crs) >= config.intermediate_stop_min_footfall
    ]
    busiest = set(sorted(on_route, key=entries, reverse=True)[: config.max_intermediate_stops])
    if not busiest:
        return
    calls = [dc.origin, *(c for c in on_route if c in busiest), dc.destination]
    legs = []
    for a, b in itertools.pairwise(calls):
        there = graph.minimum_running_time(crs_tiploc[a], crs_tiploc[b], traction)
        back = graph.minimum_running_time(crs_tiploc[b], crs_tiploc[a], traction)
        if there is None or back is None:
            return
        legs.append(max(there, back))
    dc.calls, dc.leg_minutes = calls, legs
    dc.minimum_running_minutes = sum(legs) + config.dwell_minutes * (len(legs) - 1)


def _synthesize_candidate(dc: DiscoveryCandidate, config: DiscoveryConfig) -> CandidateService:
    """Build a plausible direct candidate from a feasible pair.

    Departs at ``config.anchor_departure_time``; arrives after the BPLAN
    minimum running time plus a margin. With ``dc.calls`` set
    (:func:`_route_calls`) it calls there too, each leg timed at its
    minimum (rounded up to the minute) plus ``dwell_minutes``, the
    margin on the last leg. ``search_window`` is always set: a single
    anchor time is a guess, and sweeping a window around it gives the
    optimiser a better chance of a good slot
    (:func:`ortla.scenario._departure_frame` branches on it).
    """
    from ortla.config import (  # noqa: PLC0415 (import cycle)
        CandidateService,
        ServiceStop,
    )

    anchor_minutes = hhmm_minutes(config.anchor_departure_time) or 0.0
    calls = dc.calls or [dc.origin, dc.destination]
    legs = dc.leg_minutes or [dc.minimum_running_minutes or 0.0]
    stops = [ServiceStop(station=calls[0], departure=config.anchor_departure_time)]
    clock = anchor_minutes
    for station, leg in zip(calls[1:-1], legs, strict=False):
        clock += math.ceil(leg)
        stops.append(
            ServiceStop(
                station=station,
                arrival=format_hhmm(clock),
                departure=format_hhmm(clock + config.dwell_minutes),
            )
        )
        clock += config.dwell_minutes
    clock += math.ceil(legs[-1]) + config.running_time_margin_minutes
    stops.append(ServiceStop(station=calls[-1], arrival=format_hhmm(clock)))

    return CandidateService(
        name=f"discovery_{dc.origin}_{dc.destination}",
        traction_code=config.default_traction_code,
        days_of_week=list(config.default_days_of_week),
        stops=stops,
        search_window=config.search_window,
        return_after_minutes=config.return_after_minutes,
    )


# Stage 5: full sweep
def _promote_and_appraise(
    dc: DiscoveryCandidate,
    config: DiscoveryConfig,
    propose_template: ProposeConfig,
    assumptions: ScenarioAssumptions,
    context: ScenarioContext,
) -> DiscoveryCandidate:
    """Run the built candidate through the Scenario Studio optimiser.

    ``propose_template`` carries ``table_1410``, ``distance`` and
    ``period_matrices_dir``: ``predict_demand`` prices each leg from its
    matrix cell (a first direct link, ``gravity_new_link``) and falls
    back to the gravity model only for legs with no observed flow.

    ``context`` is the
    :class:`~ortla.scenario.ScenarioContext` built once from
    ``propose_template`` before the sweep loop, shared because every
    promoted candidate reads the same timetable calls, BPLAN graphs and
    gravity and fares models.
    """
    clear_timetable_caches()
    services = [dc.candidate]
    if dc.candidate.return_after_minutes is not None:
        services.append(dc.candidate.return_working())
    results = []
    for service in services:
        propose_cfg = propose_template.model_copy(
            update={
                "output_path": config.output_path / "discovery_sweeps" / service.name,
                "services": [service],
            }
        )
        costed = with_route_cost(assumptions, service, context.gravity)
        results.append(generate_scenarios(propose_cfg, costed, context=context))
    dc.scenario = results[0]
    dc.return_scenario = results[1] if len(results) > 1 else None
    dc.round_trip = _round_trip(dc.scenario, dc.return_scenario)
    dc.status = "promoted_feasible" if dc.round_trip else "promoted_no_option"
    return dc


def _round_trip(out: ScenarioResult, back: ScenarioResult | None) -> RoundTrip | None:
    """Pick the frequency with the best BCR, each way at its best slot.

    ``None`` when no frequency has a feasible option in every appraised
    direction.
    """

    # Simplification: each direction picks its slot alone; the return need not
    # follow the outbound train's arrival, so the fleet count is not
    # checked.
    def best_per_frequency(result: ScenarioResult) -> dict[int, ScenarioOption]:
        best: dict[int, ScenarioOption] = {}
        for option in result.options:
            held = best.get(option.trains_per_day)
            if held is None or option.bcr > held.bcr:
                best[option.trains_per_day] = option
        return best

    outward = best_per_frequency(out)
    inward = best_per_frequency(back) if back is not None else None
    trips = []
    for freq, first in outward.items():
        if inward is not None and freq not in inward:
            continue
        legs = [first] if inward is None else [first, inward[freq]]
        pvb = sum(o.pvb for o in legs)
        pvc = sum(o.pvc for o in legs)
        trips.append(
            RoundTrip(
                trains_per_day=freq,
                departure_time=first.departure_time,
                return_departure_time=legs[-1].departure_time if inward is not None else None,
                annual_passengers=sum(o.annual_passengers for o in legs),
                annual_passengers_generated=sum(o.annual_passengers_generated for o in legs),
                pvb=pvb,
                pvc=pvc,
                # As AppraisalResults.bcr: no costs to set against is
                # inf, or 0 with no benefit.
                bcr=pvb / pvc if pvc > 0 else (float("inf") if pvb > 0 else 0.0),
                npv=pvb - pvc,
            )
        )
    return max(trips, key=lambda t: (t.bcr, t.npv), default=None)


def _rank_key(dc: DiscoveryCandidate) -> tuple[int, float, float]:
    """Sort key: feasible pairs by BCR and NPV, the rest by demand."""
    if dc.round_trip is not None:
        return (0, -dc.round_trip.bcr, -dc.round_trip.npv)
    return (1, -(dc.predicted_annual_trips or 0.0), 0.0)


def _promote(
    feasible: list[DiscoveryCandidate], config: DiscoveryConfig
) -> list[DiscoveryCandidate]:
    """Take the top ``promote_to_sweep_n`` pairs, or as many a band."""
    if not config.promote_per_route_type:
        return feasible[: config.promote_to_sweep_n]
    taken: collections.Counter[str | None] = collections.Counter()
    promoted = []
    for dc in feasible:
        if taken[dc.route_type_band] < config.promote_to_sweep_n:
            taken[dc.route_type_band] += 1
            promoted.append(dc)
    return promoted


# Public entry point
def generate_discovery_candidates(
    config: DiscoveryConfig,
    tag: TagDataBook,
) -> DiscoveryResult:
    """Run the five-stage scan and return a :class:`DiscoveryResult`."""
    from ortla.config import ProposeConfig  # noqa: PLC0415 (import cycle)

    propose_template = ProposeConfig(
        output_path=config.output_path,
        services=[],
        timetable_calls_path=config.timetable_calls_path,
        bplan_output_path=config.bplan_output_path,
        period_matrices_dir=config.period_matrices_dir,
        table_1410=config.table_1410,
        distance=config.distance,
        route_type=config.route_type,
        gravity_min_loading_observations=config.gravity_min_loading_observations,
        fares_dir=config.fares_dir,
        headway_mode=config.headway_mode,
        planning_headway_minutes=config.planning_headway_minutes,
    )

    stations, _ = read_station_list(config.station_list)
    n_pairs_considered = len(stations) * max(len(stations) - 1, 0)

    logger.info(
        "Stage 1: scanning %s stations for pairs %.0f-%.0f miles apart ...",
        len(stations),
        config.min_distance_miles,
        config.max_distance_miles,
    )
    calc = TrackDistanceCalculator(config.distance)
    pairs = _pairs_within_distance_band(
        calc,
        stations,
        config.min_distance_miles,
        config.max_distance_miles,
        config.max_candidates_per_origin,
    )
    n_within_distance_band = len(pairs)
    logger.info(
        "Stage 1: %s pairs within band (of %s possible).",
        n_within_distance_band,
        n_pairs_considered,
    )

    # Without it every pair reads as unserved, and the busiest served
    # corridors would top the scan as opportunities.
    route_summary = config.timetable_output_path / "route_frequency_summary.csv"
    if not route_summary.exists():
        raise FileNotFoundError(
            f"{route_summary} not found; run `timetable` first so discovery can tell "
            "which pairs are already served."
        )
    recodes = resolve_recodes(propose_template.station_recodes)
    aliases = resolve_aliases(propose_template.station_aliases)
    pairs, frequency = _one_code_per_station(
        pairs,
        load_direct_frequency(route_summary),
        lambda code: canonical_crs(recodes.get(code, code), aliases),
    )
    served = _drop_well_served_pairs(
        pairs,
        frequency,
        config.max_existing_daily_trains,
        config.existing_service_both_directions,
    )
    n_no_or_poor_existing_service = len(served)
    logger.info(
        "Stage 2: %s pairs with no/poor existing direct service.",
        n_no_or_poor_existing_service,
    )

    scored = _score_observed_market(
        served,
        _annual_market(config.period_matrices_dir),
        aliases,
        propose_template.new_link_uplift,
        config.route_type,
        config.demand_ranked_shortlist_size,
    )
    n_demand_shortlisted = len(scored)
    logger.info("Stage 3: %s corridors ranked by their observed market.", n_demand_shortlisted)

    calls = read_public_calls(config.timetable_calls_path)
    crs_tiploc = crs_to_tiploc_map(calls)
    graph = BplanTimingGraph(config.bplan_output_path)
    feasible = _running_time_feasible(scored, crs_tiploc, graph, config.default_traction_code)
    n_running_time_feasible = len(feasible)
    logger.info(
        "Stage 4a: %s pairs feasible on the existing physical network.",
        n_running_time_feasible,
    )

    promoted = _promote(feasible, config)
    if config.max_intermediate_stops > 0:
        footfall = load_table_1410(config.table_1410)
        tiploc_crs = {tiploc: crs for crs, tiploc in crs_tiploc.items()}
        tiploc_crs.update(
            zip(
                calls["tiploc"].str.strip().str.upper(),
                calls["crs"].str.strip().str.upper(),
                strict=True,
            )
        )
        for dc in promoted:
            _route_calls(
                dc, crs_tiploc, tiploc_crs, graph, footfall, aliases, config, set(stations)
            )
    for dc in promoted:
        dc.candidate = _synthesize_candidate(dc, config)
    n_promoted_to_sweep = len(promoted)

    assumptions = ScenarioAssumptions(
        tag=tag,
        frequency_options=list(config.frequency_options),
        capital_cost_m=config.default_capital_cost_m,
        annual_operating_cost_m=config.default_annual_operating_cost_m,
        operating_cost_per_train_km=config.operating_cost_per_train_km,
        operating_hours_per_day=config.operating_hours_per_day,
    )
    if promoted:
        # Every promoted candidate reads the same timetable calls, BPLAN
        # graphs and gravity and fares models (they differ only in
        # `services`), so they are built once here (see
        # ScenarioContext).
        logger.info(
            "Stage 5: loading shared timetable/BPLAN/demand data sources for the sweep ..."
        )
        context = build_scenario_context(propose_template)
        for i, dc in enumerate(promoted, start=1):
            pair = f"{dc.origin} -> {dc.destination}"
            logger.info("Stage 5: appraising %s/%s (%s) ...", i, n_promoted_to_sweep, pair)
            _promote_and_appraise(dc, config, propose_template, assumptions, context)

    ranked = sorted(promoted, key=_rank_key)
    return DiscoveryResult(
        candidates=ranked,
        shortlist=scored,
        n_pairs_considered=n_pairs_considered,
        n_within_distance_band=n_within_distance_band,
        n_no_or_poor_existing_service=n_no_or_poor_existing_service,
        n_demand_shortlisted=n_demand_shortlisted,
        n_running_time_feasible=n_running_time_feasible,
        n_promoted_to_sweep=n_promoted_to_sweep,
    )


# CLI wrapper
_CANDIDATE_COLUMNS = [
    "origin",
    "destination",
    "distance_miles",
    "existing_daily_trains",
    "observed_annual_trips",
    "predicted_annual_trips",
    "route_type_band",
    "minimum_running_minutes",
    "calls",
    "status",
    "best_departure_time",
    "best_return_departure_time",
    "best_trains_per_day",
    "best_annual_passengers",
    "best_annual_passengers_generated",
    "best_pvb",
    "best_pvc",
    "best_bcr",
    "best_npv",
]


def _candidate_row(dc: DiscoveryCandidate) -> dict[str, object]:
    best = dc.round_trip
    return {
        "origin": dc.origin,
        "destination": dc.destination,
        "distance_miles": dc.distance_miles,
        "existing_daily_trains": dc.existing_daily_trains,
        "observed_annual_trips": dc.observed_annual_trips,
        "predicted_annual_trips": dc.predicted_annual_trips,
        "route_type_band": dc.route_type_band,
        "minimum_running_minutes": dc.minimum_running_minutes,
        "calls": "-".join(dc.calls) if dc.calls else None,
        "status": dc.status,
        "best_departure_time": best.departure_time if best else None,
        "best_return_departure_time": best.return_departure_time if best else None,
        "best_trains_per_day": best.trains_per_day if best else None,
        "best_annual_passengers": best.annual_passengers if best else None,
        "best_annual_passengers_generated": best.annual_passengers_generated if best else None,
        "best_pvb": best.pvb if best else None,
        "best_pvc": best.pvc if best else None,
        "best_bcr": best.bcr if best else None,
        "best_npv": best.npv if best else None,
    }


def run_discovery_command(config: DiscoveryConfig) -> None:
    """Run the scan and write the ranked candidate CSVs.

    Writes ``discovery_funnel_summary.csv`` (one row of stage counts)
    and ``discovery_ranked_candidates.csv`` (one row per promoted
    candidate) to ``config.output_path``. See the module docstring for
    why the ranking is a relative demand signal, not a bankable
    appraisal, when cost defaults are left at 0.0.
    """
    config.output_path.mkdir(exist_ok=True, parents=True)
    tag = TagDataBook(config.tag_databook_path)
    result = generate_discovery_candidates(config, tag)

    pd.DataFrame(
        [
            {
                "n_pairs_considered": result.n_pairs_considered,
                "n_within_distance_band": result.n_within_distance_band,
                "n_no_or_poor_existing_service": result.n_no_or_poor_existing_service,
                "n_demand_shortlisted": result.n_demand_shortlisted,
                "n_running_time_feasible": result.n_running_time_feasible,
                "n_promoted_to_sweep": result.n_promoted_to_sweep,
            }
        ]
    ).to_csv(config.output_path / "discovery_funnel_summary.csv", index=False)

    pd.DataFrame(
        [_candidate_row(dc) for dc in result.candidates], columns=_CANDIDATE_COLUMNS
    ).to_csv(config.output_path / "discovery_ranked_candidates.csv", index=False)
    pd.DataFrame(
        [_candidate_row(dc) for dc in result.shortlist], columns=_CANDIDATE_COLUMNS
    ).to_csv(config.output_path / "discovery_shortlist.csv", index=False)

    logger.info(
        "Discovery scan complete: %d candidate(s) promoted to the full sweep, "
        "%d written to discovery_ranked_candidates.csv.",
        result.n_promoted_to_sweep,
        len(result.candidates),
    )
