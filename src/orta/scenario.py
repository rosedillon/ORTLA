"""Scenario Studio: build a service, check it fits, appraise it.

It ties three parts of the pipeline into one flow:

* Build: a candidate service
  (:class:`~orta.config.CandidateService`), as the
  ``propose`` command understands it.
* Fit: ``propose``'s feasibility search
  (:func:`orta.feasibility.search_departure_window`) sweeps
  the candidate across a departure-time window, keeps only the slots
  that clear the BPLAN minimum running times and don't clash on headway,
  and predicts each slot's demand and (via the fares feed) revenue.
* Appraise: each feasible slot's predicted demand, revenue and derived
  journey-time saving go into the WebTAG engine
  (:func:`orta.appraisal.run_appraisal`), and the results are
  ranked by value for money.

It also sweeps a small set of frequency options (trains per day). Demand
is scaled sub-linearly by a frequency elasticity (the same lever
``propose``'s pivot model uses) and operating cost scales linearly, so a
half-hourly variant can be compared with an hourly one on the same
BCR/NPV basis. Every number the optimiser derives is a starting point,
to be replaced where better information exists.

Options are ranked with :func:`~orta.appraisal.evaluate`,
which builds the same cashflow as
:func:`~orta.appraisal.run_appraisal` without its logging.
The best option is then appraised again in full by
:func:`appraise_option`.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import pandas as pd

from orta.appraisal import (
    AppraisalParameters,
    AppraisalServiceInput,
    TagDataBook,
    evaluate,
    run_appraisal,
)
from orta.bplan_geography import BplanNetworkLinks, BplanTimingGraph
from orta.clock import MINUTES_PER_DAY, hhmm_minutes, period_minutes
from orta.commercial_case import (
    CommercialCaseParameters,
    CommercialServiceInput,
    run_commercial_case,
)
from orta.demand import (
    build_fares_context,
    build_gravity_context,
    candidate_calls,
    predict_demand,
    stop_time,
    transfer_demand,
)
from orta.feasibility import (
    check_headway,
    check_running_times,
    recode_stops,
    search_departure_window,
    search_flexible_slot,
    shift_candidate,
)
from orta.service_lookup import (
    best_existing_minutes,
    crs_to_tiploc_map,
    day_services,
    existing_trains,
    prepare_calls,
)
from orta.stations import recode_columns, resolve_recodes
from orta.timetable import read_public_calls, resolve_reference_date

if TYPE_CHECKING:
    from orta.appraisal import AppraisalResults
    from orta.commercial_case import CommercialCaseResults
    from orta.config import CandidateService, ProposeConfig, ScenarioConfig
    from orta.demand import FaresContext, GravityContext
    from orta.network_assignment import TransferDemand

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Inputs / outputs
# ----------------------------------------------------------------------
@dataclass
class ScenarioAssumptions:
    """Everything the appraisal side of a scenario needs.

    The demand, revenue and journey-time saving are supplied by the
    feasibility sweep, not here; this class only carries the WebTAG
    appraisal levers and the optimiser's search settings (the frequency
    options and their elasticity).
    """

    tag: TagDataBook

    # Appraisal timing
    appraisal_year: int = 2028
    appraisal_period: int = 60
    construction_period: int = 3

    # Demand ramp-up
    demand_ramp_up_years: int = 3
    demand_ramp_up_profile: list[float] = field(default_factory=lambda: [0.5, 0.75, 0.9, 1.0])

    # Trip purpose split (should sum to ~1.0)
    purpose_work: float = 0.12
    purpose_commuting: float = 0.47
    purpose_other: float = 0.41

    # Costs (£ million). ``annual_operating_cost_m`` is *per base train
    # path* and is scaled linearly by each frequency option; capital is
    # treated as fixed.
    capital_cost_m: float = 0.0
    annual_operating_cost_m: float = 0.0
    # Added to ``annual_operating_cost_m`` per candidate: this rate (£)
    # times the candidate's track distance times its service days, so a
    # long train costs more than a short one. Needs ``distance``
    # configured.
    operating_cost_per_train_km: float = 0.0
    cost_inflation_rate: float = 0.0
    optimism_bias_capex: float = 0.66
    optimism_bias_opex: float = 0.41
    indirect_tax_correction_factor: float = 1.19

    # Mode-shift assumptions
    mode_shift_car_pct: float = 0.20
    avg_car_trip_km: float = 30.0
    car_co2_per_km: float = 0.120
    indirect_tax_car_pence_per_km: float = 3.0

    # Frequency (wait-time) and crowding benefits, TAG M3.2. Leaving
    # ``operating_hours_per_day`` at 0 switches the wait-time benefit
    # off; the crowding benefit switches itself off when the corridor
    # has no existing service to relieve. ``seats_per_train`` falls back
    # to the candidate's capacity when set, else to
    # ``default_seats_per_train``.
    operating_hours_per_day: float = 0.0
    wait_time_vot_multiplier: float = 2.0
    max_average_wait_minutes: float = 15.0
    default_seats_per_train: float = 0.0

    # Optimiser search space. Each option is priced as that many trains
    # a day (see `_frequency_scaling`).
    frequency_options: list[int] = field(default_factory=lambda: [1])

    # Fallbacks used when the feasibility sweep can't supply a value: no
    # existing service to measure a saving against, or no fare in the
    # feed.
    default_time_saving_minutes: float = 0.0
    default_avg_fare: float = 0.0

    # Commercial case (operator viability) levers. The demand, revenue,
    # appraisal timing, ramp-up and cost-inflation assumptions above are
    # shared with the WebTAG side; these two are commercial only (WebTAG
    # has no equivalent of an operator's own hurdle rate or fare
    # growth).
    commercial_discount_rate: float = 0.08
    commercial_fare_growth_rate: float = 0.0


@dataclass
class StopTiming:
    """One stop's arrival, departure and dwell in one option."""

    station: str
    arrival: str | None
    departure: str | None
    dwell_minutes: float = 0.0


@dataclass
class ScenarioOption:
    """One appraised slot and frequency of the candidate."""

    label: str
    shift_minutes: float
    departure_time: str | None
    trains_per_day: int
    feasible: bool
    worst_gap_minutes: float | None
    annual_passengers: int
    annual_revenue: float  # new fare revenue: generated passengers x average fare
    avg_fare: float
    time_saving_minutes: float
    bcr: float
    npv: float
    pvb: float
    pvc: float
    # Physical attributes carried through so the option can be
    # re-appraised identically (the frequency and crowding benefits
    # depend on them).
    seats_per_train: float = 0.0
    in_vehicle_time_minutes: float = 0.0
    baseline_trains_per_day: float = 0.0
    service_days_per_year: float = 253.0
    # Full stop-by-stop schedule for this option's departure slot, so
    # the intermediate dwell times are visible as well as the origin
    # departure, and the resulting end-to-end travel time.
    stops: list[StopTiming] = field(default_factory=list)
    total_travel_minutes: float = 0.0
    # Passengers new to rail (the rest are abstracted from existing
    # trains).
    annual_passengers_generated: int = 0


@dataclass
class ScenarioResult:
    """Full output of a scenario optimisation run."""

    template_name: str
    options: list[ScenarioOption]  # feasible variants, ranked best-first
    time_saving_minutes: float
    n_departures_evaluated: int
    n_departures_feasible: int
    running_times: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str | None = None  # e.g. running-time infeasible regardless of shift

    @property
    def best(self) -> ScenarioOption | None:
        """The best value feasible option, or ``None`` if none fit."""
        return self.options[0] if self.options else None


@dataclass
class ScenarioContext:
    """What :func:`generate_scenarios` builds from ``config`` alone.

    Reading the timetable calls CSV, building the BPLAN timing and
    network-link graphs and fitting the gravity and fares models are
    most of a run's cost, and depend only on ``config``'s data sources,
    which are the same for every candidate evaluated against one
    ``ProposeConfig``. A caller sweeping many candidates against the
    same data (:mod:`orta.discovery`'s full-sweep stage, say)
    should build this once with :func:`build_scenario_context` and pass
    it to every :func:`generate_scenarios` call.
    :func:`generate_scenarios` builds its own when none is given.
    """

    calls: pd.DataFrame
    prepared: pd.DataFrame
    crs_tiploc: dict[str, str]
    reference_date: pd.Timestamp
    graph: BplanTimingGraph
    network_links: BplanNetworkLinks | None
    gravity: GravityContext | None
    fares: FaresContext | None


def build_scenario_context(
    config: ProposeConfig,
) -> ScenarioContext:
    """Build the data :func:`generate_scenarios` needs, once.

    See :class:`ScenarioContext` for why a caller evaluating many
    candidates against the same ``config`` should build this once and
    reuse it.
    """
    logger.info("Loading timetable calls from %s ...", config.timetable_calls_path)
    calls = read_public_calls(config.timetable_calls_path)
    crs_tiploc = crs_to_tiploc_map(calls)
    # Trains at a station's second code (Reading's bay platforms, RDZ)
    # serve its pairs; recoded after the TIPLOC map, as the hindcast
    # does.
    calls = recode_columns(calls, resolve_recodes(config.station_recodes), ["crs"])
    reference_date = resolve_reference_date(calls, config)
    logger.info("Loading BPLAN timing graph from %s ...", config.bplan_output_path)
    graph = BplanTimingGraph(config.bplan_output_path)
    try:
        network_links = BplanNetworkLinks(config.bplan_output_path)
    except FileNotFoundError:
        # The network links are only informational, so an older bplan
        # output without them must not stop the run.
        network_links = None
    gravity = build_gravity_context(config)
    fares = build_fares_context(config)
    prepared = prepare_calls(calls)

    return ScenarioContext(
        calls=calls,
        prepared=prepared,
        crs_tiploc=crs_tiploc,
        reference_date=reference_date,
        graph=graph,
        network_links=network_links,
        gravity=gravity,
        fares=fares,
    )


# ----------------------------------------------------------------------
# Appraisal helpers
# ----------------------------------------------------------------------
def _build_params(
    assumptions: ScenarioAssumptions,
    name: str,
    annual_passengers: int,
    avg_fare: float,
    time_saving_minutes: float,
    trains_per_day: int,
    *,
    seats_per_train: float = 0.0,
    in_vehicle_time_minutes: float = 0.0,
    baseline_trains_per_day: float = 0.0,
    service_days_per_year: float = 253.0,
    generated_passengers: int | None = None,
) -> AppraisalParameters:
    """Build :class:`AppraisalParameters` for one scenario option.

    Operating cost scales linearly with ``trains_per_day``; capital is
    fixed. ``annual_passengers`` are everyone the option carries; of
    those, ``generated_passengers`` are new to rail and the rest are
    abstracted from existing trains. The abstracted riders form the
    do-minimum: they get the full journey-time benefit, the generated
    ones half (rule of half), and only generated riders add fare
    revenue, mode shift and carbon. With ``generated_passengers`` unset
    every rider is treated as generated.

    The frequency (wait-time) and crowding benefits, however, are
    measured against the corridor's existing service:
    ``baseline_trains_per_day`` is the do-minimum frequency and
    ``baseline_trains_per_day x seats_per_train`` its daily seat
    capacity, so a service on an already-served corridor is credited for
    the extra frequency and capacity it adds rather than scoring zero.
    """
    svc = AppraisalServiceInput(
        name=name,
        annual_passengers=annual_passengers,
        journey_time_saved_minutes=time_saving_minutes,
        avg_fare=avg_fare,
        trains_per_day=float(trains_per_day),
        seats_per_train=seats_per_train,
        in_vehicle_time_minutes=in_vehicle_time_minutes,
    )
    generated = annual_passengers if generated_passengers is None else generated_passengers
    abstracted = max(0, annual_passengers - generated)
    do_minimum = (
        [
            AppraisalServiceInput(
                name=name,
                annual_passengers=abstracted,
                journey_time_saved_minutes=0.0,
                avg_fare=avg_fare,
            )
        ]
        if abstracted
        else []
    )
    return AppraisalParameters(
        tag=assumptions.tag,
        proposed_services=[svc],
        do_minimum_services=do_minimum,
        appraisal_year=assumptions.appraisal_year,
        appraisal_period=assumptions.appraisal_period,
        construction_period=assumptions.construction_period,
        demand_ramp_up_years=assumptions.demand_ramp_up_years,
        demand_ramp_up_profile=list(assumptions.demand_ramp_up_profile),
        purpose_work=assumptions.purpose_work,
        purpose_commuting=assumptions.purpose_commuting,
        purpose_other=assumptions.purpose_other,
        capital_cost_m=assumptions.capital_cost_m,
        annual_operating_cost_m=assumptions.annual_operating_cost_m * trains_per_day,
        cost_inflation_rate=assumptions.cost_inflation_rate,
        optimism_bias_capex=assumptions.optimism_bias_capex,
        optimism_bias_opex=assumptions.optimism_bias_opex,
        indirect_tax_correction_factor=assumptions.indirect_tax_correction_factor,
        mode_shift_car_pct=assumptions.mode_shift_car_pct,
        avg_car_trip_km=assumptions.avg_car_trip_km,
        car_co2_per_km=assumptions.car_co2_per_km,
        indirect_tax_car_pence_per_km=assumptions.indirect_tax_car_pence_per_km,
        operating_hours_per_day=assumptions.operating_hours_per_day,
        baseline_trains_per_day=baseline_trains_per_day,
        wait_time_vot_multiplier=assumptions.wait_time_vot_multiplier,
        max_average_wait_minutes=assumptions.max_average_wait_minutes,
        service_days_per_year=service_days_per_year,
        baseline_seats_per_day=baseline_trains_per_day * seats_per_train,
    )


def _quick_metrics(params: AppraisalParameters) -> tuple[float, float, float, float]:
    """Return ``(bcr, npv, pvb, pvc)`` for ranking an option.

    :func:`~orta.appraisal.evaluate` builds the same cashflow
    as :func:`~orta.appraisal.run_appraisal` but skips its
    log lines and do-minimum warning, which would otherwise repeat for
    every one of the many options the optimiser ranks.
    """
    result = evaluate(params)
    return result.bcr, result.npv, result.pvb, result.pvc


def appraise_option(
    assumptions: ScenarioAssumptions,
    name: str,
    annual_passengers: int,
    avg_fare: float,
    time_saving_minutes: float,
    trains_per_day: int = 1,
    *,
    seats_per_train: float = 0.0,
    in_vehicle_time_minutes: float = 0.0,
    baseline_trains_per_day: float = 0.0,
    service_days_per_year: float = 253.0,
    generated_passengers: int | None = None,
) -> AppraisalResults:
    """Run the full WebTAG appraisal for one option.

    Used for the best option, and for any option re-appraised with
    overridden inputs. The frequency and crowding keyword arguments
    carry the corridor context the optimiser used, so the result
    reproduces the ranked option's BCR.
    """
    params = _build_params(
        assumptions,
        name,
        annual_passengers,
        avg_fare,
        time_saving_minutes,
        trains_per_day,
        seats_per_train=seats_per_train,
        in_vehicle_time_minutes=in_vehicle_time_minutes,
        baseline_trains_per_day=baseline_trains_per_day,
        service_days_per_year=service_days_per_year,
        generated_passengers=generated_passengers,
    )
    return run_appraisal(params)


def commercial_case_for_option(
    assumptions: ScenarioAssumptions,
    name: str,
    opt: ScenarioOption,
) -> CommercialCaseResults:
    """Operator-viability commercial case for one selected option.

    A second view of the option the WebTAG appraisal scores. It uses the
    option's own demand and revenue
    (:attr:`ScenarioOption.annual_passengers`, ``.avg_fare``) and the
    assumptions' cost and timing levers, so whether an operator would
    run it is answered from the same optimiser output as the
    socioeconomic BCR.
    """
    # The operator's new revenue comes from generated riders only.
    svc = CommercialServiceInput(
        name=name, annual_passengers=opt.annual_passengers_generated, avg_fare=opt.avg_fare
    )
    params = CommercialCaseParameters(
        proposed_services=[svc],
        appraisal_year=assumptions.appraisal_year,
        appraisal_period=assumptions.appraisal_period,
        construction_period=assumptions.construction_period,
        demand_ramp_up_years=assumptions.demand_ramp_up_years,
        demand_ramp_up_profile=list(assumptions.demand_ramp_up_profile),
        capital_cost_m=assumptions.capital_cost_m,
        annual_operating_cost_m=assumptions.annual_operating_cost_m * opt.trains_per_day,
        cost_inflation_rate=assumptions.cost_inflation_rate,
        fare_growth_rate=assumptions.commercial_fare_growth_rate,
        discount_rate=assumptions.commercial_discount_rate,
    )
    return run_commercial_case(params)


# ----------------------------------------------------------------------
# Time-saving derivation
# ----------------------------------------------------------------------
def _candidate_journey_minutes(candidate: CandidateService) -> float | None:
    """Return the candidate's end-to-end minutes, or ``None``.

    First-stop departure to last-stop arrival, wrapping past midnight.
    Returns ``None`` when either end time is missing or there are fewer
    than two stops.
    """
    stops = candidate.stops
    if len(stops) < 2:
        return None
    dep = hhmm_minutes(stop_time(stops[0]))
    arr = hhmm_minutes(stops[-1].arrival or stops[-1].departure)
    if dep is None or arr is None:
        return None
    minutes = arr - dep
    if minutes < 0:
        minutes += MINUTES_PER_DAY
    return float(minutes)


def _stop_timings_for(candidate: CandidateService) -> tuple[list[StopTiming], float]:
    """Read the candidate's own stop times and total travel time.

    Reads only the times ``candidate`` already carries. Used by
    :func:`_stop_timeline` (a uniformly shifted candidate) and
    :func:`_flexible_slot_options` (a per-leg, dwell-adjusted candidate
    from :func:`~orta.feasibility.search_flexible_slot`), so
    both report dwell the same way.
    """
    stops: list[StopTiming] = []
    for stop in candidate.stops:
        arr_min = hhmm_minutes(stop.arrival)
        dep_min = hhmm_minutes(stop.departure)
        dwell = 0.0
        if arr_min is not None and dep_min is not None:
            dwell = dep_min - arr_min
            if dwell < 0:
                dwell += MINUTES_PER_DAY
        stops.append(
            StopTiming(
                station=stop.station,
                arrival=stop.arrival,
                departure=stop.departure,
                dwell_minutes=dwell,
            )
        )
    total_travel_minutes = _candidate_journey_minutes(candidate) or 0.0
    return stops, total_travel_minutes


def _stop_timeline(
    candidate: CandidateService, shift_minutes: float
) -> tuple[list[StopTiming], float]:
    """Stop times for one uniformly shifted departure option.

    Shifts the candidate's own typed stop times by ``shift_minutes``
    (:func:`orta.feasibility.shift_candidate`) so the
    schedule reflects the slot being evaluated rather than the
    template's nominal times, and each option shows the wait at
    intermediate stops (LIV -> WIGAN -> BOLTON, say) as well as the
    origin departure.
    """
    return _stop_timings_for(shift_candidate(candidate, shift_minutes))


def _baseline_frequency(
    candidate: CandidateService,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    cutoffs: dict[str, str],
) -> float:
    """Mean existing trains a day on the candidate's end-to-end pair.

    Counts today's scheduled services on the full origin-to-destination
    pair (:func:`orta.service_lookup.existing_trains`) for
    each running day and averages over those days. That is the
    do-minimum frequency the wait-time and crowding benefits are
    measured against; ``0`` when the corridor has no comparable through
    service.
    """
    stops = candidate.stops
    if len(stops) < 2 or not candidate.days_of_week:
        return 0.0
    origin = stops[0].station.strip().upper()
    destination = stops[-1].station.strip().upper()
    counts: list[int] = []
    for weekday in candidate.days_of_week:
        existing = existing_trains(calls, reference_date, weekday, origin, destination, cutoffs)
        counts.append(len(existing.get("dep_from_origin", [])))
    return sum(counts) / len(counts) if counts else 0.0


def _leg_time_saving(
    config: ProposeConfig,
    candidate: CandidateService,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    demand: pd.DataFrame,
) -> float | None:
    """Minutes saved per rider, weighted by trips, or ``None``.

    Each leg is compared with the best journey today's timetable offers
    for its pair (see
    :func:`~orta.service_lookup.best_existing_minutes`: the
    fastest direct train, or the quickest single change plus
    ``interchange_penalty_minutes``), floored at zero, and weighted by
    the leg's captured trips in ``demand``. A new direct link saves its
    riders the change; a leg today's trains already run faster saves
    nothing. Pairs with no journey today (all of their riders are
    generated) carry no weight.
    """
    if demand.empty:
        return None
    filtered = day_services(calls, reference_date, candidate.days_of_week[0])
    trips = demand.groupby(["leg_stop_i", "leg_stop_j"])
    trips = trips["predicted_annual_trips_captured"].sum()
    stops = candidate.stops
    saved = weight = 0.0
    for (i, j), leg_trips in trips.items():
        depart = hhmm_minutes(stops[i].departure)
        arrive = hhmm_minutes(stops[j].arrival)
        before = best_existing_minutes(
            filtered, stops[i].station, stops[j].station, config.interchange_penalty_minutes
        )
        if depart is None or arrive is None or before is None or not leg_trips > 0:
            continue
        saved += leg_trips * max(0.0, before - (arrive - depart) % MINUTES_PER_DAY)
        weight += leg_trips
    return saved / weight if weight else None


# ----------------------------------------------------------------------
# Departure evaluation
# ----------------------------------------------------------------------
def _evaluate_single_departure(
    config: ProposeConfig,
    candidate: CandidateService,
    crs_tiploc: dict[str, str],
    graph: BplanTimingGraph,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    gravity: object,
    fares: object,
    network_links: BplanNetworkLinks | None = None,
    transfers: TransferDemand | None = None,
) -> pd.DataFrame:
    """Evaluate the candidate exactly as typed (no departure window).

    Returns a one-row frame with the same shape
    :func:`orta.feasibility.search_departure_window`
    produces, so :func:`generate_scenarios` can treat both cases
    uniformly.
    """
    running = check_running_times(candidate, crs_tiploc, graph)
    running_feasible = running.empty or not (running["status"] == "infeasible").any()
    # The headway and demand lookups match the recoded calls.
    candidate = recode_stops(candidate, resolve_recodes(config.station_recodes))

    headway = check_headway(
        config,
        candidate,
        calls,
        reference_date,
        crs_tiploc=crs_tiploc,
        network_links=network_links,
    )
    conflict = bool(headway["conflict"].any()) if not headway.empty else False
    gaps = []
    if not headway.empty:
        gaps = [
            g
            for g in pd.concat(
                [headway["gap_before_minutes"], headway["gap_after_minutes"]]
            ).tolist()
            if g is not None and not pd.isna(g)
        ]
    worst_gap = min(gaps) if gaps else None

    demand = predict_demand(
        config, candidate, calls, reference_date, gravity, fares, transfers=transfers
    )
    trips = (
        demand["predicted_annual_trips_captured"].sum(skipna=True) if not demand.empty else None
    )
    generated = (
        demand["predicted_annual_trips_generated"].sum(skipna=True) if not demand.empty else None
    )
    revenue = demand["predicted_annual_revenue_gbp"].sum(skipna=True) if not demand.empty else None

    return pd.DataFrame(
        [
            {
                "shift_minutes": 0.0,
                "stop_0_departure": stop_time(candidate.stops[0]),
                "feasible": running_feasible and not conflict,
                "worst_gap_minutes": worst_gap,
                "predicted_annual_trips_captured": trips,
                "predicted_annual_trips_generated": generated,
                "predicted_annual_revenue_gbp": revenue,
            }
        ]
    )


def _departure_frame(
    config: ProposeConfig,
    candidate: CandidateService,
    crs_tiploc: dict[str, str],
    graph: BplanTimingGraph,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    gravity: object,
    fares: object,
    network_links: BplanNetworkLinks | None = None,
    transfers: TransferDemand | None = None,
) -> pd.DataFrame:
    """List the departure slots to try: the window, or the typed one."""
    if candidate.search_window is not None:
        return search_departure_window(
            config,
            candidate,
            crs_tiploc,
            graph,
            calls,
            reference_date,
            gravity,
            fares,
            transfers=transfers,
        )
    return _evaluate_single_departure(
        config,
        candidate,
        crs_tiploc,
        graph,
        calls,
        reference_date,
        gravity,
        fares,
        network_links=network_links,
        transfers=transfers,
    )


def _frequency_scaling(
    config: ProposeConfig,
    candidate: CandidateService,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    gravity: object,
    fares: object,
    transfers: TransferDemand | None,
    frequency_options: list[int],
) -> dict[int, tuple[float, float | None]]:
    """Each frequency's demand relative to one train.

    The departure sweep prices one train. ``f`` trains a day are ``f``
    copies of it spread over the day's periods, ``k = f / periods`` (at
    least one) to a period, so their demand is ``f`` times one copy's
    with the day's frequency set to ``f`` and ``k`` copies sharing its
    slot: the pivot's frequency response and the first-link step (case
    24: only at 4 or more a day) both read the day's trains, which a
    flat ``f ** elasticity`` could not. The multipliers come from the
    candidate as typed and apply to every departure slot. ``generated``
    is the option's generated trips over one train's trips, so it is
    defined when one train generates none. When the typed train predicts
    no trips at all there is nothing to scale from, so ``generated`` is
    ``None`` and each slot falls back to its own train's split, as with
    no scaling.
    """
    # ponytail: copies are assumed to load like the typed train and to
    # spread evenly over the periods; time each copy if a corridor's
    # slots differ much. The named cutoffs leave an early period before
    # the first one; cutoffs starting at midnight (hourly, half-hourly)
    # do not.
    periods = len(period_minutes(config.time_periods_cutoffs))

    def totals(per_day: int) -> tuple[float, float]:
        per_period = max(1, round(per_day / periods))
        demand = predict_demand(
            config,
            candidate,
            calls,
            reference_date,
            gravity,
            fares,
            added_trains=lambda _d, period, _o, _x: per_day if period == "day" else per_period,
            transfers=transfers,
        )
        if demand.empty:
            return 0.0, 0.0
        return (
            float(demand["predicted_annual_trips_captured"].sum(skipna=True)),
            float(demand["predicted_annual_trips_generated"].sum(skipna=True)),
        )

    one, _ = totals(1)
    scaling: dict[int, tuple[float, float | None]] = {}
    for freq in frequency_options:
        trips, generated = totals(freq)
        scaling[freq] = (
            (freq * trips / one, freq * generated / one) if one > 0 else (float(freq), None)
        )
    return scaling


def _options_for_departure(
    row: pd.Series,
    assumptions: ScenarioAssumptions,
    name: str,
    time_saving: float,
    *,
    seats_per_train: float = 0.0,
    in_vehicle_time_minutes: float = 0.0,
    baseline_trains_per_day: float = 0.0,
    service_days_per_year: float = 253.0,
    stop_schedule: list[StopTiming] | None = None,
    total_travel_minutes: float = 0.0,
    scaling: dict[int, tuple[float, float | None]] | None = None,
) -> list[ScenarioOption]:
    """Fan one feasible slot out into one option per frequency.

    ``scaling`` is :func:`_frequency_scaling`'s; without it each option
    is ``f`` times the slot's train.
    """
    base_trips = row["predicted_annual_trips_captured"]
    if pd.isna(base_trips) or base_trips <= 0:
        return []
    base_revenue = row["predicted_annual_revenue_gbp"]

    base_generated = row.get("predicted_annual_trips_generated")
    if base_generated is None or pd.isna(base_generated):
        base_generated = base_trips

    options: list[ScenarioOption] = []
    for freq in assumptions.frequency_options:
        scale, generated_scale = (scaling or {}).get(freq, (float(freq), None))
        pax = round(base_trips * scale)
        generated_pax = (
            round(base_generated * scale)
            if generated_scale is None or freq == 1
            else round(base_trips * generated_scale)
        )
        generated_pax = min(pax, generated_pax)
        if pax <= 0:
            continue
        if pd.notna(base_revenue) and base_revenue > 0:
            avg_fare = float(base_revenue) * scale / pax
        else:
            avg_fare = assumptions.default_avg_fare
        # Only generated trips are new fare revenue; abstracted riders
        # already paid.
        revenue = generated_pax * avg_fare

        bcr, npv, pvb, pvc = _quick_metrics(
            _build_params(
                assumptions,
                name,
                pax,
                avg_fare,
                time_saving,
                freq,
                generated_passengers=generated_pax,
                seats_per_train=seats_per_train,
                in_vehicle_time_minutes=in_vehicle_time_minutes,
                baseline_trains_per_day=baseline_trains_per_day,
                service_days_per_year=service_days_per_year,
            )
        )
        freq_label = "" if freq == 1 else f", {freq}/day"
        options.append(
            ScenarioOption(
                label=f"{row['stop_0_departure']}{freq_label}",
                shift_minutes=float(row["shift_minutes"]),
                departure_time=row["stop_0_departure"],
                trains_per_day=int(freq),
                feasible=True,
                worst_gap_minutes=(
                    None if pd.isna(row["worst_gap_minutes"]) else float(row["worst_gap_minutes"])
                ),
                annual_passengers=pax,
                annual_passengers_generated=generated_pax,
                annual_revenue=round(revenue, 2),
                avg_fare=round(avg_fare, 2),
                time_saving_minutes=time_saving,
                bcr=bcr,
                npv=npv,
                pvb=pvb,
                pvc=pvc,
                seats_per_train=seats_per_train,
                in_vehicle_time_minutes=in_vehicle_time_minutes,
                baseline_trains_per_day=baseline_trains_per_day,
                service_days_per_year=service_days_per_year,
                stops=list(stop_schedule) if stop_schedule else [],
                total_travel_minutes=total_travel_minutes,
            )
        )
    return options


@dataclass
class _CorridorContext:
    """Corridor context for the frequency and crowding benefits.

    Bundled to keep the helpers' signatures short.
    """

    seats_per_train: float
    in_vehicle_time_minutes: float
    baseline_trains_per_day: float
    service_days_per_year: float


def _uniform_sweep_options(
    departures: pd.DataFrame,
    candidate: CandidateService,
    assumptions: ScenarioAssumptions,
    time_saving: float,
    corridor: _CorridorContext,
    scaling: dict[int, tuple[float, float | None]] | None = None,
) -> tuple[list[ScenarioOption], str | None, int, int]:
    """Appraise every feasible row of the uniform-shift sweep.

    Returns ``(options, note, n_evaluated, n_feasible)``. ``note`` is
    only set when the running-time short circuit fired: a uniform shift
    changes no leg's running time, so a template that is running-time
    infeasible as typed is infeasible at every shift (see
    :func:`~orta.feasibility.search_departure_window`).
    """
    if departures.empty:
        return [], None, 0, 0

    if "notes" in departures.columns and departures["shift_minutes"].isna().all():
        note = str(departures.iloc[0].get("notes") or "").strip() or None
        return [], note, 0, 0

    n_evaluated = len(departures)
    feasible_rows = departures[departures["feasible"].fillna(False)]
    n_feasible = len(feasible_rows)
    logger.info(
        "%s of %s departure slot(s) feasible; appraising %s frequency option(s) each ...",
        n_feasible,
        n_evaluated,
        len(assumptions.frequency_options),
    )
    options: list[ScenarioOption] = []
    for _, row in feasible_rows.iterrows():
        stop_schedule, travel_minutes = _stop_timeline(candidate, float(row["shift_minutes"]))
        options.extend(
            _options_for_departure(
                row,
                assumptions,
                candidate.name,
                time_saving,
                seats_per_train=corridor.seats_per_train,
                in_vehicle_time_minutes=corridor.in_vehicle_time_minutes,
                baseline_trains_per_day=corridor.baseline_trains_per_day,
                service_days_per_year=corridor.service_days_per_year,
                stop_schedule=stop_schedule,
                total_travel_minutes=travel_minutes,
                scaling=scaling,
            )
        )
    return options, None, n_evaluated, n_feasible


def _flexible_slot_options(
    config: ProposeConfig,
    candidate: CandidateService,
    assumptions: ScenarioAssumptions,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    gravity: object,
    fares: object,
    time_saving: float,
    corridor: _CorridorContext,
    transfers: TransferDemand | None = None,
    scaling: dict[int, tuple[float, float | None]] | None = None,
) -> list[ScenarioOption]:
    """Appraise a multi-stop candidate's flexible-dwell slot.

    A uniform whole-candidate shift always suffices for a single leg, so
    :func:`~orta.feasibility.search_flexible_slot` (see its
    docstring for the LIV -> WGW -> BON rationale) is only worth running
    for 3+ stops, and its result only becomes an option if it found a
    slot that clears every leg; an infeasible result is logged instead.
    Returns ``[]`` when there is nothing to add.
    """
    if candidate.search_window is None or len(candidate.stops) <= 2:
        return []

    logger.info("Searching for a flexible-dwell slot that clears every leg independently ...")
    flexible = search_flexible_slot(config, candidate, calls, reference_date)
    if flexible is None:
        return []
    if not flexible.feasible:
        logger.info("Flexible-dwell slot search: %s", flexible.notes)
        return []

    stop_schedule, travel_minutes = _stop_timings_for(flexible.candidate)
    demand = predict_demand(
        config, flexible.candidate, calls, reference_date, gravity, fares, transfers=transfers
    )
    empty = demand.empty
    trips = demand["predicted_annual_trips_captured"].sum(skipna=True) if not empty else None
    generated = demand["predicted_annual_trips_generated"].sum(skipna=True) if not empty else None
    revenue = demand["predicted_annual_revenue_gbp"].sum(skipna=True) if not empty else None
    row = pd.Series(
        {
            "shift_minutes": float("nan"),
            "stop_0_departure": stop_time(flexible.candidate.stops[0]),
            "worst_gap_minutes": flexible.worst_gap_minutes,
            "predicted_annual_trips_captured": trips,
            "predicted_annual_trips_generated": generated,
            "predicted_annual_revenue_gbp": revenue,
        }
    )
    flex_options = _options_for_departure(
        row,
        assumptions,
        candidate.name,
        time_saving,
        seats_per_train=corridor.seats_per_train,
        in_vehicle_time_minutes=corridor.in_vehicle_time_minutes,
        baseline_trains_per_day=corridor.baseline_trains_per_day,
        service_days_per_year=corridor.service_days_per_year,
        stop_schedule=stop_schedule,
        total_travel_minutes=travel_minutes,
        scaling=scaling,
    )
    if flex_options:
        for opt in flex_options:
            opt.label = f"{opt.label} · flexible dwell"
        logger.info(
            "Flexible-dwell slot: %s departure, every leg independently clear of conflicts.",
            row["stop_0_departure"],
        )
    return flex_options


def _diagnose_empty_result(
    config: ProposeConfig,
    gravity: object,
    running_times: pd.DataFrame,
    n_evaluated: int,
    n_feasible: int,
) -> str:
    """Explain why the sweep appraised no options, and what to fix.

    Only called when there are no ``options`` and no running-time
    ``note`` already covers it (that case, every shift infeasible
    because one leg can never make its running time, is diagnosed in
    :func:`~orta.feasibility.search_departure_window`).
    Checked in the order the pipeline runs: no slots evaluated, then
    feasibility, then demand. The first stage that produced nothing is
    the one explained.
    """
    if n_evaluated == 0:
        return (
            "No departure slots could be evaluated. Check the candidate's "
            "stop times are set and, if a search window is configured, that its "
            "earliest/latest shift and step are set correctly."
        )
    if n_feasible == 0:
        infeasible_legs = (
            running_times[running_times["status"] == "infeasible"]
            if not running_times.empty
            else running_times
        )
        if not infeasible_legs.empty:
            return (
                f"All {n_evaluated} trial departure slot(s) failed the running-time "
                "check: the candidate's timed legs are shorter than the BPLAN "
                "minimum. Lengthen the candidate's running times, or check the "
                "traction code and timing graph."
            )
        return (
            f"All {n_evaluated} trial departure slot(s) conflict with an existing "
            "service (headway check failed at every shift). Widen the search "
            "window, or lower headway_safety_margin_minutes or "
            "min_headway_floor_minutes."
        )
    # Slots are feasible, but every one produced zero (or no) usable
    # demand.
    if config.period_matrices_dir is None:
        return (
            f"{n_feasible} departure slot(s) are feasible, but no period demand "
            "matrices are configured, so the optimiser has no way to predict "
            "passengers. Set period_matrices_dir."
        )
    if gravity is None:
        return (
            f"{n_feasible} departure slot(s) are feasible, but predicted demand "
            "was zero on every one: these O-D pairs have no existing service in "
            "the period matrices to pivot from, and no gravity/catchment model is "
            "configured to estimate demand for a new pair. Set table_1410, "
            "distance and a loading-analysis file (alongside the period matrices) "
            "to enable it."
        )
    return (
        f"{n_feasible} departure slot(s) are feasible, but predicted demand was "
        "zero on every one: the gravity/catchment model couldn't estimate these "
        "O-D pairs (too few loading observations for this route type, or the "
        "pairs fall outside its calibrated bands)."
    )


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def generate_scenarios(
    config: ProposeConfig,
    assumptions: ScenarioAssumptions,
    context: ScenarioContext | None = None,
) -> ScenarioResult:
    """Optimise the candidate and appraise every feasible variant.

    Parameters
    ----------
    config : ProposeConfig
        Propose configuration whose ``services`` list holds exactly one
        candidate (the template). Its ``search_window``, if set, drives
        the departure-time sweep; its data-source paths (timetable
        calls, BPLAN output, period matrices, fares) drive feasibility
        and demand exactly as the ``propose`` command uses them.
    assumptions : ScenarioAssumptions
        WebTAG appraisal levers plus the frequency search space.
    context : ScenarioContext | None
        Pre-built data sources from :func:`build_scenario_context`.
        Built fresh from ``config`` when omitted (the single-candidate
        case); a caller sweeping many candidates against the same
        ``config`` should build it once and pass it in, since it holds
        most of a run's cost (see :class:`ScenarioContext`).

    Returns
    -------
    ScenarioResult
        Feasible options ranked best value-for-money first, plus the
        running-time detail table and evaluation counts.
    """
    if not config.services:
        raise ValueError("Scenario needs exactly one candidate service; none given.")
    candidate = config.services[0]
    # The timetable's second codes were rewritten when it was read
    # (build_scenario_context), so the stops are recoded the same way
    # before any lookup, as `propose` does. The running-time check keeps
    # the stops as typed, because each looks up its own platform group's
    # TIPLOC.
    recoded = recode_stops(candidate, resolve_recodes(config.station_recodes))

    if context is None:
        context = build_scenario_context(config)
    calls = context.calls
    crs_tiploc = context.crs_tiploc
    reference_date = context.reference_date
    graph = context.graph
    network_links = context.network_links
    gravity = context.gravity
    fares = context.fares
    prepared = context.prepared

    # Corridor context for the frequency (wait-time) and crowding
    # benefits.
    seats_per_train = candidate.capacity or assumptions.default_seats_per_train or 0.0
    in_vehicle_time = _candidate_journey_minutes(candidate) or 0.0
    baseline_trains = _baseline_frequency(
        recoded, prepared, reference_date, config.time_periods_cutoffs
    )
    service_days = max(1.0, len(candidate.days_of_week) * 52.0)
    if assumptions.operating_hours_per_day > 0 and baseline_trains > 0:
        logger.info(
            "Corridor has ~%.1f existing train(s)/day; crediting frequency and crowding "
            "benefits for the added capacity.",
            baseline_trains,
        )

    running_times = check_running_times(candidate, crs_tiploc, graph)

    # Journeys changing onto the candidate (Manchester -> Birmingham ->
    # Gatwick) ride its legs too; routed once on the timetable with it
    # as typed. ponytail: not re-routed per departure shift; a shift
    # moving a leg into another period routes its changers in the old
    # one.
    transfers = (
        transfer_demand(
            config,
            pd.concat([calls, candidate_calls(recoded, reference_date)]),
            reference_date,
        )
        if config.period_matrices_dir is not None
        else None
    )
    scaling = _frequency_scaling(
        config,
        recoded,
        prepared,
        reference_date,
        gravity,
        fares,
        transfers,
        list(assumptions.frequency_options),
    )
    time_saving = _leg_time_saving(
        config,
        recoded,
        prepared,
        reference_date,
        predict_demand(
            config, recoded, prepared, reference_date, gravity, fares, transfers=transfers
        ),
    )
    if time_saving is None:
        time_saving = assumptions.default_time_saving_minutes
        logger.info("No leg has a journey today; journey-time saving %.1f min.", time_saving)
    else:
        logger.info("Riders save %.1f min on today's best journeys (trip-weighted).", time_saving)

    logger.info("Sweeping departure slots for feasibility and demand ...")
    departures = _departure_frame(
        config,
        candidate,
        crs_tiploc,
        graph,
        calls,
        reference_date,
        gravity,
        fares,
        network_links=network_links,
        transfers=transfers,
    )

    corridor = _CorridorContext(
        seats_per_train=seats_per_train,
        in_vehicle_time_minutes=in_vehicle_time,
        baseline_trains_per_day=baseline_trains,
        service_days_per_year=service_days,
    )
    options, note, n_evaluated, n_feasible = _uniform_sweep_options(
        departures, candidate, assumptions, time_saving, corridor, scaling
    )

    # A uniform shift can't always clear every leg of a multi-stop
    # candidate (LIV -> WGW -> BON might need a different shift per
    # leg), so also find each leg its own clearest slot, bridged by
    # flexible intermediate dwell, and add it as one more option.
    options.extend(
        _flexible_slot_options(
            config,
            recoded,
            assumptions,
            prepared,
            reference_date,
            gravity,
            fares,
            time_saving,
            corridor,
            transfers,
            scaling,
        )
    )

    # Rank by NPV, best first (BCR breaks ties for equal-NPV variants).
    options.sort(key=lambda o: (o.npv, o.bcr), reverse=True)

    if options:
        best = options[0]
        logger.info(
            "Best feasible option: %s, BCR %.2f, NPV £%.1fm, %s pax/yr.",
            best.label,
            best.bcr,
            best.npv,
            f"{best.annual_passengers:,}",
        )
    else:
        if note is None:
            note = _diagnose_empty_result(config, gravity, running_times, n_evaluated, n_feasible)
        logger.info("No feasible, demand-positive option found for this candidate: %s", note)

    return ScenarioResult(
        template_name=candidate.name,
        options=options,
        time_saving_minutes=time_saving,
        n_departures_evaluated=n_evaluated,
        n_departures_feasible=n_feasible,
        running_times=running_times,
        note=note,
    )


# ----------------------------------------------------------------------
# CLI wrapper
# ----------------------------------------------------------------------
def _option_row(candidate_name: str, rank: int, opt: ScenarioOption) -> dict[str, object]:
    return {
        "candidate_name": candidate_name,
        "rank": rank,
        "label": opt.label,
        "departure_time": opt.departure_time,
        "shift_minutes": opt.shift_minutes,
        "trains_per_day": opt.trains_per_day,
        "feasible": opt.feasible,
        "worst_gap_minutes": opt.worst_gap_minutes,
        "annual_passengers": opt.annual_passengers,
        "annual_passengers_generated": opt.annual_passengers_generated,
        "annual_revenue": opt.annual_revenue,
        "avg_fare": opt.avg_fare,
        "time_saving_minutes": opt.time_saving_minutes,
        "total_travel_minutes": opt.total_travel_minutes,
        "baseline_trains_per_day": opt.baseline_trains_per_day,
        "bcr": opt.bcr,
        "npv_m": opt.npv,
        "pvb_m": opt.pvb,
        "pvc_m": opt.pvc,
        "schedule": " ".join(
            f"{st.station}({st.arrival or ''}/{st.departure or ''})" for st in opt.stops
        ),
    }


def with_route_cost(
    assumptions: ScenarioAssumptions, candidate: CandidateService, gravity: object
) -> ScenarioAssumptions:
    """Cost ``operating_cost_per_train_km`` over the track distance.

    Returns ``assumptions`` with ``annual_operating_cost_m`` (per daily
    train) raised by rate x km x service days; unchanged when the rate
    is 0 or no distance model is loaded. A leg the distance model cannot
    route (a stop it has no location for, such as a new-build station)
    is costed at the mean of the legs it can, rather than ending the
    run.
    """
    if assumptions.operating_cost_per_train_km <= 0 or gravity is None:
        return assumptions
    legs = list(itertools.pairwise(candidate.stops))
    routed, missing = [], []
    for a, b in legs:
        try:
            leg_km = gravity.calc.calculate_distance(a.station, b.station, plausible=True)
            routed.append(leg_km)
        except ValueError:
            missing.append(f"{a.station}-{b.station}")
    if not routed:
        logger.warning(
            "%s: no leg routes on the distance model, so no train-km operating cost is added.",
            candidate.name,
        )
        return assumptions
    if missing:
        logger.warning(
            "%s: %s not routable; costed at the mean of the other legs.",
            candidate.name,
            ", ".join(missing),
        )
    km = sum(routed) * len(legs) / len(routed)
    service_days = max(1.0, len(candidate.days_of_week) * 52.0)
    per_path_m = assumptions.operating_cost_per_train_km * km * service_days / 1e6
    logger.info(
        "%s: %.0f km at £%.0f/train-km, £%.2fm a year per daily train.",
        candidate.name,
        km,
        assumptions.operating_cost_per_train_km,
        per_path_m,
    )
    return replace(
        assumptions, annual_operating_cost_m=assumptions.annual_operating_cost_m + per_path_m
    )


def run_scenario_command(config: ScenarioConfig) -> None:
    """Run Scenario Studio for every candidate in ``config.services``.

    Writes, to ``config.output_path``:

    * ``scenario_options.csv``: every feasible option of every
      candidate, ranked
    * ``scenario_summary.csv``: one row per candidate, with the best
      option, its full WebTAG appraisal and commercial case, or the
      reason nothing was feasible
    * ``scenario_{name}_running_times.csv``: per-leg running-time check
    """
    config.output_path.mkdir(parents=True, exist_ok=True)
    tag = TagDataBook(config.tag_databook_path)
    assumptions = ScenarioAssumptions(
        tag=tag,
        **{
            f.name: getattr(config.assumptions, f.name)
            for f in dataclasses.fields(ScenarioAssumptions)
            if f.name != "tag"
        },
    )
    context = build_scenario_context(config)

    option_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for candidate in config.services:
        logger.info("Scenario: optimising %s ...", candidate.name)
        single = config.model_copy(update={"services": [candidate]})
        costed = with_route_cost(assumptions, candidate, context.gravity)
        result = generate_scenarios(single, costed, context=context)
        result.running_times.to_csv(
            config.output_path / f"scenario_{candidate.name}_running_times.csv", index=False
        )
        option_rows.extend(
            _option_row(candidate.name, rank, opt)
            for rank, opt in enumerate(result.options, start=1)
        )

        row: dict[str, object] = {
            "candidate_name": candidate.name,
            "n_departures_evaluated": result.n_departures_evaluated,
            "n_departures_feasible": result.n_departures_feasible,
            "n_options": len(result.options),
            "note": result.note,
        }
        best = result.best
        if best is not None:
            appraisal = appraise_option(
                costed,
                candidate.name,
                best.annual_passengers,
                best.avg_fare,
                best.time_saving_minutes,
                best.trains_per_day,
                seats_per_train=best.seats_per_train,
                in_vehicle_time_minutes=best.in_vehicle_time_minutes,
                baseline_trains_per_day=best.baseline_trains_per_day,
                service_days_per_year=best.service_days_per_year,
                generated_passengers=best.annual_passengers_generated,
            )
            commercial = commercial_case_for_option(costed, candidate.name, best)
            row.update(
                {
                    **{
                        f"best_{k}": v
                        for k, v in _option_row(candidate.name, 1, best).items()
                        if k not in {"candidate_name", "rank"}
                    },
                    "webtag_pvb_m": appraisal.pvb,
                    "webtag_pvc_m": appraisal.pvc,
                    "webtag_npv_m": appraisal.npv,
                    "webtag_bcr": appraisal.bcr,
                    "commercial_npv_m": commercial.npv_m,
                    "commercial_irr": commercial.irr,
                    "commercial_payback_year": commercial.payback_year,
                    "commercial_total_revenue_m": commercial.total_revenue_m,
                }
            )
        summary_rows.append(row)

    pd.DataFrame(option_rows).to_csv(config.output_path / "scenario_options.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(config.output_path / "scenario_summary.csv", index=False)
    logger.info(
        "Scenario run complete: %d candidate(s), %d feasible option(s) written.",
        len(config.services),
        len(option_rows),
    )
