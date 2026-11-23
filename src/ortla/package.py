"""Price services that run together (``ScenarioConfig.package``).

``scenario`` prices each candidate as the only new train, so candidates that share station pairs
can't be added up. Here every train is priced with the others as added trains, the way a hindcast's
joint mode prices the trains it puts back, and the package is appraised as one scheme. Departures
are as entered: ``scenario`` on one service finds a clear time.

Other package trains are weighted like the train being priced when a slot is shared, so unlike
trains are given slightly less than their slot between them, never more.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import pandas as pd

from ortla import appraisal, commercial_case, ranges, scenario
from ortla.demand import candidate_calls, predict_demand, transfer_demand
from ortla.feasibility import check_headway, check_running_times, recode_stops
from ortla.hindcast import _joint_counter
from ortla.service_lookup import prepare_calls
from ortla.stations import resolve_recodes
from ortla.timetable import SERVICE_KEY

if TYPE_CHECKING:
    from ortla.config import CandidateService, ScenarioConfig

logger = logging.getLogger(__name__)

_CAPTURED = "predicted_annual_trips_captured"
_GENERATED = "predicted_annual_trips_generated"
_GENERATED_SLOT = "predicted_annual_trips_generated_slot"
_REVENUE = "predicted_annual_revenue_gbp"


@dataclasses.dataclass
class PackageResult:
    """A package priced and appraised as one scheme."""

    services: pd.DataFrame
    legs: pd.DataFrame
    running_times: pd.DataFrame
    headways: pd.DataFrame
    appraisal: appraisal.AppraisalResults
    commercial: commercial_case.CommercialCaseResults
    summary: dict[str, object]
    paths: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    ranges: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)


def package_demand(
    config: ScenarioConfig, trains: list[CandidateService], context: scenario.ScenarioContext
) -> pd.DataFrame:
    """Each train's demand by leg with the rest of the package running."""
    counts = _joint_counter(trains, config.time_periods_cutoffs)
    after = [context.calls, *(candidate_calls(t, context.reference_date) for t in trains)]
    transfers = transfer_demand(config, pd.concat(after), context.reference_date)
    return pd.concat(
        [
            predict_demand(
                config,
                train,
                context.prepared,
                context.reference_date,
                context.gravity,
                context.fares,
                added_trains=lambda day, period, o, d: counts[(day, period, o, d)],
                transfers=transfers,
            )
            for train in trains
        ],
        ignore_index=True,
    )


def package_headways(
    config: ScenarioConfig, trains: list[CandidateService], context: scenario.ScenarioContext
) -> pd.DataFrame:
    """Each train's gaps to today's trains and to the rest of the package."""
    rows = []
    # A gap is read only between the package's own stations, so only whole services calling at one
    # of them matter. Filtering the national timetable once, not once per train, gives the same rows.
    stations = {stop.station.strip().upper() for train in trains for stop in train.stops}
    base = context.prepared
    keys = [base[column] for column in SERVICE_KEY]
    local = base[base["crs"].isin(stations).groupby(keys, dropna=False).transform("any")]
    for k, train in enumerate(trains):
        others = [
            candidate_calls(other, context.reference_date)
            for i, other in enumerate(trains)
            if i != k
        ]
        calls = prepare_calls(pd.concat([local, *others])) if others else context.prepared
        rows.append(
            check_headway(
                config,
                train,
                calls,
                context.reference_date,
                crs_tiploc=context.crs_tiploc,
                network_links=context.network_links,
            )
        )
    return pd.concat(rows, ignore_index=True)


def price_package(
    config: ScenarioConfig,
    assumptions: scenario.ScenarioAssumptions,
    context: scenario.ScenarioContext,
) -> PackageResult:
    """Check, price and appraise ``config.services`` as one scheme, with its paths and range."""
    result = _price(config, assumptions, context)
    if context.paths is not None:
        recodes = resolve_recodes(config.station_recodes)
        result.paths = context.paths.check_together(
            [recode_stops(service, recodes) for service in config.services]
        )
        result.summary["paths_clear"] = bool(result.paths["path_clear"].all())
    if config.ranges is not None:
        result.ranges = _package_ranges(config, assumptions, context, result)
    return result


def _package_ranges(
    config: ScenarioConfig,
    assumptions: scenario.ScenarioAssumptions,
    context: scenario.ScenarioContext,
    central: PackageResult,
) -> pd.DataFrame:
    """Price the whole package at each case. Its trains' errors move together."""
    rows = []
    for name, load, case in ranges.cases(config):
        priced = central
        if name != "central":
            priced = _price(case, assumptions, context, load, central)
        rows.append(
            {
                "case": name,
                "load_multiplier": load,
                "frequency_elasticity": case.served_pair_frequency_elasticity,
                "new_link_uplift": case.new_link_uplift,
                **{key: priced.summary[key] for key in ranges.RANGE_COLUMNS[4:]},
            }
        )
    return pd.DataFrame(rows, columns=ranges.RANGE_COLUMNS)


def _price(  # noqa: C901 (one pass over the trains)
    config: ScenarioConfig,
    assumptions: scenario.ScenarioAssumptions,
    context: scenario.ScenarioContext,
    load: float = 1.0,
    checked: PackageResult | None = None,
) -> PackageResult:
    """Price one case, with ``load`` times the demand; ``checked`` lends its feasibility."""
    names = [s.name for s in config.services]
    if not names or len(set(names)) != len(names):
        raise ValueError("A package needs at least one service, each with its own name.")

    recodes = resolve_recodes(config.station_recodes)
    trains = [recode_stops(s, recodes) for s in config.services]
    if checked is None:
        running = pd.concat(
            [
                check_running_times(s, context.crs_tiploc, context.graph).assign(
                    candidate_name=s.name
                )
                for s in config.services
            ],
            ignore_index=True,
        )
        headways = package_headways(config, trains, context)
    else:
        running, headways = checked.running_times, checked.headways
    legs = package_demand(config, trains, context)
    if load != 1.0 and not legs.empty:
        scaled = [_CAPTURED, _GENERATED, _GENERATED_SLOT, _REVENUE]
        legs[scaled] = legs[scaled].astype(float) * load
    if legs.empty:
        raise ValueError("The package has no demand to price. Set period_matrices_dir.")

    rows, proposed, do_minimum, revenue_side = [], [], [], []
    operating_m = 0.0
    for typed, train in zip(config.services, trains, strict=True):
        own = legs[legs["candidate_name"] == train.name]
        passengers = float(own[_CAPTURED].sum())
        generated = min(passengers, float(own[_GENERATED].sum()))
        generated_slot = float(own[_GENERATED_SLOT].fillna(0).sum())
        carried = float(own[_REVENUE].sum())
        fare = carried / passengers if carried > 0 else assumptions.default_avg_fare
        saving = scenario._leg_time_saving(
            config, train, context.prepared, context.reference_date, own
        )
        if saving is None:
            saving = assumptions.default_time_saving_minutes
        minutes = scenario._candidate_journey_minutes(train) or 0.0
        own_cost = scenario.with_route_cost(
            assumptions, typed, context.gravity
        ).annual_operating_cost_m
        operating_m += own_cost
        seats = typed.capacity or assumptions.default_seats_per_train or 0.0
        service = appraisal.AppraisalServiceInput(
            name=train.name,
            annual_passengers=round(passengers),
            journey_time_saved_minutes=saving,
            avg_fare=fare,
            trains_per_day=1.0,
            seats_per_train=seats,
            in_vehicle_time_minutes=minutes,
        )
        proposed.append(service)
        if passengers > generated:
            do_minimum.append(
                appraisal.AppraisalServiceInput(
                    name=train.name,
                    annual_passengers=round(passengers) - round(generated),
                    journey_time_saved_minutes=0.0,
                    avg_fare=fare,
                )
            )
        revenue_side.append(
            commercial_case.CommercialServiceInput(train.name, round(generated), fare)
        )
        conflicts = headways[(headways["candidate_name"] == train.name)]["conflict"]
        timed = running[running["candidate_name"] == train.name]["status"]
        rows.append(
            {
                "candidate_name": train.name,
                "operator": typed.operator,
                "departure": train.stops[0].departure,
                "arrival": train.stops[-1].arrival,
                "days_a_year": len(train.days_of_week) * 52.0,
                "running_times_ok": not (timed == "infeasible").any(),
                "headway_conflicts": int(conflicts.sum()),
                "annual_passengers": passengers,
                "annual_passengers_generated": generated,
                "annual_passengers_generated_scheme": generated_slot,
                "annual_revenue_carried": carried,
                "annual_revenue_new": generated * fare,
                "avg_fare": fare,
                "time_saving_minutes": saving,
                "annual_operating_cost_m": own_cost,
            }
        )

    services = pd.DataFrame(rows)
    part_week = services[services["days_a_year"] < 7 * 52.0]
    if not part_week.empty:
        logger.warning(
            "%d of %d trains run on fewer than seven days, so their annual figures count only "
            "those days (%s: %d days a year). If they run every day, list every day.",
            len(part_week),
            len(services),
            part_week["candidate_name"].iloc[0],
            int(part_week["days_a_year"].iloc[0]),
        )
    wait_minutes = None
    if assumptions.operating_hours_per_day > 0:
        # A wrong load is a wrong base demand, so every rider on the pair moves with it.
        unscaled = legs
        if load != 1.0:
            unscaled = legs.copy()
            unscaled[scaled] = legs[scaled] / load
        wait_minutes = scenario.pair_wait_minutes(config, trains, context, unscaled, assumptions)
        if wait_minutes is not None:
            wait_minutes *= load
    service_days = float(services["days_a_year"].max())
    params = dataclasses.replace(
        scenario._build_params(
            assumptions,
            trains[0].name,
            0,
            0.0,
            0.0,
            1,
            service_days_per_year=service_days,
            wait_minutes_saved=wait_minutes,
        ),
        proposed_services=proposed,
        do_minimum_services=do_minimum,
        annual_operating_cost_m=operating_m,
    )
    results = appraisal.evaluate(params)
    commercial = commercial_case.run_commercial_case(
        commercial_case.CommercialCaseParameters(
            proposed_services=revenue_side,
            appraisal_year=assumptions.appraisal_year,
            appraisal_period=assumptions.appraisal_period,
            construction_period=assumptions.construction_period,
            demand_ramp_up_years=assumptions.demand_ramp_up_years,
            demand_ramp_up_profile=list(assumptions.demand_ramp_up_profile),
            capital_cost_m=assumptions.capital_cost_m,
            annual_operating_cost_m=operating_m,
            cost_inflation_rate=assumptions.cost_inflation_rate,
            fare_growth_rate=assumptions.commercial_fare_growth_rate,
            discount_rate=assumptions.commercial_discount_rate,
        )
    )
    summary = {
        "trains": len(trains),
        "fits": bool(services["running_times_ok"].all() and not headways["conflict"].any()),
        "annual_passengers": float(services["annual_passengers"].sum()),
        "annual_passengers_generated": float(services["annual_passengers_generated"].sum()),
        # Every journey new to rail, including those riding today's trains in the added
        # trains' slots. `annual_passengers_generated` is the added trains' own share.
        "annual_passengers_generated_scheme": float(
            services["annual_passengers_generated_scheme"].sum()
        ),
        "annual_revenue_new": float(services["annual_revenue_new"].sum()),
        "annual_operating_cost_m": operating_m,
        "wait_minutes_saved": wait_minutes,
        "webtag_pvb_m": results.pvb,
        "webtag_pvc_m": results.pvc,
        "webtag_npv_m": results.npv,
        "webtag_bcr": results.bcr,
        "commercial_npv_m": commercial.npv_m,
        "commercial_irr": commercial.irr,
        "commercial_payback_year": commercial.payback_year,
        "commercial_total_revenue_m": commercial.total_revenue_m,
    }
    return PackageResult(services, legs, running, headways, results, commercial, summary)


def run_package(config: ScenarioConfig) -> PackageResult:
    """Price the package and write its tables to ``config.output_path``.

    ``path_check`` adds ``package_paths.csv`` and ``ranges`` adds ``package_ranges.csv``.

    * ``package_summary.csv``: the scheme's totals, appraisal and commercial case
    * ``package_services.csv``: one row per train, priced with the others running
    * ``package_demand_by_leg.csv``, ``package_running_times.csv``,
      ``package_headways.csv``: the working
    """
    config.output_path.mkdir(parents=True, exist_ok=True)
    assumptions = scenario.ScenarioAssumptions(
        tag=appraisal.TagDataBook(config.tag_databook_path),
        **{
            f.name: getattr(config.assumptions, f.name)
            for f in dataclasses.fields(scenario.ScenarioAssumptions)
            if f.name != "tag"
        },
    )
    context = scenario.build_scenario_context(config)
    logger.info("Package: pricing %d trains together ...", len(config.services))
    result = price_package(config, assumptions, context)
    out = config.output_path
    pd.DataFrame([result.summary]).to_csv(out / "package_summary.csv", index=False)
    result.services.to_csv(out / "package_services.csv", index=False)
    result.legs.to_csv(out / "package_demand_by_leg.csv", index=False)
    result.running_times.to_csv(out / "package_running_times.csv", index=False)
    result.headways.to_csv(out / "package_headways.csv", index=False)
    if context.paths is not None:
        result.paths.to_csv(out / "package_paths.csv", index=False)
    if config.ranges is not None:
        result.ranges.to_csv(out / "package_ranges.csv", index=False)
    logger.info("Package run complete: %d trains priced together.", len(config.services))
    return result
