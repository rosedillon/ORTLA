"""Commercial case engine for rail service proposals.

Works out whether a proposed service pays for the operator running it,
with fare revenue as income and capital and operating costs as
outgoings. It reports Net Present Value (NPV) at the operator's discount
rate, Internal Rate of Return (IRR), payback period and a year-by-year
cashflow.

It is not a WebTAG socioeconomic appraisal. In WebTAG terms fare revenue
is a transfer payment from passenger to operator, not a net gain to
society, so it is netted off the public-sector budget rather than
counted as a benefit (see :mod:`ortla.appraisal`, which
Scenario Studio uses for that social cost-benefit case). For an operator
deciding whether to run the service, revenue is the benefit: there is no
Value of Time, mode-shift, carbon or decongestion term, and discounting
uses a flat commercial hurdle rate rather than the Green Book's stepped
social discount rates.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)


# Input / output data classes
@dataclass
class CommercialServiceInput:
    """One service (or service group) to appraise."""

    name: str
    annual_passengers: int
    avg_fare: float  # £ per passenger


@dataclass
class CommercialCaseParameters:
    """All parameters needed to run the commercial case."""

    proposed_services: list[CommercialServiceInput] = field(default_factory=list)
    do_minimum_services: list[CommercialServiceInput] = field(default_factory=list)

    appraisal_year: int = 2028
    appraisal_period: int = 30
    construction_period: int = 3

    # Demand ramp-up
    demand_ramp_up_years: int = 3
    demand_ramp_up_profile: list[float] = field(default_factory=lambda: [0.5, 0.75, 0.9, 1.0])
    demand_growth_rate: float = 0.0  # real-terms pa, over and above ramp-up

    # Costs (£ million)
    capital_cost_m: float = 0.0
    annual_operating_cost_m: float = 0.0
    cost_inflation_rate: float = 0.0  # real-terms pa

    # Revenue
    fare_growth_rate: float = 0.0  # real fare growth pa, on top of demand growth

    # The operator's own cost of capital / required return, used to
    # discount cashflows for the NPV. Not a Green Book social discount
    # rate.
    discount_rate: float = 0.08

    def total_annual_passengers(self, baseline: bool = False) -> int:
        """Annual passengers on the proposed or do-minimum services."""
        svcs = self.do_minimum_services if baseline else self.proposed_services
        return sum(s.annual_passengers for s in svcs)


@dataclass
class SensitivityCase:
    """One sensitivity-test result."""

    label: str
    parameter: str
    npv_m: float
    irr: float | None
    change_pct: float  # % change in NPV from the base case


@dataclass
class CommercialCaseResults:
    """Complete output of a commercial case run."""

    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)

    total_revenue_m: float = 0.0
    total_opex_m: float = 0.0
    total_capex_m: float = 0.0

    npv_m: float = 0.0
    irr: float | None = None
    payback_year: int | None = None  # calendar year, undiscounted
    discounted_payback_year: int | None = None

    sensitivity: list[SensitivityCase] = field(default_factory=list)

    @property
    def roi(self) -> float:
        """Return on investment: net cashflow per £ invested.

        ``inf`` when there was no capital investment but a positive net
        return; ``0.0`` when there was neither investment nor return.
        """
        if self.total_capex_m <= 0:
            net = self.total_revenue_m - self.total_opex_m
            return float("inf") if net > 0 else 0.0
        return (self.total_revenue_m - self.total_opex_m - self.total_capex_m) / self.total_capex_m


# Core calculation functions
def _demand_factor(params: CommercialCaseParameters, op_year: int) -> float:
    """Ramp-up times demand growth in operating year ``op_year``."""
    profile = params.demand_ramp_up_profile
    if op_year < params.demand_ramp_up_years and op_year < len(profile):
        ramp = profile[op_year]
    else:
        ramp = profile[-1] if profile else 1.0
    return ramp * ((1.0 + params.demand_growth_rate) ** op_year)


def _annual_revenue(
    params: CommercialCaseParameters,
    op_year: int,
    demand_factor: float,
) -> float:
    """Annual fare revenue (£) over the do-minimum, with fare growth."""
    fare_factor = (1.0 + params.fare_growth_rate) ** op_year
    baseline_rev = sum(s.annual_passengers * s.avg_fare for s in params.do_minimum_services)
    proposed_rev = sum(s.annual_passengers * s.avg_fare for s in params.proposed_services)
    return (proposed_rev - baseline_rev) * demand_factor * fare_factor


def _annual_operating_cost(params: CommercialCaseParameters, op_year: int) -> float:
    """Annual operating cost (£), grown with cost inflation.

    Inflation runs from the appraisal year.
    """
    return params.annual_operating_cost_m * 1e6 * ((1.0 + params.cost_inflation_rate) ** op_year)


def _run_cashflow(params: CommercialCaseParameters) -> pd.DataFrame:
    """Build the yearly cashflow table, undiscounted and discounted."""
    opening_year = params.appraisal_year
    end_year = opening_year + params.appraisal_period - 1
    construction_start = opening_year - params.construction_period

    annual_capital = params.capital_cost_m * 1e6 / max(1, params.construction_period)

    rows: list[dict[str, float]] = []
    cumulative_net = 0.0
    cumulative_discounted = 0.0

    for year in range(construction_start, end_year + 1):
        t = year - construction_start
        discount_factor = 1.0 / ((1.0 + params.discount_rate) ** t)

        is_operational = year >= opening_year
        is_construction = construction_start <= year < opening_year

        if is_operational:
            op_year = year - opening_year
            demand_factor = _demand_factor(params, op_year)
            revenue = _annual_revenue(params, op_year, demand_factor)
            opex = _annual_operating_cost(params, op_year)
            proposed_pax = params.total_annual_passengers(baseline=False) * demand_factor
            baseline_pax = params.total_annual_passengers(baseline=True) * demand_factor
        else:
            op_year = 0
            revenue = 0.0
            opex = 0.0
            proposed_pax = 0.0
            baseline_pax = 0.0

        capital = annual_capital if is_construction else 0.0
        net = revenue - opex - capital
        cumulative_net += net
        discounted_net = net * discount_factor
        cumulative_discounted += discounted_net

        rows.append(
            {
                "year": year,
                "revenue": revenue,
                "operating_cost": opex,
                "capital_cost": capital,
                "net_cashflow": net,
                "cumulative_net_cashflow": cumulative_net,
                "discount_factor": discount_factor,
                "discounted_net_cashflow": discounted_net,
                "cumulative_npv": cumulative_discounted,
                "proposed_passengers": proposed_pax,
                "baseline_passengers": baseline_pax,
            }
        )

    return pd.DataFrame(rows)


def _payback_year(cashflow: pd.DataFrame, column: str) -> int | None:
    """First year the cumulative column reaches zero, else ``None``.

    Counted from the first year it moves off zero: with no capital cost
    the construction years sit at exactly zero, which is no payback.
    """
    started = cashflow[column].ne(0).cummax()
    positive = cashflow[started & (cashflow[column] >= 0)]
    if positive.empty:
        return None
    return int(positive.iloc[0]["year"])


def _irr(cashflows: list[float]) -> float | None:
    """Find the internal rate of return of a series of cashflows.

    Solved by bisection on NPV(rate) over ``(-0.99, 10.0)`` (i.e. -99%
    to +1000%). Returns ``None`` when the cashflow never changes sign in
    that range, as for a service that never turns a profit or one with
    no upfront cost to recover.
    """

    def npv(rate: float) -> float:
        return sum(cf / ((1.0 + rate) ** t) for t, cf in enumerate(cashflows))

    lo, hi = -0.99, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if (f_lo > 0) == (f_hi > 0):
        return None

    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1.0:  # £1 precision is plenty for an indicative IRR
            return mid
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def run_commercial_case(params: CommercialCaseParameters) -> CommercialCaseResults:
    """Return the cashflow, NPV, IRR, payback period and sensitivity cases."""
    logger.info(
        "Running commercial case: %d proposed services, %d baseline services, "
        "%d-year period from %d",
        len(params.proposed_services),
        len(params.do_minimum_services),
        params.appraisal_period,
        params.appraisal_year,
    )

    proposed_pax = params.total_annual_passengers(baseline=False)
    baseline_pax = params.total_annual_passengers(baseline=True)
    if params.do_minimum_services and proposed_pax <= baseline_pax:
        logger.warning(
            "Do-minimum passengers (%s) >= proposed (%s): no incremental "
            "demand, so revenue will be zero. Leave do_minimum_services "
            "empty to appraise a brand-new service.",
            f"{baseline_pax:,}",
            f"{proposed_pax:,}",
        )

    cashflow = _run_cashflow(params)

    total_revenue = cashflow["revenue"].sum() / 1e6
    total_opex = cashflow["operating_cost"].sum() / 1e6
    total_capex = cashflow["capital_cost"].sum() / 1e6
    npv = cashflow["discounted_net_cashflow"].sum() / 1e6

    irr = _irr(cashflow["net_cashflow"].tolist())
    payback = _payback_year(cashflow, "cumulative_net_cashflow")
    discounted_payback = _payback_year(cashflow, "cumulative_npv")

    sensitivity = _run_sensitivity(params, npv)

    money_cols = [
        "revenue",
        "operating_cost",
        "capital_cost",
        "net_cashflow",
        "cumulative_net_cashflow",
        "discounted_net_cashflow",
        "cumulative_npv",
    ]
    display_cf = cashflow.copy()
    for col in money_cols:
        display_cf[col] = display_cf[col] / 1e6

    return CommercialCaseResults(
        cashflow=display_cf,
        total_revenue_m=total_revenue,
        total_opex_m=total_opex,
        total_capex_m=total_capex,
        npv_m=npv,
        irr=irr,
        payback_year=payback,
        discounted_payback_year=discounted_payback,
        sensitivity=sensitivity,
    )


# Sensitivity analysis
def _run_sensitivity(
    base_params: CommercialCaseParameters,
    base_npv: float,
) -> list[SensitivityCase]:
    """Rerun with varied parameters and report NPV/IRR sensitivity."""
    cases: list[SensitivityCase] = []

    variations: list[tuple[str, str, object, object]] = [
        ("Demand +20%", "demand", 1.2, None),
        ("Demand -20%", "demand", 0.8, None),
        ("Capital cost +20%", "capital", None, 1.2),
        ("Capital cost -20%", "capital", None, 0.8),
        ("Operating cost +20%", "opex", None, 1.2),
        ("Operating cost -20%", "opex", None, 0.8),
        ("Discount rate +2pp", "discount_rate", None, 0.02),
        ("Discount rate -2pp", "discount_rate", None, -0.02),
    ]

    for label, param_key, factor_a, factor_b in variations:
        p = copy.deepcopy(base_params)

        if param_key == "demand":
            for svc in p.proposed_services:
                svc.annual_passengers = int(svc.annual_passengers * factor_a)
            for svc in p.do_minimum_services:
                svc.annual_passengers = int(svc.annual_passengers * factor_a)
        elif param_key == "capital":
            p.capital_cost_m *= factor_b
        elif param_key == "opex":
            p.annual_operating_cost_m *= factor_b
        elif param_key == "discount_rate":
            p.discount_rate = max(0.0, p.discount_rate + factor_b)

        cf = _run_cashflow(p)
        npv = cf["discounted_net_cashflow"].sum() / 1e6
        irr = _irr(cf["net_cashflow"].tolist())
        change = ((npv - base_npv) / abs(base_npv) * 100.0) if base_npv != 0 else 0.0

        cases.append(
            SensitivityCase(
                label=label, parameter=param_key, npv_m=npv, irr=irr, change_pct=change
            )
        )

    return cases


# Public entry points
def import_from_propose(propose_output_dir: pathlib.Path) -> list[CommercialServiceInput]:
    """Read the proposed services from an earlier ``propose`` run."""
    services = []
    for path in propose_output_dir.glob("propose_*_demand_by_leg.csv"):
        name = path.stem.replace("propose_", "").replace("_demand_by_leg", "")
        demand_df = pd.read_csv(path)
        # One row per (leg, running day), each that day's annual trips,
        # so the service's total is their sum (as scenario takes it).
        if "predicted_annual_trips_captured" not in demand_df.columns:
            logger.warning("%s has no predicted_annual_trips_captured column; skipped.", path)
            continue
        total_pax = int(demand_df["predicted_annual_trips_captured"].sum(skipna=True))
        if "predicted_annual_revenue_gbp" in demand_df.columns and total_pax > 0:
            total_rev = demand_df["predicted_annual_revenue_gbp"].sum(skipna=True)
            avg_fare = total_rev / total_pax
        else:
            avg_fare = 15.0  # fallback

        services.append(
            CommercialServiceInput(
                name=name, annual_passengers=total_pax, avg_fare=float(avg_fare)
            )
        )

    return services


def run_commercial_case_command(config: object) -> None:
    """Run :func:`run_commercial_case` for the ``commercial`` command.

    ``config`` must be an
    :class:`~ortla.config.CommercialCaseConfig`.
    """
    from ortla.config import CommercialCaseConfig  # noqa: PLC0415 (import cycle)

    if not isinstance(config, CommercialCaseConfig):
        raise TypeError(f"Expected CommercialCaseConfig, got {type(config).__name__}")

    proposed_services = [
        CommercialServiceInput(
            name=s.name, annual_passengers=s.annual_passengers, avg_fare=s.avg_fare
        )
        for s in config.proposed_services
    ]

    if getattr(config, "propose_output_path", None) is not None:
        imported = import_from_propose(config.propose_output_path)
        logger.info("Imported %d services from %s", len(imported), config.propose_output_path)
        proposed_services.extend(imported)

    if not proposed_services:
        raise ValueError(
            "No proposed services to appraise. Check the config has a "
            "'proposed_services:' list (note the exact key) with at least "
            "one service, or set 'propose_output_path' to import them."
        )

    do_min_services = [
        CommercialServiceInput(
            name=s.name, annual_passengers=s.annual_passengers, avg_fare=s.avg_fare
        )
        for s in config.do_minimum_services
    ]

    params = CommercialCaseParameters(
        proposed_services=proposed_services,
        do_minimum_services=do_min_services,
        appraisal_year=config.appraisal_year,
        appraisal_period=config.appraisal_period,
        construction_period=config.construction_period,
        demand_ramp_up_years=config.demand_ramp_up_years,
        demand_ramp_up_profile=config.demand_ramp_up_profile,
        demand_growth_rate=config.demand_growth_rate,
        capital_cost_m=config.capital_cost,
        annual_operating_cost_m=config.annual_operating_cost,
        cost_inflation_rate=config.cost_inflation_rate,
        fare_growth_rate=config.fare_growth_rate,
        discount_rate=config.discount_rate,
    )

    results = run_commercial_case(params)

    config.output_path.mkdir(exist_ok=True, parents=True)

    results.cashflow.to_csv(config.output_path / "commercial_case_cashflow.csv", index=False)

    summary = {
        "npv_m": round(results.npv_m, 2),
        "irr": round(results.irr, 4) if results.irr is not None else "n/a",
        "payback_year": results.payback_year if results.payback_year is not None else "n/a",
        "discounted_payback_year": (
            results.discounted_payback_year
            if results.discounted_payback_year is not None
            else "n/a"
        ),
        "roi": round(results.roi, 4) if results.roi != float("inf") else "inf",
        "total_revenue_m": round(results.total_revenue_m, 2),
        "total_opex_m": round(results.total_opex_m, 2),
        "total_capex_m": round(results.total_capex_m, 2),
    }
    pd.Series(summary).to_csv(config.output_path / "commercial_case_summary.csv", header=False)

    sens_df = pd.DataFrame(
        [
            {
                "scenario": s.label,
                "npv_m": round(s.npv_m, 2),
                "irr": round(s.irr, 4) if s.irr is not None else "n/a",
                "change_pct": round(s.change_pct, 1),
            }
            for s in results.sensitivity
        ]
    )
    sens_df.to_csv(config.output_path / "commercial_case_sensitivity.csv", index=False)

    logger.info(
        "Commercial case complete: NPV=£%.1fm, IRR=%s, payback=%s",
        results.npv_m,
        f"{results.irr * 100:.1f}%" if results.irr is not None else "n/a",
        results.payback_year if results.payback_year is not None else "n/a",
    )
