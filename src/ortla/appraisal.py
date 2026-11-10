"""Economic appraisal of proposed services, on TAG Data Book values.

:class:`TagDataBook` reads the appraisal parameters from the TAG Data
Book workbook (v2.03): rail values of time (Table A 1.3.1), the Green
Book's stepped discount rates (Table A 1.1.1) and carbon values by year
(Table A 3.4.1). Values of time grow at 1.5% a year in real terms (TAG
Unit A1.1), and all money is in the data book's price year (2023).

:func:`run_appraisal` turns a service's demand, fares and costs into
value-of-time benefits by trip purpose, wait-time and crowding benefits,
carbon, decongestion and indirect-tax effects of mode shift, and the
present values of benefits and costs that give the benefit-cost ratio
(BCR) and net present value (NPV). It follows WebTAG: Green Book
discounting, the rule of half for new passengers, and fare revenue
netted off the costs as a transfer rather than counted as a benefit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import openpyxl
import pandas as pd

if TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)


def _cell(ws: openpyxl.worksheet.worksheet.Worksheet, coord: str) -> float | None:
    """Return a cell's numeric value, or ``None``."""
    val = ws[coord].value
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class ValueOfTime:
    """Values of time from TAG Table A 1.3.1 (£/hour, factor cost)."""

    work_rail: float
    commuting: float
    other: float
    price_year: int


@dataclass(frozen=True, slots=True)
class DiscountRates:
    """Green Book discount rates from TAG Table A 1.1.1."""

    standard: tuple[tuple[int, int, float], ...]  # (from_year, to_year, rate)


@dataclass(frozen=True, slots=True)
class CarbonValues:
    """Carbon appraisal values from TAG Table A 3.4.1 (£/tCO2e)."""

    # year -> (low, central, high), factor cost
    factor_cost: dict[int, tuple[float, float, float]]
    price_year: int


class TagDataBook:
    """The appraisal parameters read from a TAG Data Book ``.xlsm``.

    ``vot_growth_rate`` is the annual real growth in values of time (1.5%, TAG Unit A1.1).
    """

    # TAG guidance: all non-health values grow at 1.5% a year in real
    # terms from the appraisal year onward (TAG Unit A1.1, May 2021
    # change).
    VOT_ANNUAL_GROWTH: float = 0.015

    def __init__(self, path: str | pathlib.Path) -> None:
        self._path = str(path)
        logger.info("Loading TAG Data Book from %s", self._path)

        wb = openpyxl.load_workbook(self._path, read_only=True, data_only=True)
        try:
            self.vot = self._read_vot(wb)
            self.discount = self._read_discount(wb)
            self.carbon = self._read_carbon(wb)
        finally:
            wb.close()

        self.vot_growth_rate = self.VOT_ANNUAL_GROWTH
        logger.info(
            "TAG Data Book loaded: VoT rail work=£%.2f/hr, "
            "commuting=£%.2f/hr, other=£%.2f/hr (price year %d)",
            self.vot.work_rail,
            self.vot.commuting,
            self.vot.other,
            self.vot.price_year,
        )

    @property
    def vot_work_rail(self) -> float:
        """Rail employer's-business value of time (£/hour)."""
        return self.vot.work_rail

    @property
    def vot_commuting(self) -> float:
        """Commuting value of time (£/hour)."""
        return self.vot.commuting

    @property
    def vot_other(self) -> float:
        """Other non-work value of time (£/hour)."""
        return self.vot.other

    @property
    def price_year(self) -> int:
        """TAG price year; every monetary value is in its prices."""
        return self.vot.price_year

    @staticmethod
    def _read_vot(wb: openpyxl.Workbook) -> ValueOfTime:
        """Read Table A 1.3.1: values of time per person.

        The factor-cost values in column D.
        """
        ws = wb["A1.3.1"]
        return ValueOfTime(
            work_rail=_cell(ws, "D37") or 38.22,
            commuting=_cell(ws, "D47") or 13.04,
            other=_cell(ws, "D48") or 5.95,
            price_year=int(_cell(ws, "B14") or 2023),
        )

    @staticmethod
    def _read_discount(wb: openpyxl.Workbook) -> DiscountRates:
        """Read Table A 1.1.1: Green Book discount rates.

        Rows 25 to 30 of column D.
        """
        ws = wb["A1.1.1"]
        bands = [(0, 30), (31, 75), (76, 125), (126, 200), (201, 300), (301, 999)]
        return DiscountRates(
            standard=tuple(
                (lo, hi, _cell(ws, f"D{25 + i}") or 0.035) for i, (lo, hi) in enumerate(bands)
            )
        )

    @staticmethod
    def _read_carbon(wb: openpyxl.Workbook) -> CarbonValues:
        """Read Table A 3.4.1, carbon appraisal values (factor cost)."""
        ws = wb["A3.4"]
        price_year = int(_cell(ws, "B14") or 2023)

        factor_cost: dict[int, tuple[float, float, float]] = {}
        # Years 2010 to 2100 sit in rows 26 to 117.
        for row in range(26, 118):
            year = _cell(ws, f"B{row}")
            if year is None:
                continue
            factor_cost[int(year)] = (
                _cell(ws, f"D{row}") or 0.0,
                _cell(ws, f"E{row}") or 0.0,
                _cell(ws, f"F{row}") or 0.0,
            )
        return CarbonValues(factor_cost=factor_cost, price_year=price_year)

    def discount_rate_for_year(self, years_from_current: int) -> float:
        """Return the Green Book discount rate for a year offset."""
        for lo, hi, rate in self.discount.standard:
            if lo <= years_from_current <= hi:
                return rate
        return self.discount.standard[-1][2]

    def discount_factor(self, years_from_current: int) -> float:
        """Compound discount factor ``years_from_current`` years ahead.

        Applies the stepped Green Book rates cumulatively.
        """
        factor = 1.0
        for year in range(1, years_from_current + 1):
            factor /= 1.0 + self.discount_rate_for_year(year)
        return factor

    def carbon_price(self, year: int, scenario: str = "central") -> float:
        """Carbon price (£/tCO2e, factor cost) in ``year``; scenario low, central or high."""
        idx = {"low": 0, "central": 1, "high": 2}.get(scenario, 1)
        if year in self.carbon.factor_cost:
            return self.carbon.factor_cost[year][idx]
        # After the table ends, grow at 1.5% a year from its last year.
        max_year = max(self.carbon.factor_cost.keys())
        if year > max_year:
            base = self.carbon.factor_cost[max_year][idx]
            return base * (1.015 ** (year - max_year))
        # Before it starts, use its first year.
        min_year = min(self.carbon.factor_cost.keys())
        return self.carbon.factor_cost[min_year][idx]

    def vot_for_year(self, base_vot: float, appraisal_year: int, target_year: int) -> float:
        """Grow a value of time to ``target_year`` at 1.5% a year."""
        years = target_year - appraisal_year
        if years <= 0:
            return base_vot
        return base_vot * ((1.0 + self.vot_growth_rate) ** years)


# Default decongestion value per car-km removed (pence, 2023 prices).
# Simplified from the TAG A5.4 marginal external cost tables: a rough
# average across road types for the North of England.
_DECONGESTION_PENCE_PER_CAR_KM = 8.5


# Indicative indirect-tax (mainly fuel duty) lost to the Exchequer per
# car-km removed (pence, 2023 prices). Mode shift off the road reduces
# fuel-duty receipts; WebTAG books this as a negative "Wider Public
# Finances / indirect taxation" impact in the PVB. Simplified from TAG
# A5.4 / fuel-duty rates.
_INDIRECT_TAX_PENCE_PER_CAR_KM = 3.0


# Frequency (wait-time) and crowding benefit defaults, from TAG unit
# M3.2 ("Public Transport Assignment") and the PDFH
# generalised-journey-time framework. These are indicative
# simplifications of the full load-factor and service-interval tables in
# that guidance.
#
# Waiting time is an out-of-vehicle "walk/wait" component and is valued
# at a multiple of in-vehicle time; TAG puts this multiplier around 2.
# Beyond a roughly half-hourly headway passengers stop turning up at
# random and start timing their arrival to the timetable, so average
# perceived wait is capped rather than growing without bound.
_WAIT_TIME_VOT_MULTIPLIER = 2.0


_MAX_AVERAGE_WAIT_MINUTES = 15.0


def _crowding_multiplier(load_factor: float) -> float:
    """In-vehicle-time crowding multiplier for a given load factor.

    ``load_factor`` is passengers / seats. A simplified piecewise
    approximation of the TAG M3.2 / PDFH standing and crowding
    multipliers: 1.0 while everyone is comfortably seated, rising as
    standing sets in and the crowd builds, and flat at crush (1.90 from
    a load factor of 2), since no more riders fit. The curve is
    continuous, so adding a marginal seat always gives a small relief
    benefit without jumps.
    """
    load = max(0.0, load_factor)
    if load <= 0.85:  # noqa: PLR2004 (TAG M3.2 curve breakpoints)
        return 1.0
    if load <= 1.0:
        return 1.0 + (load - 0.85) / 0.15 * 0.05  # seated, filling up -> 1.05
    if load <= 1.5:  # noqa: PLR2004
        return 1.05 + (load - 1.0) / 0.5 * 0.35  # standing -> 1.40
    if load <= 2.0:  # noqa: PLR2004
        return 1.40 + (load - 1.5) / 0.5 * 0.50  # crush -> 1.90
    return 1.90  # crush: no more riders fit


# ----------------------------------------------------------------------
# Input / output data classes
# ----------------------------------------------------------------------
@dataclass
class AppraisalServiceInput:
    """One service (or service group) to appraise.

    ``trains_per_day``, ``seats_per_train`` and
    ``in_vehicle_time_minutes`` are only needed for the frequency
    (wait-time) and crowding-relief benefits; they default to ``0``
    (those benefits switched off) so a pure journey-time-saving
    appraisal need not supply them.
    """

    name: str
    annual_passengers: int
    journey_time_saved_minutes: float
    avg_fare: float  # £ per passenger

    # Physical service attributes for the frequency / crowding benefits.
    trains_per_day: float = 0.0
    seats_per_train: float = 0.0
    in_vehicle_time_minutes: float = 0.0  # end-to-end, for the crowding penalty


@dataclass
class AppraisalParameters:
    """All parameters needed to run the appraisal."""

    tag: TagDataBook
    proposed_services: list[AppraisalServiceInput] = field(default_factory=list)
    do_minimum_services: list[AppraisalServiceInput] = field(default_factory=list)
    appraisal_year: int = 2028
    appraisal_period: int = 60
    construction_period: int = 3

    # Demand ramp-up
    demand_ramp_up_years: int = 3
    demand_ramp_up_profile: list[float] = field(default_factory=lambda: [0.5, 0.75, 0.9, 1.0])

    # Trip purpose split (should sum to 1.0)
    purpose_work: float = 0.12
    purpose_commuting: float = 0.47
    purpose_other: float = 0.41

    # Costs
    capital_cost_m: float = 0.0  # £ million
    annual_operating_cost_m: float = 0.0  # £ million
    cost_inflation_rate: float = 0.0  # real-terms pa

    # Optimism bias (HM Treasury standard)
    optimism_bias_capex: float = 0.66  # 66% uplift on capital for rail
    optimism_bias_opex: float = 0.41  # 41% uplift on operating for rail

    # Taxation
    indirect_tax_correction_factor: float = 1.19

    # Mode shift assumptions
    mode_shift_car_pct: float = 0.20  # fraction of new pax from car
    avg_car_trip_km: float = 30.0
    car_co2_per_km: float = 0.120  # kg CO2/km (120 g)
    indirect_tax_car_pence_per_km: float = _INDIRECT_TAX_PENCE_PER_CAR_KM

    # Demand growth (real-terms, over and above VoT growth)
    demand_growth_rate: float = 0.0

    # Frequency (wait-time) benefit, TAG M3.2 / PDFH generalised journey
    # time. Higher frequency shortens average waiting time, valued at a
    # multiple of in-vehicle VoT. Left at 0 the term is switched off (no
    # operating span over which to spread the trains, so no headway can
    # be derived).
    operating_hours_per_day: float = 0.0  # daily span the service runs over
    baseline_trains_per_day: float = 0.0  # do-minimum corridor frequency
    wait_time_vot_multiplier: float = _WAIT_TIME_VOT_MULTIPLIER
    max_average_wait_minutes: float = _MAX_AVERAGE_WAIT_MINUTES
    service_days_per_year: float = 253.0  # operating days/year (weekday-only default)

    # Crowding-relief benefit, TAG M3.2 crowding multipliers on
    # in-vehicle time. Adding capacity lowers the load factor and the
    # crowding penalty borne by riders. Left at 0 the term is switched
    # off (no do-minimum capacity to relieve).
    baseline_seats_per_day: float = 0.0

    def total_annual_passengers(self, baseline: bool = False) -> int:
        """Annual passengers on the proposed or do-minimum services."""
        svcs = self.do_minimum_services if baseline else self.proposed_services
        return sum(s.annual_passengers for s in svcs)


@dataclass
class BenefitBreakdown:
    """Present-value breakdown of the PVB (£ million, market prices).

    Follows the WebTAG AMCB: user time savings plus non-user impacts
    (greenhouse gases, decongestion and the change in indirect-tax
    receipts from mode shift). Fare revenue is not here: it is a
    transfer to the operator and is booked against the PVC (see
    :class:`CostBreakdown`).
    """

    vot_work: float = 0.0
    vot_commuting: float = 0.0
    vot_other: float = 0.0
    wait_time: float = 0.0  # frequency / waiting-time saving (TAG M3.2)
    crowding: float = 0.0  # crowding relief from added capacity (TAG M3.2)
    carbon: float = 0.0
    decongestion: float = 0.0
    indirect_tax: float = 0.0  # change in Exchequer indirect-tax receipts (usually negative)

    @property
    def total_benefits(self) -> float:
        """Present value of benefits, summed over every component."""
        return (
            self.vot_work
            + self.vot_commuting
            + self.vot_other
            + self.wait_time
            + self.crowding
            + self.carbon
            + self.decongestion
            + self.indirect_tax
        )


@dataclass
class CostBreakdown:
    """Present-value breakdown of the PVC (Broad Transport Budget, £m).

    WebTAG PVC = investment + operating costs, net of fare revenue
    accruing to the public sector, all in market prices.
    """

    capital: float = 0.0
    operating: float = 0.0
    revenue: float = 0.0  # fare revenue, netted off the Broad Transport Budget

    @property
    def total_costs(self) -> float:
        """Present value of capital and operating costs less revenue."""
        return self.capital + self.operating - self.revenue


@dataclass
class AppraisalResults:
    """Complete output of an economic appraisal run."""

    benefits: BenefitBreakdown
    costs: CostBreakdown

    @property
    def pvb(self) -> float:
        """Present Value of Benefits (£m)."""
        return self.benefits.total_benefits

    @property
    def pvc(self) -> float:
        """Present Value of Costs (£m)."""
        return self.costs.total_costs

    @property
    def npv(self) -> float:
        """Net Present Value (£m)."""
        return self.pvb - self.pvc

    @property
    def bcr(self) -> float:
        """Initial Benefit-Cost Ratio (PVB / PVC).

        When revenue fully offsets the Broad Transport Budget the PVC is
        zero or negative; WebTAG treats such schemes as revenue-positive
        and the BCR is not meaningfully defined, so ``inf`` is returned.
        """
        if self.pvc <= 0:
            return float("inf") if self.pvb > 0 else 0.0
        return self.pvb / self.pvc


# ----------------------------------------------------------------------
# Core calculation functions
# ----------------------------------------------------------------------
def _annual_vot_benefits(
    params: AppraisalParameters,
    year: int,
    demand_factor: float,
) -> tuple[float, float, float]:
    """Calculate annual VoT benefits (£) by purpose for a given year.

    Returns marginal VoT benefits (work, commuting, other) in £.
    """
    tag = params.tag

    vot_work = tag.vot_for_year(tag.vot_work_rail, tag.price_year, year)
    vot_commuting = tag.vot_for_year(tag.vot_commuting, tag.price_year, year)
    vot_other = tag.vot_for_year(tag.vot_other, tag.price_year, year)

    work_benefit = 0.0
    commuting_benefit = 0.0
    other_benefit = 0.0

    # Services are matched by name to tell existing passengers from new
    # ones.
    do_minimum = {s.name: s for s in params.do_minimum_services}

    for proposed in params.proposed_services:
        baseline = do_minimum.get(proposed.name)

        baseline_pax = baseline.annual_passengers if baseline else 0
        proposed_pax = proposed.annual_passengers

        # Rule of half: existing passengers get the full time saving,
        # new passengers half of it.
        existing_pax = min(baseline_pax, proposed_pax) * demand_factor
        new_pax = max(0, proposed_pax - baseline_pax) * demand_factor

        # `journey_time_saved_minutes` is the saving of proposed over
        # baseline.
        time_hours = proposed.journey_time_saved_minutes / 60.0

        pax_equivalent = existing_pax + (new_pax * 0.5)

        work_benefit += pax_equivalent * params.purpose_work * time_hours * vot_work
        commuting_benefit += pax_equivalent * params.purpose_commuting * time_hours * vot_commuting
        other_benefit += pax_equivalent * params.purpose_other * time_hours * vot_other

    # The indirect tax correction factor takes perceived cost to market
    # price.
    itcf = params.indirect_tax_correction_factor
    return work_benefit * itcf, commuting_benefit * itcf, other_benefit * itcf


def _blended_vot(params: AppraisalParameters, year: int) -> float:
    """Purpose-weighted value of time (£/hr), grown to ``year``.

    Used for the wait-time and crowding benefits, which the cashflow
    does not split by purpose as it does the headline value-of-time
    term.
    """
    tag = params.tag
    return (
        params.purpose_work * tag.vot_for_year(tag.vot_work_rail, tag.price_year, year)
        + params.purpose_commuting * tag.vot_for_year(tag.vot_commuting, tag.price_year, year)
        + params.purpose_other * tag.vot_for_year(tag.vot_other, tag.price_year, year)
    )


def _rule_of_half_pax(params: AppraisalParameters, demand_factor: float) -> float:
    """Passenger-equivalents for a time benefit (Rule of Half).

    Existing passengers (those already travelling in the do-minimum) get
    the full benefit; newly generated passengers get half. Mirrors the
    convention in :func:`_annual_vot_benefits`.
    """
    proposed = params.total_annual_passengers(baseline=False)
    baseline = params.total_annual_passengers(baseline=True)
    existing = min(baseline, proposed) * demand_factor
    generated = max(0, proposed - baseline) * demand_factor
    return existing + 0.5 * generated


def _average_wait_minutes(trains_per_day: float, params: AppraisalParameters) -> float:
    """Average perceived wait (min) for a service frequency.

    For a turn-up-and-go service average wait is half the headway; it is
    capped at :attr:`AppraisalParameters.max_average_wait_minutes`
    because beyond a roughly half-hourly service passengers time their
    arrival to the timetable rather than waiting the full half-headway.
    """
    if trains_per_day <= 0:
        return params.max_average_wait_minutes
    headway = params.operating_hours_per_day * 60.0 / trains_per_day
    return min(0.5 * headway, params.max_average_wait_minutes)


def _annual_wait_time_benefit(
    params: AppraisalParameters,
    year: int,
    demand_factor: float,
) -> float:
    """Annual frequency (waiting-time) benefit (£), TAG M3.2 / PDFH GJT.

    Adding the scheme's trains to the do-minimum corridor frequency
    (``baseline_trains_per_day``) raises the combined frequency and so
    shortens the average wait riders experience; the benefit is that
    reduction, never negative. A bare corridor with no comparable
    service starts at the capped maximum wait. The saved wait is valued
    at ``wait_time_vot_multiplier`` times the purpose-blended in-vehicle
    VoT, with the Rule of Half applied to generated demand.
    """
    if params.operating_hours_per_day <= 0:
        return 0.0
    scheme_trains = sum(s.trains_per_day for s in params.proposed_services)
    if scheme_trains <= 0:
        return 0.0

    wait_without = _average_wait_minutes(params.baseline_trains_per_day, params)
    wait_with = _average_wait_minutes(params.baseline_trains_per_day + scheme_trains, params)
    saving_minutes = max(0.0, wait_without - wait_with)
    if saving_minutes <= 0:
        return 0.0

    pax = _rule_of_half_pax(params, demand_factor)
    vot = _blended_vot(params, year)
    return (
        pax
        * (saving_minutes / 60.0)
        * vot
        * params.wait_time_vot_multiplier
        * params.indirect_tax_correction_factor
    )


def _annual_crowding_benefit(
    params: AppraisalParameters,
    year: int,
    demand_factor: float,
) -> float:
    """Annual crowding-relief benefit (£) from TAG M3.2 multipliers.

    Adds the scheme's seat capacity to the do-minimum corridor capacity
    (``baseline_seats_per_day``), lowering the load factor riders
    experience. Without the scheme only the do-minimum's riders are on
    those seats; with it, every rider shares the combined seats. The
    benefit is the fall in the crowding multiplier applied to their
    in-vehicle time, valued at the purpose-blended VoT, with the Rule of
    Half applied to generated demand. With no do-minimum capacity to
    relieve, or no scheme seats, the term is zero.
    """
    if params.baseline_seats_per_day <= 0:
        return 0.0
    scheme_seats = sum(s.trains_per_day * s.seats_per_train for s in params.proposed_services)
    if scheme_seats <= 0:
        return 0.0

    total_pax = sum(s.annual_passengers for s in params.proposed_services)
    if total_pax <= 0:
        return 0.0
    ivt_minutes = (
        sum(s.annual_passengers * s.in_vehicle_time_minutes for s in params.proposed_services)
        / total_pax
    )
    if ivt_minutes <= 0:
        return 0.0

    days = max(1.0, params.service_days_per_year)
    baseline_pax = params.total_annual_passengers(baseline=True)
    lf_without = baseline_pax * demand_factor / days / params.baseline_seats_per_day
    lf_with = total_pax * demand_factor / days / (params.baseline_seats_per_day + scheme_seats)
    relief = _crowding_multiplier(lf_without) - _crowding_multiplier(lf_with)
    if relief <= 0:
        return 0.0

    pax = _rule_of_half_pax(params, demand_factor)
    vot = _blended_vot(params, year)
    return pax * (ivt_minutes / 60.0) * relief * vot * params.indirect_tax_correction_factor


def _annual_revenue(
    params: AppraisalParameters,
    demand_factor: float,
) -> float:
    """Marginal annual fare revenue (£).

    Fares are already a market-price financial flow, so no indirect-tax
    correction is applied. This revenue is netted off the Broad
    Transport Budget (PVC), not added to the PVB.
    """
    baseline = sum(s.annual_passengers * s.avg_fare for s in params.do_minimum_services)
    proposed = sum(s.annual_passengers * s.avg_fare for s in params.proposed_services)
    return (proposed - baseline) * demand_factor


def _annual_carbon_benefit(
    params: AppraisalParameters,
    year: int,
    demand_factor: float,
) -> float:
    """Annual carbon benefit from mode shift (£)."""
    tag = params.tag
    carbon_price = tag.carbon_price(year, "central")  # £/tCO2e

    baseline_pax = sum(s.annual_passengers for s in params.do_minimum_services)
    proposed_pax = sum(s.annual_passengers for s in params.proposed_services)
    new_pax = max(0, proposed_pax - baseline_pax) * demand_factor

    diverted_pax = new_pax * params.mode_shift_car_pct
    car_km_saved = diverted_pax * params.avg_car_trip_km
    tonnes_co2_saved = car_km_saved * params.car_co2_per_km / 1000.0

    return tonnes_co2_saved * carbon_price


def _car_km_saved(params: AppraisalParameters, demand_factor: float) -> float:
    """Car-km removed from the road by mode shift in a given year."""
    baseline_pax = sum(s.annual_passengers for s in params.do_minimum_services)
    proposed_pax = sum(s.annual_passengers for s in params.proposed_services)
    new_pax = max(0, proposed_pax - baseline_pax) * demand_factor

    diverted_pax = new_pax * params.mode_shift_car_pct
    return diverted_pax * params.avg_car_trip_km


def _annual_decongestion_benefit(
    params: AppraisalParameters,
    demand_factor: float,
) -> float:
    """Annual decongestion benefit from mode shift (£)."""
    return _car_km_saved(params, demand_factor) * _DECONGESTION_PENCE_PER_CAR_KM / 100.0


def _annual_indirect_tax_impact(
    params: AppraisalParameters,
    demand_factor: float,
) -> float:
    """Annual change in indirect-tax receipts from mode shift (£).

    Car-km removed from the road reduce fuel-duty receipts, so this is
    normally a negative entry in the PVB (WebTAG "Wider Public
    Finances").
    """
    return -_car_km_saved(params, demand_factor) * params.indirect_tax_car_pence_per_km / 100.0


def _annual_operating_cost(params: AppraisalParameters, year: int) -> float:
    """Annual operating cost (£), grown with cost inflation.

    Inflation runs from the appraisal year.
    """
    years = year - params.appraisal_year
    opex = (
        params.annual_operating_cost_m
        * 1e6
        * ((1.0 + params.cost_inflation_rate) ** max(0, years))
    )
    return opex * (1.0 + params.optimism_bias_opex)


def _run_cashflow(params: AppraisalParameters) -> pd.DataFrame:
    """Build the year-by-year cashflow table.

    Returns a DataFrame with columns year, vot_work, vot_commuting,
    vot_other, wait_time, crowding, carbon, decongestion, indirect_tax,
    total_benefits, capital, operating, revenue, total_costs,
    net_benefit, discount_factor, pv_benefits, pv_costs, pv_net and
    cumulative_npv.
    """
    tag = params.tag
    opening_year = params.appraisal_year
    end_year = opening_year + params.appraisal_period - 1
    # Capital is spent in the construction years before opening.
    construction_start = opening_year - params.construction_period

    rows: list[dict[str, float]] = []
    cumulative_npv = 0.0

    # Capital is spread evenly across the construction years, with
    # optimism bias.
    annual_capital = params.capital_cost_m * 1e6 / max(1, params.construction_period)
    annual_capital *= 1.0 + params.optimism_bias_capex

    for year in range(construction_start, end_year + 1):
        # TAG discounts to its present-value year (the data book's
        # price/value year), not to the start of construction.
        years_from_base = max(0, year - tag.price_year)
        discount = tag.discount_factor(years_from_base)

        is_operational = year >= opening_year
        is_construction = construction_start <= year < opening_year

        if is_operational:
            op_year = year - opening_year
            # The ramp-up profile covers the first years; after it, its
            # last value.
            if op_year < params.demand_ramp_up_years and op_year < len(
                params.demand_ramp_up_profile
            ):
                ramp_factor = params.demand_ramp_up_profile[op_year]
            else:
                ramp_factor = (
                    params.demand_ramp_up_profile[-1] if params.demand_ramp_up_profile else 1.0
                )
            demand_factor = ramp_factor * ((1.0 + params.demand_growth_rate) ** op_year)
        else:
            demand_factor = 0.0

        if is_operational:
            vot_work, vot_commuting, vot_other = _annual_vot_benefits(params, year, demand_factor)
            wait = _annual_wait_time_benefit(params, year, demand_factor)
            crowding = _annual_crowding_benefit(params, year, demand_factor)
            revenue = _annual_revenue(params, demand_factor)
            carbon = _annual_carbon_benefit(params, year, demand_factor)
            decongestion = _annual_decongestion_benefit(params, demand_factor)
            indirect_tax = _annual_indirect_tax_impact(params, demand_factor)
        else:
            vot_work = vot_commuting = vot_other = wait = crowding = revenue = 0.0
            carbon = decongestion = indirect_tax = 0.0

        # PVB excludes fare revenue (a transfer) and includes the
        # indirect-tax impact.
        total_benefits = (
            vot_work
            + vot_commuting
            + vot_other
            + wait
            + crowding
            + carbon
            + decongestion
            + indirect_tax
        )

        # PVC = Broad Transport Budget = capital + operating, net of
        # revenue.
        capital = annual_capital if is_construction else 0.0
        opex = _annual_operating_cost(params, year) if is_operational else 0.0
        total_costs = capital + opex - revenue

        net = total_benefits - total_costs
        pv_net = net * discount
        cumulative_npv += pv_net

        rows.append(
            {
                "year": year,
                "vot_work": vot_work,
                "vot_commuting": vot_commuting,
                "vot_other": vot_other,
                "wait_time": wait,
                "crowding": crowding,
                "carbon": carbon,
                "decongestion": decongestion,
                "indirect_tax": indirect_tax,
                "total_benefits": total_benefits,
                "capital": capital,
                "operating": opex,
                "revenue": revenue,
                "total_costs": total_costs,
                "net_benefit": net,
                "discount_factor": discount,
                "pv_benefits": total_benefits * discount,
                "pv_costs": total_costs * discount,
                "pv_net": pv_net,
                "cumulative_npv": cumulative_npv,
            }
        )

    return pd.DataFrame(rows)


def _aggregate_results(
    cashflow: pd.DataFrame,
) -> tuple[BenefitBreakdown, CostBreakdown]:
    """Sum the discounted cashflow into PV breakdowns, in £ million."""

    def present_value(column: str) -> float:
        return cashflow[column].mul(cashflow["discount_factor"]).sum() / 1e6

    benefits = BenefitBreakdown(
        vot_work=present_value("vot_work"),
        vot_commuting=present_value("vot_commuting"),
        vot_other=present_value("vot_other"),
        wait_time=present_value("wait_time"),
        crowding=present_value("crowding"),
        carbon=present_value("carbon"),
        decongestion=present_value("decongestion"),
        indirect_tax=present_value("indirect_tax"),
    )
    costs = CostBreakdown(
        capital=present_value("capital"),
        operating=present_value("operating"),
        revenue=present_value("revenue"),
    )
    return benefits, costs


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------
def run_appraisal(params: AppraisalParameters) -> AppraisalResults:
    """Run the economic appraisal: PVB and PVC breakdowns, from which NPV and BCR follow."""
    logger.info(
        "Running economic appraisal: %d proposed services, %d baseline services, "
        "%d-year period from %d",
        len(params.proposed_services),
        len(params.do_minimum_services),
        params.appraisal_period,
        params.appraisal_year,
    )

    # Warn when the do-minimum baseline cancels out incremental demand:
    # the revenue, carbon and decongestion benefits are all marginal
    # (proposed minus do-minimum), so they collapse to zero if the
    # baseline carries the same (or more) passengers as the proposed
    # services.
    proposed_pax = params.total_annual_passengers(baseline=False)
    baseline_pax = params.total_annual_passengers(baseline=True)
    if params.do_minimum_services and proposed_pax <= baseline_pax:
        logger.warning(
            "Do-minimum passengers (%s) >= proposed (%s): no incremental "
            "demand, so revenue/carbon/decongestion benefits will be zero. "
            "Leave do_minimum_services empty to appraise a new service.",
            f"{baseline_pax:,}",
            f"{proposed_pax:,}",
        )

    return evaluate(params)


def evaluate(params: AppraisalParameters) -> AppraisalResults:
    """Appraise ``params`` without logging, for ranking many options."""
    benefits, costs = _aggregate_results(_run_cashflow(params))
    return AppraisalResults(benefits=benefits, costs=costs)
