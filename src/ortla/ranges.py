"""Low, central and high cases of a service forecast.

Two uncertainties are moved together. The response (the served-pair elasticity and a new direct
link's step) changes how many passengers are new to rail, and the loading error multiplies
everyone the train carries. The ends are scenarios. Only the loading part can be given a coverage,
from the errors the multipliers were taken from, and none of them allows for growth, fares, costs
or competition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ortla.config import RangesConfig, ScenarioConfig

RANGE_COLUMNS = [
    "case",
    "load_multiplier",
    "frequency_elasticity",
    "new_link_uplift",
    "annual_passengers",
    "annual_passengers_generated",
    "annual_revenue_new",
    "webtag_pvb_m",
    "webtag_pvc_m",
    "webtag_npv_m",
    "webtag_bcr",
    "commercial_npv_m",
]


def cases(config: ScenarioConfig) -> Iterator[tuple[str, float, ScenarioConfig]]:
    """Yield each case's name, loading multiplier and config."""
    ends: RangesConfig | None = config.ranges
    if ends is None:
        return
    for name, load, elasticity, step in (
        ("low", ends.load_low, ends.elasticity_low, ends.new_link_low),
        ("central", 1.0, config.served_pair_frequency_elasticity, config.new_link_uplift),
        ("high", ends.load_high, ends.elasticity_high, ends.new_link_high),
    ):
        response = {"served_pair_frequency_elasticity": elasticity, "new_link_uplift": step}
        yield name, load, config.model_copy(update=response)
