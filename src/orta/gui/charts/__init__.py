"""Charts and maps of each command's outputs.

:func:`build_figures` returns every chart for a command's output
folder as ``(title, figure)`` pairs. The functions live in
:mod:`.plotly_charts` (the data commands), :mod:`.network` (network
charts and maps for those), :mod:`.forecasting` (hindcast, scenario,
discovery, commercial case, new station), :mod:`.assignment`
(national train loads) and :mod:`.maps`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from orta.gui.charts import assignment, forecasting, network, plotly_charts
from orta.gui.constants import ThemeMode

if TYPE_CHECKING:
    import pathlib

    import pandas as pd
    import plotly.graph_objects as go

_FORECASTING = {
    "hindcast": forecasting.hindcast_figures,
    "scenario": forecasting.scenario_figures,
    "discover": forecasting.discover_figures,
    "commercial": forecasting.commercial_figures,
    "new-station": forecasting.new_station_figures,
    "assign": assignment.assign_figures,
}


def build_figures(
    command: str,
    output_dir: pathlib.Path,
    tag: str | None = None,
    operator: str | None = None,
    candidate: str | None = None,
    theme_mode: ThemeMode = ThemeMode.LIGHT,
    stations: pd.DataFrame | None = None,
) -> list[tuple[str, go.Figure]]:
    """Build every chart and map for ``command``'s outputs.

    ``tag``, ``operator`` and ``candidate`` narrow the charts to one
    station pair, operator or proposed service. ``stations`` (CRS code
    to ``lat`` and ``lon``) places outputs on maps; without it the maps
    that need it are left out.
    """
    if not output_dir.exists():
        return []
    pc, mode = plotly_charts, theme_mode
    if command in _FORECASTING:
        return _FORECASTING[command](output_dir, mode, stations)
    if command == "loading":
        builds = [
            pc.temporal_factor_figures,
            pc.demand_grid_figures,
            pc.demand_profile_figures,
            pc.journey_type_figures,
            pc.analysis_figures,
            pc.factor_validation_figures,
        ]
        figures = [f for build in builds for f in build(output_dir, operator, mode)]
        return figures + network.loading_figures(output_dir, operator, mode, stations)
    if command == "propose":
        builds = [
            pc.running_time_figures,
            pc.headway_figures,
            pc.demand_by_leg_figures,
            pc.loading_profile_figures,
            pc.search_window_figures,
        ]
        figures = [f for build in builds for f in build(output_dir, candidate, mode)]
        return figures + network.propose_figures(output_dir, candidate, mode, stations)
    by_pair = {
        "services": pc.services_figures,
        "lookup": pc.lookup_figures,
        "fares": pc.fares_figures,
    }
    if command in by_pair:
        return by_pair[command](output_dir, tag, mode)
    if command == "timetable":
        return pc.timetable_figures(output_dir, mode) + network.timetable_figures(
            output_dir, mode, stations
        )
    if command == "matrix":
        return pc.matrix_figures(output_dir, mode) + network.matrix_figures(
            output_dir, mode, stations
        )
    if command == "bplan":
        return pc.bplan_figures(output_dir, mode) + network.bplan_figures(output_dir, mode)
    if command == "distance":
        return pc.distance_figures(output_dir, mode)
    return []
