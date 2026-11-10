"""A map of the whole track network, coloured by service level.

The track graph is the one :class:`TrackDistanceCalculator` builds for
the ``distance`` command. With a ``timetable`` run's route frequency
summary, each track segment is coloured by the busiest station pair
whose route uses it, and clicking a segment lists those pairs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import folium
import geopandas as gpd
import matplotlib as mpl
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point

from ortla.track_distance import TrackDistanceCalculator

if TYPE_CHECKING:
    import pathlib

    from ortla.config import TrackCalculatorConfig

logger = logging.getLogger(__name__)

# Purple to red stays saturated across its range, so even quiet lines
# show up against the light base map. geopandas' explore() needs the
# colour map registered by name: given a Colormap object it treats it
# as a per-value colour function and its legend fails.
_CMAP = "orm_purple_red"
if _CMAP not in mpl.colormaps:
    mpl.colormaps.register(LinearSegmentedColormap.from_list(_CMAP, ["#7C3AED", "#E63946"]))

# Pairs listed in a segment's popup; a segment near a hub can carry
# hundreds.
_MAX_LISTED = 12


def load_frequency(timetable_output: pathlib.Path) -> dict[tuple[str, str], float]:
    """Read daily trains per station pair from the frequency summary."""
    path = timetable_output / "route_frequency_summary.csv"
    df = pd.read_csv(path, usecols=["origin_crs", "destination_crs", "daily_trains"])
    return {
        (o, d): float(v)
        for o, d, v in df.itertuples(index=False)
        if isinstance(o, str) and isinstance(d, str)
    }


def _popup(pairs: list[tuple[str, str, float]]) -> str:
    ranked = sorted(pairs, key=lambda p: p[2], reverse=True)
    lines = [f"{o} → {d}: {v:g}" for o, d, v in ranked[:_MAX_LISTED]]
    if len(ranked) > _MAX_LISTED:
        lines.append(f"and {len(ranked) - _MAX_LISTED} more")
    # folium sets tooltips as innerHTML, so <br> gives a real line
    # break.
    return "<br>".join(lines)


def build_network_map(
    calculator: TrackCalculatorConfig,
    output_file: pathlib.Path,
    timetable_output: pathlib.Path | None = None,
    simplify_m: float = 25.0,
) -> tuple[int, int]:
    """Write the network map to ``output_file``.

    Parameters
    ----------
    calculator : TrackCalculatorConfig
        The track model and station list, as for ``distance``.
    output_file : pathlib.Path
        The HTML file to write.
    timetable_output : pathlib.Path | None, optional
        A ``timetable`` output folder. When given, segments are coloured
        by daily trains.
    simplify_m : float, optional
        Line simplification tolerance in metres. The full track model
        has millions of points, far too many for an embedded browser,
        and 25 m cannot be seen at network scale.

    Returns
    -------
    tuple[int, int]
        Station pairs drawn and station pairs in the frequency summary;
        ``(0, 0)`` without one.
    """
    calc = TrackDistanceCalculator(calculator)
    working = calculator.coordinate_systems.working_crs
    display = calculator.coordinate_systems.input_crs

    lines = [
        data["geometry"].simplify(simplify_m, preserve_topology=False)
        for _, _, data in calc.graph.edges(data=True)
        if data.get("geometry") is not None
    ]
    fmap = (
        gpd.GeoDataFrame(geometry=lines, crs=working)
        .to_crs(display)
        .explore(color="#B0B0B0", weight=1, name="Track")
    )

    values = load_frequency(timetable_output) if timetable_output else {}
    with_data = {code for pair in values for code in pair}
    codes = [c for c in calc.station_locations if not values or c in with_data]
    stations = gpd.GeoDataFrame(
        {"Station": codes},
        geometry=[Point(calc.station_locations[c]) for c in codes],
        crs=working,
    ).to_crs(display)
    fmap = stations.explore(m=fmap, color="#555555", marker_kwds={"radius": 2}, name="Stations")

    # One line per physical segment, coloured by the busiest pair using
    # it: drawing every pair's whole route stacks hundreds of lines on
    # the busy approaches and makes a file no browser can open.
    by_edge: dict[tuple, list[tuple[str, str, float]]] = {}
    drawn = 0
    for (origin, dest), value in values.items():
        try:
            edges = calc._route(origin, dest)[2]
        except ValueError:
            continue
        drawn += 1
        for edge in edges:
            by_edge.setdefault(edge, []).append((origin, dest, value))
    if by_edge:
        rows = [
            {
                "daily_trains": max(v for _, _, v in pairs),
                "pairs": _popup(pairs),
                "geometry": calc.graph[u][v]["geometry"].simplify(
                    simplify_m, preserve_topology=False
                ),
            }
            for (u, v), pairs in by_edge.items()
        ]
        fmap = (
            gpd.GeoDataFrame(rows, crs=working)
            .to_crs(display)
            .explore(
                m=fmap,
                column="daily_trains",
                cmap=_CMAP,
                legend=True,
                weight=6,
                tooltip=["daily_trains", "pairs"],
                name="Daily trains",
                legend_kwds={"caption": "Daily trains (busiest pair using the segment)"},
            )
        )

    folium.LayerControl().add_to(fmap)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_file))
    logger.info("Network map written to %s (%d of %d pairs)", output_file, drawn, len(values))
    return drawn, len(values)
