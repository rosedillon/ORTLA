"""Network-wide charts and maps for the data commands.

These sit beside the per-command charts in :mod:`plotly_charts`: the
timetable and matrix as maps of flows and stations, operators' counts
by station, BPLAN's timing points and a proposed service's route.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyproj import Transformer

from ortla.gui.charts import maps
from ortla.gui.charts.style import apply_theme, colour, read
from ortla.gui.constants import ThemeMode, palette

if TYPE_CHECKING:
    import pathlib

Figures = list[tuple[str, go.Figure]]

_FROM_GRID = Transformer.from_crs(27700, 4326, always_xy=True)
# Great Britain, roughly: west, east, south and north in degrees.
_GB = (-8.7, 1.8, 49.8, 61.0)


def timetable_figures(
    output_dir: pathlib.Path, mode: ThemeMode, stations: pd.DataFrame | None
) -> Figures:
    """Chart the timetable's links by hour, by frequency and on maps."""
    figures: Figures = []
    summary = read(output_dir / "route_frequency_summary.csv")
    hourly = read(output_dir / "route_frequency_hourly.csv")
    if {"hour", "trains"}.issubset(hourly.columns):
        by_hour = hourly.groupby("hour")["trains"].sum().reindex(range(24), fill_value=0)
        fig = go.Figure(
            go.Bar(x=by_hour.index, y=by_hour.to_numpy(), marker_color=colour(mode, 0))
        )
        fig.update_layout(
            title="Direct station-to-station links served, by hour of departure",
            xaxis={"title": "Hour", "dtick": 2},
            yaxis_title="Pair-trains",
        )
        figures.append(("Network by hour", apply_theme(fig, mode)))
    if {"origin_crs", "destination_crs", "daily_trains"}.issubset(summary.columns):
        fig = go.Figure(
            go.Histogram(
                x=np.log10(summary["daily_trains"].clip(lower=1)),
                nbinsx=40,
                marker_color=colour(mode, 0),
            )
        )
        ticks = [1, 3, 10, 30, 100, 300]
        fig.update_layout(
            title="How often each directly linked pair is served",
            xaxis={"title": "Trains a day", "tickvals": np.log10(ticks), "ticktext": ticks},
            yaxis_title="Station pairs",
        )
        figures.append(("Service frequency", apply_theme(fig, mode)))

        reach = summary.groupby("origin_crs").agg(
            direct_destinations=("destination_crs", "nunique"),
            busiest_link_trains=("daily_trains", "max"),
        )
        placed = maps.with_positions(reach.reset_index(names="crs"), stations)
        fig = maps.station_map(
            placed,
            "direct_destinations",
            "Stations reachable without changing",
            mode,
            hover=["busiest_link_trains"],
        )
        if fig is not None:
            figures.append(("Map: direct reach", fig))
        fig = maps.flow_map(
            summary,
            stations,
            "daily_trains",
            "The 1,500 most frequent direct links",
            mode,
            origin="origin_crs",
            destination="destination_crs",
            top=1500,
        )
        if fig is not None:
            figures.append(("Map: busiest links", fig))
    return figures


def matrix_figures(
    output_dir: pathlib.Path, mode: ThemeMode, stations: pd.DataFrame | None
) -> Figures:
    """Map the annual matrix: biggest flows and each station's trips."""
    wide = read(output_dir / "odm_matrix.csv", index_col=0)
    if wide.empty:
        return []
    wide = wide.apply(pd.to_numeric, errors="coerce")
    long = (
        wide.rename_axis("origin")
        .reset_index()
        .melt(id_vars="origin", var_name="destination", value_name="journeys")
    )
    long = long[(long["origin"] != long["destination"]) & (long["journeys"] > 0)]
    figures: Figures = []
    fig = go.Figure(
        go.Histogram(x=np.log10(long["journeys"]), nbinsx=50, marker_color=colour(mode, 0))
    )
    ticks = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    fig.update_layout(
        title="Size of each station-to-station flow",
        xaxis={
            "title": "Journeys a year",
            "tickvals": np.log10(ticks),
            "ticktext": [f"{t:,}" for t in ticks],
        },
        yaxis_title="Station pairs",
    )
    figures.append(("Flow sizes", apply_theme(fig, mode)))
    fig = maps.flow_map(long, stations, "journeys", "The 300 biggest flows", mode)
    if fig is not None:
        figures.append(("Map: biggest flows", fig))
    totals = (wide.sum(axis=1) + wide.sum(axis=0)).rename("journeys").reset_index()
    totals.columns = ["crs", "journeys"]
    fig = maps.station_map(
        maps.with_positions(totals, stations),
        "journeys",
        "Journeys starting or ending at each station",
        mode,
    )
    if fig is not None:
        figures.append(("Map: station totals", fig))
    return figures


def loading_figures(
    output_dir: pathlib.Path, operator: str | None, mode: ThemeMode, stations: pd.DataFrame | None
) -> Figures:
    """Map operators' counted passengers by station; chart occupancy."""
    figures: Figures = []
    for path in sorted(output_dir.glob(f"{operator or '*'}_loading_analysis.csv")):
        name = path.stem.removesuffix("_loading_analysis").replace("_", " ")
        df = read(path, low_memory=False)
        need = {"Departure from CRS", "Mean Passengers On"}
        if df.empty or not need.issubset(df.columns):
            continue
        by_station = df.groupby("Departure from CRS")["Mean Passengers On"].agg(["sum", "size"])
        by_station.columns = ["passengers_on_board", "counts"]
        placed = maps.with_positions(by_station.reset_index(names="crs"), stations)
        fig = maps.station_map(
            placed,
            "passengers_on_board",
            f"Passengers on board leaving each station: {name}",
            mode,
            hover=["counts"],
        )
        if fig is not None:
            figures.append((f"Map: counts by station, {name}", fig))
        seats = [
            c
            for c in df.columns
            if c.strip() in ("Standard Seated Capacity", "First Class Seated Capacity")
        ]
        if seats:
            capacity = df[seats].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            factor = (df["Mean Passengers On"] / capacity.where(capacity > 0)).dropna()
            if len(factor):
                p = palette(mode)
                fig = go.Figure(
                    go.Histogram(
                        x=100 * factor.clip(upper=2), nbinsx=40, marker_color=colour(mode, 0)
                    )
                )
                fig.add_vline(x=100, line_dash="dot", line_color=p.text_secondary)
                fig.update_layout(
                    title=f"Passengers per seat on each counted departure: {name} "
                    f"({(factor > 1).mean():.0%} above seats)",
                    xaxis_title="Passengers as % of seats",
                    yaxis_title="Counted departures",
                )
                figures.append((f"Seat occupancy, {name}", apply_theme(fig, mode)))
    return figures


def bplan_figures(output_dir: pathlib.Path, mode: ThemeMode) -> Figures:
    """Map every BPLAN timing point that has a grid reference."""
    locations = read(output_dir / "bplan_locations.csv")
    if not {"easting", "northing", "tiploc"}.issubset(locations.columns):
        return []
    grid = locations[(locations["easting"] > 0) & (locations["northing"] > 0)]
    # Grid references outside Great Britain belong to foreign locations.
    grid = grid[(grid["easting"] < 700_000) & (grid["northing"] < 1_300_000)]
    if grid.empty:
        return []
    lon, lat = _FROM_GRID.transform(grid["easting"].to_numpy(), grid["northing"].to_numpy())
    # Some references put a location out at sea; keep those on land in
    # Great Britain's rough outline.
    west, east, south, north = _GB
    inside = (lon > west) & (lon < east) & (lat > south) & (lat < north)
    grid, lon, lat = grid[inside], lon[inside], lat[inside]
    p = palette(mode)
    names = grid.get("name", grid["tiploc"]).astype(str)
    fig = go.Figure(
        go.Scattermap(
            lat=lat,
            lon=lon,
            mode="markers",
            marker={"size": 4, "color": p.chart_colors[0]},
            text=grid["tiploc"].astype(str) + " " + names,
            hovertemplate="%{text}<extra></extra>",
            name="timing points",
        )
    )
    fig.update_layout(title=f"{len(grid):,} BPLAN timing points")
    return [("Map: timing points", maps.finish(fig, mode, pd.Series(lat), pd.Series(lon)))]


def propose_figures(
    output_dir: pathlib.Path, candidate: str | None, mode: ThemeMode, stations: pd.DataFrame | None
) -> Figures:
    """Map each proposed service's calls, with trips on each leg."""
    figures: Figures = []
    pattern = f"propose_{candidate or '*'}_demand_by_leg.csv"
    for path in sorted(output_dir.glob(pattern)):
        df = read(path)
        need = {"from_station", "to_station", "leg_stop_i", "leg_stop_j"}
        if not need.issubset(df.columns):
            continue
        name = path.stem.removeprefix("propose_").removesuffix("_demand_by_leg")
        stops = (
            pd.concat(
                [
                    df[["leg_stop_i", "from_station"]].set_axis(["i", "crs"], axis=1),
                    df[["leg_stop_j", "to_station"]].set_axis(["i", "crs"], axis=1),
                ]
            )
            .drop_duplicates("i")
            .sort_values("i")
        )
        codes = stops["crs"].astype(str).tolist()
        value = "predicted_annual_trips_captured"
        values = None
        if value in df.columns:
            # Trips on board a leg: every pair that boards at or
            # before it and alights after it.
            i, j, v = df["leg_stop_i"].to_numpy(), df["leg_stop_j"].to_numpy(), df[value].fillna(0)
            values = [float(v[(i <= k) & (j > k)].sum()) for k in stops["i"].iloc[:-1]]
        fig = maps.route_map(
            codes, stations, f"Calling points of {name}", mode, values, "annual trips on board"
        )
        if fig is not None:
            figures.append((f"Map: route of {name}", fig))
    return figures
