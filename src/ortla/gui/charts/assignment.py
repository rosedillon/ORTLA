"""Charts and maps of the national train loads (`ortla assign`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ortla.gui.charts import maps
from ortla.gui.charts.style import apply_theme, colour, read
from ortla.gui.constants import ThemeMode, palette

if TYPE_CHECKING:
    import pathlib

Figures = list[tuple[str, go.Figure]]
_SAMPLE = 20_000  # points on the scatter
_SMALLER = "smaller operators"


def _one_day(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Wednesday's rows if there are any, else the first day's."""
    if frame.empty or "day" not in frame.columns:
        return frame, ""
    days = frame["day"].unique().tolist()
    day = "Wednesday" if "Wednesday" in days else days[0]
    return frame[frame["day"] == day], day


def _fold(segments: pd.DataFrame) -> pd.DataFrame:
    """Fold the least-loaded operators into one, to fit the colours."""
    ranked = segments.groupby("operator")["load"].sum().sort_values(ascending=False)
    keep = ranked.index[: len(palette(ThemeMode.LIGHT).chart_colors) - 1]
    if len(ranked) <= len(keep) + 1:
        return segments
    return segments.assign(
        operator=segments["operator"].where(segments["operator"].isin(keep), _SMALLER)
    )


def assign_figures(
    output_dir: pathlib.Path,
    mode: ThemeMode = ThemeMode.LIGHT,
    stations: pd.DataFrame | None = None,
) -> Figures:
    """Draw scores, loads by hour and train, station and link maps."""
    figures: Figures = []
    figures += _scores(output_dir, mode)
    segments = read(output_dir / "assign_segments.csv", low_memory=False)
    if {"operator", "load"}.issubset(segments.columns):
        segments = _fold(segments)
    if not segments.empty and "actual_load" in segments.columns:
        figures += _against_counts(segments, mode, stations)
    if {"departure", "boardings", "operator"}.issubset(segments.columns):
        figures += _by_hour(segments, mode)
    figures += _trains(output_dir, mode)
    figures += _maps(output_dir, mode, stations)
    return figures


def _scores(output_dir: pathlib.Path, mode: ThemeMode) -> Figures:
    scores = read(output_dir / "assign_scores.csv")
    if scores.empty or not {"operator", "rmse_log", "ratio"}.issubset(scores.columns):
        return []
    ops = scores[scores["operator"] != "all"].sort_values("rmse_log")
    p = palette(mode)
    fig = go.Figure(
        go.Bar(
            x=ops["operator"],
            y=ops["rmse_log"],
            marker_color=colour(mode, 0),
            customdata=np.c_[ops["ratio"], ops["segments"]],
            hovertemplate="%{x}<br>rmse(log) %{y:.3f}<br>predicted / counted "
            "%{customdata[0]:.2f}<br>%{customdata[1]:,} segments<extra></extra>",
        )
    )
    overall = scores[scores["operator"] == "all"]
    if not overall.empty:
        fig.add_hline(
            y=float(overall["rmse_log"].iloc[0]),
            line_dash="dot",
            line_color=p.text_secondary,
            annotation_text=f"all {float(overall['rmse_log'].iloc[0]):.3f}",
        )
    fig.update_layout(
        title="Error against counts by operator: rmse of log(predicted / counted)",
        yaxis_title="rmse(log)",
    )
    figures: Figures = [("Scorecard: error by operator", apply_theme(fig, mode))]
    shown = scores.round(3)
    table = go.Figure(
        go.Table(
            header={
                "values": [c.replace("_", " ") for c in shown.columns],
                "fill_color": p.surface_container,
                "font": {"color": p.text_primary},
                "align": "left",
            },
            cells={
                "values": [shown[c] for c in shown.columns],
                "fill_color": p.surface,
                "font": {"color": p.text_primary},
                "align": "left",
            },
        )
    )
    table.update_layout(title="Scores against counts")
    figures.append(("Scorecard: table", apply_theme(table, mode)))
    return figures


def _against_counts(
    segments: pd.DataFrame, mode: ThemeMode, stations: pd.DataFrame | None
) -> Figures:
    counted = segments[(segments["actual_load"] > 0) & (segments["load"] > 0)]
    if counted.empty:
        return []
    sample = counted.sample(min(len(counted), _SAMPLE), random_state=0)
    fig = go.Figure()
    for i, (op, group) in enumerate(sample.groupby("operator")):
        fig.add_trace(
            go.Scattergl(
                x=group["actual_load"],
                y=group["load"],
                mode="markers",
                name=str(op),
                marker={"size": 4, "color": colour(mode, i), "opacity": 0.5},
                text=group["train_uid"].astype(str) + " " + group["from_station"].astype(str),
                hovertemplate="%{text}<br>counted %{x:,.0f}<br>predicted %{y:,.0f}<extra></extra>",
            )
        )
    top = float(max(sample["actual_load"].max(), sample["load"].max()))
    p = palette(mode)
    for factor, dash in ((1, "solid"), (2, "dot"), (0.5, "dot")):
        fig.add_trace(
            go.Scatter(
                x=[1, top],
                y=[factor, factor * top],
                mode="lines",
                line={"color": p.text_secondary, "dash": dash, "width": 1},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title=f"Predicted against counted load leaving each call ({len(counted):,} segments;"
        " dotted lines at twice and half)",
        xaxis={"title": "Counted", "type": "log"},
        yaxis={"title": "Predicted", "type": "log"},
    )
    figures: Figures = [("Predicted against counted", apply_theme(fig, mode))]
    by_station = (
        counted.assign(err=np.log(counted["load"] / counted["actual_load"]))
        .groupby("from_station")
        .agg(log_error=("err", "mean"), segments=("err", "size"))
        .reset_index(names="crs")
    )
    fig = maps.station_map(
        maps.with_positions(by_station[by_station["segments"] >= 5], stations),
        "log_error",
        "Mean log(predicted / counted) by station: red over, blue under",
        mode,
        signed=True,
        hover=["segments"],
    )
    if fig is not None:
        figures.append(("Map: error by station", fig))
    return figures


def _by_hour(segments: pd.DataFrame, mode: ThemeMode) -> Figures:
    day, name = _one_day(segments)
    hour = pd.to_numeric(day["departure"].astype(str).str[:2], errors="coerce")
    grid = day.assign(hour=hour).dropna(subset=["hour"])
    table = grid.pivot_table(
        index="hour", columns="operator", values="boardings", aggfunc="sum", fill_value=0
    )
    order = table.sum().sort_values(ascending=False).index
    fig = go.Figure()
    for i, op in enumerate(order):
        fig.add_bar(x=table.index, y=table[op], name=str(op), marker_color=colour(mode, i))
    fig.update_layout(
        title=f"Passengers boarding by hour, {name}",
        barmode="stack",
        xaxis={"title": "Hour", "dtick": 2},
        yaxis_title="Boardings",
    )
    return [("Boardings by hour", apply_theme(fig, mode))]


def _trains(output_dir: pathlib.Path, mode: ThemeMode) -> Figures:
    trains, name = _one_day(read(output_dir / "assign_trains.csv"))
    if trains.empty or "peak_load" not in trains.columns:
        return []
    figures: Figures = []
    fig = go.Figure(
        go.Histogram(
            x=np.log10(trains["peak_load"].clip(lower=1)), nbinsx=50, marker_color=colour(mode, 0)
        )
    )
    ticks = [1, 3, 10, 30, 100, 300, 1000]
    fig.update_layout(
        title=f"Busiest point of each train's run, {name}",
        xaxis={"title": "Peak load", "tickvals": np.log10(ticks), "ticktext": ticks},
        yaxis_title="Trains",
    )
    figures.append(("Peak load per train", apply_theme(fig, mode)))
    top = trains.nlargest(20, "peak_load").iloc[::-1]
    label = (
        top["headcode"].astype(str)
        + " "
        + top["train_departure"].astype(str)
        + " "
        + top["train_origin"].astype(str)
        + " → "
        + top["train_destination"].astype(str)
    )
    fig = go.Figure(
        go.Bar(
            x=top["peak_load"],
            y=label,
            orientation="h",
            marker_color=colour(mode, 0),
            customdata=top["operator"],
            hovertemplate="%{y}<br>%{customdata}<br>peak %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title=f"The 20 busiest trains, {name}", xaxis_title="Peak load", height=560)
    figures.append(("Busiest trains", apply_theme(fig, mode)))
    return figures


def _maps(output_dir: pathlib.Path, mode: ThemeMode, stations: pd.DataFrame | None) -> Figures:
    figures: Figures = []
    totals, name = _one_day(read(output_dir / "assign_stations.csv"))
    if {"crs", "boardings"}.issubset(totals.columns):
        fig = maps.station_map(
            maps.with_positions(totals, stations),
            "boardings",
            f"Passengers boarding at each station, {name}",
            mode,
            hover=["alightings"],
        )
        if fig is not None:
            figures.append(("Map: station boardings", fig))
    links, name = _one_day(read(output_dir / "assign_links.csv"))
    if {"from_station", "to_station", "passengers"}.issubset(links.columns):
        fig = maps.flow_map(
            links,
            stations,
            "passengers",
            f"The 1,500 busiest links between calls, {name}",
            mode,
            origin="from_station",
            destination="to_station",
            top=1500,
        )
        if fig is not None:
            figures.append(("Map: busiest links", fig))
    return figures
