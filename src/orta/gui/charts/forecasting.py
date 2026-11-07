"""Charts for the forecasting commands.

Hindcast, scenario, discovery, commercial case and new station. Each
function takes the command's output folder, the theme and the station
positions (``None`` when no config names a station file; the maps are
then left out) and returns ``(title, figure)`` pairs.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from orta.gui.charts import maps
from orta.gui.charts.style import PERIOD_ORDER, apply_theme, colour, order, pair_label, read
from orta.gui.constants import ThemeMode, palette

if TYPE_CHECKING:
    import pathlib

Figures = list[tuple[str, go.Figure]]
_LOG2 = float(np.log(2))


def _diagonal(fig: go.Figure, top: float, mode: ThemeMode, low: float = 0.0) -> None:
    fig.add_trace(
        go.Scatter(
            x=[low, top],
            y=[low, top],
            mode="lines",
            name="predicted = counted",
            line={"dash": "dot", "color": palette(mode).text_secondary, "width": 1},
            hoverinfo="skip",
        )
    )


def _x2_band(fig: go.Figure, low: float, top: float, mode: ThemeMode) -> None:
    """Dotted lines at half and double, the usual factor-of-two band."""
    for factor, name in ((2.0, "x2"), (0.5, "/2")):
        fig.add_trace(
            go.Scatter(
                x=[low, top],
                y=[low * factor, top * factor],
                mode="lines",
                name=name,
                line={"dash": "dot", "color": palette(mode).border, "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _log_ratio(predicted: pd.Series, actual: pd.Series) -> pd.Series:
    ok = (predicted > 0) & (actual > 0)
    return np.log(predicted[ok] / actual[ok])


def _histogram(values: pd.Series, title: str, x_title: str, mode: ThemeMode) -> go.Figure:
    """Histogram a log error, with lines at zero and a factor of two."""
    p = palette(mode)
    fig = go.Figure(go.Histogram(x=values, nbinsx=40, marker_color=p.chart_colors[0], name=""))
    for x in (-_LOG2, _LOG2):
        fig.add_vline(x=x, line_dash="dot", line_color=p.border)
    fig.add_vline(x=0, line_color=p.text_secondary, line_width=1)
    inside = float((values.abs() < _LOG2).mean()) if len(values) else float("nan")
    fig.update_layout(
        title=f"{title} ({inside:.0%} within a factor of two)",
        xaxis_title=x_title,
        yaxis_title="Count",
        showlegend=False,
    )
    return apply_theme(fig, mode)


# ── hindcast ─────────────────────────────────────────────────────────


def hindcast_figures(
    output_dir: pathlib.Path,
    mode: ThemeMode = ThemeMode.LIGHT,
    stations: pd.DataFrame | None = None,
) -> Figures:
    """Chart a launched service's forecast against its counts."""
    figures: Figures = []
    figures += _hindcast_scorecard(output_dir, mode)
    trains = read(output_dir / "hindcast_trains.csv")
    columns = [c for c in trains.columns if c.startswith("predicted_load_")]
    if not trains.empty and "actual_load" in trains.columns and columns:
        figures += _hindcast_trains(trains, columns, mode)
    segments = read(output_dir / "hindcast_segments.csv")
    seg_cols = [c for c in segments.columns if c.startswith("predicted_load_")]
    if not segments.empty and "actual_load" in segments.columns and seg_cols:
        figures += _hindcast_segments(segments, seg_cols[0], mode, stations)
    figures += _hindcast_pairs(output_dir, mode)
    figures += _hindcast_stations(output_dir, mode)
    return figures


def _hindcast_scorecard(output_dir: pathlib.Path, mode: ThemeMode) -> Figures:
    summary = read(output_dir / "hindcast_summary.csv")
    need = {"level", "demand_model", "demand_mode", "ratio_total_pred_to_actual"}
    if summary.empty or not need.issubset(summary.columns):
        return []
    if "basis" in summary.columns:
        summary = summary[summary["basis"].fillna("all_rows") == "all_rows"]
    # The ODM and feasibility rows measure other things on other scales.
    summary = summary[summary["level"].isin(["segment", "train"])]
    if summary.empty:
        return []
    summary = summary.assign(variant=summary["demand_model"] + " " + summary["demand_mode"])
    figures: Figures = []
    fig = go.Figure()
    for idx, (level, group) in enumerate(summary.groupby("level", sort=False)):
        fig.add_bar(
            x=group["variant"],
            y=group["ratio_total_pred_to_actual"],
            name=str(level),
            marker_color=colour(mode, idx),
            hovertemplate="%{x}<br>predicted / counted %{y:.2f}<extra>" + str(level) + "</extra>",
        )
    fig.add_hline(y=1.0, line_dash="dot", line_color=palette(mode).text_secondary)
    fig.update_layout(
        title="Total predicted over total counted, by model variant",
        yaxis_title="Predicted / counted",
        barmode="group",
        legend_title_text="Level",
    )
    figures.append(("Scorecard: level", apply_theme(fig, mode)))

    metrics = [m for m in ("mae", "rmse", "geh_mean", "pearson_r", "mape_pct") if m in summary]
    if metrics:
        shown = summary[["level", "variant", "n", *metrics]].round(3)
        p = palette(mode)
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
        table.update_layout(title="Error measures by level and model variant")
        figures.append(("Scorecard: table", apply_theme(table, mode)))
    return figures


def _hindcast_trains(trains: pd.DataFrame, columns: list[str], mode: ThemeMode) -> Figures:
    figures: Figures = []
    fig = go.Figure()
    # Hidden series do not count towards the axes, so the diagonal only
    # spans what is shown at first.
    top = 1.05 * float(trains[["actual_load", columns[0]]].max().max())
    for idx, column in enumerate(columns):
        label = column.removeprefix("predicted_load_").replace("_", " ")
        fig.add_trace(
            go.Scatter(
                x=trains["actual_load"],
                y=trains[column],
                mode="markers",
                name=label,
                marker={"color": colour(mode, idx), "size": 8, "opacity": 0.75},
                text=trains["train_uid"].astype(str) + " " + trains["day"].astype(str),
                hovertemplate="%{text}<br>counted %{x:.0f}<br>predicted %{y:.0f}<extra></extra>",
                visible=True if idx == 0 else "legendonly",
            )
        )
    _diagonal(fig, top, mode)
    fig.update_layout(
        title="Peak load per train: predicted against counted",
        xaxis_title="Counted load",
        yaxis_title="Predicted load",
    )
    figures.append(("Predicted against counted, by train", apply_theme(fig, mode)))

    main = columns[0]
    error = _log_ratio(trains[main], trains["actual_load"])
    if len(error):
        figures.append(
            (
                "Error distribution, by train",
                _histogram(error, "Train error", "log(predicted / counted)", mode),
            )
        )

    if "period" in trains.columns:
        by_period = trains.groupby("period")[["actual_load", main]].sum()
        by_period = by_period.reindex(order(list(by_period.index), PERIOD_ORDER))
        fig = go.Figure()
        fig.add_bar(
            x=by_period.index,
            y=by_period["actual_load"],
            name="counted",
            marker_color=colour(mode, 0),
        )
        fig.add_bar(
            x=by_period.index, y=by_period[main], name="predicted", marker_color=colour(mode, 1)
        )
        fig.update_layout(
            title=f"Load by period ({main.removeprefix('predicted_load_').replace('_', ' ')})",
            barmode="group",
            yaxis_title="Sum of train loads",
        )
        figures.append(("Load by period", apply_theme(fig, mode)))

    if "train_departure" in trains.columns and len(error):
        hour = pd.to_datetime(
            trains.loc[error.index, "train_departure"], format="%H:%M", errors="coerce"
        ).dt.hour
        fig = go.Figure(
            go.Box(x=hour, y=error, marker_color=colour(mode, 0), boxpoints="outliers", name="")
        )
        fig.add_hline(y=0, line_color=palette(mode).text_secondary, line_width=1)
        fig.update_layout(
            title="Train error by departure hour",
            xaxis_title="Departure hour",
            yaxis_title="log(predicted / counted)",
            showlegend=False,
        )
        figures.append(("Error by departure hour", apply_theme(fig, mode)))
    return figures


def _hindcast_segments(
    segments: pd.DataFrame, column: str, mode: ThemeMode, stations: pd.DataFrame | None
) -> Figures:
    figures: Figures = []
    # Stations in the order the trains first reach them.
    seq = list(dict.fromkeys(segments["from_station"].astype(str)))
    by_station = segments.groupby("from_station")[["actual_load", column]].mean()
    by_station = by_station.reindex([s for s in seq if s in by_station.index])
    fig = go.Figure()
    fig.add_scatter(
        x=by_station.index,
        y=by_station["actual_load"],
        name="counted",
        mode="lines+markers",
        line={"color": colour(mode, 0), "width": 2},
    )
    fig.add_scatter(
        x=by_station.index,
        y=by_station[column],
        name="predicted",
        mode="lines+markers",
        line={"color": colour(mode, 1), "width": 2},
    )
    fig.update_layout(
        title="Mean load leaving each station",
        xaxis_title="Station, in the order trains reach it",
        yaxis_title="Mean load",
    )
    figures.append(("Load along the line", apply_theme(fig, mode)))

    sums = segments.groupby("from_station").agg(
        counted=("actual_load", "sum"), predicted=(column, "sum"), segments=("actual_load", "size")
    )
    sums = sums[(sums["counted"] > 0) & (sums["predicted"] > 0)]
    sums["log_error"] = np.log(sums["predicted"] / sums["counted"])
    placed = maps.with_positions(sums.reset_index(names="crs"), stations)
    fig = maps.station_map(
        placed,
        "log_error",
        "Where the forecast is over (red) or under (blue) the counts",
        mode,
        signed=True,
        hover=["counted", "predicted", "segments"],
    )
    if fig is not None:
        figures.append(("Map: error by station", fig))
    return figures


def _hindcast_pairs(output_dir: pathlib.Path, mode: ThemeMode) -> Figures:
    pairs = read(output_dir / "hindcast_odm_pairs.csv")
    need = {"origin_tlc", "destination_tlc", "journeys_after_actual", "journeys_after_predicted"}
    if pairs.empty or not need.issubset(pairs.columns):
        return []
    pairs = pairs[(pairs["journeys_after_actual"] > 0) & (pairs["journeys_after_predicted"] > 0)]
    if pairs.empty:
        return []
    low = float(pairs[["journeys_after_actual", "journeys_after_predicted"]].min().min())
    top = float(pairs[["journeys_after_actual", "journeys_after_predicted"]].max().max())
    new = pairs.get("new_direct_link", pd.Series(False, index=pairs.index)).astype(bool)
    fig = go.Figure()
    for idx, (flag, name) in enumerate(((False, "existing link"), (True, "new direct link"))):
        sub = pairs[new == flag]
        fig.add_scatter(
            x=sub["journeys_after_actual"],
            y=sub["journeys_after_predicted"],
            mode="markers",
            name=name,
            marker={"size": 8, "color": colour(mode, idx), "opacity": 0.75},
            text=pair_label(sub, "origin_tlc", "destination_tlc"),
            hovertemplate="%{text}<br>ODM after %{x:,.0f}<br>predicted %{y:,.0f}<extra></extra>",
        )
    _diagonal(fig, top, mode, low)
    _x2_band(fig, low, top, mode)
    fig.update_layout(
        title="Annual journeys after the change: predicted against the ODM",
        xaxis={"type": "log", "title": "ODM journeys after"},
        yaxis={"type": "log", "title": "Predicted journeys after"},
    )
    return [("Station pairs against the ODM", apply_theme(fig, mode))]


def _hindcast_stations(output_dir: pathlib.Path, mode: ThemeMode) -> Figures:
    balance = read(output_dir / "hindcast_station_balance.csv")
    if balance.empty or not {"crs", "factor"}.issubset(balance.columns):
        return []
    figures: Figures = []
    if "forecast_entries_exits" in balance.columns:
        fig = go.Figure(
            go.Bar(
                x=balance["crs"],
                y=balance["forecast_entries_exits"],
                marker_color=colour(mode, 0),
                name="",
            )
        )
        fig.update_layout(
            title="Forecast annual entries and exits at the new stations",
            yaxis_title="Entries and exits a year",
        )
        figures.append(("New stations' footfall", apply_theme(fig, mode)))
    fig = go.Figure(go.Bar(x=balance["crs"], y=balance["factor"], marker_color=colour(mode, 2)))
    fig.add_hline(y=1.0, line_dash="dot", line_color=palette(mode).text_secondary)
    fig.update_layout(
        title="Scaling applied to hold each new station to its forecast footfall",
        yaxis_title="Factor",
    )
    figures.append(("Station balance factors", apply_theme(fig, mode)))
    return figures


# ── scenario ─────────────────────────────────────────────────────────

_CALL = re.compile(r"([A-Z]{3})\(")


def scenario_figures(
    output_dir: pathlib.Path,
    mode: ThemeMode = ThemeMode.LIGHT,
    stations: pd.DataFrame | None = None,
) -> Figures:
    """Chart each candidate's best option and every option swept."""
    summary = read(output_dir / "scenario_summary.csv")
    if summary.empty or "candidate_name" not in summary.columns:
        return []
    figures: Figures = []
    if "best_bcr" in summary.columns:
        ranked = summary.sort_values("best_bcr")
        fig = go.Figure(
            go.Bar(
                x=ranked["best_bcr"],
                y=ranked["candidate_name"],
                orientation="h",
                marker_color=colour(mode, 0),
            )
        )
        fig.add_vline(x=1.0, line_dash="dot", line_color=palette(mode).text_secondary)
        fig.update_layout(
            title="Benefit-cost ratio of each candidate's best option", xaxis_title="BCR"
        )
        figures.append(("Benefit-cost ratio", apply_theme(fig, mode)))
    if {"best_pvb_m", "best_pvc_m"}.issubset(summary.columns):
        fig = go.Figure()
        fig.add_bar(
            x=summary["candidate_name"],
            y=summary["best_pvb_m"],
            name="benefits",
            marker_color=colour(mode, 0),
        )
        fig.add_bar(
            x=summary["candidate_name"],
            y=summary["best_pvc_m"],
            name="costs",
            marker_color=colour(mode, 1),
        )
        fig.update_layout(
            title="Present value of benefits and costs, best option",
            yaxis_title="£ million",
            barmode="group",
        )
        figures.append(("Benefits and costs", apply_theme(fig, mode)))
    if "best_annual_passengers" in summary.columns:
        fig = go.Figure()
        fig.add_bar(
            x=summary["candidate_name"],
            y=summary["best_annual_passengers"],
            name="all passengers",
            marker_color=colour(mode, 0),
        )
        if "best_annual_passengers_generated" in summary.columns:
            fig.add_bar(
                x=summary["candidate_name"],
                y=summary["best_annual_passengers_generated"],
                name="new to rail",
                marker_color=colour(mode, 2),
            )
        fig.update_layout(
            title="Annual passengers on the best option",
            barmode="group",
            yaxis_title="Passengers a year",
        )
        figures.append(("Annual passengers", apply_theme(fig, mode)))

    options = read(output_dir / "scenario_options.csv")
    if {"candidate_name", "departure_time", "bcr"}.issubset(options.columns):
        fig = go.Figure()
        for idx, (name, unsorted) in enumerate(options.groupby("candidate_name", sort=False)):
            group = unsorted.sort_values("departure_time")
            feasible = group.get("feasible", pd.Series(True, index=group.index)).astype(bool)
            fig.add_scatter(
                x=group["departure_time"],
                y=group["bcr"],
                mode="markers",
                name=str(name),
                marker={
                    "color": colour(mode, idx),
                    "size": 6
                    + 2
                    * group.get("trains_per_day", pd.Series(1, index=group.index)).clip(upper=8),
                    "symbol": np.where(feasible, "circle", "x"),
                },
                text=group.get("label", group["departure_time"]).astype(str),
                hovertemplate="%{text}<br>BCR %{y:.2f}<extra>" + str(name) + "</extra>",
            )
        fig.add_hline(y=1.0, line_dash="dot", line_color=palette(mode).text_secondary)
        fig.update_layout(
            title="Every option tried: BCR by first departure (size: trains a day; x: infeasible)",
            xaxis_title="First departure",
            yaxis_title="BCR",
        )
        figures.append(("Options by departure time", apply_theme(fig, mode)))

    if "best_schedule" in summary.columns and stations is not None:
        fig = go.Figure()
        lats, lons = [], []
        for idx, row in enumerate(summary.itertuples()):
            codes = [c for c in _CALL.findall(str(row.best_schedule)) if c in stations.index]
            if len(codes) < 2:
                continue
            pos = stations.loc[codes]
            lats.append(pos["lat"])
            lons.append(pos["lon"])
            fig.add_trace(
                go.Scattermap(
                    lat=pos["lat"],
                    lon=pos["lon"],
                    mode="lines+markers",
                    name=row.candidate_name,
                    line={"width": 4, "color": colour(mode, idx)},
                    marker={"size": 9, "color": colour(mode, idx)},
                    text=codes,
                    hovertemplate="%{text}<extra>" + row.candidate_name + "</extra>",
                )
            )
        if lats:
            fig.update_layout(title="Calling points of each candidate's best option")
            figures.append(
                ("Map: candidate routes", maps.finish(fig, mode, pd.concat(lats), pd.concat(lons)))
            )
    return figures


# ── discovery ────────────────────────────────────────────────────────


def discover_figures(
    output_dir: pathlib.Path,
    mode: ThemeMode = ThemeMode.LIGHT,
    stations: pd.DataFrame | None = None,
) -> Figures:
    """Chart the discovery funnel and the candidates it found."""
    figures: Figures = []
    funnel = read(output_dir / "discovery_funnel_summary.csv")
    if not funnel.empty:
        row = funnel.iloc[0]
        fig = go.Figure(
            go.Funnel(
                y=[k.removeprefix("n_").replace("_", " ") for k in row.index],
                x=list(row.to_numpy()),
                marker={"color": colour(mode, 0)},
            )
        )
        fig.update_layout(title="Station pairs left at each stage")
        figures.append(("Funnel", apply_theme(fig, mode)))

    ranked = read(output_dir / "discovery_ranked_candidates.csv")
    if not {"origin", "destination"}.issubset(ranked.columns):
        return figures
    metric = next((m for m in ("best_bcr", "predicted_annual_trips") if m in ranked.columns), None)
    if metric:
        top = ranked.dropna(subset=[metric]).nlargest(20, metric).iloc[::-1]
        fig = go.Figure(
            go.Bar(x=top[metric], y=pair_label(top), orientation="h", marker_color=colour(mode, 0))
        )
        if metric == "best_bcr":
            fig.add_vline(x=1.0, line_dash="dot", line_color=palette(mode).text_secondary)
        fig.update_layout(title=f"Top 20 candidates by {metric.replace('_', ' ')}")
        figures.append(("Top candidates", apply_theme(fig, mode)))
        fig = maps.flow_map(
            ranked,
            stations,
            metric,
            f"Candidate services, by {metric.replace('_', ' ')}",
            mode,
            top=200,
        )
        if fig is not None:
            figures.append(("Map: candidates", fig))

    if {"distance_miles", "best_bcr"}.issubset(ranked.columns):
        fig = go.Figure()
        bands = ranked.get("route_type_band", pd.Series("all", index=ranked.index)).fillna("other")
        for idx, (band, group) in enumerate(ranked.groupby(bands)):
            fig.add_scatter(
                x=group["distance_miles"],
                y=group["best_bcr"],
                mode="markers",
                name=str(band),
                marker={"size": 8, "color": colour(mode, idx), "opacity": 0.75},
                text=pair_label(group),
                hovertemplate="%{text}<br>%{x:.0f} miles<br>BCR %{y:.2f}<extra></extra>",
            )
        fig.add_hline(y=1.0, line_dash="dot", line_color=palette(mode).text_secondary)
        fig.update_layout(
            title="Value for money against distance", xaxis_title="Miles", yaxis_title="Best BCR"
        )
        figures.append(("BCR against distance", apply_theme(fig, mode)))

    if "status" in ranked.columns:
        counts = ranked["status"].fillna("unknown").value_counts().sort_values()
        fig = go.Figure(
            go.Bar(
                x=counts.to_numpy(),
                y=[s.replace("_", " ") for s in counts.index],
                orientation="h",
                marker_color=colour(mode, 0),
            )
        )
        fig.update_layout(title="What happened to each ranked candidate", xaxis_title="Pairs")
        figures.append(("Candidate status", apply_theme(fig, mode)))
    return figures


# ── commercial case ──────────────────────────────────────────────────


def commercial_figures(
    output_dir: pathlib.Path,
    mode: ThemeMode = ThemeMode.LIGHT,
    stations: pd.DataFrame | None = None,  # noqa: ARG001
) -> Figures:
    """Chart the cashflow, its parts and the NPV's sensitivity."""
    figures: Figures = []
    cash = read(output_dir / "commercial_case_cashflow.csv")
    if "year" in cash.columns:
        fig = go.Figure()
        for idx, column in enumerate(["cumulative_net_cashflow", "cumulative_npv"]):
            if column in cash.columns:
                fig.add_scatter(
                    x=cash["year"],
                    y=cash[column] / 1e6,
                    mode="lines+markers",
                    name=column.replace("_", " "),
                    line={"color": colour(mode, idx), "width": 2},
                )
        fig.add_hline(y=0, line_dash="dot", line_color=palette(mode).text_secondary)
        fig.update_layout(title="Cumulative cashflow", yaxis_title="£ million")
        figures.append(("Cumulative cashflow", apply_theme(fig, mode)))

        parts = [c for c in ("revenue", "operating_cost", "capital_cost") if c in cash.columns]
        if parts:
            fig = go.Figure()
            for idx, column in enumerate(parts):
                sign = 1 if column == "revenue" else -1
                fig.add_bar(
                    x=cash["year"],
                    y=sign * cash[column].abs() / 1e6,
                    name=column.replace("_", " "),
                    marker_color=colour(mode, idx),
                )
            if "net_cashflow" in cash.columns:
                fig.add_scatter(
                    x=cash["year"],
                    y=cash["net_cashflow"] / 1e6,
                    name="net",
                    mode="lines+markers",
                    line={"color": palette(mode).text_primary, "width": 2},
                )
            fig.update_layout(
                title="Money in and out each year", barmode="relative", yaxis_title="£ million"
            )
            figures.append(("Annual cashflow", apply_theme(fig, mode)))

        pax = [c for c in ("proposed_passengers", "baseline_passengers") if c in cash.columns]
        if pax:
            fig = go.Figure()
            for idx, column in enumerate(pax):
                fig.add_scatter(
                    x=cash["year"],
                    y=cash[column],
                    mode="lines",
                    name=column.removesuffix("_passengers"),
                    line={"color": colour(mode, idx), "width": 2},
                )
            fig.update_layout(title="Passengers each year", yaxis_title="Passengers")
            figures.append(("Passengers", apply_theme(fig, mode)))

    sens = read(output_dir / "commercial_case_sensitivity.csv")
    summary = read(output_dir / "commercial_case_summary.csv", header=None, index_col=0)
    if {"scenario", "npv_m"}.issubset(sens.columns) and "npv_m" in summary.index:
        base = float(summary.loc["npv_m"].iloc[0])
        sens = sens.assign(delta=sens["npv_m"] - base)
        sens = sens.reindex(sens["delta"].abs().sort_values().index)
        p = palette(mode)
        fig = go.Figure(
            go.Bar(
                x=sens["delta"],
                y=sens["scenario"],
                orientation="h",
                marker_color=[
                    p.chart_colors[0] if d >= 0 else p.chart_colors[1] for d in sens["delta"]
                ],
                customdata=sens["npv_m"],
                hovertemplate="%{y}<br>NPV £%{customdata:.1f}m (%{x:+.1f}m)<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color=p.text_secondary, line_width=1)
        fig.update_layout(
            title=f"How the NPV moves with each input (base £{base:.1f}m)",
            xaxis_title="Change in NPV, £ million",
        )
        figures.append(("Sensitivity", apply_theme(fig, mode)))
    return figures


# ── new station ──────────────────────────────────────────────────────


def new_station_figures(
    output_dir: pathlib.Path,
    mode: ThemeMode = ThemeMode.LIGHT,
    stations: pd.DataFrame | None = None,
) -> Figures:
    """Chart new stations' forecasts, neighbours and calibration fit."""
    figures: Figures = []
    forecasts = read(output_dir / "new_station_forecasts.csv")
    if "forecast_entries_exits" not in forecasts.columns:
        return figures
    p = palette(mode)
    fig = go.Figure()
    for level, fill, width in (("90", p.border, 0.5), ("80", p.chart_colors[0], 0.3)):
        low, high = forecasts[f"low_{level}"], forecasts[f"high_{level}"]
        fig.add_bar(
            x=forecasts["crs"],
            y=high - low,
            base=low,
            width=width,
            name=f"{level}% interval",
            marker_color=fill,
            opacity=0.6 if level == "90" else 0.85,
            hovertemplate="%{x}: %{base:,.0f} to %{y:,.0f} wide<extra></extra>",
        )
    fig.add_scatter(
        x=forecasts["crs"],
        y=forecasts["forecast_entries_exits"],
        mode="markers",
        name="forecast",
        marker={"color": p.text_primary, "size": 12, "symbol": "diamond"},
        hovertemplate="%{x}: %{y:,.0f} a year<extra></extra>",
    )
    fig.update_layout(
        title="Annual entries and exits: forecast and intervals",
        barmode="overlay",
        yaxis={"type": "log", "title": "Entries and exits a year"},
    )
    figures.append(("Forecasts", apply_theme(fig, mode)))

    inputs = [c for c in ("prw_pop", "jobs", "freq", "car_spaces") if c in forecasts.columns]
    if inputs:
        fig = go.Figure()
        for idx, column in enumerate(inputs):
            fig.add_bar(
                x=forecasts["crs"],
                y=forecasts[column],
                name=column.replace("_", " "),
                marker_color=colour(mode, idx),
                visible=True if idx == 0 else "legendonly",
            )
        fig.update_layout(
            title="What drives each forecast (click the legend to switch input)",
            barmode="group",
        )
        figures.append(("Forecast inputs", apply_theme(fig, mode)))

    near = read(output_dir / "new_station_neighbours.csv")
    if "catchment_change" in near.columns:
        losers = near[near["catchment_change"] < 0].sort_values("catchment_change")
        fig = go.Figure(
            go.Bar(
                x=100 * losers["catchment_change"],
                y=losers["neighbour"] + " (" + losers["new_station"] + ")",
                orientation="h",
                marker_color=colour(mode, 1),
            )
        )
        fig.update_layout(
            title="Catchment each neighbour loses to the new station",
            xaxis_title="Change in weighted catchment (%)",
        )
        figures.append(("Neighbours' catchments", apply_theme(fig, mode)))

    placed = maps.with_positions(forecasts, stations)
    if not placed.empty:
        near_placed = maps.with_positions(near.rename(columns={"neighbour": "crs"}), stations)
        fig = None
        if not near_placed.empty:
            fig = maps.station_map(
                near_placed.assign(change_pct=100 * near_placed["catchment_change"]),
                "change_pct",
                "New stations (black) and the catchment their neighbours lose (%)",
                mode,
                signed=True,
                hover=["new_station", "km", "catchment_before", "catchment_after"],
            )
        fig = fig or go.Figure()
        fig.add_trace(
            go.Scattermap(
                lat=placed["lat"],
                lon=placed["lon"],
                mode="markers+text",
                marker={"size": 18, "color": p.text_primary},
                text=placed["crs"],
                textposition="top right",
                textfont={"color": p.text_primary},
                customdata=placed["forecast_entries_exits"],
                name="new station",
                hovertemplate="%{text}<br>forecast %{customdata:,.0f} a year<extra></extra>",
            )
        )
        if near_placed.empty:
            fig.update_layout(title="New stations")
            maps.finish(fig, mode, placed["lat"], placed["lon"])
        figures.append(("Map: new stations and neighbours", fig))

    cal = read(output_dir / "new_station_calibration_stations.csv")
    if {"footfall", "fitted"}.issubset(cal.columns):
        cal = cal[(cal["footfall"] > 0) & (cal["fitted"] > 0)]
        low, top = (
            float(cal[["footfall", "fitted"]].min().min()),
            float(cal[["footfall", "fitted"]].max().max()),
        )
        fig = go.Figure(
            go.Scattergl(
                x=cal["footfall"],
                y=cal["fitted"],
                mode="markers",
                name="stations",
                marker={"size": 6, "color": colour(mode, 0), "opacity": 0.5},
                text=cal["crs"],
                hovertemplate="%{text}<br>counted %{x:,.0f}<br>model %{y:,.0f}<extra></extra>",
            )
        )
        _diagonal(fig, top, mode, low)
        _x2_band(fig, low, top, mode)
        fig.update_layout(
            title="The trip-end model on the stations it was calibrated on",
            xaxis={"type": "log", "title": "Counted entries and exits"},
            yaxis={"type": "log", "title": "Model entries and exits"},
        )
        figures.append(("Calibration fit", apply_theme(fig, mode)))
        if "log_residual" in cal.columns:
            figures.append(
                (
                    "Calibration residuals",
                    _histogram(
                        cal["log_residual"], "Calibration residuals", "log(counted / model)", mode
                    ),
                )
            )
            fig = maps.station_map(
                maps.with_positions(cal, stations),
                "log_residual",
                "Where the model under-predicts (red) or over-predicts (blue) today's stations",
                mode,
                signed=True,
                hover=["footfall", "fitted"],
                size=False,
            )
            if fig is not None:
                figures.append(("Map: calibration residuals", fig))
    return figures
