"""Charts for the data commands: loading, lookups, timetables and so on.

Each function reads the CSVs one command writes and returns
``(title, figure)`` pairs. :func:`orta.gui.charts.build_figures` picks
the functions for a command.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from orta import fares
from orta.gui.charts.style import DAY_ORDER, PERIOD_ORDER, apply_theme, order
from orta.gui.constants import ThemeMode, palette

if TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)

# ── loading: shared helpers ───────────────────────────────────────────


def _loading_prefix(operator: str | None) -> str:
    return operator or "*"


# ── loading: temporal factor heatmaps ─────────────────────────────────


def temporal_factor_figures(
    output_dir: pathlib.Path, operator: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart heatmaps of the temporal factors, one per journey type."""
    figures = []
    palette(mode)
    for path in sorted(output_dir.glob(f"{_loading_prefix(operator)}_temporal_factors.csv")):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty or "journey_type" not in df.columns:
            continue

        operator_name = path.stem.replace("_temporal_factors", "")
        journey_types = order(df["journey_type"].astype(str).tolist(), [])

        # Create one heatmap per journey type as separate figures/tabs
        for jt in journey_types:
            sub = df[df["journey_type"].astype(str) == jt]
            grid = sub.pivot_table(
                index="Service Week Day",
                columns="time_period",
                values="factor",
                aggfunc="sum",
            )
            grid = grid.reindex(
                index=order(grid.index.tolist(), DAY_ORDER),
                columns=order(grid.columns.tolist(), PERIOD_ORDER),
            )

            fig = go.Figure(
                data=go.Heatmap(
                    z=grid.values,
                    x=grid.columns,
                    y=grid.index,
                    colorscale="YlOrRd" if mode == ThemeMode.LIGHT else "Viridis",
                    text=np.round(grid.values, 3),
                    texttemplate="%{text}",
                    hoverinfo="x+y+z",
                )
            )

            fig.update_layout(
                title=dict(
                    text=f"Temporal Factors ({jt}): {operator_name}",
                    font=dict(size=14, weight="bold"),
                ),
                xaxis_title="Time Period",
                yaxis_title="Day of Week",
            )
            apply_theme(fig, mode)
            figures.append((f"Factors ({jt}): {operator_name}", fig))

    return figures


def journey_type_figures(
    output_dir: pathlib.Path, operator: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart total demand by journey type."""
    figures = []
    p = palette(mode)
    for path in sorted(output_dir.glob(f"{_loading_prefix(operator)}_temporal_factors.csv")):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty or "journey_type" not in df.columns or "boardings" not in df.columns:
            continue
        operator_name = path.stem.replace("_temporal_factors", "")
        totals = df.groupby("journey_type")["boardings"].sum().sort_values(ascending=True)

        fig = go.Figure(
            data=go.Bar(
                y=totals.index.astype(str),
                x=totals.values,
                orientation="h",
                marker_color=p.success,
                text=totals.to_numpy().astype(int),
                textposition="auto",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Demand by Journey Type: {operator_name}", font=dict(size=14, weight="bold")
            ),
            xaxis_title="Total Boardings",
            yaxis_title="Journey Type",
        )
        apply_theme(fig, mode)
        figures.append((f"Journey mix: {operator_name}", fig))
    return figures


def demand_profile_figures(
    output_dir: pathlib.Path, operator: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart demand by day of the week and by period."""
    figures = []
    p = palette(mode)
    for path in sorted(output_dir.glob(f"{_loading_prefix(operator)}_temporal_factors.csv")):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        needed = {"boardings", "Service Week Day", "time_period"}
        if df.empty or not needed.issubset(df.columns):
            continue
        operator_name = path.stem.replace("_temporal_factors", "")

        by_day = df.groupby("Service Week Day")["boardings"].sum()
        by_day = by_day.reindex(order(by_day.index.astype(str).tolist(), DAY_ORDER))
        by_period = df.groupby("time_period")["boardings"].sum()
        by_period = by_period.reindex(order(by_period.index.astype(str).tolist(), PERIOD_ORDER))

        # Build subplots: Day of Week (left), Time Period (right)
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Demand by Day of Week", "Demand by Time Period")
        )

        fig.add_trace(
            go.Bar(x=by_day.index.astype(str), y=by_day.values, marker_color=p.info, name="Day"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=by_period.index.astype(str),
                y=by_period.to_numpy(),
                marker_color=p.warning,
                name="Period",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title=dict(text=f"Demand Profile: {operator_name}", font=dict(size=14, weight="bold")),
            showlegend=False,
        )
        apply_theme(fig, mode)
        figures.append((f"Demand profile: {operator_name}", fig))
    return figures


def demand_grid_figures(
    output_dir: pathlib.Path, operator: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart a day by period grid of total demand."""
    figures = []
    for path in sorted(output_dir.glob(f"{_loading_prefix(operator)}_temporal_factors.csv")):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        needed = {"boardings", "Service Week Day", "time_period"}
        if df.empty or not needed.issubset(df.columns):
            continue
        operator_name = path.stem.replace("_temporal_factors", "")
        grid = df.pivot_table(
            index="Service Week Day",
            columns="time_period",
            values="boardings",
            aggfunc="sum",
        )
        grid = grid.reindex(
            index=order(grid.index.astype(str).tolist(), DAY_ORDER),
            columns=order(grid.columns.astype(str).tolist(), PERIOD_ORDER),
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=grid.values,
                x=grid.columns,
                y=grid.index,
                colorscale="YlOrRd" if mode == ThemeMode.LIGHT else "Viridis",
                text=np.round(grid.values, 0),
                texttemplate="%{text}",
                hoverinfo="x+y+z",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Total Demand (Boardings): {operator_name}",
                font=dict(size=14, weight="bold"),
            ),
            xaxis_title="Time Period",
            yaxis_title="Day of Week",
        )
        apply_theme(fig, mode)
        figures.append((f"Demand grid: {operator_name}", fig))
    return figures


# ── loading: analysis figures (loads, stations, distances)
# ───────────────────


def analysis_figures(
    output_dir: pathlib.Path,
    operator: str | None = None,
    demand_col: str = "Mean Passengers On",
    mode: ThemeMode = ThemeMode.LIGHT,
) -> list[tuple[str, go.Figure]]:
    """Chart loads, busy stations, distance and airport demand."""
    figures = []
    p = palette(mode)
    for path in sorted(output_dir.glob(f"{_loading_prefix(operator)}_loading_analysis.csv")):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty or demand_col not in df.columns:
            continue
        name = path.stem.replace("_loading_analysis", "")
        df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce")

        # 1. Distribution & Boxplot
        values = df[demand_col].dropna()
        if not values.empty:
            fig_dist = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(f"{demand_col} Distribution", "Demand by Time Period"),
            )

            # Histogram
            fig_dist.add_trace(
                go.Histogram(
                    x=values, nbinsx=40, marker_color=p.info, opacity=0.7, name="Density"
                ),
                row=1,
                col=1,
            )

            # Boxplot by time period
            if "time_period" in df.columns:
                periods = order(df["time_period"].dropna().astype(str).tolist(), PERIOD_ORDER)
                for prd in periods:
                    sub_vals = df.loc[df["time_period"].astype(str) == prd, demand_col].dropna()
                    if not sub_vals.empty:
                        fig_dist.add_trace(
                            go.Box(y=sub_vals, name=prd, boxpoints=False), row=1, col=2
                        )

            fig_dist.update_layout(
                title=dict(
                    text=f"Loading Distribution: {name}", font=dict(size=14, weight="bold")
                ),
                showlegend=False,
            )
            apply_theme(fig_dist, mode)
            figures.append((f"Loads: {name}", fig_dist))

        # 2. Busiest Stations
        if "Departure from CRS" in df.columns:
            totals = (
                df.groupby(df["Departure from CRS"].astype(str))[demand_col]
                .sum()
                .sort_values(ascending=False)
                .head(15)
                .sort_values(ascending=True)
            )
            if not totals.empty:
                fig_stn = go.Figure(
                    data=go.Bar(
                        y=totals.index.astype(str),
                        x=totals.values,
                        orientation="h",
                        marker_color=p.info,
                    )
                )
                fig_stn.update_layout(
                    title=dict(
                        text=f"Busiest Boarding Stations: {name}",
                        font=dict(size=14, weight="bold"),
                    ),
                    xaxis_title=f"Total {demand_col}",
                    yaxis_title="Departure Station (CRS)",
                )
                apply_theme(fig_stn, mode)
                figures.append((f"Top stations: {name}", fig_stn))

        # 3. Distance vs Demand Scatter
        candidates = ("distance_to_pax_off_miles", "distance_to_1410_miles")
        dist_col = next((c for c in candidates if c in df.columns), None)
        if dist_col is not None:
            pair = df[[dist_col, demand_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) >= 3 and pair[dist_col].nunique() >= 2:
                x = pair[dist_col].to_numpy()
                y = pair[demand_col].to_numpy()

                fig_scat = go.Figure()
                fig_scat.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker=dict(color=p.info, opacity=0.4, size=6),
                        name="Data Points",
                    )
                )

                # Fit line
                fit = stats.linregress(x, y)
                xs = np.linspace(x.min(), x.max(), 100)
                fig_scat.add_trace(
                    go.Scatter(
                        x=xs,
                        y=fit.intercept + fit.slope * xs,
                        mode="lines",
                        line=dict(color=p.error, width=2.5),
                        name=f"Fit (r={fit.rvalue:.2f})",
                    )
                )

                fig_scat.update_layout(
                    title=dict(
                        text=f"Distance vs Demand: {name}", font=dict(size=14, weight="bold")
                    ),
                    xaxis_title=f"{dist_col} (miles)",
                    yaxis_title=demand_col,
                )
                apply_theme(fig_scat, mode)
                figures.append((f"Distance vs demand: {name}", fig_scat))

        # 4. Airport Comparison
        if "is_airport" in df.columns:
            flag = df["is_airport"].astype(str).str.lower().isin({"true", "1", "1.0"})
            means = df.groupby(flag)[demand_col].mean()
            if not (means.empty or means.isna().all()):
                labels = {True: "Airport Touching", False: "Other Services"}
                ordered = [k for k in (False, True) if k in means.index]
                fig_air = go.Figure(
                    data=go.Bar(
                        x=[labels[k] for k in ordered],
                        y=[means[k] for k in ordered],
                        marker_color=[p.text_disabled, p.warning][: len(ordered)],
                        text=np.round([means[k] for k in ordered], 1),
                        textposition="auto",
                    )
                )
                fig_air.update_layout(
                    title=dict(
                        text=f"Airport vs Non-Airport Demand: {name}",
                        font=dict(size=14, weight="bold"),
                    ),
                    yaxis_title=f"Mean {demand_col}",
                )
                apply_theme(fig_air, mode)
                figures.append((f"Airport split: {name}", fig_air))

    return figures


def factor_validation_figures(
    output_dir: pathlib.Path, operator: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart how well the temporal factors reproduce the counts."""
    figures = []
    p = palette(mode)
    for path in sorted(output_dir.glob(f"{_loading_prefix(operator)}_factor_validation.csv")):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty or "total_variation_distance" not in df.columns:
            continue
        operator_name = path.stem.replace("_factor_validation", "")
        df = df.sort_values("total_variation_distance", ascending=True)
        labels = (df["origin"].astype(str) + "→" + df["destination"].astype(str)).tolist()
        tvd = df["total_variation_distance"].to_numpy(dtype=float)

        fig = go.Figure(
            data=go.Bar(
                y=labels,
                x=tvd,
                orientation="h",
                marker=dict(
                    color=tvd,
                    colorscale=[[0.0, p.success], [1.0, p.error]],
                    showscale=False,
                ),
                text=np.round(tvd, 2),
                textposition="outside",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Factor Fit vs Observed Loading: {operator_name}",
                font=dict(size=14, weight="bold"),
            ),
            xaxis_title="Total Variation Distance (0 = Perfect Fit)",
            yaxis_title="Flow (O→D)",
        )
        apply_theme(fig, mode)
        figures.append((f"Factor fit: {operator_name}", fig))
    return figures


# ── services: trains per hour / period / operator ─────────────────────


def services_figures(
    output_dir: pathlib.Path, tag: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart trains per hour, period and operator for each pair."""
    figures = []
    p = palette(mode)
    for path in sorted(output_dir.glob(f"services_{_pair_glob(tag)}")):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty or "dep_hour" not in df.columns:
            continue
        pair = path.stem.replace("services_", "").replace("_", " → ")

        if "weekday_fraction" in df.columns:
            weight = pd.to_numeric(df["weekday_fraction"], errors="coerce").fillna(1.0)
        else:
            weight = pd.Series(1.0, index=df.index)
        averaged = not np.allclose(weight.to_numpy(), 1.0)
        unit = "mean trains/weekday" if averaged else "trains"

        work = df.assign(weight=weight, hour=pd.to_numeric(df["dep_hour"], errors="coerce"))
        hourly = work.dropna(subset=["hour"])
        by_hour = hourly.groupby(hourly["hour"].astype(int))["weight"].sum()
        if not by_hour.empty:
            by_hour = by_hour.reindex(
                range(int(by_hour.index.min()), int(by_hour.index.max()) + 1),
                fill_value=0.0,
            )
        periods = order(work["time_period"].dropna().astype(str).tolist(), PERIOD_ORDER)
        by_period = work.groupby("time_period")["weight"].sum().reindex(periods, fill_value=0.0)

        # subplots for Hourly & Period counts
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"{unit.capitalize()} per Departure Hour",
                f"{unit.capitalize()} per Period",
            ),
        )

        fig.add_trace(
            go.Bar(x=[f"{h:02d}" for h in by_hour.index], y=by_hour.values, marker_color=p.info),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=by_period.index.astype(str),
                y=by_period.to_numpy(),
                marker_color=p.warning,
                text=[_fmt_count(v) for v in by_period.to_numpy()],
                textposition="auto",
            ),
            row=1,
            col=2,
        )

        total = weight.sum()
        headline = f"{len(df)} trains, {total:.1f}/wkday" if averaged else f"{len(df)} total"
        fig.update_layout(
            title=dict(
                text=f"Timetabled Trains {pair} ({headline})", font=dict(size=14, weight="bold")
            ),
            showlegend=False,
        )
        apply_theme(fig, mode)
        figures.append((f"Trains: {pair}", fig))

        if "operator" in df.columns:
            by_op = work.groupby(work["operator"].astype(str))["weight"].sum().sort_values()
            fig2 = go.Figure(
                data=go.Bar(
                    y=by_op.index.astype(str),
                    x=by_op.values,
                    orientation="h",
                    marker_color=p.success,
                    text=[_fmt_count(v) for v in by_op.to_numpy()],
                    textposition="auto",
                )
            )
            fig2.update_layout(
                title=dict(text=f"Trains by Operator: {pair}", font=dict(size=14, weight="bold")),
                xaxis_title=unit,
                yaxis_title="Operator",
            )
            apply_theme(fig2, mode)
            figures.append((f"Operators: {pair}", fig2))

    return figures


def _fmt_count(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def _pair_glob(tag: str | None) -> str:
    return f"{tag}.csv" if tag else "*.csv"


# ── propose: candidate-service feasibility, headway, demand, loading
# ─────────


def _propose_pattern(candidate: str | None, suffix: str) -> str:
    return f"propose_{candidate or '*'}{suffix}.csv"


def running_time_figures(
    output_dir: pathlib.Path, candidate: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart proposed against minimum running time for each leg."""
    figures = []
    p = palette(mode)
    status_colors = {"ok": p.success, "infeasible": p.error, "unverifiable": p.text_disabled}
    for path in sorted(output_dir.glob(_propose_pattern(candidate, "_running_times"))):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty or "status" not in df.columns:
            continue
        name = path.stem.removeprefix("propose_").removesuffix("_running_times")
        labels = (df["from_station"].astype(str) + " → " + df["to_station"].astype(str)).tolist()
        proposed = pd.to_numeric(df["proposed_minutes"], errors="coerce").to_numpy()
        minimum = pd.to_numeric(df["minimum_minutes"], errors="coerce").to_numpy()
        colours = [status_colors.get(s, p.text_disabled) for s in df["status"].astype(str)]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=labels,
                x=proposed,
                name="Proposed",
                orientation="h",
                marker_color=colours,
            )
        )
        fig.add_trace(
            go.Bar(
                y=labels,
                x=minimum,
                name="BPLAN Minimum",
                orientation="h",
                marker_color=p.text_disabled,
                opacity=0.4,
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Running-Time Feasibility: {name}", font=dict(size=14, weight="bold")
            ),
            xaxis_title="Minutes",
            yaxis_title="Segment",
            barmode="group",
        )
        apply_theme(fig, mode)
        figures.append((f"Running times: {name}", fig))
    return figures


def headway_figures(
    output_dir: pathlib.Path, candidate: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart the gap to the nearest existing train on each leg."""
    figures = []
    p = palette(mode)
    for path in sorted(output_dir.glob(_propose_pattern(candidate, "_headway_conflicts"))):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty or "conflict" not in df.columns:
            continue
        name = path.stem.removeprefix("propose_").removesuffix("_headway_conflicts")

        gap_before = pd.to_numeric(df["gap_before_minutes"], errors="coerce")
        gap_after = pd.to_numeric(df["gap_after_minutes"], errors="coerce")
        tightest = pd.concat([gap_before, gap_after], axis=1).min(axis=1, skipna=True)
        effective_min = pd.to_numeric(df["effective_min_gap_minutes"], errors="coerce")
        margin = (tightest - effective_min).to_numpy()

        labels = (
            df["from_station"].astype(str)
            + " → "
            + df["to_station"].astype(str)
            + " ("
            + df["day"].astype(str).str[:3]
            + " "
            + df["candidate_departure"].astype(str)
            + ")"
        ).tolist()
        colours = [p.error if c else p.success for c in df["conflict"]]

        fig = go.Figure(
            data=go.Bar(
                y=labels,
                x=margin,
                orientation="h",
                marker_color=colours,
                text=np.round(margin, 1),
                textposition="outside",
            )
        )

        fig.update_layout(
            title=dict(text=f"Headway Conflicts: {name}", font=dict(size=14, weight="bold")),
            xaxis_title="Headway Margin (min) [Negative = Conflict]",
            yaxis_title="Segment",
        )
        apply_theme(fig, mode)
        figures.append((f"Headway: {name}", fig))
    return figures


def demand_by_leg_figures(
    output_dir: pathlib.Path, candidate: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart predicted annual trips on each leg."""
    figures = []
    p = palette(mode)
    model_colors = {"pivot": p.info, "gravity": p.warning, "insufficient_data": p.text_disabled}
    for path in sorted(output_dir.glob(_propose_pattern(candidate, "_demand_by_leg"))):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        needed = {"predicted_annual_trips_captured", "demand_model_used"}
        if df.empty or not needed.issubset(df.columns):
            continue
        name = path.stem.removeprefix("propose_").removesuffix("_demand_by_leg")

        totals = (
            df.groupby(["from_station", "to_station", "demand_model_used"], sort=False)[
                "predicted_annual_trips_captured"
            ]
            .sum()
            .reset_index()
            .sort_values("predicted_annual_trips_captured", ascending=True)
        )
        if totals.empty:
            continue
        labels = (totals["from_station"] + " → " + totals["to_station"]).tolist()
        colours = [model_colors.get(m, p.text_disabled) for m in totals["demand_model_used"]]

        fig = go.Figure(
            data=go.Bar(
                y=labels,
                x=totals["predicted_annual_trips_captured"],
                orientation="h",
                marker_color=colours,
                text=np.round(totals["predicted_annual_trips_captured"], 0),
                textposition="outside",
            )
        )

        fig.update_layout(
            title=dict(text=f"Predicted Demand by Leg: {name}", font=dict(size=14, weight="bold")),
            xaxis_title="Predicted Annual Trips Captured",
            yaxis_title="Segment",
        )
        apply_theme(fig, mode)
        figures.append((f"Demand: {name}", fig))
    return figures


def loading_profile_figures(
    output_dir: pathlib.Path, candidate: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart load and load over capacity along the train."""
    figures = []
    p = palette(mode)
    for path in sorted(output_dir.glob(_propose_pattern(candidate, "_loading_profile"))):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        needed = {"from_station", "to_station", "on_board_typical_day", "voc"}
        if df.empty or not needed.issubset(df.columns):
            continue
        name = path.stem.removeprefix("propose_").removesuffix("_loading_profile")

        order = list(dict.fromkeys(zip(df["from_station"], df["to_station"], strict=False)))
        labels = [f"{o} → {d}" for o, d in order]
        agg = (
            df.groupby(["from_station", "to_station"], sort=False)
            .agg(
                load=("on_board_typical_day", "mean"),
                voc=("voc", "mean"),
            )
            .reindex(order)
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels,
                y=agg["load"],
                name="Mean On-Board Load",
                marker_color=p.info,
            )
        )

        voc_pct = agg["voc"].to_numpy(dtype=float) * 100
        if np.isfinite(voc_pct).any():
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=voc_pct,
                    name="VoC %",
                    mode="lines+markers",
                    line=dict(color=p.error, width=3),
                    yaxis="y2",
                )
            )

        fig.update_layout(
            title=dict(text=f"Loading Profile: {name}", font=dict(size=14, weight="bold")),
            xaxis_title="Segment",
            yaxis=dict(title="Passengers"),
            yaxis2=dict(
                title="VoC %",
                overlaying="y",
                side="right",
                range=[0, 120],
            ),
        )
        apply_theme(fig, mode)
        figures.append((f"Loading profile: {name}", fig))
    return figures


def search_window_figures(
    output_dir: pathlib.Path, candidate: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart every departure time tried in the window search."""
    figures = []
    p = palette(mode)
    for path in sorted(output_dir.glob(_propose_pattern(candidate, "_search_results"))):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty or "feasible" not in df.columns or df["shift_minutes"].isna().all():
            continue
        name = path.stem.removeprefix("propose_").removesuffix("_search_results")

        labels = df["stop_0_departure"].astype(str).tolist()
        gaps = pd.to_numeric(df["worst_gap_minutes"], errors="coerce").to_numpy()
        colours = [p.success if f else p.error for f in df["feasible"]]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels,
                y=gaps,
                marker_color=colours,
                name="Worst gap to existing train (min)",
            )
        )

        captured = pd.to_numeric(df["predicted_annual_trips_captured"], errors="coerce")
        if captured.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=captured,
                    name="Predicted annual trips captured",
                    mode="lines+markers",
                    line=dict(color=p.info, width=2),
                    yaxis="y2",
                )
            )
            fig.update_layout(
                yaxis2=dict(title="Predicted annual trips", overlaying="y", side="right")
            )

        fig.update_layout(
            title=dict(text=f"Departure Window Search: {name}", font=dict(size=14, weight="bold")),
            xaxis_title="First-stop departure at this trial shift",
            yaxis_title="Worst headway gap (min) [green = feasible]",
        )
        apply_theme(fig, mode)
        figures.append((f"Search: {name}", fig))
    return figures


# ── filesystem helpers ────────────────────────────────────────────────


def available_loading_operators(output_dir: pathlib.Path) -> list[str]:
    """List the operators with loading outputs in the folder."""
    operators: set[str] = set()
    for suffix in ("_temporal_factors", "_loading_analysis"):
        for path in output_dir.glob(f"*{suffix}.csv"):
            operators.add(path.stem[: -len(suffix)])
    return sorted(operators)


def available_services_pairs(output_dir: pathlib.Path) -> list[tuple[str, str]]:
    """List the station pairs with services outputs in the folder."""
    pairs: set[tuple[str, str]] = set()
    for path in output_dir.glob("services_*.csv"):
        origin, sep, dest = path.stem.replace("services_", "").partition("_")
        if sep and origin and dest:
            pairs.add((origin, dest))
    return sorted(pairs)


def available_lookup_pairs(output_dir: pathlib.Path) -> list[tuple[str, str]]:
    """List the station pairs with lookup outputs in the folder."""
    pairs: set[tuple[str, str]] = set()
    for path in output_dir.glob("lookup_*.csv"):
        origin, sep, dest = path.stem.replace("lookup_", "").partition("_")
        if sep and origin and dest:
            pairs.add((origin, dest))
    return sorted(pairs)


def available_fares_pairs(output_dir: pathlib.Path) -> list[tuple[str, str]]:
    """List the station pairs with fares outputs in the folder."""
    pairs: set[tuple[str, str]] = set()
    for path in output_dir.glob("fares_*.csv"):
        origin, sep, dest = path.stem.replace("fares_", "").partition("_")
        if sep and origin and dest:
            pairs.add((origin, dest))
    return sorted(pairs)


def available_propose_candidates(output_dir: pathlib.Path) -> list[str]:
    """Discover candidate-service names with propose outputs on disk."""
    prefix = "propose_"
    names: set[str] = set()
    for suffix in ("_running_times", "_headway_conflicts", "_demand_by_leg", "_loading_profile"):
        tail = f"{suffix}.csv"
        for path in output_dir.glob(f"{prefix}*{suffix}.csv"):
            names.add(path.name[len(prefix) : -len(tail)])
    return sorted(names)


def find_html_maps(output_dir: pathlib.Path, *extra: pathlib.Path | None) -> list[pathlib.Path]:
    """List the HTML maps under ``output_dir``, plus any extras."""
    maps = {p.resolve() for p in output_dir.rglob("*.html") if not p.name.startswith(".")}
    for path in extra:
        if path and path.exists():
            maps.add(path.resolve())
    return sorted(maps)


def propose_fix_summary(output_dir: pathlib.Path, candidate: str | None = None) -> str:
    """List every infeasible leg and headway clash, with its fix."""
    lines: list[str] = []

    for path in sorted(output_dir.glob(_propose_pattern(candidate, "_running_times"))):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty or "status" not in df.columns:
            continue
        name = path.stem.removeprefix("propose_").removesuffix("_running_times")
        has_fix = "suggested_departure" in df.columns
        for _, row in df[df["status"] == "infeasible"].iterrows():
            lines.append(
                f"[{name}] RUNNING-TIME INFEASIBLE: {row['from_station']} → {row['to_station']}"
            )
            shortfall = pd.to_numeric(row.get("shortfall_minutes"), errors="coerce")
            minimum = pd.to_numeric(row.get("minimum_minutes"), errors="coerce")
            lines.append(f"    {-shortfall:.1f} min short of the {minimum:.1f} min BPLAN minimum")
            if has_fix and pd.notna(row.get("suggested_departure")):
                lines.append(
                    f"    FIX: depart by {row['suggested_departure']} (same arrival) or "
                    f"arrive by {row['suggested_arrival']} (same departure)"
                )
            lines.append("")

    for path in sorted(output_dir.glob(_propose_pattern(candidate, "_headway_conflicts"))):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty or "conflict" not in df.columns:
            continue
        name = path.stem.removeprefix("propose_").removesuffix("_headway_conflicts")
        has_fix = "suggested_departure" in df.columns
        for _, row in df[df["conflict"]].iterrows():
            lines.append(
                f"[{name}] HEADWAY CONFLICT: {row['day']} {row['from_station']} → "
                f"{row['to_station']} dep {row['candidate_departure']}"
            )
            lines.append(f"    conflicts with {row['conflicting_train']}")
            if has_fix and pd.notna(row.get("suggested_departure")):
                shift = pd.to_numeric(row.get("shift_minutes"), errors="coerce")
                lines.append(
                    f"    FIX: try {row['suggested_departure']} instead ({shift:+.0f} min)"
                )
            lines.append("")

    return "\n".join(lines).strip()


# ── distance: leg-by-leg / cumulative track distance ──────────────────


def distance_figures(
    output_dir: pathlib.Path, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart distance of each leg and cumulative distance."""
    path = output_dir / "route_legs.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []
    needed = {"origin", "destination", "distance_miles", "cumulative_miles"}
    if df.empty or not needed.issubset(df.columns):
        return []
    p = palette(mode)
    labels = df["origin"] + " → " + df["destination"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=df["distance_miles"],
            name="Leg distance",
            marker_color=p.info,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=df["cumulative_miles"],
            name="Cumulative distance",
            mode="lines+markers",
            line=dict(color=p.error, width=3),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=dict(text="Track distance by leg", font=dict(size=14, weight="bold")),
        xaxis_title="Leg",
        yaxis=dict(title="Leg distance (miles)"),
        yaxis2=dict(title="Cumulative distance (miles)", overlaying="y", side="right"),
    )
    apply_theme(fig, mode)
    return [("Route distance", fig)]


# ── matrix: annual OD matrix top pairs
# ─────────────────────────────────


def matrix_figures(
    output_dir: pathlib.Path, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart the largest flows in the annual matrix."""
    path = output_dir / "odm_matrix.csv"
    if not path.exists():
        return []
    try:
        wide = pd.read_csv(path, index_col=0)
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []
    if wide.empty:
        return []
    p = palette(mode)
    long = wide.reset_index(names="origin").melt(
        id_vars="origin", var_name="destination", value_name="journeys"
    )
    long["journeys"] = pd.to_numeric(long["journeys"], errors="coerce")
    long = long.dropna(subset=["journeys"])
    long = long[long["origin"] != long["destination"]]
    top = long.nlargest(20, "journeys").sort_values("journeys")
    if top.empty:
        return []
    labels = top["origin"] + " → " + top["destination"]

    fig = go.Figure(
        data=go.Bar(
            y=labels,
            x=top["journeys"],
            orientation="h",
            marker_color=p.info,
        )
    )
    fig.update_layout(
        title=dict(text="Top 20 O-D pairs by annual journeys", font=dict(size=14, weight="bold")),
        xaxis_title="Annual journeys",
        yaxis_title="Origin → Destination",
    )
    apply_theme(fig, mode)
    return [("Top OD pairs", fig)]


# ── lookup: expected-trips grid
# ─────────────────────────────────────────


def lookup_figures(
    output_dir: pathlib.Path, tag: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart expected trips by day and period for each pair."""
    figures = []
    for path in sorted(output_dir.glob(f"lookup_{_pair_glob(tag)}")):
        try:
            grid = pd.read_csv(path, index_col=0)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if grid.empty:
            continue
        pair_tag = path.stem.removeprefix("lookup_")
        grid = grid.reindex(
            index=order(grid.index.astype(str).tolist(), DAY_ORDER),
            columns=order(grid.columns.astype(str).tolist(), PERIOD_ORDER),
        )
        fig = go.Figure(
            data=go.Heatmap(
                z=grid.values,
                x=grid.columns,
                y=grid.index,
                colorscale="Blues" if mode == ThemeMode.LIGHT else "Viridis",
                text=np.round(grid.values, 2),
                texttemplate="%{text}",
                hoverinfo="x+y+z",
            )
        )
        fig.update_layout(
            title=dict(text=f"Expected trips ({pair_tag})", font=dict(size=14, weight="bold")),
            xaxis_title="Time Period",
            yaxis_title="Day of Week",
        )
        apply_theme(fig, mode)
        figures.append((f"Trips: {pair_tag}", fig))
    return figures


# ── fares: standard fare tables per station pair
# ───────────────────────


def fares_figures(
    output_dir: pathlib.Path, tag: str | None = None, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart standard fares for each pair, and mean fare by class."""
    figures = []
    p = palette(mode)
    class_colors = {"Standard": p.info, "First": p.warning}
    skip_names = {fares.FLOW_CSV, fares.LOCATIONS_CSV, fares.TICKETS_CSV}
    for path in sorted(output_dir.glob(f"fares_{_pair_glob(tag)}")):
        if path.name in skip_names:
            continue
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty or "fare_gbp" not in df.columns:
            continue
        pair = path.stem.removeprefix("fares_").replace("_", " → ")
        df = df.assign(fare_gbp=pd.to_numeric(df["fare_gbp"], errors="coerce")).dropna(
            subset=["fare_gbp"]
        )
        if df.empty:
            continue

        # Group, tour operator and inclusive tour fares are
        # placeholders, not prices. They are left out of the chart, and
        # the title says so.
        if "non_retail_fare" in df.columns:
            retail = df[~df["non_retail_fare"].astype(bool)]
            excluded = len(df) - len(retail)
        else:
            retail, excluded = df, 0
        if retail.empty:
            continue
        sub = retail.sort_values("fare_gbp", ascending=True).head(40)

        label_col = "ticket_description" if "ticket_description" in sub.columns else "ticket_code"
        labels = sub[label_col].fillna(sub.get("ticket_code", "")).astype(str)
        if "restriction_code" in sub.columns:
            labels = labels + " (" + sub["restriction_code"].fillna("").astype(str) + ")"
        classes = sub["tkt_class"] if "tkt_class" in sub.columns else pd.Series([""] * len(sub))
        colours = [class_colors.get(c, p.info) for c in classes]

        fig = go.Figure(
            data=go.Bar(
                y=labels.iloc[::-1],
                x=sub["fare_gbp"].iloc[::-1],
                orientation="h",
                marker_color=colours[::-1],
                text=sub["fare_gbp"].round(2).iloc[::-1],
                textposition="outside",
            )
        )
        title = f"Standard Fares: {pair}"
        if excluded:
            title += f" (excludes {excluded} non-retail placeholder fare(s))"
        fig.update_layout(
            title=dict(text=title, font=dict(size=14, weight="bold")),
            xaxis_title="Fare (GBP)",
            yaxis_title="Ticket type",
        )
        apply_theme(fig, mode)
        figures.append((f"Fares: {pair}", fig))

        if (
            "tkt_class" in retail.columns
            and retail["tkt_class"].dropna().drop_duplicates().size > 1
        ):
            means = retail.groupby("tkt_class")["fare_gbp"].mean().sort_values()
            fig2 = go.Figure(
                data=go.Bar(
                    x=means.index.astype(str),
                    y=means.values,
                    marker_color=[class_colors.get(c, p.info) for c in means.index],
                    text=np.round(means.values, 2),
                    textposition="auto",
                )
            )
            fig2.update_layout(
                title=dict(text=f"Mean Fare by Class: {pair}", font=dict(size=14, weight="bold")),
                yaxis_title="Mean fare (GBP)",
            )
            apply_theme(fig2, mode)
            figures.append((f"Fares by class: {pair}", fig2))

    return figures


# ── timetable: scheduled route frequency
# ────────────────────────────────


def timetable_figures(
    output_dir: pathlib.Path, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart the busiest routes by daily trains."""
    figures: list[tuple[str, go.Figure]] = []
    p = palette(mode)
    path = output_dir / "route_frequency_summary.csv"
    if not path.exists():
        return figures
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return figures
    if df.empty:
        return figures

    if {"origin_crs", "destination_crs", "daily_trains"}.issubset(df.columns):
        top = df.nlargest(20, "daily_trains").sort_values("daily_trains")
        labels = top["origin_crs"] + " → " + top["destination_crs"]
        fig = go.Figure(
            data=go.Bar(
                y=labels,
                x=top["daily_trains"],
                orientation="h",
                marker_color=p.info,
            )
        )
        fig.update_layout(
            title=dict(
                text="Top 20 routes by scheduled trains/day", font=dict(size=14, weight="bold")
            ),
            xaxis_title="Trains per typical weekday",
            yaxis_title="Origin → Destination",
        )
        apply_theme(fig, mode)
        figures.append(("Top routes by frequency", fig))

    if "peak_tph" in df.columns:
        fig2 = go.Figure(data=go.Histogram(x=df["peak_tph"], marker_color=p.info))
        fig2.update_layout(
            title=dict(
                text="Distribution of peak trains-per-hour across routes",
                font=dict(size=14, weight="bold"),
            ),
            xaxis_title="Peak trains/hour",
            yaxis_title="Number of routes",
        )
        apply_theme(fig2, mode)
        figures.append(("Peak frequency distribution", fig2))
    return figures


# ── bplan: geography extract summary
# ────────────────────────────────────


def bplan_figures(
    output_dir: pathlib.Path, mode: ThemeMode = ThemeMode.LIGHT
) -> list[tuple[str, go.Figure]]:
    """Chart what the BPLAN Geography extract holds."""
    figures = []
    p = palette(mode)

    counts: dict[str, int] = {}
    for label, filename in (
        ("Locations", "bplan_locations.csv"),
        ("Network links", "bplan_network_links.csv"),
        ("Timing links", "bplan_timing_links.csv"),
        ("Traction loads", "bplan_traction_loads.csv"),
    ):
        path = output_dir / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        counts[label] = len(df)

    if counts:
        fig = go.Figure(
            data=go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                marker_color=p.info,
                text=list(counts.values()),
                textposition="auto",
            )
        )
        fig.update_layout(
            title=dict(text="BPLAN extract summary", font=dict(size=14, weight="bold")),
            yaxis_title="Record count",
        )
        apply_theme(fig, mode)
        figures.append(("BPLAN summary", fig))

    links_path = output_dir / "bplan_network_links.csv"
    if links_path.exists():
        try:
            links = pd.read_csv(links_path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", links_path, exc)
            links = pd.DataFrame()
        if not links.empty and "distance_metres" in links.columns:
            distance_m = pd.to_numeric(links["distance_metres"], errors="coerce")
            distances_km = distance_m.dropna() / 1000
            fig2 = go.Figure(data=go.Histogram(x=distances_km, marker_color=p.info))
            fig2.update_layout(
                title=dict(
                    text="Network link distance distribution",
                    font=dict(size=14, weight="bold"),
                ),
                xaxis_title="Link distance (km)",
                yaxis_title="Number of links",
            )
            apply_theme(fig2, mode)
            figures.append(("Link distance distribution", fig2))

    timing_path = output_dir / "bplan_timing_links.csv"
    if timing_path.exists():
        try:
            timing = pd.read_csv(timing_path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Could not read %s: %s", timing_path, exc)
            timing = pd.DataFrame()
        if not timing.empty and "traction_code" in timing.columns:
            top_codes = timing["traction_code"].value_counts().nlargest(20).sort_values()
            fig3 = go.Figure(
                data=go.Bar(
                    y=top_codes.index.astype(str),
                    x=top_codes.to_numpy(),
                    orientation="h",
                    marker_color=p.info,
                )
            )
            fig3.update_layout(
                title=dict(
                    text="Timing links by traction code (top 20)",
                    font=dict(size=14, weight="bold"),
                ),
                xaxis_title="Number of timing links",
                yaxis_title="Traction code",
            )
            apply_theme(fig3, mode)
            figures.append(("Traction code coverage", fig3))
    return figures
