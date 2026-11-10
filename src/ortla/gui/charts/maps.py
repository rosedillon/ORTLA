"""Maps of stations and flows, drawn as Plotly map figures.

Map figures sit in the chart list beside the other charts. Station
positions come from any station file a project's configs name (the
``distance`` section's station list, ``station_covariates`` and so on),
or from latitude and longitude columns in the output itself.
"""

from __future__ import annotations

import functools
import itertools
import logging
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ortla.gui.charts.style import SEQUENTIAL, apply_theme, diverging
from ortla.gui.constants import ThemeMode, palette

logger = logging.getLogger(__name__)

_CODE = ("crsCode", "crs", "crs_code", "CRS", "code")
_LAT = ("lat", "latitude", "Latitude")
_LON = ("long", "lon", "longitude", "Longitude")


def _pick(columns: list[str], names: tuple[str, ...]) -> str | None:
    return next((c for c in names if c in columns), None)


@functools.cache
def _read_stations(path: pathlib.Path, mtime: float) -> pd.DataFrame | None:  # noqa: ARG001
    # ``mtime`` is part of the cache key, so an edited file is reread.
    try:
        head = pd.read_csv(path, nrows=0).columns.tolist()
    except (OSError, ValueError):
        return None
    code, lat, lon = _pick(head, _CODE), _pick(head, _LAT), _pick(head, _LON)
    if not (code and lat and lon):
        return None
    df = pd.read_csv(path, usecols=[code, lat, lon])[[code, lat, lon]].dropna()
    df.columns = ["crs", "lat", "lon"]
    df["crs"] = df["crs"].astype(str).str.upper()
    return df.drop_duplicates("crs").set_index("crs")


def _paths(value: object) -> list[pathlib.Path]:
    """Every CSV path named anywhere in a config."""
    if isinstance(value, dict):
        return [p for v in value.values() for p in _paths(v)]
    if isinstance(value, list):
        return [p for v in value for p in _paths(v)]
    if isinstance(value, str | pathlib.Path) and str(value).lower().endswith(".csv"):
        return [pathlib.Path(value)]
    return []


def find_stations(*configs: dict) -> pd.DataFrame | None:
    """Station positions from the first station file the configs name.

    Returns a frame indexed by CRS code with ``lat`` and ``lon``, or
    ``None`` when no config names a file with codes and coordinates.
    """
    for path in dict.fromkeys(p for c in configs for p in _paths(c)):
        if path.is_file():
            found = _read_stations(path, path.stat().st_mtime)
            if found is not None:
                return found
    return None


def with_positions(
    frame: pd.DataFrame, stations: pd.DataFrame | None, code: str = "crs"
) -> pd.DataFrame:
    """Add ``lat`` and ``lon``, from the frame's own columns if any."""
    lat, lon = _pick(list(frame.columns), _LAT), _pick(list(frame.columns), _LON)
    if lat and lon:
        return frame.rename(columns={lat: "lat", lon: "lon"}).dropna(subset=["lat", "lon"])
    if stations is None or code not in frame.columns:
        return frame.iloc[0:0].assign(lat=[], lon=[])
    keys = frame[code].astype(str).str.upper()
    return frame.assign(lat=keys.map(stations["lat"]), lon=keys.map(stations["lon"])).dropna(
        subset=["lat", "lon"]
    )


def _view(lat: pd.Series, lon: pd.Series) -> dict:
    """Centre and zoom that fit the points."""
    span = max(float(lat.max() - lat.min()), float(lon.max() - lon.min()) * 0.6, 0.02)
    zoom = float(np.clip(np.log2(360 / span) - 0.8, 4.5, 13))
    return {"center": {"lat": float(lat.mean()), "lon": float(lon.mean())}, "zoom": zoom}


def finish(fig: go.Figure, mode: ThemeMode, lat: pd.Series, lon: pd.Series) -> go.Figure:
    """Theme a map figure and fit its view to the points."""
    apply_theme(fig, mode)
    style = "carto-positron" if mode is ThemeMode.LIGHT else "carto-darkmatter"
    fig.update_layout(
        map={"style": style, **_view(lat, lon)},
        margin={"t": 48, "b": 0, "l": 0, "r": 0},
        height=640,
        legend={"x": 0.01, "y": 0.99, "bgcolor": palette(mode).surface, "borderwidth": 0},
    )
    return fig


def station_map(
    frame: pd.DataFrame,
    value: str,
    title: str,
    mode: ThemeMode,
    *,
    label: str = "crs",
    signed: bool = False,
    hover: list[str] | None = None,
    size: bool = True,
) -> go.Figure | None:
    """Stations as circles coloured, and sized, by ``value``.

    ``signed`` colours a quantity either side of zero (an error, say)
    blue to red, symmetric about zero; otherwise one hue light to dark.
    """
    df = frame.dropna(subset=["lat", "lon", value])
    if df.empty:
        return None
    v = df[value].astype(float)
    if signed:
        top = float(np.nanpercentile(v.abs(), 95)) or 1.0
        colours = {"colorscale": diverging(mode), "cmin": -top, "cmax": top}
        sizes = 7 + 14 * np.clip(v.abs() / top, 0, 1)
    else:
        # A few large values (London's termini) would wash out the rest.
        top = float(np.nanpercentile(v, 98)) or float(v.max()) or 1.0
        colours = {"colorscale": SEQUENTIAL, "cmin": float(v.min()), "cmax": top}
        sizes = 6 + 18 * np.sqrt(np.clip(v / top, 0, 1))
    extra = hover or []
    text = df[label].astype(str)
    for col in extra:
        text = text + f"<br>{col.replace('_', ' ')}: " + df[col].map(_fmt)
    fig = go.Figure(
        go.Scattermap(
            lat=df["lat"],
            lon=df["lon"],
            mode="markers",
            marker={
                "size": sizes if size else 9,
                "color": v,
                "colorbar": {"title": value.replace("_", " "), "thickness": 12},
                "opacity": 0.85,
                **colours,
            },
            text=text,
            customdata=v,
            hovertemplate="%{text}<br>"
            + value.replace("_", " ")
            + ": %{customdata:,.3~g}<extra></extra>",
            name=value.replace("_", " "),
        )
    )
    fig.update_layout(title=title)
    return finish(fig, mode, df["lat"], df["lon"])


def flow_map(
    pairs: pd.DataFrame,
    stations: pd.DataFrame | None,
    value: str,
    title: str,
    mode: ThemeMode,
    *,
    origin: str = "origin",
    destination: str = "destination",
    top: int = 300,
    bands: int = 5,
) -> go.Figure | None:
    """Draw straight lines between pairs, darker and thicker for more.

    Only the ``top`` pairs by ``value`` are drawn. Lines share one trace
    per band of ``value``, so even hundreds of pairs stay quick.
    """
    if stations is None or pairs.empty:
        return None
    df = pairs.dropna(subset=[value]).nlargest(top, value).copy()
    for end, col in (("o", origin), ("d", destination)):
        keys = df[col].astype(str).str.upper()
        df[f"{end}_lat"] = keys.map(stations["lat"])
        df[f"{end}_lon"] = keys.map(stations["lon"])
    df = df.dropna(subset=["o_lat", "o_lon", "d_lat", "d_lon"])
    if df.empty:
        return None
    edges = np.unique(np.quantile(df[value], np.linspace(0, 1, bands + 1)))
    if len(edges) < 2:
        edges = np.array([df[value].min(), df[value].max() + 1])
    df["band"] = np.clip(np.searchsorted(edges, df[value], side="right") - 1, 0, len(edges) - 2)
    steps = SEQUENTIAL[-(len(edges) - 1) :]
    fig = go.Figure()
    for band, group in df.groupby("band"):
        lat = np.column_stack(
            [group["o_lat"], group["d_lat"], np.full(len(group), np.nan)]
        ).ravel()
        lon = np.column_stack(
            [group["o_lon"], group["d_lon"], np.full(len(group), np.nan)]
        ).ravel()
        low, high = edges[int(band)], edges[int(band) + 1]
        fig.add_trace(
            go.Scattermap(
                lat=lat,
                lon=lon,
                mode="lines",
                line={"width": 1.5 + 1.2 * int(band), "color": steps[int(band)]},
                name=f"{_fmt(low)} to {_fmt(high)}",
                hoverinfo="skip",
            )
        )
    # Hover sits on each line's midpoint, where it can be found.
    fig.add_trace(
        go.Scattermap(
            lat=(df["o_lat"] + df["d_lat"]) / 2,
            lon=(df["o_lon"] + df["d_lon"]) / 2,
            mode="markers",
            marker={"size": 8, "color": palette(mode).text_secondary, "opacity": 0.05},
            text=df[origin].astype(str) + " → " + df[destination].astype(str),
            customdata=df[value],
            hovertemplate="%{text}<br>"
            + value.replace("_", " ")
            + ": %{customdata:,.3~g}<extra></extra>",
            name="pairs (hover)",
            showlegend=False,
        )
    )
    fig.update_layout(title=title, legend={"title": {"text": value.replace("_", " ")}})
    return finish(
        fig, mode, pd.concat([df["o_lat"], df["d_lat"]]), pd.concat([df["o_lon"], df["d_lon"]])
    )


def route_map(
    codes: list[str],
    stations: pd.DataFrame | None,
    title: str,
    mode: ThemeMode,
    values: list[float] | None = None,
    value_name: str = "",
) -> go.Figure | None:
    """Join a service's calling points in order.

    With ``values`` (one per leg) each leg is labelled with its value.
    """
    if stations is None:
        return None
    known = [c for c in codes if c in stations.index]
    if len(known) < 2:
        return None
    pos = stations.loc[known]
    p = palette(mode)
    fig = go.Figure(
        go.Scattermap(
            lat=pos["lat"],
            lon=pos["lon"],
            mode="lines+markers+text",
            line={"width": 4, "color": p.chart_colors[0]},
            marker={"size": 10, "color": p.chart_colors[0]},
            text=known,
            textposition="top right",
            textfont={"color": p.text_primary},
            name="calling points",
            hovertemplate="%{text}<extra></extra>",
        )
    )
    if values is not None and len(values) == len(codes) - 1 and len(known) == len(codes):
        fig.add_trace(
            go.Scattermap(
                lat=(pos["lat"].to_numpy()[:-1] + pos["lat"].to_numpy()[1:]) / 2,
                lon=(pos["lon"].to_numpy()[:-1] + pos["lon"].to_numpy()[1:]) / 2,
                mode="markers",
                marker={
                    "size": 16,
                    "color": values,
                    "colorscale": SEQUENTIAL,
                    "showscale": True,
                    "colorbar": {"title": value_name, "thickness": 12},
                },
                customdata=values,
                text=[f"{a} → {b}" for a, b in itertools.pairwise(codes)],
                hovertemplate="%{text}<br>" + value_name + ": %{customdata:,.0f}<extra></extra>",
                name=value_name,
            )
        )
    fig.update_layout(title=title, showlegend=False)
    return finish(fig, mode, pos["lat"], pos["lon"])


def _fmt(value: object) -> str:
    if isinstance(value, float | int | np.floating | np.integer):
        v = float(value)
        if abs(v) >= 1e6:  # noqa: PLR2004
            return f"{v / 1e6:.2f}m"
        if abs(v) >= 1e4:  # noqa: PLR2004
            return f"{v / 1e3:.0f}k"
        if abs(v) >= 1e3:  # noqa: PLR2004
            return f"{v:,.0f}"
        return f"{v:,.3g}"
    return str(value)
