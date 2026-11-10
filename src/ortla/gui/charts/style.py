"""What every chart shares: theme, colour scales, orderings, reading.

The categorical colours are a fixed order checked for colour blindness
(``Palette.chart_colors``). Magnitude uses one blue ramp, light to
dark, and signed quantities such as an error use blue to red through a
grey that means "no difference".
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from ortla.gui.constants import ThemeMode, palette

if TYPE_CHECKING:
    import pathlib

    import plotly.graph_objects as go

logger = logging.getLogger(__name__)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PERIOD_ORDER = ["early", "am_peak", "interpeak", "pm_peak", "evening"] + [
    f"{h:02d}:00" for h in range(24)
]

# One hue, near-zero to most; the lightest step stays visible on the
# chart surface.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def diverging(mode: ThemeMode) -> list[list[float | str]]:
    """Blue (under) to red (over) through a neutral grey at zero."""
    mid = "#f0efec" if mode is ThemeMode.LIGHT else "#383835"
    return [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, mid], [0.75, "#ec835a"], [1.0, "#b02a2a"]]


def colour(mode: ThemeMode, index: int) -> str:
    """Return categorical colour ``index``; past eight, the last."""
    colours = palette(mode).chart_colors
    return colours[min(index, len(colours) - 1)]


def order(values: list[str], preferred: list[str]) -> list[str]:
    """Put ``values`` in ``preferred``'s order, unknown ones last."""
    present = list(dict.fromkeys(values))
    ordered = [v for v in preferred if v in present]
    return ordered + [v for v in present if v not in preferred]


def read(path: pathlib.Path, **kwargs: object) -> pd.DataFrame:
    """Read a CSV, or return an empty frame if it cannot be read."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


def pair_label(
    frame: pd.DataFrame, origin: str = "origin", dest: str = "destination"
) -> pd.Series:
    """``"ABC → XYZ"`` for each row."""
    return frame[origin].astype(str) + " → " + frame[dest].astype(str)


def apply_theme(fig: go.Figure, mode: ThemeMode) -> go.Figure:
    """Style a figure for the active theme: recessive grid and axes."""
    p = palette(mode)
    fig.update_layout(
        paper_bgcolor=p.surface,
        plot_bgcolor=p.surface,
        font={
            "family": "system-ui, -apple-system, Segoe UI, sans-serif",
            "color": p.text_primary,
            "size": 12,
        },
        title={"font": {"size": 15}, "x": 0.01, "xanchor": "left"},
        margin={"t": 56, "b": 44, "l": 56, "r": 28},
        hoverlabel={"bgcolor": p.surface_elevated, "font": {"color": p.text_primary}},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
        bargap=0.25,
        colorway=list(p.chart_colors),
    )
    axis = {"gridcolor": p.border_subtle, "linecolor": p.border, "zeroline": False}
    fig.update_xaxes(showgrid=False, **axis)
    fig.update_yaxes(showgrid=True, **axis)
    return fig
