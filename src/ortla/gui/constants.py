"""Design tokens, colour palettes, and application constants.

All visual parameters (colours, spacing, typography, timing) are defined
here so every widget draws from a single source of truth.  The
:func:`palette` helper returns the active colours for the current theme
mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# Theme mode
class ThemeMode(Enum):
    """Light / dark appearance modes."""

    LIGHT = "light"
    DARK = "dark"


# Colour palette
@dataclass(frozen=True, slots=True)
class Palette:
    """Resolved colour values for a single theme mode."""

    # Brand / accent
    primary: str
    primary_hover: str
    primary_pressed: str
    accent: str
    accent_hover: str

    # Semantic
    success: str
    warning: str
    error: str
    info: str

    # Surfaces
    background: str
    surface: str
    surface_variant: str
    surface_container: str
    surface_elevated: str

    # Borders
    border: str
    border_subtle: str

    # Text
    text_primary: str
    text_secondary: str
    text_disabled: str
    text_on_primary: str
    text_on_accent: str

    # Activity bar
    activity_bar_bg: str
    activity_bar_fg: str
    activity_bar_active: str
    activity_bar_badge: str

    # Status bar
    status_bar_bg: str
    status_bar_fg: str

    # Sidebar
    sidebar_bg: str
    sidebar_fg: str
    sidebar_hover: str
    sidebar_active: str
    sidebar_border: str

    # Input fields
    input_bg: str
    input_border: str
    input_focus_border: str
    input_placeholder: str

    # Scrollbar
    scrollbar_bg: str
    scrollbar_thumb: str
    scrollbar_thumb_hover: str

    # Chart colours (categorical, in a fixed order checked for colour
    # blindness; give series these in order, never cycled)
    chart_colors: tuple[str, ...] = field(default_factory=tuple)


_LIGHT = Palette(
    # Brand
    primary="#0F5C5A",
    primary_hover="#137370",
    primary_pressed="#0B4645",
    accent="#1A9C94",
    accent_hover="#21B3AA",
    # Semantic
    success="#0CA30C",
    warning="#FAB219",
    error="#D03B3B",
    info="#2A78D6",
    # Surfaces
    background="#F4F4F1",
    surface="#FCFCFB",
    surface_variant="#F0EFEC",
    surface_container="#E8E7E2",
    surface_elevated="#FFFFFF",
    # Borders
    border="#C3C2B7",
    border_subtle="#E1E0D9",
    # Text
    text_primary="#0B0B0B",
    text_secondary="#52514E",
    text_disabled="#A3A29C",
    text_on_primary="#FFFFFF",
    text_on_accent="#FFFFFF",
    # Activity bar
    activity_bar_bg="#0E2A2C",
    activity_bar_fg="#8FB3B0",
    activity_bar_active="#FFFFFF",
    activity_bar_badge="#D03B3B",
    # Status bar
    status_bar_bg="#0F5C5A",
    status_bar_fg="#FFFFFF",
    # Sidebar
    sidebar_bg="#F0EFEC",
    sidebar_fg="#0B0B0B",
    sidebar_hover="#E1E0D9",
    sidebar_active="#D5E6E4",
    sidebar_border="#C3C2B7",
    # Inputs
    input_bg="#FFFFFF",
    input_border="#C3C2B7",
    input_focus_border="#1A9C94",
    input_placeholder="#A3A29C",
    # Scrollbar
    scrollbar_bg="transparent",
    scrollbar_thumb="#C3C2B7",
    scrollbar_thumb_hover="#898781",
    # Charts
    chart_colors=(
        "#2A78D6",
        "#EB6834",
        "#1BAF7A",
        "#EDA100",
        "#E87BA4",
        "#008300",
        "#6250D6",
        "#E34948",
    ),
)

_DARK = Palette(
    # Brand
    primary="#2BA59C",
    primary_hover="#36BDB3",
    primary_pressed="#228A83",
    accent="#3CC2B8",
    accent_hover="#5AD3CA",
    # Semantic
    success="#0CA30C",
    warning="#FAB219",
    error="#E66767",
    info="#3987E5",
    # Surfaces
    background="#0F1111",
    surface="#1A1A19",
    surface_variant="#222322",
    surface_container="#2A2B2A",
    surface_elevated="#2C2C2A",
    # Borders
    border="#383835",
    border_subtle="#2C2C2A",
    # Text
    text_primary="#F0EFEC",
    text_secondary="#C3C2B7",
    text_disabled="#6B6A65",
    text_on_primary="#0B0B0B",
    text_on_accent="#0B0B0B",
    # Activity bar
    activity_bar_bg="#0A1415",
    activity_bar_fg="#6F8F8C",
    activity_bar_active="#F0EFEC",
    activity_bar_badge="#E66767",
    # Status bar
    status_bar_bg="#0A1415",
    status_bar_fg="#C3C2B7",
    # Sidebar
    sidebar_bg="#161716",
    sidebar_fg="#F0EFEC",
    sidebar_hover="#242524",
    sidebar_active="#1E3836",
    sidebar_border="#383835",
    # Inputs
    input_bg="#222322",
    input_border="#383835",
    input_focus_border="#3CC2B8",
    input_placeholder="#6B6A65",
    # Scrollbar
    scrollbar_bg="transparent",
    scrollbar_thumb="#383835",
    scrollbar_thumb_hover="#6B6A65",
    # Charts
    chart_colors=(
        "#3987E5",
        "#D95926",
        "#199E70",
        "#C98500",
        "#D55181",
        "#008300",
        "#9085E9",
        "#E66767",
    ),
)


def palette(mode: ThemeMode) -> Palette:
    """Return the colour palette for the requested theme mode."""
    return _LIGHT if mode is ThemeMode.LIGHT else _DARK


# Spacing
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 16
SPACE_LG = 24
SPACE_XL = 32
SPACE_2XL = 48

# Border radius
RADIUS_SM = 4
RADIUS_MD = 8
RADIUS_LG = 12
RADIUS_XL = 16

# Layout dimensions
ACTIVITY_BAR_WIDTH = 48
SIDEBAR_WIDTH = 280
SIDEBAR_MIN_WIDTH = 220
SIDEBAR_MAX_WIDTH = 420
STATUS_BAR_HEIGHT = 28
WINDOW_MIN_WIDTH = 1100
WINDOW_MIN_HEIGHT = 700
WINDOW_DEFAULT_WIDTH = 1440
WINDOW_DEFAULT_HEIGHT = 900

# Typography
FONT_FAMILY = "Inter"
FONT_FAMILY_MONO = "JetBrains Mono"
FONT_FAMILY_FALLBACK = "system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO_FALLBACK = "Menlo, Consolas, 'Courier New', monospace"

FONT_SIZE_H1 = 24
FONT_SIZE_H2 = 20
FONT_SIZE_H3 = 16
FONT_SIZE_BODY = 14
FONT_SIZE_SMALL = 12
FONT_SIZE_TINY = 11
FONT_SIZE_MONO = 13

# Animation timing (milliseconds)
ANIM_FAST = 150
ANIM_NORMAL = 250
ANIM_SLOW = 350

# Application metadata
APP_NAME = "ORTLA"
APP_FULL_NAME = "Open Rail Train-level Assignment"
APP_ORG = "ORTLA"

# Commands: name -> (label, icon). The description and the config model
# come from ortla.config.COMMANDS, which the command line uses
# too, so the two never disagree.
_COMMAND_LOOK: dict[str, tuple[str, str]] = {
    "loading": ("Loading analysis", "fa5s.chart-bar"),
    "matrix": ("OD matrix", "fa5s.th"),
    "lookup": ("Trip lookup", "fa5s.search-location"),
    "timetable": ("Timetable", "fa5s.calendar-alt"),
    "services": ("Service lookup", "fa5s.train"),
    "distance": ("Track distance", "fa5s.ruler"),
    "bplan": ("BPLAN geography", "fa5s.project-diagram"),
    "propose": ("Service proposal", "fa5s.lightbulb"),
    "scenario": ("Scenario studio", "fa5s.sliders-h"),
    "discover": ("Service discovery", "fa5s.compass"),
    "commercial": ("Commercial case", "fa5s.pound-sign"),
    "new-station": ("New station", "fa5s.map-marker-alt"),
    "hindcast": ("Hindcast", "fa5s.history"),
    "fares": ("Fares lookup", "fa5s.ticket-alt"),
}


def _commands() -> dict[str, tuple[str, str, str]]:
    from ortla.config import COMMANDS as REGISTRY  # noqa: PLC0415

    out = {}
    for key, (label, icon) in _COMMAND_LOOK.items():
        if key in REGISTRY:
            help_text = REGISTRY[key][1]
            out[key] = (label, icon, help_text[:1].upper() + help_text[1:])
    for key, (_model, help_text) in REGISTRY.items():
        out.setdefault(key, (key.title(), "fa5s.cog", help_text[:1].upper() + help_text[1:]))
    return out


# name -> (label, icon, description), in the order the app lists them.
COMMANDS: dict[str, tuple[str, str, str]] = _commands()
