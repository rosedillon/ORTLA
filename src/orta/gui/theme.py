"""QSS theme engine that generates and applies light/dark stylesheets.

The stylesheet is built programmatically from :mod:`constants` tokens so
that every colour, radius, and spacing value traces back to one place.
Switching themes at runtime is a single call to
:meth:`ThemeEngine.apply`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import darkdetect

from orta.gui.constants import (
    FONT_FAMILY,
    FONT_FAMILY_MONO,
    FONT_SIZE_BODY,
    FONT_SIZE_MONO,
    FONT_SIZE_SMALL,
    FONT_SIZE_TINY,
    RADIUS_LG,
    RADIUS_MD,
    RADIUS_SM,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    STATUS_BAR_HEIGHT,
    ThemeMode,
    palette,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication


class ThemeEngine:
    """Manage and apply application-wide QSS themes."""

    def __init__(self) -> None:
        self._mode = ThemeMode.LIGHT

    @property
    def mode(self) -> ThemeMode:
        """Currently active theme mode."""
        return self._mode

    def detect_system(self) -> ThemeMode:
        """Return the system's light or dark mode (light if unknown)."""
        try:
            return ThemeMode.DARK if darkdetect.isDark() else ThemeMode.LIGHT
        except Exception:  # noqa: BLE001
            return ThemeMode.LIGHT

    def toggle(self) -> ThemeMode:
        """Switch to the other theme and return the new mode."""
        self._mode = ThemeMode.DARK if self._mode is ThemeMode.LIGHT else ThemeMode.LIGHT
        return self._mode

    def apply(self, app: QApplication, mode: ThemeMode | None = None) -> None:
        """Generate and apply the QSS stylesheet for *mode* to *app*."""
        if mode is not None:
            self._mode = mode
        app.setStyleSheet(self._build_stylesheet())

    def _build_stylesheet(self) -> str:
        """Return the complete QSS string for the current palette."""
        p = palette(self._mode)

        return f"""
/* ================================================================
   ORTA: Generated QSS ({self._mode.value} theme)
   ================================================================ */

/* --- Global reset ------------------------------------------------ */
* {{
    font-family: "{FONT_FAMILY}", system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: {FONT_SIZE_BODY}px;
    outline: none;
}}

/* --- QMainWindow ------------------------------------------------- */
QMainWindow {{
    background-color: {p.background};
}}

QMainWindow::separator {{
    background-color: {p.border};
    width: 1px;
    height: 1px;
}}

/* --- Central / generic frames ------------------------------------ */
QWidget {{
    background-color: transparent;
    color: {p.text_primary};
}}

QFrame[frameShape="4"],  /* HLine */
QFrame[frameShape="5"]   /* VLine */
{{
    color: {p.border_subtle};
    max-height: 1px;
}}

/* --- Scroll areas ------------------------------------------------ */
QScrollArea, QScrollArea::viewport {{
    border: none;
    background-color: transparent;
    background: transparent;
}}

QScrollBar:vertical {{
    background: {p.scrollbar_bg};
    width: 10px;
    margin: 0;
    border: none;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {p.scrollbar_thumb};
    min-height: 30px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: {p.scrollbar_thumb_hover};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    background: {p.scrollbar_bg};
    height: 10px;
    margin: 0;
    border: none;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background: {p.scrollbar_thumb};
    min-width: 30px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {p.scrollbar_thumb_hover};
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* --- Labels ------------------------------------------------------ */
QLabel {{
    background-color: transparent;
    color: {p.text_primary};
    padding: 0;
}}

QLabel[class="display"] {{
    font-size: 40px;
    font-weight: 800;
    color: {p.primary};
    letter-spacing: 2px;
}}

QLabel[class="tagline"] {{
    font-size: 16px;
    color: {p.text_secondary};
}}

QLabel[class="heading"] {{
    font-size: 20px;
    font-weight: 600;
    color: {p.text_primary};
}}

QLabel[class="subheading"] {{
    font-size: 16px;
    font-weight: 600;
    color: {p.text_primary};
}}

QLabel[class="section"] {{
    font-size: 13px;
    font-weight: 600;
    color: {p.text_secondary};
    text-transform: uppercase;
    letter-spacing: 1px;
}}

QLabel[class="caption"] {{
    font-size: {FONT_SIZE_SMALL}px;
    color: {p.text_secondary};
}}

QLabel[class="error"] {{
    color: {p.error};
    font-size: {FONT_SIZE_SMALL}px;
}}

QLabel[class="success"] {{
    color: {p.success};
    font-size: {FONT_SIZE_SMALL}px;
}}

/* --- Push buttons ------------------------------------------------ */
QPushButton {{
    background-color: {p.surface_container};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {RADIUS_MD}px;
    padding: 4px 12px;
    font-weight: 500;
    min-height: 28px;
}}
QPushButton:hover {{
    background-color: {p.sidebar_hover};
    border-color: {p.border};
}}
QPushButton:pressed {{
    background-color: {p.sidebar_active};
}}
QPushButton:disabled {{
    color: {p.text_disabled};
    border-color: {p.border_subtle};
}}

QPushButton[class="primary"] {{
    background-color: {p.primary};
    color: {p.text_on_primary};
    border: none;
    font-weight: 600;
}}
QPushButton[class="primary"]:hover {{
    background-color: {p.primary_hover};
}}
QPushButton[class="primary"]:pressed {{
    background-color: {p.primary_pressed};
}}
QPushButton[class="primary"]:disabled {{
    background-color: {p.surface_container};
    color: {p.text_disabled};
}}

QPushButton[class="accent"] {{
    background-color: {p.accent};
    color: {p.text_on_accent};
    border: none;
    font-weight: 600;
}}
QPushButton[class="accent"]:hover {{
    background-color: {p.accent_hover};
}}

QPushButton[class="danger"] {{
    background-color: {p.error};
    color: #FFFFFF;
    border: none;
    font-weight: 600;
}}

QPushButton[class="ghost"] {{
    background-color: transparent;
    border: none;
    color: {p.text_secondary};
    padding: 4px 8px;
    min-height: 24px;
}}
QPushButton[class="ghost"]:hover {{
    color: {p.text_primary};
    background-color: {p.sidebar_hover};
}}

QPushButton[class="link"] {{
    background-color: transparent;
    border: none;
    color: {p.accent};
    text-decoration: underline;
    padding: 0;
    min-height: 0;
}}
QPushButton[class="link"]:hover {{
    color: {p.accent_hover};
}}

/* --- Line edits / inputs ----------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {p.input_bg};
    color: {p.text_primary};
    border: 1px solid {p.input_border};
    border-radius: {RADIUS_SM}px;
    padding: 4px 8px;
    min-height: 28px;
    selection-background-color: {p.accent};
    selection-color: {p.text_on_accent};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {p.input_focus_border};
    border-width: 2px;
    padding: 3px 7px;
}}
QLineEdit:disabled {{
    background-color: {p.surface_variant};
    color: {p.text_disabled};
}}
QLineEdit[readOnly="true"] {{
    background-color: {p.surface_variant};
}}

/* --- Text edits -------------------------------------------------- */
QTextEdit, QPlainTextEdit {{
    background-color: {p.input_bg};
    color: {p.text_primary};
    border: 1px solid {p.input_border};
    border-radius: {RADIUS_SM}px;
    padding: 8px;
    font-family: "{FONT_FAMILY_MONO}", Menlo, Consolas, monospace;
    font-size: {FONT_SIZE_MONO}px;
    selection-background-color: {p.accent};
    selection-color: {p.text_on_accent};
}}
QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {p.input_focus_border};
    border-width: 2px;
    padding: 7px;
}}

/* --- Combo boxes ------------------------------------------------- */
QComboBox {{
    background-color: {p.input_bg};
    color: {p.text_primary};
    border: 1px solid {p.input_border};
    border-radius: {RADIUS_SM}px;
    padding: 4px 28px 4px 8px;
    min-height: 28px;
}}
QComboBox:focus {{
    border-color: {p.input_focus_border};
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {p.text_secondary};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {p.surface_elevated};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px;
    selection-background-color: {p.sidebar_active};
    selection-color: {p.text_primary};
    outline: none;
}}

/* --- Check boxes ------------------------------------------------- */
QCheckBox {{
    spacing: {SPACE_SM}px;
    color: {p.text_primary};
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {p.input_border};
    border-radius: {RADIUS_SM}px;
    background-color: {p.input_bg};
}}
QCheckBox::indicator:checked {{
    background-color: {p.accent};
    border-color: {p.accent};
}}
QCheckBox::indicator:hover {{
    border-color: {p.accent};
}}

/* --- Group boxes ------------------------------------------------- */
QGroupBox {{
    background-color: {p.surface};
    border: 1px solid {p.border_subtle};
    border-radius: {RADIUS_MD}px;
    margin-top: 20px;
    padding: {SPACE_MD}px;
    padding-top: {SPACE_LG}px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 {SPACE_SM}px;
    color: {p.text_primary};
    background-color: {p.surface};
    border-radius: {RADIUS_SM}px;
}}

/* --- Tab widget -------------------------------------------------- */
QTabWidget::pane {{
    background-color: {p.surface};
    border: 1px solid {p.border_subtle};
    border-top: none;
    border-radius: 0 0 {RADIUS_MD}px {RADIUS_MD}px;
}}

QTabBar {{
    background-color: transparent;
}}
QTabBar::tab {{
    background-color: transparent;
    color: {p.text_secondary};
    padding: {SPACE_SM + 2}px {SPACE_MD}px;
    border-bottom: 2px solid transparent;
    font-weight: 500;
    min-width: 80px;
}}
QTabBar::tab:hover {{
    color: {p.text_primary};
    background-color: {p.sidebar_hover};
}}
QTabBar::tab:selected {{
    color: {p.accent};
    border-bottom-color: {p.accent};
}}

/* --- Table views ------------------------------------------------- */
QTableView, QTreeView, QListView {{
    background-color: {p.surface};
    color: {p.text_primary};
    border: 1px solid {p.border_subtle};
    border-radius: {RADIUS_SM}px;
    gridline-color: {p.border_subtle};
    selection-background-color: {p.sidebar_active};
    selection-color: {p.text_primary};
    alternate-background-color: {p.surface_variant};
}}

QHeaderView::section {{
    background-color: {p.surface_variant};
    color: {p.text_secondary};
    border: none;
    border-bottom: 1px solid {p.border};
    border-right: 1px solid {p.border_subtle};
    padding: {SPACE_SM}px {SPACE_SM + 2}px;
    font-weight: 600;
    font-size: {FONT_SIZE_SMALL}px;
}}

/* --- Splitter ---------------------------------------------------- */
QSplitter::handle {{
    background-color: {p.border_subtle};
}}
QSplitter::handle:horizontal {{
    width: 1px;
}}
QSplitter::handle:vertical {{
    height: 1px;
}}
QSplitter::handle:hover {{
    background-color: {p.accent};
}}

/* --- Progress bars ----------------------------------------------- */
QProgressBar {{
    background-color: {p.surface_container};
    border: none;
    border-radius: {RADIUS_SM}px;
    text-align: center;
    color: {p.text_secondary};
    font-size: {FONT_SIZE_TINY}px;
    min-height: 8px;
    max-height: 8px;
}}
QProgressBar::chunk {{
    background-color: {p.accent};
    border-radius: {RADIUS_SM}px;
}}

/* --- Tooltips ---------------------------------------------------- */
QToolTip {{
    background-color: {p.surface_elevated};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_SM}px {SPACE_SM + 2}px;
    font-size: {FONT_SIZE_SMALL}px;
}}

/* --- Menu bar / menus -------------------------------------------- */
QMenuBar {{
    background-color: {p.surface};
    color: {p.text_primary};
    border-bottom: 1px solid {p.border_subtle};
    padding: 2px;
}}
QMenuBar::item:selected {{
    background-color: {p.sidebar_hover};
    border-radius: {RADIUS_SM}px;
}}

QMenu {{
    background-color: {p.surface_elevated};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {RADIUS_MD}px;
    padding: {SPACE_XS}px;
}}
QMenu::item {{
    padding: {SPACE_SM}px {SPACE_MD}px;
    border-radius: {RADIUS_SM}px;
}}
QMenu::item:selected {{
    background-color: {p.sidebar_hover};
}}
QMenu::separator {{
    height: 1px;
    background-color: {p.border_subtle};
    margin: {SPACE_XS}px {SPACE_SM}px;
}}

/* --- Status bar -------------------------------------------------- */
QStatusBar {{
    background-color: {p.status_bar_bg};
    color: {p.status_bar_fg};
    font-size: {FONT_SIZE_SMALL}px;
    min-height: {STATUS_BAR_HEIGHT}px;
    max-height: {STATUS_BAR_HEIGHT}px;
    border: none;
}}
QStatusBar QLabel {{
    color: {p.status_bar_fg};
    font-size: {FONT_SIZE_SMALL}px;
    padding: 0 {SPACE_SM}px;
}}
QStatusBar::item {{
    border: none;
}}

/* --- Dialogs ----------------------------------------------------- */
QDialog {{
    background-color: {p.surface};
}}

/* --- Message boxes ----------------------------------------------- */
QMessageBox {{
    background-color: {p.surface};
}}
QMessageBox QLabel {{
    color: {p.text_primary};
    font-size: {FONT_SIZE_BODY}px;
}}

/* --- Custom class selectors -------------------------------------- */

/* Card-style panels */
QFrame[class="card"] {{
    background-color: {p.surface};
    border: 1px solid {p.border_subtle};
    border-radius: {RADIUS_LG}px;
    padding: {SPACE_MD}px;
}}

QFrame[class="card"]:hover {{
    border-color: {p.border};
}}

/* Elevated card */
QFrame[class="card-elevated"] {{
    background-color: {p.surface_elevated};
    border: 1px solid {p.border_subtle};
    border-radius: {RADIUS_LG}px;
    padding: {SPACE_MD}px;
}}

/* Panel backgrounds */
QFrame[class="panel"] {{
    background-color: {p.surface};
    border: none;
}}

QFrame[class="panel-sidebar"] {{
    background-color: {p.sidebar_bg};
    border-right: 1px solid {p.sidebar_border};
}}

/* Badge / tag */
QLabel[class="badge"] {{
    background-color: {p.accent};
    color: {p.text_on_accent};
    border-radius: 10px;
    padding: 2px 8px;
    font-size: {FONT_SIZE_TINY}px;
    font-weight: 600;
}}

QLabel[class="badge-success"] {{
    background-color: {p.success};
    color: #FFFFFF;
    border-radius: 10px;
    padding: 2px 8px;
    font-size: {FONT_SIZE_TINY}px;
    font-weight: 600;
}}

QLabel[class="badge-error"] {{
    background-color: {p.error};
    color: #FFFFFF;
    border-radius: 10px;
    padding: 2px 8px;
    font-size: {FONT_SIZE_TINY}px;
    font-weight: 600;
}}

QLabel[class="badge-warning"] {{
    background-color: {p.warning};
    color: #FFFFFF;
    border-radius: 10px;
    padding: 2px 8px;
    font-size: {FONT_SIZE_TINY}px;
    font-weight: 600;
}}
"""
