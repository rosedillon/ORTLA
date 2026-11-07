"""VS Code-style activity bar: a narrow icon rail along the left edge.

Each icon represents a top-level view.  Clicking an icon switches the
content area; clicking the already-active icon toggles the sidebar.
"""

from __future__ import annotations

from typing import ClassVar

import qtawesome as qta
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from orta.gui.constants import (
    ACTIVITY_BAR_WIDTH,
    SPACE_SM,
    ThemeMode,
    palette,
)


class _ActivityButton(QToolButton):
    """A single icon button in the activity bar."""

    def __init__(
        self,
        icon_name: str,
        tooltip: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon_name = icon_name
        self.setToolTip(tooltip)
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setIconSize(QSize(22, 22))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(ACTIVITY_BAR_WIDTH, ACTIVITY_BAR_WIDTH)
        self._update_icon(ThemeMode.LIGHT)

    def _update_icon(self, mode: ThemeMode) -> None:
        p = palette(mode)
        colour = p.activity_bar_active if self.isChecked() else p.activity_bar_fg
        self.setIcon(qta.icon(self._icon_name, color=colour))

    def update_theme(self, mode: ThemeMode) -> None:
        """Refresh the icon colour for the given theme."""
        self._update_icon(mode)

    # Override to refresh icon on check-state change
    def nextCheckState(self) -> None:  # noqa: N802
        super().nextCheckState()
        # Will be refreshed by the parent when it re-themes


class ActivityBar(QFrame):
    """Vertical icon rail for top-level navigation.

    Signals
    -------
    view_changed(str)
        Emitted when the user clicks an icon. Payload is the view key.
    sidebar_toggled()
        Emitted when the user clicks the already-active icon (toggle
        sidebar).
    """

    view_changed = Signal(str)
    sidebar_toggled = Signal()

    # Each item is (key, icon, tooltip).
    ITEMS: ClassVar[list[tuple[str, str, str]]] = [
        ("dashboard", "fa5s.home", "Dashboard"),
        ("analysis", "fa5s.flask", "Analysis"),
        ("pipeline", "fa5s.network-wired", "Pipeline"),
        ("jobs", "fa5s.tasks", "Jobs"),
        ("maps", "fa5s.map-marked-alt", "Maps"),
    ]

    BOTTOM_ITEMS: ClassVar[list[tuple[str, str, str]]] = [
        ("help", "fa5s.question-circle", "Help"),
        ("settings", "fa5s.cog", "Settings"),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(ACTIVITY_BAR_WIDTH)
        self._buttons: dict[str, _ActivityButton] = {}
        self._current_key: str | None = None
        self._mode = ThemeMode.LIGHT
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACE_SM, 0, SPACE_SM)
        layout.setSpacing(0)

        for key, icon, tip in self.ITEMS:
            btn = _ActivityButton(icon, tip, self)
            btn.clicked.connect(lambda _checked, k=key: self._on_click(k))
            layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
            self._buttons[key] = btn

        layout.addStretch(1)

        for key, icon, tip in self.BOTTOM_ITEMS:
            btn = _ActivityButton(icon, tip, self)
            btn.clicked.connect(lambda _checked, k=key: self._on_click(k))
            layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
            self._buttons[key] = btn

    def _on_click(self, key: str) -> None:
        if key == self._current_key:
            self.sidebar_toggled.emit()
        else:
            self._current_key = key
            self._buttons[key].setChecked(True)
            self.view_changed.emit(key)
            self._refresh_icons()

    def set_active(self, key: str) -> None:
        """Programmatically select a view."""
        if key in self._buttons:
            self._current_key = key
            self._buttons[key].setChecked(True)
            self._refresh_icons()

    def update_theme(self, mode: ThemeMode) -> None:
        """Refresh all icon colours for the given theme."""
        self._mode = mode
        p = palette(mode)
        self.setStyleSheet(f"ActivityBar {{ background-color: {p.activity_bar_bg}; }}")
        self._refresh_icons()

    def _refresh_icons(self) -> None:
        for btn in self._buttons.values():
            btn.update_theme(self._mode)
