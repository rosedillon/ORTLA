"""Dashboard / home view, the first thing users see on launch.

Shows a welcome banner, recent projects, and quick-action cards for each
command.
"""

from __future__ import annotations

import pathlib

import qtawesome as qta
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ortla.gui.constants import (
    APP_FULL_NAME,
    APP_NAME,
    COMMANDS,
    FONT_SIZE_H3,
    SPACE_2XL,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    ThemeMode,
    palette,
)


class _CommandCard(QFrame):
    """A clickable card representing one analysis command."""

    clicked = Signal(str)

    def __init__(
        self,
        key: str,
        label: str,
        icon_name: str,
        description: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._key = key
        self._icon_name = icon_name
        self.setProperty("class", "card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(SPACE_SM)

        # Icon + title row
        top = QHBoxLayout()
        top.setSpacing(SPACE_SM)
        self._icon_label = QLabel()
        self._icon_label.setFixedSize(28, 28)
        top.addWidget(self._icon_label)
        title = QLabel(label)
        title.setFont(QFont("Inter", FONT_SIZE_H3, QFont.Weight.DemiBold))
        top.addWidget(title)
        top.addStretch()
        layout.addLayout(top)

        # Description
        desc = QLabel(description)
        desc.setProperty("class", "caption")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addStretch()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Open the analysis this card stands for."""
        self.clicked.emit(self._key)
        super().mousePressEvent(event)

    def update_theme(self, mode: ThemeMode) -> None:
        p = palette(mode)
        self._icon_label.setPixmap(qta.icon(self._icon_name, color=p.accent).pixmap(28, 28))


class DashboardView(QScrollArea):
    """The home / welcome screen.

    Signals
    -------
    command_selected(str)
        Emitted when a quick-action card is clicked.
    project_selected(str)
        Emitted when a recent project is selected.
    """

    command_selected = Signal(str)
    project_selected = Signal(str)
    new_project_requested = Signal()
    open_project_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._cards: list[_CommandCard] = []

        # Opaque QFrame (not a plain QWidget, which the global
        # stylesheet leaves transparent) so the scroll area always shows
        # a real background instead of the raw window buffer bleeding
        # through.
        container = QFrame()
        container.setProperty("class", "panel")
        self._main_layout = QVBoxLayout(container)
        self._main_layout.setContentsMargins(SPACE_2XL, SPACE_2XL, SPACE_2XL, SPACE_2XL)
        self._main_layout.setSpacing(SPACE_XL)

        # ── Welcome banner ───────────────────────────────────────────
        banner = QFrame()
        banner.setProperty("class", "card")
        banner_layout = QVBoxLayout(banner)
        banner_layout.setContentsMargins(SPACE_XL, SPACE_XL, SPACE_XL, SPACE_XL)
        banner_layout.setSpacing(SPACE_SM)

        title = QLabel(APP_NAME)
        title.setProperty("class", "display")
        banner_layout.addWidget(title)

        subtitle = QLabel(APP_FULL_NAME)
        subtitle.setProperty("class", "tagline")
        banner_layout.addWidget(subtitle)

        desc = QLabel(
            "Open demand-modelling toolkit for the UK rail network. "
            "New here? Start a project to save your configuration and results, "
            "or jump straight into an analysis below."
        )
        desc.setWordWrap(True)
        desc.setProperty("class", "caption")
        banner_layout.addWidget(desc)

        # Primary calls-to-action: New / Open project
        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, SPACE_SM, 0, 0)
        actions_row.setSpacing(SPACE_SM)

        btn_new = QPushButton("  New Project")
        btn_new.setProperty("class", "primary")
        btn_new.setIcon(qta.icon("fa5s.plus", color="#FFFFFF"))
        btn_new.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_new.clicked.connect(self.new_project_requested.emit)
        actions_row.addWidget(btn_new)

        btn_open = QPushButton("  Open Project…")
        btn_open.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_open_icon = btn_open
        btn_open.clicked.connect(self.open_project_requested.emit)
        actions_row.addWidget(btn_open)

        actions_row.addStretch()
        banner_layout.addLayout(actions_row)

        self._main_layout.addWidget(banner)

        # ── Recent Projects container ────────────────────────────────
        self._recent_widget = QWidget()
        self._recent_layout = QVBoxLayout(self._recent_widget)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(SPACE_SM)
        self._main_layout.addWidget(self._recent_widget)
        self._recent_widget.hide()

        # ── Quick actions ────────────────────────────────────────────
        section_label = QLabel("ANALYSES")
        section_label.setProperty("class", "section")
        self._main_layout.addWidget(section_label)

        grid = QGridLayout()
        grid.setSpacing(SPACE_MD)
        col, row = 0, 0
        for key, (label, icon, desc_text) in COMMANDS.items():
            card = _CommandCard(key, label, icon, desc_text)
            card.clicked.connect(self.command_selected.emit)
            self._cards.append(card)
            grid.addWidget(card, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1
        self._main_layout.addLayout(grid)

        self._main_layout.addStretch(1)
        self.setWidget(container)

    def set_recent_projects(self, project_paths: list[str]) -> None:
        """Populate the recent projects list section."""
        # Clear old list widgets
        while self._recent_layout.count():
            item = self._recent_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        valid_paths = [p for p in project_paths if pathlib.Path(p).exists()]

        if not valid_paths:
            self._recent_widget.hide()
            return

        lbl = QLabel("RECENT PROJECTS")
        lbl.setProperty("class", "section")
        self._recent_layout.addWidget(lbl)

        for path_str in valid_paths:
            path = pathlib.Path(path_str)
            row = QFrame()
            row.setProperty("class", "card")
            row.setFixedHeight(50)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(SPACE_MD, 0, SPACE_MD, 0)

            icon_lbl = QLabel()
            icon_lbl.setPixmap(qta.icon("fa5s.folder-open", color="#5A6169").pixmap(16, 16))
            row_layout.addWidget(icon_lbl)

            name_lbl = QLabel(path.stem)
            name_lbl.setStyleSheet("font-weight: 600;")
            row_layout.addWidget(name_lbl)

            path_lbl = QLabel(str(path))
            path_lbl.setProperty("class", "caption")
            row_layout.addWidget(path_lbl, stretch=1)

            btn = QPushButton("Open")
            btn.setProperty("class", "ghost")
            btn.setFixedWidth(60)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _ch, p=path_str: self.project_selected.emit(p))
            row_layout.addWidget(btn)

            self._recent_layout.addWidget(row)

        self._recent_widget.show()

    def update_theme(self, mode: ThemeMode) -> None:
        """Refresh card icons for the given theme."""
        p = palette(mode)
        for card in self._cards:
            card.update_theme(mode)
        self._btn_open_icon.setIcon(qta.icon("fa5s.folder-open", color=p.text_secondary))
