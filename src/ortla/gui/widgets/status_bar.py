"""The status bar: jobs, project name and messages."""

from __future__ import annotations

import qtawesome as qta
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QStatusBar, QWidget

from ortla.gui.constants import SPACE_SM, SPACE_XS, ThemeMode, palette


class AppStatusBar(QStatusBar):
    """Bottom status bar showing application state at a glance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizeGripEnabled(False)
        self.setContentsMargins(SPACE_SM, 0, SPACE_SM, 0)

        # Left: status message
        self._status_label = QLabel("Ready")
        self.addWidget(self._status_label, stretch=1)

        # Centre: job indicator
        self._jobs_label = QLabel("No active jobs")
        self._jobs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.addWidget(self._jobs_label)

        # Right: the project's icon, name and unsaved marker.
        self._project_icon = QLabel()
        self._project_name = QLabel("No project")
        self._project_dirty = QLabel("")  # "●" when unsaved

        project_box = QWidget()
        pb_layout = QHBoxLayout(project_box)
        pb_layout.setContentsMargins(SPACE_SM, 0, SPACE_XS, 0)
        pb_layout.setSpacing(SPACE_XS + 2)
        pb_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        pb_layout.addWidget(self._project_icon, alignment=Qt.AlignmentFlag.AlignVCenter)
        pb_layout.addWidget(self._project_name, alignment=Qt.AlignmentFlag.AlignVCenter)
        pb_layout.addWidget(self._project_dirty, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.addPermanentWidget(project_box)

        self._fg = "#FFFFFF"
        self._refresh_project_icon()
        self.set_project(None)

    # ── public API ────────────────────────────────────────────────────
    def set_status(self, text: str) -> None:
        """Update the main status message."""
        self._status_label.setText(text)

    def set_jobs(self, running: int, queued: int) -> None:
        """Update the job indicator."""
        if running == 0 and queued == 0:
            self._jobs_label.setText("No active jobs")
        elif running > 0 and queued > 0:
            self._jobs_label.setText(f"● {running} running · {queued} queued")
        elif running > 0:
            self._jobs_label.setText(f"● {running} running")
        else:
            self._jobs_label.setText(f"○ {queued} queued")

    def set_project(self, name: str | None, dirty: bool = False) -> None:
        """Show the project's name and whether it is saved."""
        self._project_icon.setVisible(bool(name))
        if name:
            self._project_name.setText(name)
            self._project_dirty.setText("●" if dirty else "")
            tip = f"{name}: unsaved changes" if dirty else name
            self._project_name.setToolTip(tip)
        else:
            self._project_name.setText("No project")
            self._project_dirty.setText("")
            self._project_name.setToolTip("No project open")

    def update_theme(self, mode: ThemeMode) -> None:
        """Recolour the project icon to match the bar."""
        self._fg = palette(mode).status_bar_fg
        self._refresh_project_icon()

    def _refresh_project_icon(self) -> None:
        """Draw the folder icon in the bar's colour."""
        self._project_icon.setPixmap(qta.icon("fa5s.folder", color=self._fg).pixmap(13, 13))
        self._project_dirty.setStyleSheet(f"color: {self._fg}; font-size: 11px;")
