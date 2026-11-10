"""A card for one background job in the Jobs view.

Visualises status (queued, running, completed, error), progress,
execution elapsed duration, and offers action buttons to cancel, retry
or view logs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import qtawesome as qta
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ortla.gui.constants import SPACE_MD, SPACE_SM

if TYPE_CHECKING:
    from ortla.gui.services.job_queue import Job


class JobCard(QFrame):
    """Visual panel for a queued or running job."""

    def __init__(self, job: Job, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.job = job
        self.setProperty("class", "card")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedHeight(100)

        self._build_ui()
        self._connect_signals()
        self._update_status(self.job.status)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_MD, SPACE_SM, SPACE_MD, SPACE_SM)
        layout.setSpacing(SPACE_SM)

        # Header: Name + Badge + Cancel Button
        header = QHBoxLayout()
        header.setSpacing(SPACE_SM)

        self._lbl_name = QLabel(self.job.label)
        self._lbl_name.setStyleSheet("font-weight: 600; font-size: 13px;")
        header.addWidget(self._lbl_name)

        self._lbl_status_badge = QLabel("Queued")
        self._lbl_status_badge.setProperty("class", "badge")
        header.addWidget(self._lbl_status_badge)

        self._lbl_duration = QLabel("0.0s")
        self._lbl_duration.setProperty("class", "caption")
        header.addWidget(self._lbl_duration)

        header.addStretch()

        self._btn_cancel = QPushButton()
        self._btn_cancel.setFixedSize(28, 28)
        self._btn_cancel.setIcon(qta.icon("fa5s.times", color="#D03B3B"))
        self._btn_cancel.setToolTip("Cancel Job")
        self._btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_cancel.clicked.connect(self.job.cancel)
        header.addWidget(self._btn_cancel)

        layout.addLayout(header)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        # Detailed subtext (current log snippet or error message)
        self._lbl_detail = QLabel("Waiting in queue…")
        self._lbl_detail.setProperty("class", "caption")
        self._lbl_detail.setWordWrap(True)
        layout.addWidget(self._lbl_detail)

    def _connect_signals(self) -> None:
        self.job.status_changed.connect(self._update_status)
        self.job.progress_changed.connect(self._update_progress)
        self.job.log_received.connect(self._update_detail)

    def _update_status(self, status: str) -> None:
        self._lbl_status_badge.setText(status.upper())

        if status == "queued":
            self._lbl_status_badge.setStyleSheet("background-color: #B0B7C6; color: #FFFFFF;")
            self._progress.setRange(0, 0)  # indeterminate loading pulse
            self._btn_cancel.show()
        elif status == "running":
            self._lbl_status_badge.setStyleSheet("background-color: #1A9C94; color: #FFFFFF;")
            self._progress.setRange(0, 100)
            self._btn_cancel.hide()
        elif status == "done":
            self._lbl_status_badge.setStyleSheet("background-color: #54A24B; color: #FFFFFF;")
            self._progress.setRange(0, 100)
            self._progress.setValue(100)
            self._btn_cancel.hide()
            self._lbl_detail.setText("Success")
            self._lbl_duration.setText(f"{self.job.elapsed:.1f}s")
        elif status == "error":
            self._lbl_status_badge.setStyleSheet("background-color: #D03B3B; color: #FFFFFF;")
            self._progress.setRange(0, 100)
            self._progress.setValue(100)
            self._btn_cancel.hide()
            self._lbl_detail.setText(self.job.error_message)
            self._lbl_detail.setStyleSheet("color: #D03B3B;")
            self._lbl_duration.setText(f"{self.job.elapsed:.1f}s")
        elif status == "cancelled":
            self._lbl_status_badge.setStyleSheet("background-color: #F58518; color: #FFFFFF;")
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._btn_cancel.hide()
            self._lbl_detail.setText("Cancelled by user")

    def _update_progress(self, val: float) -> None:
        self._progress.setValue(int(val * 100))
        self._lbl_duration.setText(f"{self.job.elapsed:.1f}s")

    def _update_detail(self, line: str) -> None:
        # Show the latest non-empty log line
        cleaned = line.strip()
        if cleaned:
            self._lbl_detail.setText(cleaned)
            self._lbl_duration.setText(f"{self.job.elapsed:.1f}s")
        self._lbl_detail.setStyleSheet("")
