"""Job queue view that visualises scheduled background tasks.

Hosts a list of JobCards in a scrollable frame, providing utility
buttons to clear finished runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ortla.gui.constants import SPACE_LG, SPACE_MD, SPACE_SM
from ortla.gui.widgets.job_card import JobCard

if TYPE_CHECKING:
    from ortla.gui.services.job_queue import JobQueue


class JobQueueView(QScrollArea):
    """List of all background operations and scheduler state."""

    def __init__(self, queue: JobQueue, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._queue = queue
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)

        # Opaque QFrame (not a plain QWidget, which the global
        # stylesheet leaves transparent) so the scroll area always shows
        # a real background instead of the raw window buffer bleeding
        # through.
        container = QFrame()
        container.setProperty("class", "panel")
        self._main_layout = QVBoxLayout(container)
        self._main_layout.setContentsMargins(SPACE_LG, SPACE_LG, SPACE_LG, SPACE_LG)
        self._main_layout.setSpacing(SPACE_MD)

        # Header toolbar
        header = QHBoxLayout()
        header.setSpacing(SPACE_SM)

        title = QLabel("Execution Queue")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        header.addWidget(title)

        header.addStretch()

        self._btn_clear = QPushButton("Clear Completed")
        self._btn_clear.setProperty("class", "ghost")
        self._btn_clear.clicked.connect(self._queue.clear_completed)
        header.addWidget(self._btn_clear)

        self._main_layout.addLayout(header)

        # Inner list
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(SPACE_SM)
        self._main_layout.addWidget(self._list_widget)

        # Empty state panel
        self._empty_widget = QWidget()
        empty_l = QVBoxLayout(self._empty_widget)
        empty_l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_l.setSpacing(SPACE_SM)
        empty_lbl = QLabel("No operations scheduled")
        empty_lbl.setStyleSheet("font-size: 16px; font-weight: 600; color: rgba(0,0,0,0.4);")
        empty_l.addWidget(empty_lbl)
        sub_lbl = QLabel("Run analyses or pipelines to queue background jobs here.")
        sub_lbl.setProperty("class", "caption")
        empty_l.addWidget(sub_lbl)
        self._main_layout.addWidget(self._empty_widget)

        self._main_layout.addStretch(1)
        self.setWidget(container)

        # Wire queue notifications
        self._queue.job_added.connect(self._refresh_list)
        self._queue.job_updated.connect(self._check_empty_state)
        self._queue.queue_cleared.connect(self._refresh_list)

        self._refresh_list()

    def _refresh_list(self) -> None:
        # Clear layout
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Build card for each active job in queue
        for job in self._queue.jobs:
            card = JobCard(job, self._list_widget)
            self._list_layout.addWidget(card)

        self._check_empty_state()

    def _check_empty_state(self) -> None:
        if not self._queue.jobs:
            self._empty_widget.show()
            self._btn_clear.setEnabled(False)
        else:
            self._empty_widget.hide()
            self._btn_clear.setEnabled(
                any(j.status in ("done", "error", "cancelled") for j in self._queue.jobs)
            )
