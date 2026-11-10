"""The Maps view: the track network, coloured by daily trains.

The map is built by :mod:`ortla.gui.network_map` on a
background job and shown in a :class:`MapView`.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ortla import config
from ortla.gui import network_map
from ortla.gui.constants import SPACE_MD, SPACE_SM
from ortla.gui.views.analysis import load_example_config
from ortla.gui.widgets.config_form import ConfigForm
from ortla.gui.widgets.map_view import MapView

if TYPE_CHECKING:
    from ortla.gui.services.job_queue import Job, JobQueue

# Default location the map is (re-)built to, so re-opening this view
# without rebuilding can still find the last map (not currently
# auto-loaded, but kept stable so the user can find the file easily
# outside the app too).
_DEFAULT_OUTPUT_FILE = pathlib.Path.home() / ".ortla" / "network_map.html"


class _DirField(QFrame):
    """A labelled folder picker."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACE_SM, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label)
        lbl.setProperty("class", "caption")
        lbl.setStyleSheet("font-weight: 600; font-size: 12px;")
        layout.addWidget(lbl)

        row = QHBoxLayout()
        row.setSpacing(4)
        self._entry = QLineEdit()
        self._entry.setPlaceholderText("Folder…")
        row.addWidget(self._entry, stretch=1)
        btn = QPushButton("Folder")
        btn.setFixedWidth(56)
        btn.setProperty("class", "ghost")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self._pick)
        row.addWidget(btn)
        layout.addLayout(row)

    def _pick(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select directory")
        if path:
            self._entry.setText(path)

    def path(self) -> pathlib.Path | None:
        text = self._entry.text().strip()
        return pathlib.Path(text) if text else None


class NetworkMapView(QWidget):
    """Build and show the whole-network map."""

    def __init__(self, queue: JobQueue, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._queue = queue
        self._output_file: pathlib.Path | None = None

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left = QFrame()
        left.setProperty("class", "panel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        left_layout.setSpacing(SPACE_SM)

        title = QLabel("Network map")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        left_layout.addWidget(title)
        desc = QLabel(
            "Draw the whole track network and its stations. Give a timetable "
            "run's output folder to colour each line by its daily trains."
        )
        desc.setProperty("class", "caption")
        desc.setWordWrap(True)
        left_layout.addWidget(desc)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._calc_form = ConfigForm(config.TrackCalculatorConfig)
        initial = load_example_config("distance.yml")
        if initial and "calculator" in initial:
            self._calc_form.set_data(initial["calculator"])
        scroll.setWidget(self._calc_form)
        left_layout.addWidget(scroll, stretch=1)

        self._timetable_dir = _DirField("Timetable output folder (optional)")
        left_layout.addWidget(self._timetable_dir)

        self._btn_build = QPushButton("Build map")
        self._btn_build.setProperty("class", "primary")
        self._btn_build.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_build.clicked.connect(self._build_map)
        left_layout.addWidget(self._btn_build)

        self._status_lbl = QLabel("")
        self._status_lbl.setProperty("class", "caption")
        self._status_lbl.setWordWrap(True)
        left_layout.addWidget(self._status_lbl)

        splitter.addWidget(left)

        self._map_view = MapView()
        splitter.addWidget(self._map_view)
        splitter.setSizes([420, 780])

    def _build_map(self) -> None:
        try:
            calculator = config.TrackCalculatorConfig.model_validate(self._calc_form.get_data())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Invalid Config", str(exc))
            return

        self._output_file = _DEFAULT_OUTPUT_FILE
        self._status_lbl.setText(
            "Building network map… this can take a while for a large shapefile."
        )
        self._btn_build.setEnabled(False)
        self._btn_build.setText("Building…")

        result_holder: dict[str, tuple[int, int]] = {}

        timetable_dir = self._timetable_dir.path()

        def target() -> None:
            result_holder["result"] = network_map.build_network_map(
                calculator, _DEFAULT_OUTPUT_FILE, timetable_dir
            )

        job = self._queue.submit("network_map", "Network Map", target)
        job.status_changed.connect(
            lambda status, j=job: self._on_job_status(j, status, result_holder)
        )

    def _on_job_status(self, job: Job, status: str, result_holder: dict) -> None:
        if status == "done":
            self._btn_build.setEnabled(True)
            self._btn_build.setText("Build map")
            shown, total = result_holder.get("result", (0, 0))
            if total:
                self._status_lbl.setText(f"Map built: {shown:,} of {total:,} station pairs drawn.")
            else:
                self._status_lbl.setText("Map built.")
            if self._output_file:
                self._map_view.set_maps([self._output_file])
        elif status == "error":
            self._btn_build.setEnabled(True)
            self._btn_build.setText("Build map")
            self._status_lbl.setText(f"Failed: {job.error_message}")
            QMessageBox.critical(self, "Build Failed", job.error_message)
        elif status == "cancelled":
            self._btn_build.setEnabled(True)
            self._btn_build.setText("Build map")
