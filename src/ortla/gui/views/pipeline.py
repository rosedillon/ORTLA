"""The Pipeline view: run commands one after another.

The user builds a list of stages, from scratch or from a preset, edits
each stage's config and runs them in order. A stage's inputs that an
earlier stage writes, such as ``propose``'s ``timetable_calls_path``,
are filled in from that stage's actual output
(:mod:`ortla.gui.services.pipeline`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import qtawesome as qta
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ortla.gui.constants import COMMANDS, SPACE_LG, SPACE_MD, SPACE_SM
from ortla.gui.services.pipeline import PipelineRunner, PipelineStage
from ortla.gui.views.analysis import CONFIG_MAP, load_example_config
from ortla.gui.widgets.config_form import ConfigForm

if TYPE_CHECKING:
    from ortla.gui.services.job_queue import JobQueue

# Preset stage sequences, each a shortcut for a common chain of
# commands.
_PRESETS: dict[str, list[str]] = {
    "Demand": ["matrix", "loading", "lookup"],
    "New service": ["timetable", "loading", "bplan", "propose", "commercial"],
    "Scenario": ["timetable", "loading", "bplan", "scenario"],
    "Network scan": ["timetable", "loading", "bplan", "discover"],
}

_STATUS_STYLES: dict[str, str] = {
    "Queued": "color: #1A9C94;",
    "Running…": "color: #1A9C94;",
    "Done ✓": "color: #54A24B;",
    "Failed ✕": "color: #D03B3B;",
    "Cancelled": "color: #F58518;",
}


class _PipelineNode(QFrame):
    """A visual step in the pipeline flow chart."""

    def __init__(
        self, step_key: str, label: str, icon_name: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.step_key = step_key
        self.setProperty("class", "card")
        self.setFixedSize(160, 90)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._icon_lbl = QLabel()
        self._icon_lbl.setPixmap(qta.icon(icon_name, color="#1A9C94").pixmap(24, 24))
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._icon_lbl)

        self._title_lbl = QLabel(label)
        self._title_lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_lbl.setWordWrap(True)
        layout.addWidget(self._title_lbl)

        self._lbl_status = QLabel("Idle")
        self._lbl_status.setProperty("class", "caption")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_status)

    def set_status(self, text: str, color_qss: str = "") -> None:
        """Update node status subtext."""
        self._lbl_status.setText(text)
        self._lbl_status.setStyleSheet(color_qss)


class _StageRow(QFrame):
    """One stage in the list, with its settings and remove buttons."""

    configure_requested = Signal()
    remove_requested = Signal()

    def __init__(self, index: int, command: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "card")
        label, icon, _desc = COMMANDS.get(command, (command, "fa5s.cog", ""))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        layout.setSpacing(SPACE_MD)

        order_lbl = QLabel(f"{index + 1}.")
        order_lbl.setStyleSheet("font-weight: bold; color: #5A6169;")
        layout.addWidget(order_lbl)

        self.node = _PipelineNode(command, label, icon)
        layout.addWidget(self.node)

        layout.addStretch(1)

        btn_configure = QPushButton("Configure")
        btn_configure.setProperty("class", "ghost")
        btn_configure.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_configure.clicked.connect(self.configure_requested.emit)
        layout.addWidget(btn_configure)

        btn_remove = QPushButton("✕")
        btn_remove.setProperty("class", "ghost")
        btn_remove.setFixedWidth(28)
        btn_remove.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_remove.clicked.connect(self.remove_requested.emit)
        layout.addWidget(btn_remove)

    def set_status(self, text: str) -> None:
        self.node.set_status(text, _STATUS_STYLES.get(text, ""))


class _StageConfigDialog(QDialog):
    """Edit one pipeline stage's config before it runs."""

    def __init__(self, model_cls: type, data: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure Stage")
        self.resize(520, 640)

        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._form = ConfigForm(model_cls)
        if data:
            self._form.set_data(data)
        scroll.setWidget(self._form)
        layout.addWidget(scroll)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_ok = QPushButton("Save")
        btn_ok.setProperty("class", "primary")
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def data(self) -> dict:
        return self._form.get_data()


class PipelineView(QScrollArea):
    """Build a list of stages and run them in order."""

    def __init__(self, queue: JobQueue, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._queue = queue
        self._stages: list[PipelineStage] = []
        self._rows: list[_StageRow] = []
        self._runner: PipelineRunner | None = None
        self._failed = False

        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)

        # Opaque QFrame (not a plain QWidget, which the global
        # stylesheet leaves transparent) so the scroll area always shows
        # a real background instead of the raw window buffer bleeding
        # through.
        container = QFrame()
        container.setProperty("class", "panel")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(SPACE_LG, SPACE_LG, SPACE_LG, SPACE_LG)
        layout.setSpacing(SPACE_LG)

        # Title
        title_box = QVBoxLayout()
        title = QLabel("Pipeline")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        title_box.addWidget(title)
        desc = QLabel(
            "Run several commands in order. When a stage needs a file an earlier "
            "stage writes, such as the timetable calls a proposal reads, it is "
            "filled in for you once that stage finishes."
        )
        desc.setProperty("class", "caption")
        desc.setWordWrap(True)
        title_box.addWidget(desc)
        layout.addLayout(title_box)

        # Presets row
        preset_panel = QFrame()
        preset_panel.setProperty("class", "card")
        preset_layout = QHBoxLayout(preset_panel)
        preset_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        preset_layout.addWidget(QLabel("Presets:"))
        for name, sequence in _PRESETS.items():
            btn = QPushButton(name)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, seq=sequence: self._load_preset(seq))
            preset_layout.addWidget(btn)
        preset_layout.addStretch()
        layout.addWidget(preset_panel)

        # Add-stage row
        add_panel = QFrame()
        add_panel.setProperty("class", "card")
        add_layout = QHBoxLayout(add_panel)
        add_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        add_layout.addWidget(QLabel("Add stage:"))
        self._add_combo = QComboBox()
        for key, (label, icon, _desc) in COMMANDS.items():
            self._add_combo.addItem(qta.icon(icon, color="#5A6169"), label, key)
        add_layout.addWidget(self._add_combo, stretch=1)
        btn_add = QPushButton("+ Add")
        btn_add.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_add.clicked.connect(self._add_stage_from_combo)
        add_layout.addWidget(btn_add)
        layout.addWidget(add_panel)

        # Stage list
        self._stage_list_layout = QVBoxLayout()
        self._stage_list_layout.setSpacing(SPACE_SM)
        layout.addLayout(self._stage_list_layout)

        # Execution controls
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        self._btn_run_pipeline = QPushButton("Execute Pipeline ▶")
        self._btn_run_pipeline.setProperty("class", "primary")
        self._btn_run_pipeline.setMinimumWidth(180)
        self._btn_run_pipeline.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_run_pipeline.setEnabled(False)
        self._btn_run_pipeline.clicked.connect(self._run_pipeline)
        btn_box.addWidget(self._btn_run_pipeline)
        layout.addLayout(btn_box)

        layout.addStretch(1)
        self.setWidget(container)

    # ── stage list management ────────────────────────────────────────

    def _add_stage_from_combo(self) -> None:
        command = self._add_combo.currentData()
        if command:
            self._add_stage(command)

    def _add_stage(self, command: str) -> None:
        _, filename = CONFIG_MAP[command]
        data = load_example_config(filename) or {}
        self._stages.append(PipelineStage(command=command, config_data=data))
        self._rebuild_stage_list()

    def _load_preset(self, sequence: list[str]) -> None:
        self._stages = []
        for command in sequence:
            _, filename = CONFIG_MAP[command]
            data = load_example_config(filename) or {}
            self._stages.append(PipelineStage(command=command, config_data=data))
        self._rebuild_stage_list()

    def _rebuild_stage_list(self) -> None:
        while self._stage_list_layout.count():
            item = self._stage_list_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self._rows = []

        for i, stage in enumerate(self._stages):
            row = _StageRow(i, stage.command)
            row.configure_requested.connect(lambda s=stage: self._configure_stage(s))
            row.remove_requested.connect(lambda s=stage: self._remove_stage(s))
            self._stage_list_layout.addWidget(row)
            self._rows.append(row)

        self._btn_run_pipeline.setEnabled(len(self._stages) > 0)

    def _configure_stage(self, stage: PipelineStage) -> None:
        model_cls, _ = CONFIG_MAP[stage.command]
        dlg = _StageConfigDialog(model_cls, stage.config_data, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            stage.config_data = dlg.data()

    def _remove_stage(self, stage: PipelineStage) -> None:
        self._stages.remove(stage)
        self._rebuild_stage_list()

    # ── execution ─────────────────────────────────────────────────────

    def _run_pipeline(self) -> None:
        if not self._stages:
            return
        self._failed = False
        self._btn_run_pipeline.setEnabled(False)
        for row in self._rows:
            row.set_status("Idle")

        self._runner = PipelineRunner(list(self._stages), self._queue, CONFIG_MAP, self)
        self._runner.stage_started.connect(self._on_stage_started)
        self._runner.stage_finished.connect(self._on_stage_finished)
        self._runner.pipeline_finished.connect(self._on_pipeline_finished)
        self._runner.run()

    def _on_stage_started(self, index: int) -> None:
        self._rows[index].set_status("Running…")

    def _on_stage_finished(self, index: int, status: str, detail: str) -> None:
        if status == "done":
            self._rows[index].set_status("Done ✓")
        elif status == "error":
            self._rows[index].set_status("Failed ✕")
            self._failed = True
            QMessageBox.critical(
                self,
                "Pipeline Stage Failed",
                f"Stage {index + 1} ({self._stages[index].command}) failed:\n\n{detail}",
            )
        elif status == "cancelled":
            self._rows[index].set_status("Cancelled")
            self._failed = True

    def _on_pipeline_finished(self) -> None:
        self._btn_run_pipeline.setEnabled(True)
        if not self._failed:
            QMessageBox.information(
                self,
                "Pipeline Complete",
                f"All {len(self._stages)} stage(s) completed successfully.",
            )
        self._runner = None
