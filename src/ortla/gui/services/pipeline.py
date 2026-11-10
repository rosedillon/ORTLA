"""Pipeline chaining service.

Runs an ordered sequence of commands, auto-wiring each stage's dependent
config fields from earlier stages' outputs. Stages must run strictly
sequentially (not just be submitted together), since a later stage's
auto-filled fields depend on files an earlier stage has actually
finished writing.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from collections.abc import Callable

    from ortla import config
    from ortla.gui.services.job_queue import Job, JobQueue


@dataclasses.dataclass
class PipelineStage:
    """One command in a pipeline, with its config."""

    command: str
    config_data: dict
    output_path: pathlib.Path | None = None


def _first_period_matrices_dir(loading_stage: PipelineStage) -> pathlib.Path | None:
    """Find one operator's period matrices folder from a loading run.

    Picks the first match alphabetically when more than one operator was
    run The user can still change it in the stage's settings before
    running.
    """
    if loading_stage.output_path is None:
        return None
    matches = sorted(loading_stage.output_path.glob("*_period_matrices"))
    return matches[0] if matches else None


def _timetable_calls_csv(timetable_stage: PipelineStage) -> pathlib.Path | None:
    if timetable_stage.output_path is None:
        return None
    return timetable_stage.output_path / "timetable_calls.csv"


def _route_frequency_summary_csv(timetable_stage: PipelineStage) -> pathlib.Path | None:
    if timetable_stage.output_path is None:
        return None
    return timetable_stage.output_path / "route_frequency_summary.csv"


def _output_dir(stage: PipelineStage) -> pathlib.Path | None:
    return stage.output_path


# The timetable, running-time graph and period matrices that every
# command predicting a service's demand reads.
_SERVICE_INPUTS: dict[str, tuple[str, Callable[[PipelineStage], object]]] = {
    "timetable_calls_path": ("timetable", _timetable_calls_csv),
    "bplan_output_path": ("bplan", _output_dir),
    "period_matrices_dir": ("loading", _first_period_matrices_dir),
}


# For each consuming command, the fields it can auto-fill from an
# earlier stage's output: producing command key -> resolver(that stage)
# -> value. Fields not listed here (raw source data shared by
# convention, e.g. source_odm/station_list/table_1410) are left for the
# user to fill in.
PIPELINE_DEPENDENCIES: dict[str, dict[str, tuple[str, Callable[[PipelineStage], object]]]] = {
    "services": {
        "timetable_calls_path": ("timetable", _timetable_calls_csv),
    },
    "lookup": {
        "period_matrices_dir": ("loading", _first_period_matrices_dir),
    },
    "propose": _SERVICE_INPUTS,
    "scenario": _SERVICE_INPUTS,
    "hindcast": _SERVICE_INPUTS,
    "discover": {
        **_SERVICE_INPUTS,
        "timetable_output_path": ("timetable", _output_dir),
    },
    "loading": {
        "route_frequency_path": ("timetable", _route_frequency_summary_csv),
    },
    "commercial": {
        "propose_output_path": ("propose", _output_dir),
    },
}


def apply_dependencies(
    command: str, config_data: dict, completed: dict[str, PipelineStage]
) -> dict:
    """``config_data`` with fields filled from earlier stages.

    A field is filled only when the command that writes it has already
    finished in this pipeline, and then it replaces whatever the stage
    held: chaining exists to wire this run's own outputs together.
    """
    data = dict(config_data)
    for field, (producer, resolver) in PIPELINE_DEPENDENCIES.get(command, {}).items():
        stage = completed.get(producer)
        if stage is None:
            continue
        value = resolver(stage)
        if value is not None:
            data[field] = str(value)
    return data


class PipelineRunner(QObject):
    """Run :class:`PipelineStage` items in order on a ``JobQueue``.

    Signals
    -------
    stage_started(int)
        The stage at this index has been submitted to the queue.
    stage_finished(int, str, str)
        The stage at this index finished with a status
        (``"done"``/``"error"``/``"cancelled"``) and a detail message
        (empty for a clean success).
    pipeline_finished()
        Every stage ran, or the chain halted after a failure.
    """

    stage_started = Signal(int)
    stage_finished = Signal(int, str, str)
    pipeline_finished = Signal()

    def __init__(
        self,
        stages: list[PipelineStage],
        queue: JobQueue,
        config_map: dict[str, tuple[type[config.BaseRunConfig], str]],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._stages = stages
        self._queue = queue
        self._config_map = config_map
        self._completed: dict[str, PipelineStage] = {}
        self._index = -1

    def run(self) -> None:
        """Start the chain from its first stage."""
        self._index = -1
        self._completed = {}
        self._advance()

    def _advance(self) -> None:
        self._index += 1
        if self._index >= len(self._stages):
            self.pipeline_finished.emit()
            return

        stage = self._stages[self._index]
        model_cls, _ = self._config_map[stage.command]
        stage.config_data = apply_dependencies(stage.command, stage.config_data, self._completed)

        try:
            config = model_cls.model_validate(stage.config_data)
        except Exception as exc:  # noqa: BLE001
            self.stage_finished.emit(self._index, "error", f"Invalid config: {exc}")
            self.pipeline_finished.emit()
            return

        stage.output_path = pathlib.Path(config.output_path)
        self.stage_started.emit(self._index)

        job = self._queue.submit(stage.command, stage.command, config.run)
        job.status_changed.connect(
            lambda status, j=job, s=stage: self._on_job_status(s, j, status)
        )

    def _on_job_status(self, stage: PipelineStage, job: Job, status: str) -> None:
        if status == "done":
            self._completed[stage.command] = stage
            self.stage_finished.emit(self._index, "done", "")
            self._advance()
        elif status == "error":
            self.stage_finished.emit(self._index, "error", job.error_message)
            self.pipeline_finished.emit()
        elif status == "cancelled":
            self.stage_finished.emit(self._index, "cancelled", "")
            self.pipeline_finished.emit()
