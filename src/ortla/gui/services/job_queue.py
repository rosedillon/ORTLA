"""Job queue service that manages concurrent background pipeline tasks.

Implements a queue manager that executes up to *max_concurrent* jobs in
parallel, managing their states (queued, running, done, error,
cancelled) and routing log and progress messages.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from ortla.gui.services.runner import JobRunner

if TYPE_CHECKING:
    from collections.abc import Callable


class Job(QObject):
    """Represents a single command execution unit in the queue."""

    status_changed = Signal(str)  # new status
    progress_changed = Signal(float)  # 0.0 - 1.0
    log_received = Signal(str)

    def __init__(self, key: str, label: str, target: Callable[[], None]) -> None:
        super().__init__()
        self.key = key
        self.label = label
        self.target = target
        self.status = "queued"  # queued, running, done, error, cancelled
        self.progress: float = 0.0
        self.log_text: str = ""
        self.error_message: str = ""
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.runner: JobRunner | None = None

    @property
    def elapsed(self) -> float:
        """Get elapsed run time in seconds."""
        if self.start_time == 0:
            return 0.0
        if self.end_time == 0:
            return time.monotonic() - self.start_time
        return self.end_time - self.start_time

    def cancel(self) -> None:
        """Cancel the job if it is still queued.

        Running jobs cannot be cancelled: the target callable executes
        on a QThread with no cooperative interruption support, so there
        is no safe way to stop it mid-run without corrupting partial
        output files.
        """
        if self.status == "queued":
            self.status = "cancelled"
            self.status_changed.emit(self.status)


class JobQueue(QObject):
    """Schedules, limits concurrency, and coordinates background jobs.

    Signals
    -------
    job_added(Job)
        Emitted when a new job joins the queue.
    job_updated(Job)
        Emitted when a job's status/progress updates.
    queue_cleared()
        Emitted when completed jobs are removed from list.
    """

    job_added = Signal(Job)
    job_updated = Signal(Job)
    queue_cleared = Signal()

    def __init__(self, max_concurrent: int = 2) -> None:
        super().__init__()
        self.max_concurrent = max_concurrent
        self.jobs: list[Job] = []

    def submit(self, key: str, label: str, target: Callable[[], None]) -> Job:
        """Create and queue a new job."""
        job = Job(key, label, target)
        self.jobs.append(job)
        self.job_added.emit(job)
        self._process_queue()
        return job

    def clear_completed(self) -> None:
        """Remove completed, errored, and cancelled jobs from memory."""
        self.jobs = [j for j in self.jobs if j.status in ("queued", "running")]
        self.queue_cleared.emit()

    # ── internal scheduling ──────────────────────────────────────────

    def _process_queue(self) -> None:
        running_count = sum(1 for j in self.jobs if j.status == "running")
        if running_count >= self.max_concurrent:
            return

        for job in self.jobs:
            if job.status == "queued":
                self._start_job(job)
                running_count += 1
                if running_count >= self.max_concurrent:
                    break

    def _start_job(self, job: Job) -> None:
        job.status = "running"
        job.start_time = time.monotonic()
        job.status_changed.emit(job.status)
        self.job_updated.emit(job)

        runner = JobRunner(self)
        job.runner = runner

        # Connect runner signals to job
        runner.log_message.connect(lambda msg, j=job: self._append_log(j, msg))
        runner.job_finished.connect(lambda _elapsed, j=job: self._on_job_success(j))
        runner.job_error.connect(lambda err, j=job: self._on_job_error(j, err))

        try:
            runner.start(job.target)
        except Exception as exc:  # noqa: BLE001
            self._on_job_error(job, str(exc))

    def _append_log(self, job: Job, message: str) -> None:
        job.log_text += message + "\n"
        job.log_received.emit(message)

        # A log line may carry an "NN%" progress token; use the first.
        token = next((part for part in message.split() if part.endswith("%")), None)
        if token is not None:
            with contextlib.suppress(ValueError):
                job.progress = float(token.rstrip("%")) / 100.0
                job.progress_changed.emit(job.progress)
                self.job_updated.emit(job)

    def _on_job_success(self, job: Job) -> None:
        if job.status == "cancelled":
            job.runner = None
            return
        job.status = "done"
        job.end_time = time.monotonic()
        job.progress = 1.0
        job.progress_changed.emit(1.0)
        job.status_changed.emit(job.status)
        self.job_updated.emit(job)
        job.runner = None
        self._process_queue()

    def _on_job_error(self, job: Job, error: str) -> None:
        if job.status == "cancelled":
            job.runner = None
            return
        job.status = "error"
        job.error_message = error
        job.end_time = time.monotonic()
        job.status_changed.emit(job.status)
        self.job_updated.emit(job)
        job.runner = None
        self._process_queue()
