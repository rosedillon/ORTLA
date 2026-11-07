"""QThread-based job runner with signal-driven progress reporting.

Replaces the legacy ``runner.py`` (threading + queue) with Qt's native
threading model.  Log messages, completion, and errors are delivered as
signals, no polling required.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal

if TYPE_CHECKING:
    from collections.abc import Callable


class _QueueLogHandler(logging.Handler):
    """Logging handler that emits records as Qt signals."""

    def __init__(self, signal: Signal) -> None:
        super().__init__()
        self._signal = signal
        self.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._signal.emit(self.format(record))
        except Exception:  # noqa: BLE001
            self.handleError(record)


class _Worker(QObject):
    """Runs a callable on a QThread, emitting progress signals."""

    log = Signal(str)
    finished = Signal(float)  # elapsed seconds
    error = Signal(str)

    def __init__(self, target: Callable[[], None]) -> None:
        super().__init__()
        self._target = target

    def run(self) -> None:
        """Execute the target callable."""
        logger = logging.getLogger("orta")
        handler = _QueueLogHandler(self.log)
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        t0 = time.monotonic()
        try:
            self._target()
            elapsed = time.monotonic() - t0
            self.finished.emit(elapsed)
        except Exception as exc:
            logger.exception("Run failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)


class JobRunner(QObject):
    """Run a callable on a background thread, reporting by signal.

    Signals
    -------
    log_message(str)
        Each log line produced by the running command.
    job_finished(float)
        Emitted on success with elapsed seconds.
    job_error(str)
        Emitted on failure with the formatted error message.
    """

    log_message = Signal(str)
    job_finished = Signal(float)
    job_error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _Worker | None = None

    @property
    def running(self) -> bool:
        """Whether a job is currently executing."""
        return self._thread is not None and self._thread.isRunning()

    def start(self, target: Callable[[], None]) -> None:
        """Run *target* on a worker thread.

        Raises
        ------
        RuntimeError
            If a job is already running.
        """
        if self.running:
            raise RuntimeError("A job is already running.")

        self._thread = QThread()
        self._worker = _Worker(target)
        self._worker.moveToThread(self._thread)

        # Wire signals
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log_message.emit)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._thread.start()

    def _on_finished(self, elapsed: float) -> None:
        self.job_finished.emit(elapsed)
        self._cleanup()

    def _on_error(self, message: str) -> None:
        self.job_error.emit(message)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
