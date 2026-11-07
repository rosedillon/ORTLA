"""The main window.

``RailModelApp`` holds the activity bar, the stack of views and the
status bar, and hands the real work to the views and services.
"""

from __future__ import annotations

import pathlib

from PySide6.QtCore import QSize
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from orta.gui.constants import (
    APP_NAME,
    COMMANDS,
    WINDOW_DEFAULT_HEIGHT,
    WINDOW_DEFAULT_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    ThemeMode,
)
from orta.gui.services.job_queue import Job, JobQueue
from orta.gui.services.project import Project
from orta.gui.services.settings import Settings
from orta.gui.theme import ThemeEngine
from orta.gui.views.analysis import AnalysisView
from orta.gui.views.dashboard import DashboardView
from orta.gui.views.help import HelpView
from orta.gui.views.job_queue import JobQueueView
from orta.gui.views.network_map import NetworkMapView
from orta.gui.views.pipeline import PipelineView
from orta.gui.widgets.activity_bar import ActivityBar
from orta.gui.widgets.notification import ToastNotification
from orta.gui.widgets.search_dialog import SearchDialog
from orta.gui.widgets.settings_dialog import SettingsDialog
from orta.gui.widgets.status_bar import AppStatusBar


class RailModelApp(QMainWindow):
    """The application's top-level window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(QSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT))
        self.resize(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)

        # ── Load Services ────────────────────────────────────────────
        self._settings = Settings()
        self._project = Project()
        self._theme = ThemeEngine()
        self._job_queue = JobQueue()

        # Transient UI state
        self._active_key_before_settings = "dashboard"
        self._active_toasts: list[ToastNotification] = []

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self._connect_signals()

        # Apply initial theme preference
        self._apply_initial_theme()

        # Update dashboard recent projects
        self._dashboard.set_recent_projects(self._settings.recent_projects)
        self._refresh_recent_menu()

        # Default view
        self._activity_bar.set_active("dashboard")
        self._switch_view("dashboard")

        # Restore the previous session, or open a fresh default analysis
        self._restore_startup_state()
        self._refresh_project_indicators()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Activity bar (far left)
        self._activity_bar = ActivityBar()
        root.addWidget(self._activity_bar)

        # Content stack (fills remaining space)
        content_wrapper = QFrame()
        content_layout = QVBoxLayout(content_wrapper)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self._stack = QStackedWidget()

        # Create views
        self._dashboard = DashboardView()
        self._analysis = AnalysisView(self._job_queue)
        self._pipeline = PipelineView(self._job_queue)
        self._jobs_view = JobQueueView(self._job_queue)
        self._network_map = NetworkMapView(self._job_queue)
        self._help_view = HelpView()

        self._stack.addWidget(self._dashboard)  # 0
        self._stack.addWidget(self._analysis)  # 1
        self._stack.addWidget(self._pipeline)  # 2
        self._stack.addWidget(self._jobs_view)  # 3
        self._stack.addWidget(self._network_map)  # 4
        self._stack.addWidget(self._help_view)  # 5

        self._view_index = {
            "dashboard": 0,
            "analysis": 1,
            "pipeline": 2,
            "jobs": 3,
            "maps": 4,
            "help": 5,
        }

        content_layout.addWidget(self._stack, stretch=1)
        root.addWidget(content_wrapper, stretch=1)

        # Status bar
        self._status_bar = AppStatusBar()
        self.setStatusBar(self._status_bar)

    def _build_menu(self) -> None:
        """Create file menu actions."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        action_new = QAction("New Project", self)
        action_new.setShortcut(QKeySequence.StandardKey.New)
        action_new.triggered.connect(self._new_project)
        file_menu.addAction(action_new)

        action_open = QAction("Open Project…", self)
        action_open.setShortcut(QKeySequence.StandardKey.Open)
        action_open.triggered.connect(self._open_project_dialog)
        file_menu.addAction(action_open)

        # Recent projects submenu (populated by _refresh_recent_menu)
        self._recent_menu = file_menu.addMenu("Open Recent")

        file_menu.addSeparator()

        action_save = QAction("Save Project", self)
        action_save.setShortcut(QKeySequence.StandardKey.Save)
        action_save.triggered.connect(self._save_project)
        file_menu.addAction(action_save)

        action_save_as = QAction("Save Project As…", self)
        action_save_as.triggered.connect(self._save_project_as)
        file_menu.addAction(action_save_as)

        file_menu.addSeparator()

        action_settings = QAction("Settings…", self)
        action_settings.setShortcut(QKeySequence("Ctrl+,"))
        action_settings.triggered.connect(self._open_settings)
        file_menu.addAction(action_settings)

        file_menu.addSeparator()

        action_exit = QAction("Exit", self)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

    def _build_shortcuts(self) -> None:
        """Register application-wide keyboard shortcuts."""
        shortcuts = [
            (QKeySequence("Ctrl+1"), lambda: self._go_to("dashboard")),
            (QKeySequence("Ctrl+2"), lambda: self._go_to("analysis")),
            (QKeySequence("Ctrl+3"), lambda: self._go_to("pipeline")),
            (QKeySequence("Ctrl+4"), lambda: self._go_to("jobs")),
            (QKeySequence("Ctrl+5"), lambda: self._go_to("maps")),
            (QKeySequence("Ctrl+K"), self._open_search_palette),
            (QKeySequence("Ctrl+T"), self._toggle_theme),
        ]
        for seq, callback in shortcuts:
            action = QAction(self)
            action.setShortcut(seq)
            action.triggered.connect(callback)
            self.addAction(action)

    def _connect_signals(self) -> None:
        self._activity_bar.view_changed.connect(self._switch_view)

        # Settings click at bottom of activity bar
        self._activity_bar.view_changed.connect(self._on_activity_bar_click)

        # Dashboard quick-action cards → switch to analysis
        self._dashboard.command_selected.connect(self._on_dashboard_command)
        self._dashboard.project_selected.connect(self._load_project_file)
        self._dashboard.new_project_requested.connect(self._new_project)
        self._dashboard.open_project_requested.connect(self._open_project_dialog)

        # Analysis status messages → status bar
        self._analysis.status_message.connect(self._status_bar.set_status)

        # Track unsaved edits and autosave completed runs
        self._analysis.config_changed.connect(self._on_config_changed)
        self._analysis.job_ended.connect(self._on_run_finished)

        # Connect JobQueue signals
        self._job_queue.job_added.connect(self._on_job_added)
        self._job_queue.job_updated.connect(self._on_job_updated)

    def _on_job_added(self, job: Job) -> None:
        self._show_toast("Job Queued", f"'{job.label}' added to background queue.", "info")
        self._update_status_bar_jobs()

    def _on_job_updated(self, job: Job) -> None:
        self._update_status_bar_jobs()
        if job.status == "done":
            self._show_toast("Job Completed", f"'{job.label}' finished successfully.", "success")
        elif job.status == "error":
            self._show_toast("Job Failed", f"'{job.label}' encountered an error.", "error")
        elif job.status == "cancelled":
            self._show_toast("Job Cancelled", f"'{job.label}' cancelled by user.", "warning")

    # ── toast notifications ───────────────────────────────────────────

    def _show_toast(self, title: str, message: str, kind: str = "info") -> None:
        """Show a toast above any already on screen."""
        toast = ToastNotification(title, message, kind, self._theme.mode, self)
        toast.closed.connect(lambda t=toast: self._remove_toast(t))
        self._active_toasts.append(toast)
        toast.set_slot(len(self._active_toasts) - 1)
        toast.show_toast()

    def _remove_toast(self, toast: ToastNotification) -> None:
        if toast in self._active_toasts:
            self._active_toasts.remove(toast)
        # Re-flow the survivors so no gaps are left in the stack.
        for slot, remaining in enumerate(self._active_toasts):
            remaining.set_slot(slot)

    def _update_status_bar_jobs(self) -> None:
        running = sum(1 for j in self._job_queue.jobs if j.status == "running")
        queued = sum(1 for j in self._job_queue.jobs if j.status == "queued")
        self._status_bar.set_jobs(running, queued)

    # ── view switching ───────────────────────────────────────────────

    def _switch_view(self, key: str) -> None:
        # Settings triggers dynamic dialog rather than stacked index
        if key == "settings":
            self._open_settings()
            # Restore previous active selection
            self._activity_bar.set_active(self._active_key_before_settings or "dashboard")
            return

        idx = self._view_index.get(key)
        if idx is not None:
            self._stack.setCurrentIndex(idx)

    def _go_to(self, key: str) -> None:
        self._activity_bar.set_active(key)
        self._switch_view(key)

    def _on_dashboard_command(self, command: str) -> None:
        self._go_to("analysis")
        self._analysis.select_command(command)

    def _on_activity_bar_click(self, key: str) -> None:
        if key != "settings":
            self._active_key_before_settings = key

    # ── project state helpers ─────────────────────────────────────────

    def _default_analysis(self) -> str:
        """Return the analysis a new project opens on."""
        cmd = self._settings.default_analysis
        return cmd if cmd in COMMANDS else "loading"

    def _on_config_changed(self) -> None:
        """Mark the project unsaved after a user edit."""
        if not self._project.is_dirty:
            self._project.mark_dirty()
            self._refresh_project_indicators()

    def _refresh_project_indicators(self) -> None:
        """Show the project's name and saved state."""
        name = self._project.name
        dirty = self._project.is_dirty
        self._status_bar.set_project(name, dirty)
        marker = " •" if dirty else ""
        self.setWindowTitle(f"{APP_NAME} · {name}{marker}")

    def _capture_active_config(self) -> None:
        """Store every command's config in the project."""
        self._project.configs = self._analysis.capture_all()
        self._project.active_command = self._analysis._current_command

    def _confirm_discard_if_dirty(self) -> bool:
        """Offer to save unsaved work; False means cancel."""
        if not self._project.is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"'{self._project.name}' has unsaved changes. Save before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Save:
            return self._save_project()
        return True  # Discard

    def _on_run_finished(self, success: bool) -> None:
        """Save a named project after a run, if set to."""
        if success and self._settings.autosave and self._project.filepath:
            self._save_project()

    # ── project management slots ──────────────────────────────────────

    def _new_project(self) -> None:
        if not self._confirm_discard_if_dirty():
            return

        self._project = Project()
        # Clear the per-command store and reset to the default analysis'
        # example.
        self._analysis.load_all({}, self._default_analysis())
        self._project.mark_clean()
        self._refresh_project_indicators()
        self._go_to("dashboard")

    def _open_project_dialog(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "ORTA project (*.orta)")
        if path:
            self._load_project_file(path)

    def _load_project_file(self, filepath: str) -> None:
        try:
            self._project = Project(pathlib.Path(filepath))
            # Record in recent history
            self._settings.add_recent_project(filepath)
            self._dashboard.set_recent_projects(self._settings.recent_projects)
            self._refresh_recent_menu()

            # Restore every remembered command's config; show the active
            # one.
            self._analysis.load_all(self._project.configs, self._project.active_command)

            self._project.mark_clean()
            self._refresh_project_indicators()
            self._go_to("analysis")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Failed", f"Could not load project: {exc}")
            self._settings.remove_recent_project(filepath)
            self._dashboard.set_recent_projects(self._settings.recent_projects)
            self._refresh_recent_menu()

    def _save_project(self) -> bool:
        if not self._project.filepath:
            return self._save_project_as()

        self._capture_active_config()
        try:
            self._project.save()
            self._settings.last_project = str(self._project.filepath)
            self._settings.save()
            self._refresh_project_indicators()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"Could not save the project: {exc}")
            return False
        self.status_bar_flash(f"Saved '{self._project.name}'.")
        return True

    def _save_project_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", "ORTA project (*.orta)")
        if not path:
            return False

        # Ensure .orta extension
        filepath = pathlib.Path(path)
        if filepath.suffix != ".orta":
            filepath = filepath.with_suffix(".orta")

        self._capture_active_config()
        try:
            self._project.save(filepath)
            self._settings.add_recent_project(str(filepath))
            self._dashboard.set_recent_projects(self._settings.recent_projects)
            self._refresh_recent_menu()
            self._refresh_project_indicators()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"Could not save the project: {exc}")
            return False
        self.status_bar_flash(f"Saved '{self._project.name}'.")
        return True

    def status_bar_flash(self, message: str) -> None:
        """Show a transient message in the status bar."""
        self._status_bar.set_status(message)

    # ── recent menu / startup / shutdown ──────────────────────────────

    def _refresh_recent_menu(self) -> None:
        """Rebuild the Open Recent menu."""
        self._recent_menu.clear()
        recents = [p for p in self._settings.recent_projects if pathlib.Path(p).exists()]
        if not recents:
            empty = QAction("No recent projects", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for path in recents:
            act = QAction(pathlib.Path(path).name, self)
            act.setToolTip(path)
            act.triggered.connect(lambda _checked=False, p=path: self._load_project_file(p))
            self._recent_menu.addAction(act)
        self._recent_menu.addSeparator()
        clear = QAction("Clear Recent", self)
        clear.triggered.connect(self._clear_recent)
        self._recent_menu.addAction(clear)

    def _clear_recent(self) -> None:
        self._settings.recent_projects = []
        self._settings.save()
        self._dashboard.set_recent_projects([])
        self._refresh_recent_menu()

    def _restore_startup_state(self) -> None:
        """Reopen the last project, or the default analysis."""
        last = self._settings.last_project
        if self._settings.restore_last_session and last and pathlib.Path(last).exists():
            self._load_project_file(last)
        else:
            self._analysis.select_command(self._default_analysis())
            self._project.mark_clean()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Check for unsaved changes and remember the project."""
        if self._settings.confirm_on_exit and not self._confirm_discard_if_dirty():
            event.ignore()
            return
        if self._project.filepath:
            self._settings.last_project = str(self._project.filepath)
            self._settings.save()
        super().closeEvent(event)

    # ── settings slots ────────────────────────────────────────────────

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Re-apply styling/theme preferences
            mode = self._settings.theme
            if mode == "system":
                mode = self._theme.detect_system().value
            self._apply_theme(ThemeMode(mode))

    # ── theme ────────────────────────────────────────────────────────

    def _apply_initial_theme(self) -> None:
        mode = self._settings.theme
        if mode == "system":
            mode = self._theme.detect_system()
        else:
            mode = ThemeMode(mode)
        self._apply_theme(mode)

    def _toggle_theme(self) -> None:
        mode = self._theme.toggle()
        self._apply_theme(mode)

    def _apply_theme(self, mode: ThemeMode) -> None:
        app = QApplication.instance()
        self._theme.apply(app, mode)
        self._activity_bar.update_theme(mode)
        self._dashboard.update_theme(mode)
        self._analysis.update_theme(mode)
        self._status_bar.update_theme(mode)

    # ── search palette slots ──────────────────────────────────────────

    def _open_search_palette(self) -> None:
        dlg = SearchDialog(self)
        # Position search palette dynamically centered at the top of the
        # window
        geom = self.geometry()
        dlg.move(
            geom.x() + (geom.width() - dlg.width()) // 2,
            geom.y() + 80,
        )
        dlg.action_triggered.connect(self._on_search_action_triggered)
        dlg.exec()

    def _on_search_action_triggered(self, category: str, key: str) -> None:
        if category == "view":
            self._go_to(key)
        elif category == "analysis":
            self._go_to("analysis")
            self._analysis.select_command(key)
        elif category == "action":
            if key == "settings":
                self._open_settings()
            elif key == "theme":
                self._toggle_theme()
