"""Analysis view: config editor + results viewer for any command.

This replaces the monolithic ``app.py`` content area with a clean split:
left panel for config editing (form + YAML tabs), right panel for
results (log, charts, data, maps).
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
from typing import TYPE_CHECKING

import pandas as pd
import qtawesome as qta
import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from orta import config
from orta.gui import charts
from orta.gui.charts import maps, plotly_charts
from orta.gui.constants import (
    COMMANDS,
    FONT_FAMILY_MONO,
    FONT_SIZE_MONO,
    ThemeMode,
    palette,
)
from orta.gui.services.settings import Settings
from orta.gui.widgets.chart_view import ChartView
from orta.gui.widgets.config_form import ConfigForm
from orta.gui.widgets.data_table import DataTable
from orta.gui.widgets.filter_bar import FilterBar
from orta.gui.widgets.map_view import MapView
from orta.gui.widgets.yaml_editor import YamlEditor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from orta.gui.services.job_queue import Job, JobQueue

# Command -> (config model, example file name). Shared with PipelineView
# so both views resolve a command the same way.
CONFIG_MAP: dict[str, tuple[type[config.BaseRunConfig], str]] = {
    key: (model, f"{key}.yml") for key, (model, _help) in config.COMMANDS.items()
}


logger = logging.getLogger(__name__)


def template_dirs() -> list[pathlib.Path]:
    """Return the folders of config templates, in search order.

    A ``configs/`` folder (the working directory's, then the
    repository's) holds the user's own paths, so it wins over the
    repository's ``examples/``.
    """
    repo = pathlib.Path(__file__).resolve().parents[4]
    candidates = [pathlib.Path.cwd() / "configs", repo / "configs", repo / "examples"]
    chosen = Settings().template_dir
    if chosen:
        candidates.insert(0, pathlib.Path(chosen))
    return [c for c in dict.fromkeys(candidates) if c.is_dir()]


def load_example_config(filename: str) -> dict | None:
    """Return the first template called ``filename``, or None."""
    for folder in template_dirs():
        path = folder / filename
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        except (OSError, yaml.YAMLError):
            logger.warning("Could not read the template %s", path)
    return None


class AnalysisView(QWidget):
    """Split view: config editor (left) + results (right).

    Signals
    -------
    status_message(str)
        Request to update the status bar.
    job_started()
        A job has been launched.
    job_ended(bool)
        A job completed (True = success, False = failure).
    """

    status_message = Signal(str)
    job_started = Signal()
    job_ended = Signal(bool)
    config_changed = Signal()

    def __init__(self, queue: JobQueue, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._queue = queue
        self._current_command: str = "loading"
        self._form: ConfigForm | None = None
        self._mode = ThemeMode.LIGHT
        self._output_dir: pathlib.Path | None = None
        self._active_filters: dict = {}
        # Per-command config store. A project remembers *every*
        # command's config, not just the active one: the outgoing
        # command is folded in here on each switch (see
        # ``_capture_current``) and restored when the user returns to
        # it. Keyed by command.
        self._command_configs: dict[str, dict] = {}
        # When True, edits are programmatic (loading a project, syncing
        # panes) and must NOT be reported as user changes.
        self._suppress_change = False

        self._build_ui()
        self._connect_signals()
        self.select_command("loading")

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Top toolbar ──────────────────────────────────────────────
        toolbar = QFrame()
        toolbar.setProperty("class", "panel")
        toolbar.setFixedHeight(52)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(16, 8, 16, 8)
        tb_layout.setSpacing(8)

        tb_layout.addWidget(QLabel("Analysis:"))
        self._command_combo = QComboBox()
        for key, (label, icon, _desc) in COMMANDS.items():
            self._command_combo.addItem(qta.icon(icon, color="#5A6169"), label, key)
        self._command_combo.setMinimumWidth(200)
        tb_layout.addWidget(self._command_combo)

        tb_layout.addStretch()

        self._validate_btn = QPushButton("Validate")
        self._validate_btn.setProperty("class", "ghost")
        self._validate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        tb_layout.addWidget(self._validate_btn)

        self._load_btn = QPushButton("Load Results")
        self._load_btn.setProperty("class", "ghost")
        self._load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        tb_layout.addWidget(self._load_btn)

        self._run_btn = QPushButton("  Run ▶")
        self._run_btn.setProperty("class", "primary")
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.setMinimumWidth(100)
        tb_layout.addWidget(self._run_btn)

        main_layout.addWidget(toolbar)

        # ── Splitter: config (left) | results (right) ────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # LEFT: Config editor
        left_panel = QFrame()
        left_panel.setProperty("class", "panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._config_tabs = QTabWidget()
        self._form_page = QWidget()
        self._form_layout = QVBoxLayout(self._form_page)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        self._config_tabs.addTab(self._form_page, "Form")

        yaml_page = QWidget()
        yaml_layout = QVBoxLayout(yaml_page)
        yaml_layout.setContentsMargins(0, 0, 0, 0)
        yaml_layout.setSpacing(0)

        yaml_bar = QFrame()
        yb_layout = QHBoxLayout(yaml_bar)
        yb_layout.setContentsMargins(8, 4, 8, 4)
        self._form_to_yaml_btn = QPushButton("Form → YAML")
        self._form_to_yaml_btn.setProperty("class", "ghost")
        self._form_to_yaml_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        yb_layout.addWidget(self._form_to_yaml_btn)
        self._yaml_to_form_btn = QPushButton("YAML → Form")
        self._yaml_to_form_btn.setProperty("class", "ghost")
        self._yaml_to_form_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        yb_layout.addWidget(self._yaml_to_form_btn)
        yb_layout.addStretch()
        yaml_layout.addWidget(yaml_bar)

        self._yaml_editor = YamlEditor()
        yaml_layout.addWidget(self._yaml_editor)
        self._config_tabs.addTab(yaml_page, "YAML")

        left_layout.addWidget(self._config_tabs)
        splitter.addWidget(left_panel)

        # RIGHT: Results area
        right_panel = QFrame()
        right_panel.setProperty("class", "panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Dynamic filter bar above results tabs
        self._filter_bar = FilterBar()
        right_layout.addWidget(self._filter_bar)

        self._results_tabs = QTabWidget()

        # Log tab (always index 0)
        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        log_layout.setContentsMargins(0, 0, 0, 0)
        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont(FONT_FAMILY_MONO, FONT_SIZE_MONO))
        self._log_text.setPlaceholderText("Run an analysis or load results to see outputs here…")
        log_layout.addWidget(self._log_text)
        self._results_tabs.addTab(log_page, "Log")

        right_layout.addWidget(self._results_tabs)
        splitter.addWidget(right_panel)

        # Splitter proportions (40% config, 60% results)
        splitter.setSizes([450, 650])

        main_layout.addWidget(splitter, stretch=1)

    def _connect_signals(self) -> None:
        self._command_combo.currentIndexChanged.connect(self._on_command_changed)
        self._run_btn.clicked.connect(self._run)
        self._load_btn.clicked.connect(self._load_existing)
        self._validate_btn.clicked.connect(self._validate)
        self._form_to_yaml_btn.clicked.connect(self._sync_form_to_yaml)
        self._yaml_to_form_btn.clicked.connect(self._sync_yaml_to_form)
        self._filter_bar.filter_changed.connect(self._on_filter_changed)
        self._yaml_editor.textChanged.connect(self._on_config_edited)

    # ── command switching ────────────────────────────────────────────

    def select_command(self, key: str) -> None:
        """Programmatically switch to a command."""
        idx = self._command_combo.findData(key)
        if idx >= 0:
            if self._command_combo.currentIndex() == idx:
                self._rebuild_form()
            else:
                self._command_combo.setCurrentIndex(idx)

    def _on_command_changed(self, _index: int) -> None:
        key = self._command_combo.currentData()
        if key:
            # Preserve the command we're leaving so its edits survive
            # the switch.
            self._capture_current()
            self._current_command = key
            self._filter_bar.set_command(key)
            self._rebuild_form()
            self._clear_results()

    def _rebuild_form(self) -> None:
        if self._current_command not in CONFIG_MAP:
            return
        model_cls, filename = CONFIG_MAP[self._current_command]

        # Remove old form and scroll wrapper
        if hasattr(self, "_form_scroll") and self._form_scroll is not None:
            self._form_layout.removeWidget(self._form_scroll)
            # Hidden now: deleteLater only acts once control returns to
            # the event loop, and until then the old form would show.
            self._form_scroll.hide()
            self._form_scroll.deleteLater()
            self._form_scroll = None
            self._form = None
        elif self._form is not None:
            self._form_layout.removeWidget(self._form)
            self._form.deleteLater()
            self._form = None

        self._form = ConfigForm(model_cls)
        self._form.changed.connect(self._on_config_edited)

        # Wrap top-level form in a QScrollArea
        self._form_scroll = QScrollArea()
        self._form_scroll.setWidgetResizable(True)
        self._form_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._form_scroll.setWidget(self._form)

        self._form_layout.addWidget(self._form_scroll)

        # Pre-fill, which is not a user edit: the value this project
        # remembered for the command, else the template.
        initial = self._command_configs.get(self._current_command)
        if initial is None:
            initial = self._load_example(filename)
        with self._programmatic():
            if initial:
                self._form.set_data(initial)
            self._sync_form_to_yaml()

    def _load_example(self, filename: str) -> dict | None:
        return load_example_config(filename)

    # ── change tracking ──────────────────────────────────────────────

    @contextlib.contextmanager
    def _programmatic(self) -> Iterator[None]:
        """Hold back ``config_changed`` during programmatic edits."""
        prev = self._suppress_change
        self._suppress_change = True
        try:
            yield
        finally:
            self._suppress_change = prev

    def _on_config_edited(self) -> None:
        """Emit ``config_changed`` for genuine user edits only."""
        if not self._suppress_change:
            self.config_changed.emit()

    def _capture_current(self) -> None:
        """Save the active command's config in the store.

        A no-op during programmatic edits (e.g. while :meth:`load_all`
        is installing a project) so a freshly-loaded store is never
        clobbered by the config that happened to be on screen
        beforehand.
        """
        if self._suppress_change:
            return
        if self._current_command not in CONFIG_MAP or self._form is None:
            return
        # Invalid YAML in the editor keeps whatever was stored before.
        with contextlib.suppress(yaml.YAMLError):
            self._command_configs[self._current_command] = self._current_data()

    def capture_all(self) -> dict[str, dict]:
        """Every command's config, the active one included.

        Folds the currently-displayed command into the store first, so a
        save persists edits the user has not yet switched away from.
        """
        self._capture_current()
        return {cmd: dict(data) for cmd, data in self._command_configs.items()}

    def load_all(self, configs: dict[str, dict], active: str) -> None:
        """Replace the per-command config store and show *active*.

        Used when opening a project (restores every remembered command)
        and when starting a new one (pass ``{}`` to reset the store to
        examples).
        """
        self._command_configs = {cmd: dict(data) for cmd, data in configs.items()}
        with self._programmatic():
            self.select_command(active)

    # ── YAML ↔ form sync ─────────────────────────────────────────────

    def _sync_form_to_yaml(self) -> None:
        if self._form:
            with self._programmatic():
                self._yaml_editor.set_yaml(self._form.to_yaml())
            self._on_config_edited()

    def _sync_yaml_to_form(self) -> None:
        if not self._form:
            return
        try:
            data = yaml.safe_load(self._yaml_editor.get_yaml()) or {}
        except yaml.YAMLError as exc:
            QMessageBox.warning(self, "Invalid YAML", str(exc))
            return
        with self._programmatic():
            self._form.set_data(data)
        self._on_config_edited()

    def _current_data(self) -> dict:
        if self._config_tabs.currentIndex() == 1:  # YAML tab active
            return yaml.safe_load(self._yaml_editor.get_yaml()) or {}
        return self._form.get_data() if self._form else {}

    # ── validation ───────────────────────────────────────────────────

    def _validate(self) -> None:
        if self._current_command not in CONFIG_MAP:
            return
        model_cls, _ = CONFIG_MAP[self._current_command]
        try:
            data = self._current_data()
            model_cls.model_validate(data)
            QMessageBox.information(self, "Config Valid", "✅ Configuration is valid.")
            self.status_message.emit("Config valid")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Validation Error", str(exc))
            self.status_message.emit("Config invalid")

    # ── running & loading ────────────────────────────────────────────

    def _resolve_output_dir(self) -> pathlib.Path | None:
        data = self._current_data()
        output = data.get("output_path")
        if not output:
            QMessageBox.critical(self, "No output path", "Set 'output_path' before executing.")
            return None
        path = pathlib.Path(output)
        path.mkdir(exist_ok=True, parents=True)
        return path

    def _run(self) -> None:
        if self._current_command not in CONFIG_MAP:
            return
        model_cls, _ = CONFIG_MAP[self._current_command]
        try:
            data = self._current_data()
        except yaml.YAMLError as exc:
            QMessageBox.warning(self, "Invalid YAML", str(exc))
            return
        try:
            config = model_cls.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Invalid Config", str(exc))
            return

        self._output_dir = pathlib.Path(config.output_path)
        self._active_filters = {}

        self._log_text.clear()
        self._clear_results()
        self._append_log(f"Submitting '{self._current_command}' to queue…\n")
        self._run_btn.setEnabled(False)
        self._run_btn.setText("  Running…")
        self.status_message.emit(f"Queued {self._current_command}…")
        self.job_started.emit()
        self._results_tabs.setCurrentIndex(0)  # switch to log

        # Submit to central job queue
        label = COMMANDS.get(self._current_command, ("", "", ""))[0]
        job = self._queue.submit(self._current_command, label, config.run)

        # Wire individual job signals to append logs to our text pane
        job.log_received.connect(self._append_log)
        job.status_changed.connect(lambda status, j=job: self._on_job_status_changed(j, status))

    def _on_job_status_changed(self, job: Job, status: str) -> None:
        if status == "running":
            self.status_message.emit(f"Running {job.label}…")
        elif status == "done":
            self._append_log(f"\n✅ Done in {job.elapsed:.1f}s.\n")
            self._run_btn.setEnabled(True)
            self._run_btn.setText("  Run ▶")
            self.status_message.emit(f"{job.label} completed successfully.")
            self.job_ended.emit(True)
            self._populate_filters()
            self._render_outputs()
        elif status == "error":
            self._append_log(f"\n❌ ERROR: {job.error_message}\n")
            self._run_btn.setEnabled(True)
            self._run_btn.setText("  Run ▶")
            self.status_message.emit(f"{job.label} failed.")
            self.job_ended.emit(False)
            QMessageBox.critical(self, "Run Failed", job.error_message)
        elif status == "cancelled":
            self._append_log("\n❌ Job cancelled by user.\n")
            self._run_btn.setEnabled(True)
            self._run_btn.setText("  Run ▶")
            self.status_message.emit("Job cancelled.")
            self.job_ended.emit(False)

    def _load_existing(self) -> None:
        path = self._resolve_output_dir()
        if not path or not path.exists():
            return
        self._output_dir = path
        self._active_filters = {}
        self._clear_results()
        self._append_log(f"\nLoading existing results from {self._output_dir}…\n")
        self._populate_filters()
        count = self._render_outputs()
        if count == 0:
            QMessageBox.information(
                self,
                "No results",
                "Nothing to show in the output folder yet. Run the analysis first.",
            )

    # ── dynamic results rendering ────────────────────────────────────

    def _clear_results(self) -> None:
        """Clear all results tabs (index >= 1)."""
        while self._results_tabs.count() > 1:
            widget = self._results_tabs.widget(1)
            self._results_tabs.removeTab(1)
            widget.deleteLater()

    def _populate_filters(self) -> None:
        """Fill the filter bar from the output folder."""
        if not self._output_dir or not self._output_dir.exists():
            return

        if self._current_command in ("services", "lookup", "fares"):
            pair_finders = {
                "services": plotly_charts.available_services_pairs,
                "lookup": plotly_charts.available_lookup_pairs,
                "fares": plotly_charts.available_fares_pairs,
            }
            pairs = pair_finders[self._current_command](self._output_dir)
            self._filter_bar.set_available_pairs(pairs)
        elif self._current_command == "loading":
            ops = plotly_charts.available_loading_operators(self._output_dir)
            self._filter_bar.set_available_operators(ops)
        elif self._current_command == "propose":
            cands = plotly_charts.available_propose_candidates(self._output_dir)
            self._filter_bar.set_available_candidates(cands)

    def _on_filter_changed(self, filters: dict) -> None:
        self._active_filters = filters
        self._render_outputs()

    def _render_outputs(self) -> int:
        """Show the outputs as charts, tables and maps."""
        if not self._output_dir or not self._output_dir.exists():
            return 0

        self._clear_results()
        count = 0

        tag = self._active_filters.get("tag")
        operator = self._active_filters.get("operator")
        candidate = self._active_filters.get("candidate")

        # Timetable fixes for a proposed service come first: they say
        # what to change before the numbers are worth reading.
        if self._current_command == "propose":
            fixes_text = plotly_charts.propose_fix_summary(self._output_dir, candidate)
            if fixes_text:
                p = palette(self._mode)
                fixes_page = QWidget()
                fl = QVBoxLayout(fixes_page)
                fl.setContentsMargins(8, 8, 8, 8)
                lbl = QLabel("The timetable checks found problems; each has a suggested fix.")
                lbl.setStyleSheet(f"color: {p.error}; font-weight: 600;")
                fl.addWidget(lbl)
                txt = QPlainTextEdit()
                txt.setReadOnly(True)
                txt.setPlainText(fixes_text)
                txt.setFont(QFont(FONT_FAMILY_MONO, FONT_SIZE_MONO))
                fl.addWidget(txt)
                self._results_tabs.addTab(fixes_page, "Fixes")
                count += 1

        stations = maps.find_stations(self._current_data(), *self.capture_all().values())
        figures = charts.build_figures(
            self._current_command,
            self._output_dir,
            tag=tag,
            operator=operator,
            candidate=candidate,
            theme_mode=self._mode,
            stations=stations,
        )
        if figures:
            self._results_tabs.addTab(self._chart_browser(figures), "Charts")
            count += len(figures)

        # ── 3. Render Data Tables Tab (CSVs) ──────────────────────────
        # Skip hidden files, such as the "._" copies macOS leaves on
        # external drives.
        csv_files = sorted(p for p in self._output_dir.glob("*.csv") if not p.name.startswith("."))
        if csv_files:
            data_container = QWidget()
            data_layout = QVBoxLayout(data_container)
            data_layout.setContentsMargins(12, 12, 12, 12)
            data_layout.setSpacing(8)

            # Dropdown selector row
            sel_row = QHBoxLayout()
            sel_row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel("Table")
            lbl.setStyleSheet("font-weight: 600;")
            sel_row.addWidget(lbl)

            data_combo = QComboBox()
            sel_row.addWidget(data_combo, stretch=1)
            sel_row.addStretch(2)
            data_layout.addLayout(sel_row)

            # Stacked widget for data tables
            data_stack = QStackedWidget()
            data_layout.addWidget(data_stack)

            added_count = 0
            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        tbl = DataTable()
                        tbl.set_data(df)
                        data_stack.addWidget(tbl)
                        data_combo.addItem(csv_path.name)
                        added_count += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not read CSV %s: %s", csv_path.name, exc)

            if added_count > 0:
                data_combo.currentIndexChanged.connect(data_stack.setCurrentIndex)
                self._results_tabs.addTab(data_container, "Data")
                count += added_count

        # ── 4. Render Maps Tab ────────────────────────────────────────
        html_maps = plotly_charts.find_html_maps(self._output_dir)
        if html_maps:
            m_view = MapView()
            m_view.set_maps(html_maps)
            self._results_tabs.addTab(m_view, "Maps")
            count += len(html_maps)

        # Switch to the first result tab (index 1) if anything was
        # rendered
        if count > 0:
            self._results_tabs.setCurrentIndex(1)

        self.status_message.emit(
            f"Results loaded: {len(figures)} charts and maps"
            + (f", and {len(html_maps)} saved map pages." if html_maps else ".")
        )
        return count

    def _chart_browser(self, figures: list) -> QWidget:
        """Build a list of chart titles beside one chart drawn on pick.

        Drawing only the chart on show keeps a command with twenty
        charts and maps as quick to open as one with two.
        """
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        titles = QListWidget()
        titles.setFixedWidth(250)
        titles.setStyleSheet("QListWidget::item { padding: 6px 6px; }")
        view = ChartView()
        for title, _fig in figures:
            item = QListWidgetItem(title)
            if title.startswith("Map:"):
                item.setIcon(qta.icon("fa5s.map", color=palette(self._mode).text_secondary))
            titles.addItem(item)
        titles.currentRowChanged.connect(
            lambda row: view.set_figure(figures[row][1], self._mode) if row >= 0 else None
        )
        layout.addWidget(titles)
        layout.addWidget(view, stretch=1)
        titles.setCurrentRow(0)
        return page

    # ── log helpers ───────────────────────────────────────────────────

    def _append_log(self, text: str) -> None:
        self._log_text.appendPlainText(text)
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── theme ────────────────────────────────────────────────────────

    def update_theme(self, mode: ThemeMode) -> None:
        """Refresh theme-dependent elements."""
        self._mode = mode
        self._yaml_editor.update_theme(mode)

        # Update command combo icons
        p = palette(mode)
        for i in range(self._command_combo.count()):
            key = self._command_combo.itemData(i)
            if key in COMMANDS:
                _, icon_name, _ = COMMANDS[key]
                self._command_combo.setItemIcon(i, qta.icon(icon_name, color=p.text_secondary))

        # Re-render active outputs so charts align with the theme
        if self._output_dir:
            self._render_outputs()
