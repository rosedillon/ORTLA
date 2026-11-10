"""Preferences / Settings dialog.

Allows configuring user preferences such as theme, default output paths,
startup behaviour, and importing/exporting the preference file.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ortla.gui.constants import COMMANDS, SPACE_MD, SPACE_SM

if TYPE_CHECKING:
    from ortla.gui.services.settings import Settings


class SettingsDialog(QDialog):
    """Application preferences dialog."""

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)
        self.setModal(True)

        self._settings = settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(SPACE_MD)

        # ── Appearance ────────────────────────────────────────────────
        layout.addWidget(self._section_label("APPEARANCE"))
        appearance = QFormLayout()
        appearance.setSpacing(SPACE_SM)

        self._cmb_theme = QComboBox()
        self._cmb_theme.addItem("System Default", "system")
        self._cmb_theme.addItem("Light Mode", "light")
        self._cmb_theme.addItem("Dark Mode", "dark")
        self._select_data(self._cmb_theme, self._settings.theme)
        appearance.addRow("Theme:", self._cmb_theme)
        layout.addLayout(appearance)

        # ── Defaults ──────────────────────────────────────────────────
        layout.addWidget(self._section_label("DEFAULTS"))
        defaults = QFormLayout()
        defaults.setSpacing(SPACE_SM)

        # Default output directory
        self._dir_container = QWidget()
        dir_layout = QHBoxLayout(self._dir_container)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.setSpacing(4)
        self._inp_output_dir = QLineEdit(self._settings.default_output_dir)
        self._inp_output_dir.setPlaceholderText("Folder where results are written…")
        dir_layout.addWidget(self._inp_output_dir)
        btn_browse = QPushButton("Browse")
        btn_browse.setProperty("class", "ghost")
        btn_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(btn_browse)
        defaults.addRow("Default output folder:", self._dir_container)

        # Default analysis opened for a new project
        self._cmb_analysis = QComboBox()
        for key, (label, _icon, _desc) in COMMANDS.items():
            self._cmb_analysis.addItem(label, key)
        self._select_data(self._cmb_analysis, self._settings.default_analysis)
        defaults.addRow("Default analysis:", self._cmb_analysis)
        layout.addLayout(defaults)

        # ── Startup & saving ──────────────────────────────────────────
        layout.addWidget(self._section_label("STARTUP & SAVING"))
        toggles = QVBoxLayout()
        toggles.setSpacing(SPACE_SM)

        self._chk_autosave = QCheckBox("Autosave the project after each run")
        self._chk_autosave.setChecked(self._settings.autosave)
        toggles.addWidget(self._chk_autosave)

        self._chk_confirm_exit = QCheckBox("Warn about unsaved changes before exit")
        self._chk_confirm_exit.setChecked(self._settings.confirm_on_exit)
        toggles.addWidget(self._chk_confirm_exit)

        self._chk_restore = QCheckBox("Reopen the last project on startup")
        self._chk_restore.setChecked(self._settings.restore_last_session)
        toggles.addWidget(self._chk_restore)
        layout.addLayout(toggles)

        # ── Maintenance (save/load preferences, clear history) ────────
        layout.addWidget(self._section_label("PREFERENCES FILE"))
        maint = QHBoxLayout()
        maint.setSpacing(SPACE_SM)

        btn_import = QPushButton("Load…")
        btn_import.setProperty("class", "ghost")
        btn_import.setToolTip("Load preferences from a file")
        btn_import.clicked.connect(self._import_settings)
        maint.addWidget(btn_import)

        btn_export = QPushButton("Save As…")
        btn_export.setProperty("class", "ghost")
        btn_export.setToolTip("Save these preferences to a file to share or back up")
        btn_export.clicked.connect(self._export_settings)
        maint.addWidget(btn_export)

        btn_reset = QPushButton("Reset to defaults")
        btn_reset.setProperty("class", "ghost")
        btn_reset.clicked.connect(self._reset_settings)
        maint.addWidget(btn_reset)

        btn_clear_history = QPushButton("Clear recent projects")
        btn_clear_history.setProperty("class", "ghost")
        btn_clear_history.clicked.connect(self._clear_history)
        maint.addWidget(btn_clear_history)
        maint.addStretch()
        layout.addLayout(maint)

        # ── Action buttons ────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

        buttons = QHBoxLayout()
        buttons.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(btn_cancel)
        btn_save = QPushButton("Save Preferences")
        btn_save.setProperty("class", "primary")
        btn_save.clicked.connect(self._save_settings)
        buttons.addWidget(btn_save)
        layout.addLayout(buttons)

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("class", "section")
        return lbl

    @staticmethod
    def _select_data(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Default Output Directory")
        if path:
            self._inp_output_dir.setText(path)

    def _clear_history(self) -> None:
        self._settings.recent_projects = []
        self._settings.save()
        QMessageBox.information(self, "History Cleared", "Project history has been cleared.")

    def _export_settings(self) -> None:
        # Commit the on-screen values first so what you save is what you
        # see.
        self._collect_into_settings()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preferences", "ortla-settings.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self._settings.export_to(pathlib.Path(path))
            QMessageBox.information(self, "Preferences Saved", f"Preferences written to:\n{path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Failed", f"Could not save preferences: {exc}")

    def _import_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Preferences", "", "JSON (*.json)")
        if not path:
            return
        try:
            self._settings.import_from(pathlib.Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Failed", f"Could not load preferences: {exc}")
            return
        self._refresh_from_settings()
        QMessageBox.information(self, "Preferences Loaded", "Preferences loaded successfully.")

    def _reset_settings(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset Preferences",
            "Restore all preferences to their defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._settings.reset()
        self._refresh_from_settings()

    def _refresh_from_settings(self) -> None:
        """Push current settings values back into the widgets."""
        self._select_data(self._cmb_theme, self._settings.theme)
        self._select_data(self._cmb_analysis, self._settings.default_analysis)
        self._inp_output_dir.setText(self._settings.default_output_dir)
        self._chk_autosave.setChecked(self._settings.autosave)
        self._chk_confirm_exit.setChecked(self._settings.confirm_on_exit)
        self._chk_restore.setChecked(self._settings.restore_last_session)

    def _collect_into_settings(self) -> None:
        """Copy the dialog's values into the settings, unsaved."""
        self._settings.theme = self._cmb_theme.currentData()
        self._settings.default_output_dir = self._inp_output_dir.text().strip()
        self._settings.default_analysis = self._cmb_analysis.currentData()
        self._settings.autosave = self._chk_autosave.isChecked()
        self._settings.confirm_on_exit = self._chk_confirm_exit.isChecked()
        self._settings.restore_last_session = self._chk_restore.isChecked()

    def _save_settings(self) -> None:
        self._collect_into_settings()
        self._settings.save()
        self.accept()
