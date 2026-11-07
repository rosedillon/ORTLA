"""The Quick Start page: from a new machine to a first run.

It sets up a data folder (the subfolders every template expects, and a
copy of each template pointing at it) and has one card per data source
saying where to download it and where to save it.
"""

from __future__ import annotations

import pathlib

import qtawesome as qta
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from orta.gui.constants import (
    FONT_SIZE_H1,
    FONT_SIZE_H3,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
)
from orta.gui.services import scaffold
from orta.gui.services.settings import Settings

_GRID_COLUMNS = 2


class _SourceCard(QFrame):
    """One data source: what it is, where it goes, where to get it."""

    def __init__(self, source: scaffold.DataSource, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source = source
        self.setProperty("class", "card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(4)

        title = QLabel(source.label)
        title.setFont(QFont("Inter", FONT_SIZE_H3, QFont.Weight.DemiBold))
        layout.addWidget(title)

        desc = QLabel(source.description)
        desc.setProperty("class", "caption")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        used_by = QLabel("Used by: " + ", ".join(source.used_by))
        used_by.setProperty("class", "caption")
        used_by.setWordWrap(True)
        layout.addWidget(used_by)

        self._path_label = QLabel(f"Save as: {source.relative_path}")
        self._path_label.setProperty("class", "caption")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(self._path_label)

        layout.addStretch(1)

        btn_row = QHBoxLayout()
        self._download_btn = QPushButton("  Download")
        self._download_btn.setIcon(qta.icon("fa5s.external-link-alt", color="#FFFFFF"))
        self._download_btn.setProperty("class", "primary")
        self._download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._download_btn.clicked.connect(self._on_download_clicked)
        btn_row.addWidget(self._download_btn)

        self._open_folder_btn = QPushButton("Open Folder")
        self._open_folder_btn.setProperty("class", "ghost")
        self._open_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._open_folder_btn.setVisible(False)
        self._open_folder_btn.clicked.connect(self._on_open_folder_clicked)
        btn_row.addWidget(self._open_folder_btn)

        layout.addLayout(btn_row)
        self._target: pathlib.Path | None = None

    def _on_download_clicked(self) -> None:
        url = self._source.download_url
        if not url:
            save_as = self._path_label.text().removeprefix("Save as: ").removeprefix("Save to: ")
            QMessageBox.information(
                self,
                "Link Not Set Up Yet",
                f"No download link has been added for '{self._source.label}' yet.\n\n"
                f"Once you have the file, save it as:\n{save_as}",
            )
            return
        QDesktopServices.openUrl(QUrl(url))

    def _on_open_folder_clicked(self) -> None:
        if self._target is not None:
            folder = self._target if self._source.is_dir else self._target.parent
            folder.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def set_base_dir(self, base_dir: pathlib.Path | None) -> None:
        """Show where to save the file in the chosen folder."""
        if base_dir is None:
            self._target = None
            self._path_label.setText(f"Save as: {self._source.relative_path}")
            self._open_folder_btn.setVisible(False)
            return
        self._target = scaffold.target_path(base_dir, self._source.key)
        self._path_label.setText(f"Save to: {self._target}")
        self._open_folder_btn.setVisible(True)


class QuickStartPanel(QScrollArea):
    """The Quick Start page."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._base_dir: pathlib.Path | None = None
        self._cards: list[_SourceCard] = []

        container = QFrame()
        container.setProperty("class", "panel")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(SPACE_XL, SPACE_XL, SPACE_XL, SPACE_XL)
        layout.setSpacing(SPACE_XL)

        # ── Intro ────────────────────────────────────────────────────
        intro = QFrame()
        intro.setProperty("class", "card")
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(SPACE_LG, SPACE_LG, SPACE_LG, SPACE_LG)
        intro_layout.setSpacing(SPACE_SM)

        title = QLabel("Quick Start Guide")
        title.setFont(QFont("Inter", FONT_SIZE_H1, QFont.Weight.Bold))
        intro_layout.addWidget(title)

        desc = QLabel(
            "Welcome to ORTA, demand-modelling software for the UK rail network. "
            "Three steps to your first run:"
        )
        desc.setWordWrap(True)
        intro_layout.addWidget(desc)

        steps = QLabel(
            "1. Set up a data folder below (creates the right subfolders automatically).\n"
            "2. Download each data source into the folder shown on its card.\n"
            "3. Open an analysis from the Dashboard, check the pre-filled paths, and click "
            "Run."
        )
        steps.setWordWrap(True)
        intro_layout.addWidget(steps)
        layout.addWidget(intro)

        # ── Step 1: folder wizard ────────────────────────────────────
        wizard = QFrame()
        wizard.setProperty("class", "card")
        wizard_layout = QVBoxLayout(wizard)
        wizard_layout.setContentsMargins(SPACE_LG, SPACE_LG, SPACE_LG, SPACE_LG)
        wizard_layout.setSpacing(SPACE_SM)

        wtitle = QLabel("Set Up Data Folder Structure")
        wtitle.setFont(QFont("Inter", FONT_SIZE_H3, QFont.Weight.DemiBold))
        wizard_layout.addWidget(wtitle)

        wdesc = QLabel(
            "Choose (or create) one folder on your computer. We'll build the subfolders "
            "every analysis expects inside it, and write ready-to-use example configs "
            "pointing at them."
        )
        wdesc.setWordWrap(True)
        wdesc.setProperty("class", "caption")
        wizard_layout.addWidget(wdesc)

        wbtn_row = QHBoxLayout()
        self._wizard_btn = QPushButton("  Choose Folder && Set Up ▶")
        self._wizard_btn.setIcon(qta.icon("fa5s.folder-plus", color="#FFFFFF"))
        self._wizard_btn.setProperty("class", "primary")
        self._wizard_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._wizard_btn.clicked.connect(self._run_wizard)
        wbtn_row.addWidget(self._wizard_btn)
        wbtn_row.addStretch()
        wizard_layout.addLayout(wbtn_row)

        self._wizard_status = QLabel("No data folder chosen yet.")
        self._wizard_status.setProperty("class", "caption")
        self._wizard_status.setWordWrap(True)
        wizard_layout.addWidget(self._wizard_status)

        layout.addWidget(wizard)

        # ── Step 2: downloads ────────────────────────────────────────
        section_label = QLabel("DOWNLOAD YOUR DATA")
        section_label.setProperty("class", "section")
        layout.addWidget(section_label)

        grid = QGridLayout()
        grid.setSpacing(SPACE_MD)
        col, row = 0, 0
        for source in scaffold.DATA_SOURCES:
            card = _SourceCard(source)
            self._cards.append(card)
            grid.addWidget(card, row, col)
            col += 1
            if col >= _GRID_COLUMNS:
                col = 0
                row += 1
        layout.addLayout(grid)

        layout.addStretch(1)
        self.setWidget(container)

    # ── wizard ────────────────────────────────────────────────────────

    def _run_wizard(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose a data folder")
        if not chosen:
            return
        base_dir = pathlib.Path(chosen)

        examples = pathlib.Path(__file__).resolve().parents[4] / "examples"
        if not examples.is_dir():
            QMessageBox.critical(
                self, "Setup failed", f"Could not find the config templates in {examples}."
            )
            return

        try:
            created = scaffold.create_structure(base_dir)
            written = scaffold.write_configs(base_dir, examples)
        except OSError as exc:
            QMessageBox.critical(self, "Setup failed", f"Could not set up the folder: {exc}")
            return

        self._base_dir = base_dir
        for card in self._cards:
            card.set_base_dir(base_dir)
        self._wizard_status.setText(f"Data folder ready: {base_dir}")

        settings = Settings()
        settings.template_dir = str(base_dir / "configs")
        settings.save()
        QMessageBox.information(
            self,
            "Data folder ready",
            f"Created {len(created)} folder(s) under {base_dir}.\n\n"
            f"Wrote {len(written)} config(s) to {base_dir / 'configs'}. Forms now "
            "open with these paths filled in.\n\n"
            "Each card below shows where to save its download.",
        )
