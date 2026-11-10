"""Virtual-scrolling data table widget for large pandas DataFrames.

Uses QAbstractTableModel to display pandas data frames in a QTableView
efficiently without copying data, keeping memory usage low and rendering
instant even for datasets with millions of rows.  Includes search
filtering and exporting.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ortla.gui.constants import SPACE_MD, SPACE_SM


class _DataFrameModel(QAbstractTableModel):
    """Bridge between a pandas DataFrame and QTableView."""

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def rowCount(self, _parent: QModelIndex | None = None) -> int:  # noqa: N802
        return len(self._df)

    def columnCount(self, _parent: QModelIndex | None = None) -> int:  # noqa: N802
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            val = self._df.iloc[index.row(), index.column()]
            if pd.isna(val):
                return ""
            if isinstance(val, float):
                return f"{val:.4f}" if val.round(4) != val.round(0) else f"{val:.0f}"
            return str(val)

        return None

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        if orientation == Qt.Orientation.Vertical:
            return str(section + 1)

        return None


class DataTable(QWidget):
    """High-performance virtual-scrolling table view for CSV data."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._filtered_df = pd.DataFrame()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(SPACE_SM)

        # Search box and export button.
        toolbar = QHBoxLayout()
        toolbar.setSpacing(SPACE_SM)

        toolbar.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Filter rows…")
        self._search_input.textChanged.connect(self._on_search_changed)
        toolbar.addWidget(self._search_input)

        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.setProperty("class", "ghost")
        self._export_btn.clicked.connect(self._export_csv)
        toolbar.addWidget(self._export_btn)

        layout.addLayout(toolbar)

        # Table View
        self._view = QTableView(self)
        self._view.setAlternatingRowColors(True)
        self._view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._view.horizontalHeader().setDefaultSectionSize(120)
        self._view.verticalHeader().setDefaultSectionSize(26)
        layout.addWidget(self._view)

    def set_data(self, df: pd.DataFrame) -> None:
        """Load a DataFrame into the virtual table."""
        self._df = df
        self._filtered_df = df
        self._update_model()

    def _update_model(self) -> None:
        model = _DataFrameModel(self._filtered_df)
        self._view.setModel(model)

    def _on_search_changed(self, query: str) -> None:
        if not query.strip():
            self._filtered_df = self._df
        else:
            # Simple substring matching across all stringified columns
            mask = (
                self._df.astype(str)
                .apply(lambda x: x.str.contains(query, case=False, na=False))
                .any(axis=1)
            )
            self._filtered_df = self._df[mask]

        self._update_model()

    def _export_csv(self) -> None:
        from PySide6.QtWidgets import QFileDialog  # noqa: PLC0415

        path, _ = QFileDialog.getSaveFileName(self, "Export Data", "", "CSV Files (*.csv)")
        if path:
            try:
                self._filtered_df.to_csv(path, index=False)
            except OSError as exc:
                from PySide6.QtWidgets import QMessageBox  # noqa: PLC0415

                QMessageBox.critical(self, "Export Failed", str(exc))
