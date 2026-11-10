"""Spotlight Search / Command Palette dialog.

Enables universal search and fast navigation (Ctrl+K) across all views,
analyses, settings options, and documentation topics.
"""

from __future__ import annotations

import qtawesome as qta
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ortla.gui.constants import COMMANDS, SPACE_MD, SPACE_SM


class SearchDialog(QDialog):
    """spotlight search dialog.

    Signals
    -------
    action_triggered(str, str)
        Emitted when an item is selected. Payload is (category, key).
    """

    action_triggered = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(500, 300)

        # Style the card frame
        self.setStyleSheet("""
            SearchDialog {
                background: transparent;
            }
            #container {
                background-color: #22243A;
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 8px;
            }
            QLineEdit {
                background-color: #1A1B2E;
                color: #E4E7F0;
                border: 1px solid #454868;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 14px;
            }
            QListWidget {
                background-color: transparent;
                border: none;
                color: #CDD6F4;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: #2E3150;
            }
            QListWidget::item:selected {
                background-color: #3A3E60;
                color: #FFFFFF;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        container.setObjectName("container")
        container_l = QVBoxLayout(container)
        container_l.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        container_l.setSpacing(SPACE_SM)

        self._inp_search = QLineEdit()
        self._inp_search.setPlaceholderText("Search commands, views, documentation…")
        self._inp_search.textChanged.connect(self._filter_list)
        self._inp_search.installEventFilter(self)
        container_l.addWidget(self._inp_search)

        self._list_widget = QListWidget()
        self._list_widget.itemActivated.connect(self._on_item_selected)
        container_l.addWidget(self._list_widget)

        layout.addWidget(container)

        self._items: list[tuple[str, str, str, str]] = []  # (label, category, key, icon)
        self._build_database()
        self._filter_list("")

    def _build_database(self) -> None:
        """Register all searchable command palette terms."""
        # 1. Analyses
        for key, (label, icon, _desc) in COMMANDS.items():
            self._items.append((f"Run: {label}", "analysis", key, icon))

        # 2. Views
        self._items.append(("Go to Dashboard", "view", "dashboard", "fa5s.home"))
        self._items.append(("Go to Pipeline Wizard", "view", "pipeline", "fa5s.network-wired"))
        self._items.append(("Go to Job Queue", "view", "jobs", "fa5s.tasks"))
        self._items.append(("Go to Interactive Maps", "view", "maps", "fa5s.map-marked-alt"))

        # 3. Preferences / Actions
        self._items.append(("Preferences: Settings Dialog", "action", "settings", "fa5s.cog"))
        self._items.append(("Preferences: Toggle Theme", "action", "theme", "fa5s.adjust"))

    def _filter_list(self, query: str) -> None:
        self._list_widget.clear()
        query = query.lower().strip()

        for label, cat, key, icon in self._items:
            if not query or query in label.lower() or query in cat.lower():
                item = QListWidgetItem(label)
                item.setIcon(qta.icon(icon, color="#8FB3B0"))
                item.setData(Qt.ItemDataRole.UserRole, (cat, key))
                self._list_widget.addItem(item)

        if self._list_widget.count() > 0:
            self._list_widget.setCurrentRow(0)

    def _on_item_selected(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            cat, key = data
            self.action_triggered.emit(cat, key)
        self.accept()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        """Handle keyboard navigation inside input field."""
        if obj is self._inp_search and event.type() == event.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Down:
                row = (self._list_widget.currentRow() + 1) % self._list_widget.count()
                self._list_widget.setCurrentRow(row)
                return True
            if key == Qt.Key.Key_Up:
                row = (
                    self._list_widget.currentRow() - 1 + self._list_widget.count()
                ) % self._list_widget.count()
                self._list_widget.setCurrentRow(row)
                return True
            if key == Qt.Key.Key_Return:
                curr = self._list_widget.currentItem()
                if curr:
                    self._on_item_selected(curr)
                return True
            if key == Qt.Key.Key_Escape:
                self.reject()
                return True

        return super().eventFilter(obj, event)
