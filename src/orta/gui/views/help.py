"""The Help view: Quick Start and a page per command.

The Quick Start entry is a live panel (data folder setup and download
links). Every other entry is a page from
:mod:`orta.gui.docs`, written from the command's config
model.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from orta.gui.constants import COMMANDS, SPACE_LG, SPACE_SM
from orta.gui.docs import command_page
from orta.gui.views.quickstart import QuickStartPanel

QUICK_START_KEY = "Quick start"


class HelpView(QWidget):
    """Embedded tutorial and analysis documentation browser."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(1)

        # Left list panel. An opaque QFrame (not a plain QWidget, which
        # the global stylesheet leaves transparent) so it always shows a
        # real background instead of the raw window buffer bleeding
        # through.
        self._list_panel = QFrame()
        self._list_panel.setProperty("class", "panel")
        list_layout = QVBoxLayout(self._list_panel)
        list_layout.setContentsMargins(SPACE_LG, SPACE_LG, SPACE_LG, SPACE_LG)
        list_layout.setSpacing(SPACE_SM)

        title = QLabel("User guide")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        list_layout.addWidget(title)

        self._doc_list = QListWidget()
        self._doc_list.currentRowChanged.connect(self._on_doc_selected)
        list_layout.addWidget(self._doc_list)

        self._splitter.addWidget(self._list_panel)

        # Right panel: live Quick Start widget, or a static doc browser.
        self._right_stack = QStackedWidget()

        self._quick_start = QuickStartPanel()
        self._right_stack.addWidget(self._quick_start)  # index 0

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        self._right_stack.addWidget(self._browser)  # index 1

        self._splitter.addWidget(self._right_stack)

        self._splitter.setSizes([200, 600])
        layout.addWidget(self._splitter)

        self._populate_list()

    def _populate_list(self) -> None:
        self._keys = [QUICK_START_KEY, *COMMANDS]
        for key in self._keys:
            label = key if key == QUICK_START_KEY else COMMANDS[key][0]
            self._doc_list.addItem(QListWidgetItem(label))
        if self._keys:
            self._doc_list.setCurrentRow(0)

    def _on_doc_selected(self, index: int) -> None:
        if not (0 <= index < len(self._keys)):
            return
        key = self._keys[index]

        if key == QUICK_START_KEY:
            self._right_stack.setCurrentIndex(0)
            return

        self._right_stack.setCurrentIndex(1)
        html = command_page(key)
        # Apply styling reset matching the app design system
        styled_html = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, system-ui, BlinkMacSystemFont,
                        "Segoe UI", Roboto, sans-serif;
                    font-size: 13px;
                    line-height: 1.6;
                    color: #1A1C20;
                    padding: {SPACE_LG}px;
                }}
                h1 {{ font-size: 18px; font-weight: 700; color: #0F5C5A; margin-top: 0; }}
                h2 {{ font-size: 14px; font-weight: 600; color: #1A9C94; margin-top: 20px; }}
                h3 {{ font-size: 13px; font-weight: 600; color: #1A1C20; }}
                hr {{ border: none; border-top: 1px solid #CED4DA; margin: 16px 0; }}
                ul, ol {{ padding-left: 20px; }}
                th {{ background: #F0F1F4; text-align: left; }}
                td {{ vertical-align: top; }}
                li {{ margin-bottom: 6px; }}
                code {{
                    background: #F0F1F4; padding: 1px 4px; border-radius: 3px;
                    font-family: monospace;
                }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        self._browser.setHtml(styled_html)
