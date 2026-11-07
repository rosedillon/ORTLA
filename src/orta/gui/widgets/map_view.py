"""Interactive map viewer widget.

Loads local Folium/Leaflet HTML map files directly into a
``QWebEngineView``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from orta.gui.constants import SPACE_MD, SPACE_SM

if TYPE_CHECKING:
    import pathlib

# setHtml() puts its content in a data: URL, which Chromium caps at 2
# MB.
# A route map fits; a whole-network map is tens of MB, so above this
# size the file is loaded from disk instead.
_SET_HTML_SIZE_LIMIT = 2 * 1024 * 1024


class MapView(QWidget):
    """Host local interactive HTML maps (Folium/Leaflet) in the app."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._maps: list[pathlib.Path] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(1)

        # An opaque QFrame: the stylesheet leaves a plain QWidget
        # transparent, and beside a QWebEngineView's native surface a
        # transparent panel paints black.
        self._list_panel = QFrame()
        self._list_panel.setProperty("class", "panel")
        list_layout = QVBoxLayout(self._list_panel)
        list_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        list_layout.setSpacing(SPACE_SM)

        title = QLabel("Available Maps")
        title.setStyleSheet("font-weight: 600;")
        list_layout.addWidget(title)

        self._map_list = QListWidget()
        self._map_list.currentRowChanged.connect(self._on_map_selected)
        list_layout.addWidget(self._map_list)

        self._splitter.addWidget(self._list_panel)

        # Right map engine
        self._web_view = QWebEngineView()
        # QWebEngineView's Chromium surface defaults to a black backing
        # buffer until a page has painted; switching views/dialogs over
        # it can expose that raw buffer as a black flash instead of the
        # page's white background. Pin it to white so it never shows
        # through.
        self._web_view.page().setBackgroundColor(QColor(Qt.GlobalColor.white))
        # Folium maps pull Leaflet's JS/CSS from a CDN; when loaded via
        # file:// (the large-file path below), that's a cross-scheme
        # request QtWebEngine blocks unless explicitly allowed.
        settings = self._web_view.settings()
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self._splitter.addWidget(self._web_view)

        # Initial splitter sizes: list 25%, map 75%
        self._splitter.setSizes([200, 600])
        layout.addWidget(self._splitter)

    def set_maps(self, paths: list[pathlib.Path]) -> None:
        """Populate the map list and load the first map if available."""
        self._maps = paths
        self._map_list.clear()

        for path in paths:
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            self._map_list.addItem(item)

        if paths:
            self._splitter.setSizes([200, 600])
            self._map_list.setCurrentRow(0)
        else:
            self._web_view.setHtml("<h3>No interactive maps generated.</h3>")

    def _on_map_selected(self, index: int) -> None:
        if 0 <= index < len(self._maps):
            path = self._maps[index]
            try:
                if path.stat().st_size > _SET_HTML_SIZE_LIMIT:
                    # Too big for setHtml(); load it from disk.
                    self._web_view.load(QUrl.fromLocalFile(str(path)))
                else:
                    # Read HTML string directly to bypass file:// CORS
                    # sandbox restrictions
                    with path.open("r", encoding="utf-8") as f:
                        html_content = f.read()
                    self._web_view.setHtml(html_content, QUrl("about:blank"))
            except OSError as exc:
                self._web_view.setHtml(f"<h3>Failed to read map file: {exc}</h3>")
