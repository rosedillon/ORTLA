"""Shows one Plotly chart or map in an embedded browser."""

from __future__ import annotations

import functools
import itertools
import pathlib
import tempfile

import plotly.io as pio
import plotly.offline
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ortla.gui.constants import ThemeMode, palette

_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}
_COUNTER = itertools.count()


@functools.cache
def _folder() -> pathlib.Path:
    """Return a temporary folder holding plotly.js, for offline charts.

    Pages are written to files rather than passed to ``setHtml``, which
    refuses anything over 2 MB, and a big map passes that easily.
    """
    folder = pathlib.Path(tempfile.mkdtemp(prefix="ortla-charts-"))
    (folder / "plotly.min.js").write_text(plotly.offline.get_plotlyjs(), encoding="utf-8")
    return folder


class ChartView(QWidget):
    """Hosts an interactive Plotly figure."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._web_view = QWebEngineView(self)
        self._web_view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._web_view)
        # Each view alternates between two files of its own, so the
        # folder does not grow and a page is not rewritten while
        # shown.
        self._names = itertools.cycle([f"chart_{next(_COUNTER)}_{i}.html" for i in (0, 1)])

    def set_figure(self, fig, mode: ThemeMode = ThemeMode.LIGHT) -> None:  # noqa: ANN001
        """Draw ``fig`` on the theme's background."""
        p = palette(mode)
        html = pio.to_html(
            fig,
            config=_CONFIG,
            include_plotlyjs="directory",
            full_html=True,
            default_height="100%",
        )
        style = (
            f"<style>html, body {{ background: {p.surface}; margin: 0; height: 100%;"
            " overflow: hidden; }</style>"
        )
        path = _folder() / next(self._names)
        path.write_text(html.replace("<head>", f"<head>{style}", 1), encoding="utf-8")
        self._web_view.load(QUrl.fromLocalFile(str(path)))
