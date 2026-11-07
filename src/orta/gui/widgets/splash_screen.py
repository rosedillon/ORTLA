"""The splash screen shown while the app starts."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import QSplashScreen

from orta._version import __version__
from orta.gui.constants import APP_FULL_NAME, APP_NAME


class AppSplashScreen(QSplashScreen):
    """The app's name, version and what it is loading."""

    def __init__(self) -> None:
        # Create a blank pixmap for drawing
        pixmap = QPixmap(500, 300)
        super().__init__(pixmap)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        self._status = "Starting…"
        self._draw_splash()

    def set_status(self, text: str) -> None:
        """Update the progress text and repaint."""
        self._status = text
        self._draw_splash()
        self.repaint()

    def _draw_splash(self) -> None:
        pixmap = self.pixmap()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        gradient = QLinearGradient(0, 0, 500, 300)
        gradient.setColorAt(0, QColor("#0F5C5A"))
        gradient.setColorAt(1, QColor("#0E2A2C"))
        painter.fillRect(0, 0, 500, 300, gradient)

        # A stylised track: two rails and their sleepers.
        painter.setPen(QColor(255, 255, 255, 40))
        for x in range(0, 500, 18):
            painter.drawLine(x, 118, x + 8, 142)
        painter.setPen(QColor(255, 255, 255, 90))
        painter.drawLine(0, 122, 500, 122)
        painter.drawLine(0, 138, 500, 138)

        painter.setPen(QColor("#8FB3B0"))
        painter.setFont(QFont("Inter", 12))
        painter.drawText(40, 210, APP_FULL_NAME)

        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Inter", 28, QFont.Weight.Bold))
        painter.drawText(40, 190, APP_NAME)

        painter.setPen(QColor("#8FB3B0"))
        painter.setFont(QFont("Inter", 12))
        painter.drawText(40, 232, f"Version {__version__}")

        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Inter", 11, QFont.Weight.Medium))
        painter.drawText(40, 260, self._status)

        painter.end()
        self.setPixmap(pixmap)
