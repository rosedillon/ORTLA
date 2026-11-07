"""PySide6 desktop front-end for orta.

Launch with ``python -m orta.gui`` or the
``orta-gui`` console script.
"""

from __future__ import annotations

import sys
import time

_MIN_STAGE_SECONDS = 0.6


def launch() -> None:
    """Create and run the main application window."""
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    from orta.gui.widgets.splash_screen import AppSplashScreen  # noqa: PLC0415

    app = QApplication.instance() or QApplication(sys.argv)

    # Show branded splash screen and force it to the foreground
    # immediately
    splash = AppSplashScreen()
    splash.show()
    splash.raise_()
    splash.activateWindow()
    app.processEvents()

    def _hold(status: str) -> None:
        splash.set_status(status)
        splash.raise_()
        app.processEvents()
        time.sleep(_MIN_STAGE_SECONDS)

    _hold("Loading configuration templates…")
    from orta.gui.app import RailModelApp  # noqa: PLC0415

    _hold("Initializing main window…")
    window = RailModelApp()

    _hold("Ready.")
    window.show()
    splash.finish(window)

    sys.exit(app.exec())
