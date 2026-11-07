"""A toast that slides in at the bottom right.

The toast is drawn from the same palette tokens as the rest of the app,
so it matches whichever light/dark theme is active instead of a fixed
dark card. Multiple toasts stack upward, and each slides smoothly in,
waits, then slides back out.
"""

from __future__ import annotations

import qtawesome as qta
from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from orta.gui.constants import (
    RADIUS_MD,
    SPACE_MD,
    SPACE_SM,
    ThemeMode,
    palette,
)

# Toast geometry / layout constants shared with the app's stacking
# logic.
TOAST_WIDTH = 320
TOAST_HEIGHT = 76
TOAST_MARGIN = 16
TOAST_GAP = 10


class ToastNotification(QFrame):
    """A short-lived popup anchored to the main window.

    Signals
    -------
    closed()
        Emitted once the toast has finished sliding out and is about to
        be destroyed, so a manager can re-flow any remaining toasts.
    """

    closed = Signal()

    def __init__(
        self,
        title: str,
        message: str,
        kind: str = "info",  # info, success, warning, error
        mode: ThemeMode = ThemeMode.LIGHT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(TOAST_WIDTH, TOAST_HEIGHT)

        self._title = title
        self._message = message
        self._kind = kind
        self._mode = mode
        self._slot = 0
        self._dismissed = False

        self._build_ui()

        # Automatic dismiss timer
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._slide_out)

    def _build_ui(self) -> None:
        p = palette(self._mode)

        # Semantic accent + icon per kind, drawn from the active palette
        # so the toast reads as part of the app rather than a foreign
        # dark card.
        accent_map = {
            "success": p.success,
            "warning": p.warning,
            "error": p.error,
            "info": p.accent,
        }
        icon_map = {
            "success": "fa5s.check-circle",
            "warning": "fa5s.exclamation-triangle",
            "error": "fa5s.exclamation-circle",
            "info": "fa5s.info-circle",
        }
        accent_col = accent_map.get(self._kind, p.accent)
        icon_name = icon_map.get(self._kind, "fa5s.info-circle")

        # Visual wrapper styled like an elevated card (matches
        # QFrame[class=card-elevated]).
        card = QFrame(self)
        card.setGeometry(0, 0, TOAST_WIDTH, TOAST_HEIGHT)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {p.surface_elevated};
                border: 1px solid {p.border_subtle};
                border-left: 3px solid {accent_col};
                border-radius: {RADIUS_MD}px;
            }}
        """)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(SPACE_MD, SPACE_SM, SPACE_SM + 2, SPACE_SM)
        layout.setSpacing(SPACE_SM + 2)

        # Icon (vertically centred against the text block)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(qta.icon(icon_name, color=accent_col).pixmap(22, 22))
        icon_lbl.setFixedWidth(24)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(icon_lbl, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Text column, vertically centred within the card
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.addStretch()

        title_lbl = QLabel(self._title)
        title_lbl.setStyleSheet(
            f"color: {p.text_primary}; font-weight: 600; font-size: 13px;"
            " border: none; background: transparent;"
        )
        text_layout.addWidget(title_lbl)

        msg_lbl = QLabel(self._message)
        msg_lbl.setStyleSheet(
            f"color: {p.text_secondary}; font-size: 12px; border: none; background: transparent;"
        )
        msg_lbl.setWordWrap(True)
        text_layout.addWidget(msg_lbl)
        text_layout.addStretch()

        layout.addLayout(text_layout, stretch=1)

        # Dismiss button
        btn_close = QPushButton()
        btn_close.setFixedSize(18, 18)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setIcon(qta.icon("fa5s.times", color=p.text_secondary))
        btn_close.setStyleSheet(
            "QPushButton { border: none; background: transparent; border-radius: 4px; }"
            " QPushButton:hover { background: rgba(128,128,128,0.18); }"
        )
        btn_close.clicked.connect(self._slide_out)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignTop)

    # ── stacking
    # ──────────────────────────────────────────────────────────

    def set_slot(self, slot: int) -> None:
        """Set the position in the stack (0 is the bottom)."""
        self._slot = slot
        if self.isVisible() and not self._dismissed:
            self.move(self._resting_pos())

    def _anchor_rect(self) -> QRect:
        """Return the area the toast sits in."""
        parent = self.parentWidget()
        if parent is not None:
            return parent.frameGeometry()
        return self.screen().availableGeometry()

    def _resting_pos(self) -> QPoint:
        """Bottom-right resting position for this toast's stack slot."""
        rect = self._anchor_rect()
        x = rect.x() + rect.width() - TOAST_WIDTH - TOAST_MARGIN
        offset = self._slot * (TOAST_HEIGHT + TOAST_GAP)
        y = rect.y() + rect.height() - TOAST_HEIGHT - TOAST_MARGIN - offset
        return QPoint(x, y)

    def _hidden_pos(self) -> QPoint:
        """Return where the toast slides from, just below its place."""
        resting = self._resting_pos()
        return QPoint(resting.x(), self._anchor_rect().bottom() + TOAST_HEIGHT)

    # ── show / hide
    # ───────────────────────────────────────────────────────

    def show_toast(self, duration_ms: int = 4000) -> None:
        """Slide the toast in and schedule its automatic dismissal."""
        start_pos = self._hidden_pos()
        end_pos = self._resting_pos()

        self.setGeometry(QRect(start_pos.x(), start_pos.y(), TOAST_WIDTH, TOAST_HEIGHT))
        self.show()
        self.raise_()

        self._anim = QPropertyAnimation(self, b"pos")
        self._anim.setDuration(300)
        self._anim.setStartValue(start_pos)
        self._anim.setEndValue(end_pos)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()

        self._timer.start(duration_ms)

    def _slide_out(self) -> None:
        if self._dismissed:
            return
        self._dismissed = True
        self._timer.stop()

        self._anim = QPropertyAnimation(self, b"pos")
        self._anim.setDuration(220)
        self._anim.setStartValue(self.pos())
        self._anim.setEndValue(self._hidden_pos())
        self._anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._anim.finished.connect(self._on_slid_out)
        self._anim.start()

    def _on_slid_out(self) -> None:
        self.closed.emit()
        self.close()
