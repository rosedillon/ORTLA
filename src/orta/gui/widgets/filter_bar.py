"""Unified filter bar widget for analysis results.

Provides search and combo-box filters for Capacity, Services, Lookup,
Loading, and Propose outputs. Emits structured filter-changed events.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from orta.gui.constants import SPACE_MD, SPACE_SM


class FilterBar(QFrame):
    """Context-aware filter strip matching command outputs.

    Signals
    -------
    filter_changed(dict)
        Emitted when any filter criteria updates. Payload contains
        active parameters.
    """

    filter_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(48)
        self.setProperty("class", "panel")

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(SPACE_MD, 0, SPACE_MD, 0)
        self._layout.setSpacing(SPACE_SM)

        self._active_command = ""

        # Pre-create all filter components so we can toggle them
        # ── Capacity / Services controls ─────────────────────────────
        self._lbl_origin = QLabel("Origin CRS:")
        self._inp_origin = QLineEdit()
        self._inp_origin.setFixedWidth(50)
        self._lbl_dest = QLabel("Destination CRS:")
        self._inp_dest = QLineEdit()
        self._inp_dest.setFixedWidth(50)
        self._btn_show = QPushButton("Show")
        self._btn_show.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_show_all = QPushButton("Show all")
        self._btn_show_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lbl_combo = QLabel("or pick:")
        self._cmb_pairs = QComboBox()
        self._cmb_pairs.setMinimumWidth(150)

        # ── Loading controls ──────────────────────────────────────────
        self._lbl_operator = QLabel("Operator:")
        self._cmb_operators = QComboBox()
        self._cmb_operators.setMinimumWidth(180)

        # ── Propose controls ──────────────────────────────────────────
        self._lbl_candidate = QLabel("Candidate:")
        self._cmb_candidates = QComboBox()
        self._cmb_candidates.setMinimumWidth(180)

        # Wire events
        self._btn_show.clicked.connect(self._emit_pair_filter)
        self._inp_origin.returnPressed.connect(self._emit_pair_filter)
        self._inp_dest.returnPressed.connect(self._emit_pair_filter)
        self._btn_show_all.clicked.connect(self._emit_show_all)
        self._cmb_pairs.currentIndexChanged.connect(self._on_pair_combo_selected)
        self._cmb_operators.currentIndexChanged.connect(self._emit_operator_filter)
        self._cmb_candidates.currentIndexChanged.connect(self._emit_candidate_filter)

    # ── public API ────────────────────────────────────────────────────

    def set_command(self, command: str) -> None:
        """Adjust filter bar layout for the selected command."""
        self._active_command = command
        self._clear_layout()

        if command in ("services", "lookup", "fares"):
            self._layout.addWidget(self._lbl_origin)
            self._layout.addWidget(self._inp_origin)
            self._layout.addWidget(self._lbl_dest)
            self._layout.addWidget(self._inp_dest)
            self._layout.addWidget(self._btn_show)
            self._layout.addWidget(self._btn_show_all)
            self._layout.addWidget(self._lbl_combo)
            self._layout.addWidget(self._cmb_pairs)
            self._layout.addStretch()
            self.show()
        elif command == "loading":
            self._layout.addWidget(self._lbl_operator)
            self._layout.addWidget(self._cmb_operators)
            self._layout.addWidget(self._btn_show_all)
            self._layout.addStretch()
            self.show()
        elif command == "propose":
            self._layout.addWidget(self._lbl_candidate)
            self._layout.addWidget(self._cmb_candidates)
            self._layout.addWidget(self._btn_show_all)
            self._layout.addStretch()
            self.show()
        else:
            self.hide()

    def set_available_pairs(self, pairs: list[tuple[str, str]]) -> None:
        """Populate the O→D pair dropdown values."""
        self._cmb_pairs.blockSignals(True)
        self._cmb_pairs.clear()
        self._cmb_pairs.addItem("Select a station pair", None)
        for origin, dest in pairs:
            self._cmb_pairs.addItem(f"{origin} → {dest}", (origin, dest))
        self._cmb_pairs.blockSignals(False)

    def set_available_operators(self, operators: list[str]) -> None:
        """Populate the Operator dropdown values."""
        self._cmb_operators.blockSignals(True)
        self._cmb_operators.clear()
        self._cmb_operators.addItem("Select an operator", None)
        for op in operators:
            label = op.replace("_", " ").title()
            self._cmb_operators.addItem(label, op)
        self._cmb_operators.blockSignals(False)

    def set_available_candidates(self, candidates: list[str]) -> None:
        """Populate the Candidate dropdown values."""
        self._cmb_candidates.blockSignals(True)
        self._cmb_candidates.clear()
        self._cmb_candidates.addItem("Select a candidate", None)
        for cand in candidates:
            self._cmb_candidates.addItem(cand, cand)
        self._cmb_candidates.blockSignals(False)

    # ── internal event handlers ──────────────────────────────────────

    def _clear_layout(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

    def _on_pair_combo_selected(self, index: int) -> None:
        if index <= 0:
            return
        data = self._cmb_pairs.currentData()
        if data:
            orig, dest = data
            self._inp_origin.setText(orig)
            self._inp_dest.setText(dest)
            self._emit_pair_filter()

    def _emit_pair_filter(self) -> None:
        orig = self._inp_origin.text().strip().upper()
        dest = self._inp_dest.text().strip().upper()
        if orig and dest:
            self.filter_changed.emit(
                {
                    "tag": f"{orig}_{dest}",
                    "origin": orig,
                    "destination": dest,
                }
            )

    def _emit_operator_filter(self) -> None:
        op = self._cmb_operators.currentData()
        self.filter_changed.emit({"operator": op})

    def _emit_candidate_filter(self) -> None:
        cand = self._cmb_candidates.currentData()
        self.filter_changed.emit({"candidate": cand})

    def _emit_show_all(self) -> None:
        self._inp_origin.clear()
        self._inp_dest.clear()
        self._cmb_pairs.setCurrentIndex(0)
        self._cmb_operators.setCurrentIndex(0)
        self._cmb_candidates.setCurrentIndex(0)
        self.filter_changed.emit({})
