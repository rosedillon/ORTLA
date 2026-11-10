"""Syntax-highlighted YAML editor widget.

Provides a ``QPlainTextEdit`` with a custom ``QSyntaxHighlighter`` that
colours keys, values, comments, strings, and booleans/nulls.
Theme-aware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import QPlainTextEdit, QWidget

from ortla.gui.constants import (
    FONT_FAMILY_MONO,
    FONT_SIZE_MONO,
    ThemeMode,
    palette,
)

if TYPE_CHECKING:
    from PySide6.QtGui import QTextDocument


class _YamlHighlighter(QSyntaxHighlighter):
    """A small YAML highlighter."""

    def __init__(self, parent: QTextDocument, mode: ThemeMode) -> None:
        super().__init__(parent)
        self._mode = mode
        self._build_rules()

    def set_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        self._build_rules()
        self.rehighlight()

    def _build_rules(self) -> None:
        p = palette(self._mode)
        self._rules: list[tuple[QRegularExpression, QTextCharFormat]] = []

        # Comments
        fmt_comment = QTextCharFormat()
        fmt_comment.setForeground(QColor(p.text_disabled))
        fmt_comment.setFontItalic(True)
        self._rules.append((QRegularExpression(r"#.*$"), fmt_comment))

        # Keys (word before colon at start of line, possibly indented)
        fmt_key = QTextCharFormat()
        fmt_key.setForeground(QColor(p.accent))
        fmt_key.setFontWeight(QFont.Weight.DemiBold)
        self._rules.append((QRegularExpression(r"^\s*[\w\-\.]+(?=\s*:)"), fmt_key))

        # Quoted strings
        fmt_string = QTextCharFormat()
        fmt_string.setForeground(QColor(p.success))
        self._rules.append((QRegularExpression(r'"[^"]*"'), fmt_string))
        self._rules.append((QRegularExpression(r"'[^']*'"), fmt_string))

        # Booleans, null
        fmt_bool = QTextCharFormat()
        fmt_bool.setForeground(QColor(p.warning))
        fmt_bool.setFontWeight(QFont.Weight.DemiBold)
        self._rules.append(
            (
                QRegularExpression(
                    r"\b(true|false|yes|no|null|none)\b",
                    QRegularExpression.PatternOption.CaseInsensitiveOption,
                ),
                fmt_bool,
            )
        )

        # Numbers
        fmt_num = QTextCharFormat()
        fmt_num.setForeground(QColor(p.info))
        self._rules.append((QRegularExpression(r"\b\d+(\.\d+)?\b"), fmt_num))

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        for pattern, fmt in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


class YamlEditor(QPlainTextEdit):
    """A YAML-aware text editor with syntax highlighting.

    Parameters
    ----------
    parent
        Parent widget.
    mode
        Initial theme mode for syntax colouring.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        mode: ThemeMode = ThemeMode.LIGHT,
    ) -> None:
        super().__init__(parent)
        font = QFont(FONT_FAMILY_MONO, FONT_SIZE_MONO)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._highlighter = _YamlHighlighter(self.document(), mode)

    def update_theme(self, mode: ThemeMode) -> None:
        """Refresh syntax colours for the given theme."""
        self._highlighter.set_mode(mode)

    def get_yaml(self) -> str:
        """Return the current text."""
        return self.toPlainText()

    def set_yaml(self, text: str) -> None:
        """Replace the current text."""
        self.setPlainText(text)
