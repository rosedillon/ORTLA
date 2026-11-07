"""A Qt form built from a command's pydantic config model.

Each field gets a widget that suits its type: a checkbox for a flag, a
drop-down for a fixed set of choices, a path picker for a file or
folder, one line per item for a simple list, and a YAML box for anything
nested. The help shown with a field is the comment written above it in
``orta.config``, so the form and the code never disagree.
"""

from __future__ import annotations

import ast
import functools
import inspect
import pathlib
import textwrap
import types
import typing

import pydantic
import yaml
from pydantic_core import PydanticUndefined
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from orta.gui.constants import SPACE_MD, SPACE_SM

_SCALARS = (str, int, float, pathlib.Path)


def unwrap_optional(annotation: object) -> tuple[object, bool]:
    """Return ``(inner type, is optional)`` for an annotation."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return (args[0] if len(args) == 1 else annotation), True
    return annotation, False


def _is_model(annotation: object) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel)


def human_label(name: str) -> str:
    """``some_field_name`` becomes ``Some field name``."""
    text = name.replace("_", " ")
    return text[:1].upper() + text[1:]


@functools.cache
def field_comments(model_cls: type[pydantic.BaseModel]) -> dict[str, str]:
    """Return the comment written above each field of ``model_cls``.

    Fields inherited from a parent class take the parent's comment. A
    field with no comment is left out.
    """
    out: dict[str, str] = {}
    for cls in reversed(model_cls.__mro__):
        if not _is_model(cls) or cls.__module__.startswith(("pydantic", "caf")):
            continue
        try:
            lines, _start = inspect.getsourcelines(cls)
        except (OSError, TypeError):
            continue
        source = textwrap.dedent("".join(lines))
        body = ast.parse(source).body[0]
        src_lines = source.splitlines()
        for node in getattr(body, "body", []):
            if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
                continue
            comment = []
            idx = node.lineno - 2
            while idx >= 0 and src_lines[idx].strip().startswith("#"):
                comment.insert(0, src_lines[idx].strip().lstrip("#").strip())
                idx -= 1
            if comment:
                out[node.target.id] = " ".join(comment)
    return out


def _first_sentence(text: str, limit: int = 160) -> str:
    end = text.find(". ")
    short = text if end < 0 else text[: end + 1]
    return short if len(short) <= limit else short[: limit - 1].rstrip() + "…"


def _default(field: pydantic.fields.FieldInfo) -> object:
    """Return the field's plain default, or None."""
    if field.default is PydanticUndefined:
        return None
    return field.default


class _Field:
    """Gets and sets one field's value through its widget."""

    def __init__(
        self,
        widget: QWidget,
        getter: typing.Callable[[], object],
        setter: typing.Callable[[object], None],
    ) -> None:
        self.widget = widget
        self.get = getter
        self.set = setter


class ConfigForm(QWidget):
    """A form for one pydantic config model.

    Signals
    -------
    changed()
        A field's value changed.
    """

    changed = Signal()

    def __init__(self, model_cls: type[pydantic.BaseModel], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model_cls = model_cls
        self._fields: dict[str, _Field] = {}
        self._defaults: dict[str, object] = {}
        self._subforms: dict[str, tuple[ConfigForm, bool]] = {}
        self._help = field_comments(model_cls)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        self._layout.setSpacing(SPACE_SM)
        # Required fields first, so a new user sees what they must fill
        # in.
        fields = sorted(model_cls.model_fields.items(), key=lambda item: not item[1].is_required())
        for name, field in fields:
            self._add_field(name, field)
        self._layout.addStretch(1)

    def _add_field(self, name: str, field: pydantic.fields.FieldInfo) -> None:
        inner, optional = unwrap_optional(field.annotation)
        help_text = self._help.get(name, field.description or "")
        if _is_model(inner):
            title = human_label(name) + ("  (optional)" if optional else "")
            group = QGroupBox(title)
            group.setToolTip(help_text)
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(SPACE_SM, SPACE_MD, SPACE_SM, SPACE_SM)
            sub = ConfigForm(inner)
            sub.changed.connect(self.changed.emit)
            group_layout.addWidget(sub)
            self._subforms[name] = (sub, optional)
            self._layout.addWidget(group)
            return

        self._defaults[name] = _default(field)
        holder = self._make_widget(name, inner, self._defaults[name])
        holder.widget.setToolTip(help_text)
        self._fields[name] = holder

        row = QFrame()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, SPACE_SM, 0, 0)
        row_layout.setSpacing(4)
        label = QLabel(human_label(name) + ("" if field.is_required() else "  (optional)"))
        label.setStyleSheet("font-weight: 600; font-size: 12px;")
        label.setToolTip(f"YAML key: {name}")
        row_layout.addWidget(label)
        row_layout.addWidget(holder.widget)
        if help_text:
            tip = QLabel(_first_sentence(help_text))
            tip.setProperty("class", "caption")
            tip.setWordWrap(True)
            tip.setToolTip(help_text)
            row_layout.addWidget(tip)
        self._layout.addWidget(row)

    def _make_widget(self, name: str, inner: object, default: object) -> _Field:
        origin = typing.get_origin(inner)
        if inner is bool:
            return self._bool_widget(bool(default))
        if origin is typing.Literal:
            return self._choice_widget(typing.get_args(inner), default)
        if inner is pathlib.Path:
            return self._path_widget()
        if origin is list and typing.get_args(inner)[0] in _SCALARS:
            return self._lines_widget()
        if origin in (list, dict, tuple) or _is_model(inner):
            return self._yaml_widget(inner)
        return self._entry_widget(inner, default, secret="password" in name)

    def _emit_changed(self, *_args: object) -> None:
        self.changed.emit()

    # ── widgets ──────────────────────────────────────────────────────

    def _entry_widget(self, inner: object, default: object, *, secret: bool) -> _Field:
        entry = QLineEdit()
        entry.setPlaceholderText("" if default is None else f"default: {default}")
        if secret:
            entry.setEchoMode(QLineEdit.EchoMode.Password)
        entry.textChanged.connect(self._emit_changed)

        def getter() -> object:
            text = entry.text().strip()
            if not text:
                return None
            if inner in (int, float):
                try:
                    return inner(text)
                except ValueError:
                    return text
            return text

        return _Field(entry, getter, lambda v: entry.setText("" if v is None else str(v)))

    def _bool_widget(self, default: bool) -> _Field:
        box = QCheckBox()
        box.setChecked(default)
        box.stateChanged.connect(self._emit_changed)
        return _Field(
            box, box.isChecked, lambda v: box.setChecked(default if v is None else bool(v))
        )

    def _choice_widget(self, choices: tuple, default: object) -> _Field:
        combo = QComboBox()
        for choice in choices:
            combo.addItem(str(choice), choice)
        if default in choices:
            combo.setCurrentIndex(choices.index(default))
        combo.currentIndexChanged.connect(self._emit_changed)

        def setter(value: object) -> None:
            idx = combo.findData(default if value is None else value)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        return _Field(combo, combo.currentData, setter)

    def _path_widget(self) -> _Field:
        container = QFrame()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        entry = QLineEdit()
        entry.setPlaceholderText("Path…")
        entry.textChanged.connect(self._emit_changed)
        row.addWidget(entry, stretch=1)

        def picker(label: str, pick: typing.Callable[[], str]) -> None:
            button = QPushButton(label)
            button.setFixedWidth(56)
            button.setProperty("class", "ghost")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda: (path := pick()) and entry.setText(path))
            row.addWidget(button)

        picker("File", lambda: QFileDialog.getOpenFileName(container, "Choose a file")[0])
        picker("Folder", lambda: QFileDialog.getExistingDirectory(container, "Choose a folder"))
        return _Field(
            container,
            lambda: entry.text().strip() or None,
            lambda v: entry.setText("" if v is None else str(v)),
        )

    def _lines_widget(self) -> _Field:
        text = QPlainTextEdit()
        text.setMaximumHeight(100)
        text.setPlaceholderText("One item per line")
        text.textChanged.connect(self.changed.emit)

        def getter() -> object:
            items = [line.strip() for line in text.toPlainText().splitlines() if line.strip()]
            return items or None

        def setter(value: object) -> None:
            if isinstance(value, (list, tuple)):
                value = "\n".join(str(v) for v in value)
            text.setPlainText("" if value is None else str(value))

        return _Field(text, getter, setter)

    def _yaml_widget(self, inner: object) -> _Field:
        container = QFrame()
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        args = typing.get_args(inner)
        if typing.get_origin(inner) is list and args and _is_model(args[0]):
            hint = f"YAML: a list of {args[0].__name__} entries"
        else:
            hint = "YAML"
        hint_label = QLabel(hint)
        hint_label.setProperty("class", "caption")
        column.addWidget(hint_label)
        text = QPlainTextEdit()
        text.setMaximumHeight(200)
        text.textChanged.connect(self.changed.emit)
        column.addWidget(text)

        def getter() -> object:
            raw = text.toPlainText().strip()
            if not raw:
                return None
            try:
                return yaml.safe_load(raw)
            except yaml.YAMLError:
                return raw  # validation reports the problem against the field

        def setter(value: object) -> None:
            text.setPlainText(
                ""
                if value in (None, [], {})
                else yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
            )

        return _Field(container, getter, setter)

    # ── data ─────────────────────────────────────────────────────────

    def get_data(self) -> dict:
        """Return the form as a nested dict, without defaults."""
        data: dict = {}
        for name, holder in self._fields.items():
            value = holder.get()
            if value is None or (isinstance(value, bool) and value == self._defaults[name]):
                continue
            data[name] = value
        for name, (sub, optional) in self._subforms.items():
            if not (optional and sub.is_empty()):
                data[name] = sub.get_data()
        return data

    def is_empty(self) -> bool:
        """Return whether nothing is set here or in a nested section."""
        own = {k: v for k, v in self.get_data().items() if k not in self._subforms}
        return not own and all(sub.is_empty() for sub, _ in self._subforms.values())

    def set_data(self, data: dict) -> None:
        """Fill the form from a nested dict."""
        for name, holder in self._fields.items():
            holder.set(data.get(name))
        for name, (sub, _optional) in self._subforms.items():
            sub.set_data(data.get(name) or {})

    def to_yaml(self) -> str:
        """Return the form as YAML."""
        return yaml.safe_dump(self.get_data(), sort_keys=False, allow_unicode=True)
