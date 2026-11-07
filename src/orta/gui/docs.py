"""Reference pages for the Help view, written from the config models.

Each page lists a command's settings with their types, defaults and the
comment above each field in ``orta.config``. Nothing here is
written by hand, so the pages cannot fall behind the code.
"""

from __future__ import annotations

import html
import inspect
import typing

import pydantic
from pydantic_core import PydanticUndefined

from orta.config import COMMANDS
from orta.gui.widgets.config_form import field_comments, human_label, unwrap_optional


def _type_name(annotation: object) -> str:
    inner, optional = unwrap_optional(annotation)
    if typing.get_origin(inner) is typing.Literal:
        name = " | ".join(repr(a) for a in typing.get_args(inner))
    elif isinstance(inner, type) and not typing.get_args(inner):
        name = inner.__name__
    else:
        name = str(inner).replace("pathlib.", "").replace("typing.", "")
        name = name.replace("orta.config.", "")
    return name + (" (optional)" if optional else "")


def _default(field: pydantic.fields.FieldInfo) -> str:
    if field.is_required():
        return "required"
    if field.default is PydanticUndefined:
        return "empty"
    return repr(field.default)


def _field_rows(model: type[pydantic.BaseModel], prefix: str = "") -> list[str]:
    comments = field_comments(model)
    rows = []
    for name, field in model.model_fields.items():
        inner, _optional = unwrap_optional(field.annotation)
        key = f"{prefix}{name}"
        note = html.escape(comments.get(name, ""))
        rows.append(
            f"<tr><td><code>{key}</code></td><td>{html.escape(_type_name(field.annotation))}"
            f"</td><td>{html.escape(_default(field))}</td><td>{note}</td></tr>"
        )
        if isinstance(inner, type) and issubclass(inner, pydantic.BaseModel):
            rows += _field_rows(inner, prefix=f"{key}.")
    return rows


def _label(command: str) -> str:
    from orta.gui.constants import COMMANDS as LOOK  # noqa: PLC0415

    return LOOK.get(command, (human_label(command),))[0]


def command_page(command: str) -> str:
    """Write the HTML reference page for ``command``."""
    model, help_text = COMMANDS[command]
    doc = inspect.cleandoc(model.__doc__ or "")
    paragraphs = "".join(f"<p>{html.escape(p)}</p>" for p in doc.split("\n\n") if p)
    rows = "".join(_field_rows(model))
    return (
        f"<h1>{html.escape(_label(command))}</h1>"
        f"<p><b>{html.escape(help_text[:1].upper() + help_text[1:])}.</b></p>"
        f"{paragraphs}"
        "<h2>Running it</h2>"
        "<p>Fill in the form on the Analysis page and press Run, or from a terminal:</p>"
        f"<p><code>orta {command} configs/{command}.yml</code></p>"
        "<h2>Settings</h2>"
        "<table cellspacing='0' cellpadding='4' border='1' width='100%'>"
        "<tr><th>Key</th><th>Type</th><th>Default</th><th>Notes</th></tr>"
        f"{rows}</table>"
    )
