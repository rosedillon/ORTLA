"""Every examples/*.yml validates, with no key its model ignores."""

from __future__ import annotations

import pathlib
import typing

import pydantic
import pytest
import yaml

from orta.config import COMMANDS

_EXAMPLES = sorted(
    p
    for p in (pathlib.Path(__file__).parents[1] / "examples").glob("*.yml")
    if not p.name.startswith(".")
)


def _unknown_keys(model: type[pydantic.BaseModel], data: object, where: str = "") -> list[str]:
    """Keys in ``data`` that ``model`` would silently drop."""
    if not isinstance(data, dict):
        return []
    out = [f"{where}{k}" for k in data if k not in model.model_fields]
    for name, field in model.model_fields.items():
        if name not in data:
            continue
        # An optional field's types are its union's members.
        kinds = typing.get_args(field.annotation)
        if typing.get_origin(field.annotation) in (list, dict) or not kinds:
            kinds = (field.annotation,)
        for kind in kinds:
            origin, args = typing.get_origin(kind), typing.get_args(kind)
            inner = args[0] if origin is list else args[1] if origin is dict else kind
            if isinstance(inner, type) and issubclass(inner, pydantic.BaseModel):
                value = data[name]
                values = (
                    value
                    if isinstance(value, list)
                    else list(value.values())
                    if origin is dict and isinstance(value, dict)
                    else [value]
                )
                for item in values:
                    out += _unknown_keys(inner, item, f"{where}{name}.")
    return out


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda p: p.stem)
def test_example_matches_its_config(path: pathlib.Path) -> None:
    """Pydantic drops a stale key silently, so check for it here."""
    model = COMMANDS[path.stem][0]
    data = yaml.safe_load(path.read_text())
    model.model_validate(data)
    assert sorted(set(_unknown_keys(model, data))) == []
