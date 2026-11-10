"""Fitted models that ship with ORTLA, usable by name in a config.

* ``m3``: the slot share of cases 23 and 33, departure half-hour timing
  and theta 0.06 per in-train minute;
* ``neural``: ``m3`` with case 117's network (:mod:`ortla.slot_net`);
* ``national``: the direction split fitted on every counted operator
  (case 16's national fit).
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).parent
BUILTIN = {
    "m3": HERE / "slot_share_m3.json",
    "neural": HERE / "slot_share_neural.json",
    "national": HERE / "direction_split_national.json",
}


def resolve(path: pathlib.Path | str) -> pathlib.Path:
    """Return a model file's path, or a built-in's packaged file."""
    name = str(path)
    if name in BUILTIN and not pathlib.Path(name).exists():
        return BUILTIN[name]
    return pathlib.Path(path)
