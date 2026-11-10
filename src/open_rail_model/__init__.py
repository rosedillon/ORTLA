"""The model's old name, kept so the research cases still import.

The package is :mod:`ortla` now. ``import open_rail_model.config`` hands
back the very module ``ortla.config``, so the case scripts, and pickles
written before the rename, work unchanged. New code should import
:mod:`ortla`.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

import ortla


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``open_rail_model.x`` to the existing ``ortla.x``."""

    def find_spec(
        self, name: str, path: object = None, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        """Claim the old name's submodules."""
        del path, target
        if name.startswith(__name__ + "."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        """Hand back the ``ortla`` module itself."""
        return importlib.import_module("ortla" + spec.name.removeprefix(__name__))

    def exec_module(self, module: types.ModuleType) -> None:
        """Do nothing: the module has already run as ``ortla``."""


sys.meta_path.insert(0, _Alias())
sys.modules[__name__] = ortla
