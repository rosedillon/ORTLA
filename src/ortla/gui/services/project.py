"""Project file management service.

Tracks configuration state across all command forms, active view
selections, and project pathing.  Serialises to a light-weight JSON
format (``.ortla``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib


class Project:
    """Represents a user study/project session."""

    def __init__(self, filepath: pathlib.Path | None = None) -> None:
        self.filepath: pathlib.Path | None = filepath
        self.name: str = filepath.stem if filepath else "Untitled Analysis"
        self._dirty: bool = False

        # Config data for each command
        self.configs: dict[str, dict] = {}
        self.active_command: str = "loading"
        self.notes: str = ""

        if filepath:
            self.load()

    @property
    def is_dirty(self) -> bool:
        """Whether there are unsaved changes."""
        return self._dirty

    def mark_dirty(self) -> None:
        """Mark project as modified."""
        self._dirty = True

    def mark_clean(self) -> None:
        """Mark project as saved."""
        self._dirty = False

    def load(self) -> None:
        """Read state from filepath."""
        if not self.filepath or not self.filepath.exists():
            return

        try:
            with self.filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.configs = data.get("configs", {})
            self.active_command = data.get("active_command", "loading")
            self.notes = data.get("notes", "")
            self.name = self.filepath.stem
            self.mark_clean()
        except Exception as exc:
            raise OSError(f"Failed to load project: {exc}") from exc

    def save(self, filepath: pathlib.Path | None = None) -> None:
        """Write current state to filepath."""
        target = filepath or self.filepath
        if not target:
            raise ValueError("No destination filepath specified.")

        self.filepath = target
        self.name = target.stem

        data = {
            "configs": self.configs,
            "active_command": self.active_command,
            "notes": self.notes,
        }

        try:
            with target.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.mark_clean()
        except Exception as exc:
            raise OSError(f"Failed to save project: {exc}") from exc
