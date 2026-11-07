"""Application preferences and the recent-project list.

Saved as JSON in the user's home folder.
"""

from __future__ import annotations

import json
import logging
import pathlib

logger = logging.getLogger(__name__)

# Every preference and its default. Loading, saving and resetting all
# work from this, so a new preference only needs a line here.
_DEFAULTS: dict = {
    "theme": "system",
    "default_output_dir": "",
    "default_analysis": "loading",
    "autosave": False,
    "confirm_on_exit": True,
    "restore_last_session": False,
    "recent_projects": [],
    "last_project": "",
    # A folder of the user's own configs, searched before examples/ when
    # a form is pre-filled. The Quick Start page sets it.
    "template_dir": "",
}

_MAX_RECENT = 10


class Settings:
    """The user's preferences, read from and written to a JSON file."""

    theme: str
    default_output_dir: str
    default_analysis: str
    autosave: bool
    confirm_on_exit: bool
    restore_last_session: bool
    recent_projects: list[str]
    last_project: str
    template_dir: str

    def __init__(self, filepath: pathlib.Path | None = None) -> None:
        self.filepath = filepath or pathlib.Path.home() / ".orta_settings.json"
        self._apply({})
        self.load()

    def _apply(self, data: dict) -> None:
        """Take each preference from ``data``, else its default."""
        for key, default in _DEFAULTS.items():
            value = data.get(key, default)
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, list):
                value = list(value)
            setattr(self, key, value)

    def to_dict(self) -> dict:
        """Return the preferences as a plain dict."""
        return {key: getattr(self, key) for key in _DEFAULTS}

    def load(self) -> None:
        """Read the settings file; keep the defaults if it fails."""
        if not self.filepath.exists():
            return
        try:
            self._apply(json.loads(self.filepath.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            logger.warning("Settings file %s could not be read; using defaults", self.filepath)

    def save(self) -> None:
        """Write the settings file."""
        try:
            self.export_to(self.filepath)
        except OSError:
            logger.warning("Could not save settings to %s", self.filepath)

    def export_to(self, path: pathlib.Path) -> None:
        """Write the preferences to ``path``."""
        path.write_text(json.dumps(self.to_dict(), indent=4, ensure_ascii=False), encoding="utf-8")

    def import_from(self, path: pathlib.Path) -> None:
        """Read preferences from ``path`` and make them current."""
        self._apply(json.loads(path.read_text(encoding="utf-8")))
        self.save()

    def reset(self) -> None:
        """Put every preference back to its default and save."""
        self._apply({})
        self.save()

    def add_recent_project(self, path: str) -> None:
        """Put ``path`` at the top of the recent projects."""
        norm_path = str(pathlib.Path(path).resolve())
        if norm_path in self.recent_projects:
            self.recent_projects.remove(norm_path)
        self.recent_projects = [norm_path, *self.recent_projects][:_MAX_RECENT]
        self.last_project = norm_path
        self.save()

    def remove_recent_project(self, path: str) -> None:
        """Drop ``path`` from the recent projects."""
        norm_path = str(pathlib.Path(path).resolve())
        if norm_path in self.recent_projects:
            self.recent_projects.remove(norm_path)
            self.save()
