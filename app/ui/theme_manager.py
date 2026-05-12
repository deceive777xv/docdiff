"""ThemeManager singleton — owns the active theme and applies it app-wide."""
from __future__ import annotations

import logging
from enum import Enum

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class ThemeMode(Enum):
    LIGHT = "light"
    DARK = "dark"


class ThemeManager(QObject):
    theme_changed = Signal()

    _instance: "ThemeManager | None" = None

    def __init__(self) -> None:
        super().__init__()
        self._mode = ThemeMode.LIGHT
        self._app = None
        self._settings = None

    @classmethod
    def instance(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = ThemeManager()
        return cls._instance

    def setup(self, settings, app) -> None:
        """Initialize from persisted settings and apply QSS to app."""
        self._app = app
        self._settings = settings
        raw = getattr(settings, "theme", "light")
        self._mode = ThemeMode.DARK if raw == "dark" else ThemeMode.LIGHT
        self._apply()

    def toggle(self) -> None:
        """Switch between LIGHT and DARK, persist, and notify."""
        self._mode = ThemeMode.DARK if self._mode == ThemeMode.LIGHT else ThemeMode.LIGHT
        self._apply()
        self._save()
        self.theme_changed.emit()

    def mode(self) -> ThemeMode:
        return self._mode

    def palette(self) -> dict:
        from app.ui.theme import LATTE, MOCHA
        return LATTE if self._mode == ThemeMode.LIGHT else MOCHA

    def _apply(self) -> None:
        from app.ui.theme import LATTE, MOCHA, Theme, build_stylesheet
        p = LATTE if self._mode == ThemeMode.LIGHT else MOCHA
        for attr, value in p.items():
            if hasattr(Theme, attr):
                setattr(Theme, attr, value)
        if self._app is not None:
            self._app.setStyleSheet(build_stylesheet(p))

    def _save(self) -> None:
        if self._settings is None:
            return
        self._settings.theme = self._mode.value
        try:
            from app.config import settings as settings_module
            settings_module.save(self._settings)
        except Exception:
            logger.warning("Failed to save theme setting", exc_info=True)
