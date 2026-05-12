"""Unit tests for theme palette, build_stylesheet, and ThemeManager."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.ui.theme import LATTE, MOCHA, build_stylesheet
from app.ui.theme_manager import ThemeManager, ThemeMode


@pytest.fixture(autouse=True)
def _reset_theme_manager():
    from app.ui.theme_manager import ThemeManager
    ThemeManager._instance = None
    yield
    ThemeManager._instance = None


def test_build_stylesheet_latte_contains_primary():
    """build_stylesheet with LATTE includes the LATTE primary accent color."""
    qss = build_stylesheet(LATTE)
    assert LATTE["NAV_ACTIVE_BG"] in qss


def test_build_stylesheet_mocha_contains_primary():
    """build_stylesheet with MOCHA includes the MOCHA primary accent color."""
    qss = build_stylesheet(MOCHA)
    assert MOCHA["NAV_ACTIVE_BG"] in qss


def test_theme_manager_toggle_cycles_light_dark_light():
    """toggle() cycles LIGHT → DARK → LIGHT."""
    tm = ThemeManager.instance()
    assert tm.mode() == ThemeMode.LIGHT

    tm._app = MagicMock()
    tm._settings = MagicMock()

    tm.toggle()
    assert tm.mode() == ThemeMode.DARK

    tm.toggle()
    assert tm.mode() == ThemeMode.LIGHT


def test_theme_manager_persists_mode_to_settings():
    """After toggle(), settings.theme reflects the new mode value."""
    tm = ThemeManager.instance()
    mock_settings = MagicMock()
    mock_settings.theme = "light"
    tm._app = MagicMock()
    tm._settings = mock_settings

    tm.toggle()
    assert mock_settings.theme == "dark"


def test_fa_font_otf_exists():
    """FontAwesome Solid OTF file is present in assets."""
    otf_path = (
        Path(__file__).parent.parent.parent
        / "assets/fonts/fontawesome-free-7.2.0-desktop/otfs"
        / "Font Awesome 7 Free-Solid-900.otf"
    )
    assert otf_path.exists(), f"OTF not found: {otf_path}"
