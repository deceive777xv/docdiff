# Catppuccin Theme Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing hardcoded light theme with Catppuccin Latte (light) and Mocha (dark) palettes, add a FontAwesome toggle button in the main window top-right, and persist the theme choice to config.json.

**Architecture:** `ThemeManager` (singleton `QObject`) owns the active mode. On `toggle()` it: (1) bulk-updates `Theme` class attributes from the new palette dict so existing `Theme.X` callsites stay valid, (2) calls `app.setStyleSheet(build_stylesheet(palette))` for structural elements targeted by objectName/property selectors, (3) emits `theme_changed` so each page's `_apply_theme()` can re-call its own `setStyleSheet()` calls. `NavButton` active/inactive styles live entirely in the global QSS via the Qt dynamic property `nav_active`.

**Tech Stack:** PySide6 (`QFontDatabase`, `QToolButton`, `QApplication.setStyleSheet`, dynamic properties), FontAwesome 7 Free Solid OTF.

---

## File Map

| File | Action |
|------|--------|
| `app/config/settings.py` | Add `theme: str = "light"` to `AppSettings`; update `load()`/`save()` |
| `app/ui/theme.py` | Add `LATTE`/`MOCHA` dicts; add `build_stylesheet(p)`; update `Theme` class attrs to Catppuccin Latte defaults; add layout-only constants back |
| `app/ui/theme_manager.py` | **New.** `ThemeManager` singleton + `ThemeMode` enum |
| `app/ui/main_window.py` | `NavButton`: dynamic property `nav_active`; `SideBar`: `setObjectName("sidebar")`; `logo_row`: `setObjectName("logo_row")`; add `ThemeToggleButton`; remove hardcoded `#3498db` |
| `app/ui/pages/compare_page.py` | `_DIFF_CSS`/`_RISK_COLORS` → functions; scroll area gets `setObjectName("detail_scroll")`; `_apply_theme()` + `theme_changed` connection; fix hardcoded `#dde1ea`/`#374151` |
| `app/ui/pages/qa_page.py` | `_USER_BUBBLE_STYLE`/`_ASST_BUBBLE_STYLE` → functions; scroll area `setObjectName("chat_scroll")`; `_apply_theme()` + `theme_changed`; fix hardcoded `#dde1ea` |
| `app/ui/pages/home_page.py` | `_StatCard` stores color key; `_apply_theme()` on page + card |
| `app/ui/pages/library_page.py` | Store button/table refs; `_apply_theme()` + `theme_changed` |
| `app/ui/pages/settings_page.py` | `_apply_theme()` + `theme_changed` |
| `main.py` | Call `ThemeManager.instance().setup(settings, app)`; connect `theme_changed` to `_on_theme_changed` which updates toggle button |
| `tests/test_config.py` | Add 2 tests for `theme` field persistence |
| `tests/test_ui/test_theme.py` | **New.** 4 unit tests |

---

## Task 1: Add `theme` field to `AppSettings`

**Files:**
- Modify: `app/config/settings.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_config.py`:

```python
def test_settings_theme_defaults_to_light():
    s = AppSettings()
    assert s.theme == "light"


def test_settings_theme_persists_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    s = AppSettings(theme="dark")
    save(s)
    s2 = load()
    assert s2.theme == "dark"
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_config.py::test_settings_theme_defaults_to_light tests/test_config.py::test_settings_theme_persists_round_trip -v
```

Expected: FAIL (`AppSettings` has no `theme` field)

- [ ] **Step 3: Add `theme` field to `AppSettings` and update `load()`/`save()`**

In `app/config/settings.py`, change `AppSettings`:

```python
@dataclass
class AppSettings:
    providers: list[ProviderConfig] = field(default_factory=list)
    local_embedding: LocalEmbeddingConfig = field(default_factory=LocalEmbeddingConfig)
    active_provider: str = ""
    data_dir: str = field(default_factory=_default_data_dir)
    theme: str = "light"
```

In `load()`, add after `data_dir=...`:

```python
        theme=raw.get("theme", "light"),
```

In `save()`, add `"theme": settings.theme,` to the `raw` dict:

```python
    raw = {
        "providers": providers_raw,
        "local_embedding": {
            "enabled": settings.local_embedding.enabled,
            "model_path": settings.local_embedding.model_path,
        },
        "active_provider": settings.active_provider,
        "data_dir": settings.data_dir,
        "theme": settings.theme,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_config.py::test_settings_theme_defaults_to_light tests/test_config.py::test_settings_theme_persists_round_trip -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite to check no regressions**

```
uv run pytest tests/test_config.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add app/config/settings.py tests/test_config.py
git commit -m "feat: add theme field to AppSettings"
```

---

## Task 2: Rewrite `theme.py` with Catppuccin palettes

**Files:**
- Modify: `app/ui/theme.py`

- [ ] **Step 1: Replace `theme.py` entirely**

```python
"""Centralized UI theme — palette dicts, global QSS builder, and Theme compat shim."""
from __future__ import annotations


# ── Layout constants (theme-independent) ──────────────────────────────────────
SIDEBAR_WIDTH = 140
PAGE_MARGIN = 24
CARD_RADIUS = 8


# ── Catppuccin Latte (light) ──────────────────────────────────────────────────
LATTE: dict[str, str] = {
    # backgrounds
    "BG_SIDEBAR":      "#e6e9ef",
    "BG_PAGE":         "#eff1f5",
    "BG_CARD":         "#ffffff",
    "BG_HEADER":       "#dce0e8",
    # text
    "TEXT_PRIMARY":    "#209fb5",   # Sapphire — accent / heading color
    "TEXT_SECONDARY":  "#6c6f85",   # Subtext0 — muted labels
    "TEXT_PLACEHOLDER":"#9ca0b0",   # Overlay0
    # navigation
    "NAV_ACTIVE_BG":   "#209fb5",
    "NAV_ACTIVE_TEXT": "#ffffff",
    "NAV_TEXT":        "#6c6f85",
    "LOGO_COLOR":      "#04a5e5",   # Sky
    # borders
    "BORDER":          "#bcc0cc",   # Surface2
    # action palette
    "COLOR_PRIMARY":   "#209fb5",
    "COLOR_SUCCESS":   "#179299",   # Teal
    "COLOR_DANGER":    "#d20f39",   # Red
    "COLOR_WARNING":   "#df8e1d",   # Yellow
    "COLOR_COMPLETED": "#179299",
    # diff
    "DIFF_ADDED":      "#209fb5",
    "DIFF_DELETED":    "#dd7878",
    "DIFF_MINOR":      "#dc8a78",
    "DIFF_MAJOR":      "#df8e1d",
    "DIFF_REWRITE":    "#ea76cb",
    "DIFF_FORMAT":     "#8839ef",
}

# ── Catppuccin Mocha (dark) ───────────────────────────────────────────────────
MOCHA: dict[str, str] = {
    # backgrounds
    "BG_SIDEBAR":      "#181825",
    "BG_PAGE":         "#1e1e2e",
    "BG_CARD":         "#313244",
    "BG_HEADER":       "#11111b",
    # text
    "TEXT_PRIMARY":    "#74c7ec",   # Sapphire — accent / heading color
    "TEXT_SECONDARY":  "#a6adc8",   # Subtext0
    "TEXT_PLACEHOLDER":"#6c7086",   # Overlay0
    # navigation
    "NAV_ACTIVE_BG":   "#74c7ec",
    "NAV_ACTIVE_TEXT": "#1e1e2e",   # dark text on light Sapphire
    "NAV_TEXT":        "#7f849c",   # Overlay0
    "LOGO_COLOR":      "#89dceb",   # Sky
    # borders
    "BORDER":          "#45475a",   # Surface1
    # action palette
    "COLOR_PRIMARY":   "#74c7ec",
    "COLOR_SUCCESS":   "#94e2d5",   # Teal
    "COLOR_DANGER":    "#f38ba8",   # Red
    "COLOR_WARNING":   "#f9e2af",   # Yellow
    "COLOR_COMPLETED": "#94e2d5",
    # diff
    "DIFF_ADDED":      "#74c7ec",
    "DIFF_DELETED":    "#f2cdcd",
    "DIFF_MINOR":      "#f5e0dc",
    "DIFF_MAJOR":      "#f9e2af",
    "DIFF_REWRITE":    "#f5c2e7",
    "DIFF_FORMAT":     "#cba6f7",
}


def build_stylesheet(p: dict) -> str:
    """Return global QSS for objectName/property-targeted elements.

    Only covers widgets that use setObjectName or dynamic properties.
    Per-widget inline styles are handled by each page's _apply_theme().
    """
    return f"""
    QWidget#sidebar {{
        background-color: {p["BG_SIDEBAR"]};
    }}
    QWidget#logo_row {{
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
        border-bottom: 2px solid {p["TEXT_PRIMARY"]};
        border-right: 2px solid {p["TEXT_PRIMARY"]};
    }}
    QPushButton[nav_active="true"] {{
        background-color: {p["NAV_ACTIVE_BG"]};
        color: {p["NAV_ACTIVE_TEXT"]};
        border: none;
        padding: 12px 8px;
        text-align: left;
        font-size: 14px;
        border-radius: 6px;
    }}
    QPushButton[nav_active="false"] {{
        background-color: transparent;
        color: {p["NAV_TEXT"]};
        border: none;
        padding: 12px 8px;
        text-align: left;
        font-size: 14px;
        border-radius: 6px;
    }}
    QScrollArea#detail_scroll {{
        border: 1px solid {p["BORDER"]};
        border-radius: 4px;
    }}
    QScrollArea#chat_scroll {{
        border: 1px solid {p["BORDER"]};
        border-radius: 4px;
    }}
    """


# ── Theme compatibility shim ──────────────────────────────────────────────────
# Attributes are populated from LATTE at import; ThemeManager updates them on switch.
class Theme:
    # Layout (never change)
    SIDEBAR_WIDTH = SIDEBAR_WIDTH
    PAGE_MARGIN = PAGE_MARGIN
    CARD_RADIUS = CARD_RADIUS

    # Color attributes — initial values from LATTE, updated by ThemeManager
    BG_SIDEBAR      = LATTE["BG_SIDEBAR"]
    BG_PAGE         = LATTE["BG_PAGE"]
    BG_CARD         = LATTE["BG_CARD"]
    BG_HEADER       = LATTE["BG_HEADER"]
    TEXT_PRIMARY    = LATTE["TEXT_PRIMARY"]
    TEXT_SECONDARY  = LATTE["TEXT_SECONDARY"]
    TEXT_PLACEHOLDER= LATTE["TEXT_PLACEHOLDER"]
    NAV_ACTIVE_BG   = LATTE["NAV_ACTIVE_BG"]
    NAV_ACTIVE_TEXT = LATTE["NAV_ACTIVE_TEXT"]
    NAV_TEXT        = LATTE["NAV_TEXT"]
    LOGO_COLOR      = LATTE["LOGO_COLOR"]
    BORDER          = LATTE["BORDER"]
    COLOR_PRIMARY   = LATTE["COLOR_PRIMARY"]
    COLOR_SUCCESS   = LATTE["COLOR_SUCCESS"]
    COLOR_DANGER    = LATTE["COLOR_DANGER"]
    COLOR_WARNING   = LATTE["COLOR_WARNING"]
    COLOR_COMPLETED = LATTE["COLOR_COMPLETED"]
    DIFF_ADDED      = LATTE["DIFF_ADDED"]
    DIFF_DELETED    = LATTE["DIFF_DELETED"]
    DIFF_MINOR      = LATTE["DIFF_MINOR"]
    DIFF_MAJOR      = LATTE["DIFF_MAJOR"]
    DIFF_REWRITE    = LATTE["DIFF_REWRITE"]
    DIFF_FORMAT     = LATTE["DIFF_FORMAT"]

    # QSS helper classmethods — read cls.X at call time, so they reflect current theme
    @classmethod
    def btn_primary(cls) -> str:
        return (
            f"background-color:{cls.COLOR_PRIMARY};color:white;"
            f"border:none;border-radius:{CARD_RADIUS}px;padding:8px 16px;font-size:13px;"
        )

    @classmethod
    def btn_success(cls) -> str:
        return (
            f"background-color:{cls.COLOR_SUCCESS};color:white;"
            f"border:none;border-radius:{CARD_RADIUS}px;padding:8px 16px;font-size:13px;"
        )

    @classmethod
    def btn_danger(cls) -> str:
        return (
            f"background-color:{cls.COLOR_DANGER};color:white;"
            f"border:none;border-radius:{CARD_RADIUS}px;padding:8px 16px;font-size:13px;"
        )

    @classmethod
    def card(cls) -> str:
        return (
            f"background:{cls.BG_CARD};border:1px solid {cls.BORDER};"
            f"border-radius:{CARD_RADIUS}px;"
        )

    @classmethod
    def label_primary(cls) -> str:
        return f"color:{cls.TEXT_PRIMARY};font-size:13px;"

    @classmethod
    def label_secondary(cls) -> str:
        return f"color:{cls.TEXT_SECONDARY};font-size:12px;"

    @classmethod
    def page_title(cls) -> str:
        return f"color:{cls.TEXT_PRIMARY};font-size:22px;font-weight:bold;"
```

- [ ] **Step 2: Verify import works**

```
uv run python -c "from app.ui.theme import Theme, LATTE, MOCHA, build_stylesheet; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/ui/theme.py
git commit -m "feat: add Catppuccin LATTE/MOCHA palettes and build_stylesheet to theme.py"
```

---

## Task 3: Create `theme_manager.py`

**Files:**
- Create: `app/ui/theme_manager.py`

- [ ] **Step 1: Create `app/ui/theme_manager.py`**

```python
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
            logger.warning("Failed to save theme setting")
```

- [ ] **Step 2: Verify import**

```
uv run python -c "from app.ui.theme_manager import ThemeManager, ThemeMode; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/ui/theme_manager.py
git commit -m "feat: add ThemeManager singleton"
```

---

## Task 4: Write and run unit tests

**Files:**
- Create: `tests/test_ui/test_theme.py`

- [ ] **Step 1: Create `tests/test_ui/test_theme.py`**

```python
"""Unit tests for theme palette, build_stylesheet, and ThemeManager."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_build_stylesheet_latte_contains_primary():
    from app.ui.theme import LATTE, build_stylesheet
    qss = build_stylesheet(LATTE)
    assert LATTE["NAV_ACTIVE_BG"] in qss


def test_build_stylesheet_mocha_contains_primary():
    from app.ui.theme import MOCHA, build_stylesheet
    qss = build_stylesheet(MOCHA)
    assert MOCHA["NAV_ACTIVE_BG"] in qss


def test_theme_manager_toggle_cycles_light_dark_light():
    from app.ui.theme_manager import ThemeManager, ThemeMode

    # Reset singleton for test isolation
    ThemeManager._instance = None
    tm = ThemeManager.instance()

    assert tm.mode() == ThemeMode.LIGHT
    tm._app = MagicMock()
    tm._settings = MagicMock()

    tm.toggle()
    assert tm.mode() == ThemeMode.DARK

    tm.toggle()
    assert tm.mode() == ThemeMode.LIGHT

    # Cleanup
    ThemeManager._instance = None


def test_theme_manager_persists_mode_to_settings():
    from app.ui.theme_manager import ThemeManager, ThemeMode

    ThemeManager._instance = None
    tm = ThemeManager.instance()
    mock_settings = MagicMock()
    mock_settings.theme = "light"
    tm._app = MagicMock()
    tm._settings = mock_settings

    tm.toggle()  # light → dark
    assert mock_settings.theme == "dark"

    ThemeManager._instance = None


def test_fa_font_load_returns_true_when_otf_exists():
    otf_path = (
        Path(__file__).parent.parent.parent
        / "assets/fonts/fontawesome-free-7.2.0-desktop/otfs"
        / "Font Awesome 7 Free-Solid-900.otf"
    )
    assert otf_path.exists(), f"OTF not found: {otf_path}"
```

Note: `test_fa_font_load_returns_true_when_otf_exists` only checks the file exists (not that `QFontDatabase` loads it — that requires a `QApplication` which is out of scope for unit tests).

- [ ] **Step 2: Run tests**

```
uv run pytest tests/test_ui/test_theme.py -v
```

Expected: all 5 pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_ui/test_theme.py
git commit -m "test: add unit tests for theme palettes and ThemeManager"
```

---

## Task 5: Update `main_window.py`

**Files:**
- Modify: `app/ui/main_window.py`

Changes:
1. `NavButton` uses dynamic property `nav_active` — no individual `setStyleSheet()` calls
2. `SideBar` sets `objectName("sidebar")` and `logo_row` sets `objectName("logo_row")` — removes hardcoded `#3498db`
3. `SideBar` connects `theme_changed` to `_apply_theme()` to re-style logo text and settings row
4. Add `ThemeToggleButton` (`QToolButton` with FA glyph) to `MainWindow` top bar
5. `MainWindow` keeps its `setStyleSheet` but updates on `theme_changed`

- [ ] **Step 1: Rewrite `app/ui/main_window.py`**

Replace the entire file with:

```python
"""Main application window with sidebar navigation."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.app_context import AppContext
from app.ui.theme import Theme

_NAV_ITEMS = [
    ("首页",     0),
    ("文档对比", 1),
    ("标准库",   2),
    ("智能问答", 3),
]

_WINDOW_TITLE = "Doc-Diff-Agent"
_WINDOW_SIZE = (1280, 800)
_ICON_PATH = Path(__file__).parent.parent.parent / "assets" / "icons" / "docdiff.png"
_FA_SOLID_OTF = (
    Path(__file__).parent.parent.parent
    / "assets/fonts/fontawesome-free-7.2.0-desktop/otfs"
    / "Font Awesome 7 Free-Solid-900.otf"
)
_FA_SOLID_FAMILY: str = ""

_HARMONY_SANS_DIR = Path(__file__).parent.parent.parent / "assets" / "fonts" / "HarmonyOS_Sans"
_HARMONY_FAMILY: str = ""

# FA glyphs: moon = shown in light mode (click → go dark), sun = shown in dark mode
_FA_MOON = "\uf186"
_FA_SUN  = "\uf185"


def _load_fa_solid() -> str:
    global _FA_SOLID_FAMILY
    if not _FA_SOLID_FAMILY and _FA_SOLID_OTF.exists():
        fid = QFontDatabase.addApplicationFont(str(_FA_SOLID_OTF))
        if fid >= 0:
            families = QFontDatabase.applicationFontFamilies(fid)
            if families:
                _FA_SOLID_FAMILY = families[0]
    return _FA_SOLID_FAMILY


def load_harmony_sans() -> str:
    global _HARMONY_FAMILY
    if _HARMONY_FAMILY:
        return _HARMONY_FAMILY
    for weight_file in (
        "HarmonyOS_SansSC_Regular.ttf",
        "HarmonyOS_SansSC_Bold.ttf",
        "HarmonyOS_SansSC_Medium.ttf",
        "HarmonyOS_SansSC_Light.ttf",
    ):
        path = _HARMONY_SANS_DIR / weight_file
        if path.exists():
            fid = QFontDatabase.addApplicationFont(str(path))
            if fid >= 0 and not _HARMONY_FAMILY:
                families = QFontDatabase.applicationFontFamilies(fid)
                if families:
                    _HARMONY_FAMILY = families[0]
    return _HARMONY_FAMILY


class NavButton(QPushButton):
    """Navigation button whose active/inactive style is driven by global QSS
    via the dynamic property ``nav_active``."""

    def __init__(self, label: str, parent=None):
        super().__init__(label, parent)
        self._is_active = False
        self.setProperty("nav_active", "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(Theme.SIDEBAR_WIDTH - 16)

    def set_active(self, active: bool) -> None:
        self._is_active = active
        self.setProperty("nav_active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class ThemeToggleButton(QToolButton):
    """Top-right button that shows a moon (light mode) or sun (dark mode) glyph."""

    def __init__(self, parent=None):
        super().__init__(parent)
        fa_family = _load_fa_solid()
        if fa_family:
            self.setFont(QFont(fa_family, 16))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("border:none;padding:4px 8px;background:transparent;")
        self._update_glyph()

    def _update_glyph(self) -> None:
        from app.ui.theme_manager import ThemeManager, ThemeMode
        if ThemeManager.instance().mode() == ThemeMode.LIGHT:
            self.setText(_FA_MOON)
        else:
            self.setText(_FA_SUN)
        if not self.font().family():
            # FA font not loaded — fall back to emoji
            self.setText("☀" if ThemeManager.instance().mode() == ThemeMode.DARK else "🌙")

    def on_theme_changed(self) -> None:
        self._update_glyph()


class SideBar(QWidget):
    def __init__(self, on_navigate, on_settings, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(Theme.SIDEBAR_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 16, 8, 16)
        layout.setSpacing(4)

        # Logo row — objectName used by global QSS for border color
        logo_row = QWidget()
        logo_row.setObjectName("logo_row")
        logo_layout = QHBoxLayout(logo_row)
        logo_layout.setContentsMargins(4, 4, 4, 4)
        logo_layout.setSpacing(8)

        logo_img = QLabel()
        if _ICON_PATH.exists():
            pix = QPixmap(str(_ICON_PATH)).scaled(
                28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )  # type: ignore
            logo_img.setPixmap(pix)
            logo_img.setStyleSheet("border: none;")
        logo_layout.addWidget(logo_img)

        self._logo_text = QLabel("DocDiff")
        _harmony = load_harmony_sans()
        self._logo_text.setFont(QFont(_harmony or "Segoe UI", 14, QFont.Weight.Bold))
        self._logo_text.setStyleSheet(f"color:{Theme.LOGO_COLOR};border: none;")
        logo_layout.addWidget(self._logo_text)
        logo_layout.addStretch()
        layout.addWidget(logo_row)
        layout.addSpacing(16)

        self._buttons: list[NavButton] = []
        for label, idx in _NAV_ITEMS:
            btn = NavButton(label)
            btn.setFont(QFont(_harmony or "Segoe UI", 14, QFont.Weight.Bold))
            btn.clicked.connect(lambda checked, i=idx: on_navigate(i))
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # Settings row
        self._settings_row = QWidget()
        self._settings_row.setFixedWidth(Theme.SIDEBAR_WIDTH - 16)
        self._settings_row.setCursor(Qt.CursorShape.PointingHandCursor)
        _sr_layout = QHBoxLayout(self._settings_row)
        _sr_layout.setContentsMargins(8, 12, 8, 12)
        _sr_layout.setSpacing(6)

        self._settings_icon = QLabel("\uf013")
        fa_family = _load_fa_solid()
        if fa_family:
            self._settings_icon.setFont(QFont(fa_family, 14))
        _sr_layout.addWidget(self._settings_icon)

        self._settings_text = QLabel("设置")
        if _harmony:
            self._settings_text.setFont(QFont(_harmony, 16, QFont.Weight.Bold))
        _sr_layout.addWidget(self._settings_text)
        _sr_layout.addStretch()

        self._settings_row.mousePressEvent = lambda e: on_settings()
        layout.addWidget(self._settings_row)

        self._set_active(0)
        self._apply_theme()

        from app.ui.theme_manager import ThemeManager
        ThemeManager.instance().theme_changed.connect(self._apply_theme)

    def _apply_theme(self) -> None:
        self.setStyleSheet("")   # let global QSS handle sidebar bg via objectName
        self._logo_text.setStyleSheet(f"color:{Theme.LOGO_COLOR};border:none;")
        self._settings_icon.setStyleSheet(f"color:{Theme.NAV_TEXT};border:none;")
        self._settings_text.setStyleSheet(f"color:{Theme.NAV_TEXT};border:none;font-size:16px;")
        self._settings_row.setStyleSheet(
            f"background-color:transparent;border-top:1px solid {Theme.BORDER};"
        )
        # Re-apply nav button active states so QSS re-polishes them
        for btn in self._buttons:
            btn.set_active(btn._is_active)

    def _set_active(self, index: int) -> None:
        for i, btn in enumerate(self._buttons):
            btn.set_active(i == index)

    def navigate(self, index: int) -> None:
        self._set_active(index)


class MainWindow(QMainWindow):

    settings_requested = Signal()

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle(_WINDOW_TITLE)
        self.resize(*_WINDOW_SIZE)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._sidebar = SideBar(self._on_navigate, self._on_settings_btn)
        root_layout.addWidget(self._sidebar)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Top bar with theme toggle
        top_bar = QWidget()
        top_bar.setFixedHeight(40)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(0, 0, 8, 0)
        top_bar_layout.addStretch()

        self._theme_btn = ThemeToggleButton()
        self._theme_btn.clicked.connect(self._on_theme_toggle)
        top_bar_layout.addWidget(self._theme_btn)
        right_layout.addWidget(top_bar)

        self._stack = QStackedWidget()
        right_layout.addWidget(self._stack, 1)

        root_layout.addWidget(right_panel, 1)

        for label, _ in _NAV_ITEMS:
            placeholder = QLabel(f"{label} 页面")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(f"font-size:24px;color:{Theme.TEXT_PLACEHOLDER};")
            self._stack.addWidget(placeholder)

        self._apply_theme()

        from app.ui.theme_manager import ThemeManager
        ThemeManager.instance().theme_changed.connect(self._apply_theme)
        ThemeManager.instance().theme_changed.connect(self._theme_btn.on_theme_changed)

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
        self._stack.setStyleSheet(f"background-color:{Theme.BG_PAGE};")

    def add_page(self, index: int, widget: QWidget) -> None:
        old = self._stack.widget(index)
        self._stack.insertWidget(index, widget)
        if old is not None:
            self._stack.removeWidget(old)
            old.deleteLater()

    def _on_navigate(self, index: int) -> None:
        self._sidebar.navigate(index)
        self._stack.setCurrentIndex(index)

    def _on_settings_btn(self) -> None:
        self.settings_requested.emit()

    def _on_theme_toggle(self) -> None:
        from app.ui.theme_manager import ThemeManager
        ThemeManager.instance().toggle()

    def navigate_to(self, index: int) -> None:
        self._on_navigate(index)
```

- [ ] **Step 2: Verify existing main_window import test still passes**

```
uv run pytest tests/test_ui/test_main_window_import.py -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add app/ui/main_window.py
git commit -m "feat: add ThemeToggleButton and NavButton dynamic QSS property in main_window"
```

---

## Task 6: Update `compare_page.py`

**Files:**
- Modify: `app/ui/pages/compare_page.py`

Changes:
1. `_DIFF_CSS` and `_RISK_COLORS` → become functions (called at use-time, reading current `Theme.X`)
2. `self._detail_scroll.setObjectName("detail_scroll")` — removes hardcoded `#dde1ea`
3. `_apply_theme()` method to re-style all styled widgets
4. Connect `theme_changed` to `_apply_theme()`
5. Fix `exp_lbl.setStyleSheet("color:#374151...")` to use `Theme.TEXT_SECONDARY`

- [ ] **Step 1: Convert `_DIFF_CSS` and `_RISK_COLORS` to functions**

Find `_DIFF_CSS = {...}` and `_RISK_COLORS = {...}` at module level (lines ~41-52) and replace with:

```python
def _diff_css() -> dict:
    return {
        "新增":     ("added",   Theme.DIFF_ADDED),
        "删减":     ("deleted", Theme.DIFF_DELETED),
        "微调":     ("minor",   Theme.DIFF_MINOR),
        "实质修改": ("major",   Theme.DIFF_MAJOR),
        "重写":     ("rewrite", Theme.DIFF_REWRITE),
        "格式变化": ("format",  Theme.DIFF_FORMAT),
    }


def _risk_colors() -> dict:
    return {
        "high":   Theme.DIFF_DELETED,
        "medium": Theme.DIFF_MAJOR,
        "low":    Theme.DIFF_ADDED,
    }
```

- [ ] **Step 2: Update all use-sites of `_DIFF_CSS` and `_RISK_COLORS`**

Search for `_DIFF_CSS.get(` and `_RISK_COLORS.get(` and replace:

```python
# was: css_cls, _ = _DIFF_CSS.get(item.diff_type, ("format", Theme.DIFF_FORMAT))
css_cls, _ = _diff_css().get(item.diff_type, ("format", Theme.DIFF_FORMAT))

# was: css_cls, color = _DIFF_CSS.get(item.diff_type, ("format", Theme.DIFF_FORMAT))
css_cls, color = _diff_css().get(item.diff_type, ("format", Theme.DIFF_FORMAT))

# was: risk_color = _RISK_COLORS.get(item.risk_level, Theme.TEXT_SECONDARY)
risk_color = _risk_colors().get(item.risk_level, Theme.TEXT_SECONDARY)
```

- [ ] **Step 3: Set objectName on detail scroll area**

Find `self._detail_scroll = QScrollArea()` and add after it:

```python
self._detail_scroll.setObjectName("detail_scroll")
```

Remove its `setStyleSheet()` call (the `border: 1px solid #dde1ea` one — now handled by global QSS).

- [ ] **Step 4: Store widget refs and add `_apply_theme()`**

After `self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")` at the top of `__init__` (or `_build_ui()`), store references to styled widgets that need refreshing. Then add a `_apply_theme()` method.

Find the widget creation for `self._run_btn`, `self._loading_label`, `self._export_btn` (the outline button). Add `_apply_theme()`:

```python
def _apply_theme(self) -> None:
    self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
    self._run_btn.setStyleSheet(Theme.btn_primary())
    self._loading_label.setStyleSheet(Theme.label_secondary())
    self._export_btn.setStyleSheet(
        f"background-color:transparent;color:{Theme.TEXT_PRIMARY};"
        f"border:1px solid {Theme.TEXT_PRIMARY};padding:6px 14px;"
        f"border-radius:{Theme.CARD_RADIUS}px;font-size:13px;"
    )
```

(Identify the export button — it's the outline-style button near `self._run_btn`. Store it as `self._export_btn` during `_build_ui()`.)

- [ ] **Step 5: Fix `exp_lbl` inline style and connect `theme_changed`**

Find `exp_lbl.setStyleSheet("color:#374151;font-size:12px;")` and replace with:

```python
exp_lbl.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:12px;")
```

In `__init__` (after `_build_ui()`), add:

```python
from app.ui.theme_manager import ThemeManager
ThemeManager.instance().theme_changed.connect(self._apply_theme)
```

- [ ] **Step 6: Run retrieval + agent tests to check nothing broke**

```
uv run pytest tests/test_diff/ tests/test_services/ -v
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add app/ui/pages/compare_page.py
git commit -m "feat: theme-aware compare_page — _diff_css() functions, objectName scroll area, _apply_theme()"
```

---

## Task 7: Update `qa_page.py`

**Files:**
- Modify: `app/ui/pages/qa_page.py`

Changes:
1. `_USER_BUBBLE_STYLE` and `_ASST_BUBBLE_STYLE` → module-level functions
2. `self._chat_scroll.setObjectName("chat_scroll")` — removes hardcoded `#dde1ea`
3. `_apply_theme()` + `theme_changed` connection

- [ ] **Step 1: Convert module-level style strings to functions**

Find `_USER_BUBBLE_STYLE = (...)` and `_ASST_BUBBLE_STYLE = (...)` and replace with:

```python
def _user_bubble_style() -> str:
    return (
        f"background:{Theme.COLOR_PRIMARY};color:white;"
        "border-radius:12px;padding:10px;margin:4px 0;"
    )


def _asst_bubble_style() -> str:
    return (
        f"background:{Theme.BG_CARD};border:1px solid {Theme.BORDER};"
        "border-radius:12px;padding:10px;margin:4px 0;"
    )
```

- [ ] **Step 2: Update all use-sites**

Search for `_USER_BUBBLE_STYLE` and `_ASST_BUBBLE_STYLE` in the file. Replace with calls:

```python
# was: bubble.setStyleSheet(_USER_BUBBLE_STYLE)
bubble.setStyleSheet(_user_bubble_style())

# was: bubble.setStyleSheet(_ASST_BUBBLE_STYLE)
bubble.setStyleSheet(_asst_bubble_style())
```

- [ ] **Step 3: Set objectName on chat scroll area and remove its hardcoded setStyleSheet**

Find `self._chat_scroll = QScrollArea()` and add:

```python
self._chat_scroll.setObjectName("chat_scroll")
```

Remove the `self._chat_scroll.setStyleSheet("""...""")` call that had `border: 1px solid #dde1ea`.

- [ ] **Step 4: Add `_apply_theme()` and connect `theme_changed`**

Add method to `QaPage`:

```python
def _apply_theme(self) -> None:
    self.setStyleSheet(f"background-color:{Theme.BG_CARD};")
    self._new_session_btn.setStyleSheet(Theme.btn_primary())
    self._send_btn.setStyleSheet(Theme.btn_primary())
```

(Store `new_session_btn` as `self._new_session_btn` and `send_btn` as `self._send_btn` during `_build_ui()`.)

In `__init__` after `_build_ui()`:

```python
from app.ui.theme_manager import ThemeManager
ThemeManager.instance().theme_changed.connect(self._apply_theme)
```

- [ ] **Step 5: Run QA tests**

```
uv run pytest tests/test_agent/ -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add app/ui/pages/qa_page.py
git commit -m "feat: theme-aware qa_page — bubble style functions, objectName scroll area, _apply_theme()"
```

---

## Task 8: Update `home_page.py`, `library_page.py`, `settings_page.py`

**Files:**
- Modify: `app/ui/pages/home_page.py`
- Modify: `app/ui/pages/library_page.py`
- Modify: `app/ui/pages/settings_page.py`

### `home_page.py`

- [ ] **Step 1: Make `_StatCard` theme-aware**

Change `_StatCard.__init__` to store the color hex (read at construction time — already updated by ThemeManager before cards are created). Add `_apply_color(color)` method:

```python
class _StatCard(QWidget):
    def __init__(self, label: str, value: str, color: str = Theme.COLOR_PRIMARY, parent=None):
        super().__init__(parent)
        self._label_text = label
        self._color_hex = color   # store to re-apply on theme change
        self._val_lbl = QLabel(value)
        self._lbl = QLabel(label)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        self._val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._val_lbl)
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl)
        self._apply_color(color)

    def _apply_color(self, color: str) -> None:
        self._color_hex = color
        from PySide6.QtGui import QColor
        _c = QColor(color)
        _c.setAlpha(50)
        self.setStyleSheet(
            f"background:{_c.name(QColor.NameFormat.HexArgb)};border:1px solid {color};"
            f"border-radius:{Theme.CARD_RADIUS}px;padding:12px;"
        )
        self._val_lbl.setStyleSheet(f"font-size:26px;font-weight:bold;color:{color};")
        self._lbl.setStyleSheet(Theme.label_secondary() + f"font-size:14px;color:{color};")

    def update_value(self, value: str) -> None:
        self._val_lbl.setText(value)

    def refresh_theme(self) -> None:
        self._apply_color(self._color_hex)
```

- [ ] **Step 2: Store action button refs and add `_apply_theme()` to `HomePage`**

During `_build_ui()`, store styled buttons as instance attributes. Add `_apply_theme()`:

```python
def _apply_theme(self) -> None:
    self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
    self._title.setStyleSheet(Theme.page_title())
    self._subtitle.setStyleSheet(Theme.label_secondary() + "font-size:14px;")
    # Stat cards: re-apply with current palette colors
    self._card_docs.refresh_theme()
    self._card_tasks.refresh_theme()
    self._card_done.refresh_theme()
    # Action buttons
    for btn, color in self._action_buttons:
        btn.setStyleSheet(
            f"background-color:{color};color:white;padding:10px 20px;"
            f"border:none;border-radius:{Theme.CARD_RADIUS}px;font-size:16px;"
        )
```

Store `self._title`, `self._subtitle`, and `self._action_buttons: list[tuple[QPushButton, str]]` (button + color_key_value) during `_build_ui()`. For action buttons, the colors come from `Theme.COLOR_PRIMARY`, `Theme.COLOR_SUCCESS`, `Theme.COLOR_COMPLETED` — read those at `_apply_theme()` call time by fetching them fresh from `Theme`.

Simplest approach for action buttons — store them with their palette attribute name:

```python
self._action_buttons = [
    (btn_import, "COLOR_PRIMARY"),
    (btn_compare, "COLOR_SUCCESS"),
    (btn_qa,     "COLOR_COMPLETED"),
]
```

Then `_apply_theme()`:

```python
for btn, color_attr in self._action_buttons:
    color = getattr(Theme, color_attr)
    btn.setStyleSheet(
        f"background-color:{color};color:white;padding:10px 20px;"
        f"border:none;border-radius:{Theme.CARD_RADIUS}px;font-size:16px;"
    )
```

In `__init__` after `_build_ui()`:

```python
from app.ui.theme_manager import ThemeManager
ThemeManager.instance().theme_changed.connect(self._apply_theme)
```

### `library_page.py`

- [ ] **Step 3: Store refs and add `_apply_theme()` to `LibraryPage`**

In `_build_ui()`, store:
- `self._import_btn` (the "导入标准文档" button)
- `self._table` (already stored as `self._table`)

Add `_apply_theme()`:

```python
def _apply_theme(self) -> None:
    self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
    self._title.setStyleSheet(Theme.page_title())
    self._import_btn.setStyleSheet(Theme.btn_primary())
    self._add_version_btn.setStyleSheet(Theme.btn_success())
    self._table.setStyleSheet(
        f"QTableWidget {{ background:{Theme.BG_CARD};gridline-color:{Theme.BORDER}; }}"
        f"QHeaderView::section {{ background:{Theme.BG_HEADER};color:{Theme.TEXT_PRIMARY};"
        f"border:1px solid {Theme.BORDER};padding:4px; }}"
    )
    self._status.setStyleSheet(Theme.label_secondary())
```

Store `self._title` and `self._import_btn` during `_build_ui()`.

In `__init__` after `_build_ui()`:

```python
from app.ui.theme_manager import ThemeManager
ThemeManager.instance().theme_changed.connect(self._apply_theme)
```

### `settings_page.py`

- [ ] **Step 4: Add `_apply_theme()` to `SettingsDialog`**

Read `settings_page.py` to understand its structure, then add `_apply_theme()` that re-applies all `Theme.X` references. Connect to `theme_changed`.

The key styled elements (from the grep output) are:
- Page background: `self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")`
- `self._backup_label`, `self._restore_label`, `self._update_status` labels with `Theme.TEXT_SECONDARY`
- Outline buttons with `Theme.TEXT_PRIMARY` and `Theme.BORDER`
- Save button with `Theme.TEXT_PRIMARY` background

Store these widgets as instance attributes during `_build_ui()` and re-apply in `_apply_theme()`:

```python
def _apply_theme(self) -> None:
    self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
    self._backup_label.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:14px;")
    self._restore_label.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:14px;")
    self._update_status.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:14px;")
    for btn in self._outline_btns:
        btn.setStyleSheet(
            f"background-color:transparent;color:{Theme.TEXT_PRIMARY};"
            f"border:1px solid {Theme.BORDER};padding:8px 20px;"
            f"border-radius:{Theme.CARD_RADIUS}px;font-size:14px;"
        )
    for btn in self._primary_btns:
        btn.setStyleSheet(
            f"background-color:{Theme.TEXT_PRIMARY};color:white;"
            f"border:none;padding:8px 20px;border-radius:{Theme.CARD_RADIUS}px;font-size:14px;"
        )
```

Store `self._outline_btns: list[QPushButton]` and `self._primary_btns: list[QPushButton]` during `_build_ui()`.

In `__init__` after `_build_ui()`:

```python
from app.ui.theme_manager import ThemeManager
ThemeManager.instance().theme_changed.connect(self._apply_theme)
```

- [ ] **Step 5: Run full test suite**

```
uv run pytest -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add app/ui/pages/home_page.py app/ui/pages/library_page.py app/ui/pages/settings_page.py
git commit -m "feat: theme-aware home, library, settings pages — _apply_theme() methods"
```

---

## Task 9: Wire up `main.py`

**Files:**
- Modify: `main.py`

Changes: call `ThemeManager.instance().setup(settings, app)` after `QApplication` is created and fonts are loaded.

- [ ] **Step 1: Add ThemeManager setup to `main()`**

In `main.py`, after `harmony_family = load_harmony_sans()` (and its font application), add:

```python
from app.ui.theme_manager import ThemeManager
ThemeManager.instance().setup(settings, app)
```

Full context of the change (the relevant section of `main()`):

```python
    settings = load()
    # ... data_dir, conn setup ...

    harmony_family = load_harmony_sans()
    if harmony_family:
        app.setFont(QFont(harmony_family, 13))

    # Apply theme before building any UI widgets
    from app.ui.theme_manager import ThemeManager
    ThemeManager.instance().setup(settings, app)

    from app.core.model.factory import get_embedder, get_provider
    # ... rest of main() unchanged ...
```

- [ ] **Step 2: Verify app starts without errors**

```
uv run python -c "
import sys
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)
from app.config.settings import load
settings = load()
from app.ui.theme_manager import ThemeManager
ThemeManager.instance().setup(settings, app)
print('ThemeManager setup OK, mode:', ThemeManager.instance().mode())
"
```

Expected: `ThemeManager setup OK, mode: ThemeMode.LIGHT`

- [ ] **Step 3: Run full test suite**

```
uv run pytest -v
```

Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: wire ThemeManager setup in main() entry point"
```

---

## Task 10: Manual verification

- [ ] **Start the app**

```
python main.py
```

- [ ] **Verify light theme**
  - App launches with Catppuccin Latte colors (sidebar `#e6e9ef`, page bg `#eff1f5`)
  - Top-right shows moon icon (`🌙` or FA moon glyph)
  - Active nav button is highlighted in Sapphire `#209fb5`

- [ ] **Toggle to dark theme**
  - Click the moon icon → app switches to Catppuccin Mocha
  - Sidebar becomes `#181825`, page bg `#1e1e2e`
  - Toggle button now shows sun icon
  - Nav button active state in light Sapphire `#74c7ec`

- [ ] **Verify persistence**
  - Close and relaunch the app — dark theme is restored

- [ ] **Check compare page**
  - Run a comparison — diff cards render with correct theme colors

- [ ] **Check QA page**
  - Ask a question — user/AI bubbles use correct theme colors

- [ ] **Final commit**

```bash
git add -p   # stage only intentional changes
git commit -m "feat: Catppuccin Latte/Mocha theme system with FontAwesome toggle button"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ Catppuccin Latte and Mocha palettes defined
- ✅ FontAwesome toggle button in top-right (`ThemeToggleButton`)
- ✅ Theme persists to `config.json` via `AppSettings.theme`
- ✅ Instant switch (no animation) via `app.setStyleSheet()`
- ✅ All hardcoded colors migrated: `#3498db` → global QSS `TEXT_PRIMARY`; `#dde1ea` → objectName selectors; `#374151` → `Theme.TEXT_SECONDARY`
- ✅ `NavButton` styles in global QSS via `nav_active` property
- ✅ Module-level style strings that referenced `Theme.X` at import time → converted to functions
- ✅ 4 unit tests + 2 config tests

**Type consistency:**
- `ThemeManager.instance()` used consistently throughout; no alternative accessor names
- `Theme.CARD_RADIUS` used (not `Theme.card_radius`) in all style methods
- `_apply_theme()` method name used consistently across all pages
