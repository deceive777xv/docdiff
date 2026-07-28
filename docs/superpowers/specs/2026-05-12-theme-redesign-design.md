# Theme Redesign Design — Catppuccin Light/Dark with Toggle

**Date:** 2026-05-12  
**Status:** Approved

## Overview

Redesign the application's color scheme using the Catppuccin palette (Latte for light, Mocha for dark), add a dark theme, and provide a FontAwesome icon toggle button in the top-right corner of the main window. Theme preference persists to `config.json`. Switching is instant (no animation), implemented via global QSS replacement.

---

## Architecture

### New / Modified Files

| File | Change |
|------|--------|
| `app/ui/theme.py` | Add `LATTE` and `MOCHA` palette dicts; add `build_stylesheet(palette)` function; add `load_fa_font()` to register FontAwesome OTF via `QFontDatabase` |
| `app/ui/theme_manager.py` | **New.** `ThemeManager` singleton: holds current `Theme` enum, exposes `toggle()`, `apply(app)`, `palette()`. Emits `theme_changed` Qt signal. Reads/writes `AppSettings.theme`. |
| `app/ui/main_window.py` | Add `ThemeToggleButton` (`QToolButton` with FontAwesome glyph `\uf186`/`\uf185`). Connect to `ThemeManager.toggle()`. Fix `NavButton` styles to call a function instead of class-level constants. Move inline hardcoded colors (resize handle) into global QSS. |
| `app/ui/pages/compare_page.py` | Remove inline `setStyleSheet()` with hardcoded colors; set `objectName` on scroll area (moved to global QSS); use `ThemeManager.palette()["text_muted"]` for dynamically-created explanation labels. |
| `app/ui/pages/qa_page.py` | Remove inline `setStyleSheet()` with hardcoded border color; set `objectName` on chat scroll area (moved to global QSS). |
| `app/config/settings.py` | Add `theme: str = "light"` field to `AppSettings`. |

### Runtime Flow

```
App startup
  → read config.json → AppSettings.theme ("light" | "dark")
  → ThemeManager.apply(app) → app.setStyleSheet(build_stylesheet(palette))
  → load_fa_font() → register FontAwesome OTF

User clicks ThemeToggleButton
  → ThemeManager.toggle()
  → app.setStyleSheet(build_stylesheet(new_palette))
  → ThemeToggleButton updates glyph (moon ↔ sun)
  → AppSettings.theme updated → settings.save()
```

---

## Color Palettes

### Catppuccin Latte (Light)

| Key | Hex | Role |
|-----|-----|------|
| `bg_page` | `#eff1f5` | Page background |
| `bg_sidebar` | `#e6e9ef` | Sidebar background |
| `bg_card` | `#ffffff` | Card / panel background |
| `bg_table_header` | `#dce0e8` | Table header row |
| `border` | `#bcc0cc` | Borders, dividers, scroll area outlines |
| `text_primary` | `#4c4f69` | Primary text |
| `text_muted` | `#6c6f85` | Secondary / muted text |
| `primary` | `#209fb5` | Accent (Sapphire), buttons, active nav, resize handle |
| `success` | `#179299` | Success states (Teal) |
| `warning` | `#df8e1d` | Warnings (Yellow) |
| `danger` | `#d20f39` | Errors / high-risk (Red) |
| `diff_add` | `#209fb5` | Diff: addition |
| `diff_del` | `#dd7878` | Diff: deletion |
| `diff_tweak` | `#dc8a78` | Diff: minor tweak |
| `diff_edit` | `#df8e1d` | Diff: substantive edit |
| `diff_rewrite` | `#ea76cb` | Diff: rewrite |
| `diff_fmt` | `#8839ef` | Diff: format change |

### Catppuccin Mocha (Dark)

| Key | Hex | Role |
|-----|-----|------|
| `bg_page` | `#1e1e2e` | Page background |
| `bg_sidebar` | `#181825` | Sidebar background |
| `bg_card` | `#313244` | Card / panel background |
| `bg_table_header` | `#11111b` | Table header row |
| `border` | `#45475a` | Borders, dividers, scroll area outlines |
| `text_primary` | `#cdd6f4` | Primary text |
| `text_muted` | `#a6adc8` | Secondary / muted text |
| `primary` | `#74c7ec` | Accent (Sapphire), buttons, active nav, resize handle |
| `success` | `#94e2d5` | Success states (Teal) |
| `warning` | `#f9e2af` | Warnings (Yellow) |
| `danger` | `#f38ba8` | Errors / high-risk (Red) |
| `diff_add` | `#74c7ec` | Diff: addition |
| `diff_del` | `#f2cdcd` | Diff: deletion |
| `diff_tweak` | `#f5e0dc` | Diff: minor tweak |
| `diff_edit` | `#f9e2af` | Diff: substantive edit |
| `diff_rewrite` | `#f5c2e7` | Diff: rewrite |
| `diff_fmt` | `#cba6f7` | Diff: format change |

---

## Hardcoded Color Migration

All previously scattered hardcoded hex values are replaced as follows:

| Location | Old Hex | Mapped Key | Strategy |
|----------|---------|------------|----------|
| `main_window.py:111-112` | `#3498db` | `primary` | Move to global QSS via `objectName` selector |
| `compare_page.py:284` | `#dde1ea` | `border` | Move to global QSS via `objectName` on scroll area |
| `compare_page.py:592` | `#374151` | `text_muted` | Use `ThemeManager.palette()["text_muted"]` at label creation time (labels are recreated on each compare run) |
| `qa_page.py:180` | `#dde1ea` | `border` | Move to global QSS via `objectName` on chat scroll area |

---

## FontAwesome Integration

- **Font file:** `assets/fonts/fontawesome-free-7.2.0-desktop/otfs/Font Awesome 7 Free-Solid-900.otf`
- **Glyphs:** moon `\uf186` (shown in light mode — click to go dark), sun `\uf185` (shown in dark mode — click to go light)
- **Loading:** `load_fa_font()` calls `QFontDatabase.addApplicationFont()` at startup. Returns `True` on success.
- **Fallback:** If OTF file is missing or loading fails, `ThemeToggleButton` displays text `"☀"` / `"🌙"` instead.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| FontAwesome OTF missing | Button falls back to emoji text; no crash |
| `config.json` write failure on theme save | Log warning; theme remains active in memory |
| `AppSettings.theme` has invalid value at startup | Reset to `"light"` |

---

## Tests

New file: `tests/test_ui/test_theme.py`

| Test | Description |
|------|-------------|
| `test_theme_manager_toggle` | LIGHT → toggle → DARK → toggle → LIGHT |
| `test_theme_manager_persists` | After toggle, `settings.theme` reflects new value |
| `test_build_stylesheet_contains_colors` | Both palettes generate QSS containing their primary hex |
| `test_fa_font_load` | `load_fa_font()` returns `True` when OTF file exists |

UI interaction (button glyph switching) is verified manually.

---

## Out of Scope

- Animated theme transitions
- "Follow system" auto-detection
- Per-page theme overrides
