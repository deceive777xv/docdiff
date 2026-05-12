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
    """Return global QSS for objectName/property-targeted elements."""
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

    # QSS helper classmethods — read cls.X at call time, so they reflect current theme
    @classmethod
    def btn_primary(cls) -> str:
        return (
            f"background-color:{cls.COLOR_PRIMARY};color:{cls.NAV_ACTIVE_TEXT};"
            f"border:none;border-radius:{CARD_RADIUS}px;padding:8px 16px;font-size:13px;"
        )

    @classmethod
    def btn_success(cls) -> str:
        return (
            f"background-color:{cls.COLOR_SUCCESS};color:{cls.NAV_ACTIVE_TEXT};"
            f"border:none;border-radius:{CARD_RADIUS}px;padding:8px 16px;font-size:13px;"
        )

    @classmethod
    def btn_danger(cls) -> str:
        return (
            f"background-color:{cls.COLOR_DANGER};color:{cls.NAV_ACTIVE_TEXT};"
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


# Populate Theme color attributes from LATTE at import time.
# ThemeManager can call setattr(Theme, k, v) for each key in MOCHA to switch themes.
for _k, _v in LATTE.items():
    setattr(Theme, _k, _v)
del _k, _v
