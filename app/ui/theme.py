"""Centralized UI theme constants — import this instead of hardcoding colors."""


class Theme:
    # Layout
    SIDEBAR_WIDTH = 140
    PAGE_MARGIN = 24
    CARD_RADIUS = 8

    # Sidebar
    BG_SIDEBAR = "#efe8e5"
    NAV_ACTIVE_BG = "#209fb5"
    NAV_ACTIVE_TEXT = "#ffffff"
    NAV_TEXT = "#9ea4b2"
    LOGO_COLOR = "#04a5e5"

    # Page areas
    BG_PAGE = "#ebf2f1"
    BG_CARD = "#ffffff"
    BG_HEADER = "#edf0f5"

    # Text
    TEXT_PRIMARY = "#209fb5"
    TEXT_SECONDARY = "#99d1db"
    TEXT_PLACEHOLDER = "#4c4f69"

    # Borders
    BORDER = "#dde1ea" #dde1ea

    # Action colors
    COLOR_PRIMARY = "#85c1dc"
    COLOR_SUCCESS = "#81c8be"
    COLOR_DANGER = "#e78284"
    COLOR_WARNING = "#ef9f76"
    COLOR_COMPLETED = "#179299"

    # Diff highlight colors — also used in assets/diff_template.html
    DIFF_ADDED = "#209fb5"
    DIFF_DELETED = "#dd7878"
    DIFF_MINOR = "#dc8a78"
    DIFF_MAJOR = "#df8e1d"
    DIFF_REWRITE = "#ea76cb"
    DIFF_FORMAT = "#ca9ee6"

    # QSS helper classmethods
    @classmethod
    def btn_primary(cls) -> str:
        return (
            f"background-color:{cls.TEXT_PRIMARY};color:white;"
            f"border:none;border-radius:6px;padding:8px 16px;font-size:13px;"
        )

    @classmethod
    def btn_success(cls) -> str:
        return (
            f"background-color:{cls.COLOR_SUCCESS};color:white;"
            f"border:none;border-radius:6px;padding:8px 16px;font-size:13px;"
        )

    @classmethod
    def btn_danger(cls) -> str:
        return (
            f"background-color:{cls.COLOR_DANGER};color:white;"
            f"border:none;border-radius:6px;padding:8px 16px;font-size:13px;"
        )

    @classmethod
    def card(cls) -> str:
        return (
            f"background:{cls.BG_CARD};border:1px solid {cls.BORDER};"
            f"border-radius:{cls.CARD_RADIUS}px;"
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
