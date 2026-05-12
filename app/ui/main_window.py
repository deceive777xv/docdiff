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
