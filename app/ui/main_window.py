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


def _load_fa_solid() -> str:
    """Load FA Solid font on first call; return the font family name."""
    global _FA_SOLID_FAMILY
    if not _FA_SOLID_FAMILY and _FA_SOLID_OTF.exists():
        fid = QFontDatabase.addApplicationFont(str(_FA_SOLID_OTF))
        if fid >= 0:
            families = QFontDatabase.applicationFontFamilies(fid)
            if families:
                _FA_SOLID_FAMILY = families[0]
    return _FA_SOLID_FAMILY


def load_harmony_sans() -> str:
    """Load all HarmonyOS Sans weights; return the font family name."""
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
    _ACTIVE_STYLE = (
        f"background-color:{Theme.NAV_ACTIVE_BG};color:{Theme.NAV_ACTIVE_TEXT};"
        "border:none;padding:12px 8px;text-align:left;font-size:14px;border-radius:6px;"
    )
    _INACTIVE_STYLE = (
        f"background-color:transparent;color:{Theme.NAV_TEXT};"
        "border:none;padding:12px 8px;text-align:left;font-size:14px;border-radius:6px;"
    )

    def __init__(self, label: str, parent=None):
        super().__init__(label, parent)
        self.setStyleSheet(self._INACTIVE_STYLE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(Theme.SIDEBAR_WIDTH - 16)

    def set_active(self, active: bool) -> None:
        self.setStyleSheet(self._ACTIVE_STYLE if active else self._INACTIVE_STYLE)


class SideBar(QWidget):
    def __init__(self, on_navigate, on_settings, parent=None):
        super().__init__(parent)
        self.setFixedWidth(Theme.SIDEBAR_WIDTH)
        self.setStyleSheet(f"background-color:{Theme.BG_SIDEBAR};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 16, 8, 16)
        layout.setSpacing(4)

        # Logo row: icon + text
        logo_row = QWidget()
        logo_row.setStyleSheet("""
            QWidget {
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            border-bottom: 2px solid #3498db;
            border-right: 2px solid #3498db;
            }
        """)
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

        logo_text = QLabel("DocDiff")
        _harmony = load_harmony_sans()
        logo_text.setFont(QFont(_harmony or "Segoe UI", 14, QFont.Weight.Bold))
        logo_text.setStyleSheet(f"color:{Theme.LOGO_COLOR};border: none;")
        logo_layout.addWidget(logo_text)
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

        # Settings row: FA icon (FA font) + "设置" (HarmonyOS Sans)
        settings_row = QWidget()
        settings_row.setFixedWidth(Theme.SIDEBAR_WIDTH - 16)
        settings_row.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_row.setStyleSheet(
            f"background-color:transparent;border-top:1px solid {Theme.BORDER};"
        )
        _sr_layout = QHBoxLayout(settings_row)
        _sr_layout.setContentsMargins(8, 12, 8, 12)
        _sr_layout.setSpacing(6)

        _icon_lbl = QLabel("\uf013")
        fa_family = _load_fa_solid()
        if fa_family:
            _icon_lbl.setFont(QFont(fa_family, 14))
        _icon_lbl.setStyleSheet(f"color:{Theme.NAV_TEXT};border:none;")
        _sr_layout.addWidget(_icon_lbl)

        _text_lbl = QLabel("设置")
        if _harmony:
            _text_lbl.setFont(QFont(_harmony, 16, QFont.Weight.Bold))
        _text_lbl.setStyleSheet(f"color:{Theme.NAV_TEXT};border:none;font-size:16px;")
        _sr_layout.addWidget(_text_lbl)
        _sr_layout.addStretch()

        settings_row.mousePressEvent = lambda e: on_settings()
        layout.addWidget(settings_row)

        self._set_active(0)

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
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._sidebar = SideBar(self._on_navigate, self._on_settings_btn)
        root_layout.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
        root_layout.addWidget(self._stack, 1)

        for label, _ in _NAV_ITEMS:
            placeholder = QLabel(f"{label} 页面")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(f"font-size:24px;color:{Theme.TEXT_PLACEHOLDER};")
            self._stack.addWidget(placeholder)

    def add_page(self, index: int, widget: QWidget) -> None:
        """Replace the placeholder at index with a real page widget."""
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

    def navigate_to(self, index: int) -> None:
        self._on_navigate(index)
