"""Home page — dashboard with recent tasks and quick actions."""
from __future__ import annotations
import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui.app_context import AppContext
from app.ui.theme import Theme

logger = logging.getLogger(__name__)


class _StatCard(QWidget):
    """A small statistic card widget."""

    def __init__(self, label: str, value: str, color: str = Theme.COLOR_PRIMARY, parent=None):
        super().__init__(parent)
        self._color_hex = color
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        val_lbl = QLabel(value)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(val_lbl)

        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        self._val_lbl = val_lbl
        self._lbl = lbl
        self._apply_color(color)

    def _apply_color(self, color: str) -> None:
        self._color_hex = color
        _c = QColor(color)
        _c.setAlpha(50)
        self.setStyleSheet(
            f"background:{_c.name(QColor.NameFormat.HexArgb)};border:1px solid {color};"
            f"border-radius:{Theme.CARD_RADIUS}px;padding:12px;"
        )
        self._val_lbl.setStyleSheet(f"font-size:26px;font-weight:bold;color:{color};")
        self._lbl.setStyleSheet(Theme.label_secondary() + f"font-size:14px;color:{color};")

    def refresh_theme(self) -> None:
        self._apply_color(self._color_hex)

    def update_value(self, value: str) -> None:
        self._val_lbl.setText(value)


class HomePage(QWidget):
    """Dashboard home page."""

    navigate_requested = Signal(int)   # page index to navigate to

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._build_ui()
        self._apply_theme()
        from app.ui.theme_manager import ThemeManager
        ThemeManager.instance().theme_changed.connect(self._apply_theme)
        self.refresh()

    def _build_ui(self) -> None:
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Theme.PAGE_MARGIN, Theme.PAGE_MARGIN, Theme.PAGE_MARGIN, Theme.PAGE_MARGIN)
        layout.setSpacing(24)

        # Title
        title = QLabel("Doc-Diff-Agent")
        title.setStyleSheet(Theme.page_title())
        self._title = title
        layout.addWidget(title)

        subtitle = QLabel("智能文档对比与问答平台")
        subtitle.setStyleSheet(Theme.label_secondary()+f"font-size:14px;")
        self._subtitle = subtitle
        layout.addWidget(subtitle)

        # Stat cards
        cards_layout = QGridLayout()
        cards_layout.setSpacing(16)

        self._card_docs = _StatCard("标准文档", "0", Theme.COLOR_PRIMARY)
        self._card_tasks = _StatCard("对比任务", "0", Theme.COLOR_SUCCESS)
        self._card_done = _StatCard("已完成", "0", Theme.COLOR_COMPLETED)

        cards_layout.addWidget(self._card_docs, 0, 0)
        cards_layout.addWidget(self._card_tasks, 0, 1)
        cards_layout.addWidget(self._card_done, 0, 2)
        layout.addLayout(cards_layout)

        # Quick actions
        actions_group = QGroupBox("快捷操作")
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setSpacing(12)

        actions = [
            ("导入标准文档", 2, "COLOR_PRIMARY"),
            ("开始文档对比", 1, "COLOR_SUCCESS"),
            ("智能问答",      3, "COLOR_COMPLETED"),
        ]
        self._action_buttons = []
        for label, page_idx, color_attr in actions:
            btn = QPushButton(label)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda checked, i=page_idx: self.navigate_requested.emit(i))
            actions_layout.addWidget(btn)
            self._action_buttons.append((btn, color_attr))

        layout.addWidget(actions_group)

        # Recent tasks
        recent_group = QGroupBox("最近对比任务")
        recent_layout = QVBoxLayout(recent_group)

        self._tasks_table = QTableWidget(0, 4)
        self._tasks_table.setHorizontalHeaderLabels(["任务ID", "状态", "创建时间", "完成时间"])
        self._tasks_table.horizontalHeader().setStretchLastSection(True)
        self._tasks_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._tasks_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tasks_table.setAlternatingRowColors(True)
        self._tasks_table.setMaximumHeight(200)
        recent_layout.addWidget(self._tasks_table)

        layout.addWidget(recent_group)
        layout.addStretch()

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
        self._title.setStyleSheet(Theme.page_title())
        self._subtitle.setStyleSheet(Theme.label_secondary() + "font-size:14px;")
        self._card_docs.refresh_theme()
        self._card_tasks.refresh_theme()
        self._card_done.refresh_theme()
        for btn, color_attr in self._action_buttons:
            color = getattr(Theme, color_attr)
            btn.setStyleSheet(
                f"background-color:{color};color:white;padding:10px 20px;"
                f"border:none;border-radius:{Theme.CARD_RADIUS}px;font-size:16px;"
            )

    def refresh(self) -> None:
        """Reload stats and recent tasks from DB."""
        try:
            docs = self.ctx.conn.execute(
                "SELECT COUNT(*) FROM documents WHERE source_type='standard'"
            ).fetchone()[0]
            tasks = self.ctx.conn.execute(
                "SELECT COUNT(*) FROM compare_tasks"
            ).fetchone()[0]
            done = self.ctx.conn.execute(
                "SELECT COUNT(*) FROM compare_tasks WHERE status='completed'"
            ).fetchone()[0]

            self._card_docs.update_value(str(docs))
            self._card_tasks.update_value(str(tasks))
            self._card_done.update_value(str(done))

            recent = self.ctx.conn.execute(
                "SELECT * FROM compare_tasks ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            self._tasks_table.setRowCount(len(recent))
            for row, task in enumerate(recent):
                self._tasks_table.setItem(row, 0, QTableWidgetItem(task["id"][:8] + "…"))
                status_map = {
                    "pending": "等待中",
                    "running": "进行中",
                    "completed": "已完成",
                    "failed": "失败",
                }
                self._tasks_table.setItem(
                    row, 1,
                    QTableWidgetItem(status_map.get(task["status"], task["status"]))
                )
                self._tasks_table.setItem(
                    row, 2, QTableWidgetItem(str(task["created_at"])[:16])
                )
                self._tasks_table.setItem(
                    row, 3, QTableWidgetItem(str(task["finished_at"] or "")[:16])
                )
        except Exception as e:
            logger.warning("Home page refresh failed: %s", e)
