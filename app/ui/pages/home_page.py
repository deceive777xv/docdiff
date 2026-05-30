"""Home page — dashboard with recent tasks and quick actions."""
from __future__ import annotations
import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.db import compare_repo
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
    compare_task_open_requested = Signal(str)
    compare_task_recover_requested = Signal(str)

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
        subtitle.setStyleSheet(Theme.form_label())
        self._subtitle = subtitle
        layout.addWidget(subtitle)

        # Stat cards
        cards_layout = QGridLayout()
        cards_layout.setSpacing(16)

        self._card_docs = _StatCard("文档", "0", Theme.COLOR_PRIMARY)
        self._card_tasks = _StatCard("对比任务", "0", Theme.COLOR_SUCCESS)
        self._card_done = _StatCard("已完成", "0", Theme.COLOR_COMPLETED)

        cards_layout.addWidget(self._card_docs, 0, 0)
        cards_layout.addWidget(self._card_tasks, 0, 1)
        cards_layout.addWidget(self._card_done, 0, 2)
        layout.addLayout(cards_layout)

        # Quick actions
        actions_group = QGroupBox()
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setSpacing(12)

        actions = [
            ("导入文档", 2, "COLOR_PRIMARY"),
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

        self._tasks_table = QTableWidget(0, 6)
        self._tasks_table.setHorizontalHeaderLabels(
            ["任务ID", "版本", "状态", "结果", "创建时间", "操作"]
        )
        tasks_header = self._tasks_table.horizontalHeader()
        tasks_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        tasks_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        tasks_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        tasks_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        tasks_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        tasks_header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self._tasks_table.setColumnWidth(1, 260)
        self._tasks_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._tasks_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tasks_table.setAlternatingRowColors(True)
        self._tasks_table.setMaximumHeight(240)
        self._tasks_table.itemDoubleClicked.connect(self._on_task_row_activated)
        recent_layout.addWidget(self._tasks_table)

        layout.addWidget(recent_group)
        layout.addStretch()

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
        self._title.setStyleSheet(Theme.page_title())
        self._subtitle.setStyleSheet(Theme.form_label())
        self._card_docs._apply_color(Theme.COLOR_PRIMARY)
        self._card_tasks._apply_color(Theme.COLOR_SUCCESS)
        self._card_done._apply_color(Theme.COLOR_COMPLETED)
        
        for btn, color_attr in self._action_buttons:
            color = getattr(Theme, color_attr)
            btn.setStyleSheet(
                f"background-color:{color};color:{Theme.NAV_ACTIVE_TEXT};padding:10px 20px;"
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

            recent = compare_repo.list_recent_task_summaries(self.ctx.conn, limit=10)
            self._tasks_table.setRowCount(len(recent))
            for row, task in enumerate(recent):
                self._populate_task_row(row, task)
        except Exception as e:
            logger.warning("Home page refresh failed: %s", e)

    def _populate_task_row(self, row: int, task) -> None:
        task_id = task["id"]
        values = [
            task_id[:8] + "…",
            self._format_version_pair(task),
            self._format_status(task),
            self._format_result_summary(task),
            str(task["created_at"])[:16],
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setData(Qt.ItemDataRole.UserRole, task_id)
            self._tasks_table.setItem(row, col, item)

        action = self._make_task_action_button(task)
        self._tasks_table.setCellWidget(row, 5, action)

    def _format_version_pair(self, task) -> str:
        return f"{self._format_version(task, 'baseline')} → {self._format_version(task, 'target')}"

    def _format_version(self, task, prefix: str) -> str:
        doc_name = task[f"{prefix}_doc_name"] or "未知文档"
        version_no = task[f"{prefix}_version_no"]
        version = f"v{version_no}" if version_no is not None else "未知版本"
        label = task[f"{prefix}_version_label"] or ""
        return f"{doc_name} {version}({label})" if label else f"{doc_name} {version}"

    def _format_status(self, task) -> str:
        status_map = {
            "pending": "等待中",
            "running": "进行中",
            "completed": "已完成",
            "failed": "失败",
        }
        return status_map.get(task["status"], task["status"])

    def _format_result_summary(self, task) -> str:
        if task["status"] != "completed":
            return "暂无结果"
        return (
            f"{int(task['diff_count'])}处差异 / "
            f"高{int(task['high_count'])} 中{int(task['medium_count'])} "
            f"低{int(task['low_count'])} 无{int(task['none_count'])}"
        )

    def _make_task_action_button(self, task) -> QPushButton:
        task_id = task["id"]
        is_active = task_id in self.ctx.active_compare_task_ids
        if task["status"] == "completed":
            label = "打开"
            callback = self.compare_task_open_requested.emit
            enabled = True
        elif is_active:
            label = "进行中"
            callback = self.compare_task_open_requested.emit
            enabled = False
        else:
            label = "恢复"
            callback = self.compare_task_recover_requested.emit
            enabled = True

        btn = QPushButton(label)
        btn.setEnabled(enabled)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            f"background-color:{Theme.COLOR_PRIMARY};color:{Theme.NAV_ACTIVE_TEXT};"
            "border:none;border-radius:4px;padding:4px 10px;font-size:12px;"
        )
        btn.clicked.connect(lambda checked=False, tid=task_id, cb=callback: cb(tid))
        return btn

    def _on_task_row_activated(self, item: QTableWidgetItem) -> None:
        task_id = item.data(Qt.ItemDataRole.UserRole)
        if not task_id:
            return
        task = compare_repo.get_task_by_id(self.ctx.conn, task_id)
        if task is None:
            return
        if task["status"] == "completed" or task_id in self.ctx.active_compare_task_ids:
            self.compare_task_open_requested.emit(task_id)
        else:
            self.compare_task_recover_requested.emit(task_id)
