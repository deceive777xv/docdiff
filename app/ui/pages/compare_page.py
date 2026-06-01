"""Compare page — side-by-side document diff view with WebEngine rendering."""
from __future__ import annotations

import html
import json
import logging
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot, QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtGui import QColor
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.markdown_utils import (
    render_inline_markdown as _core_render_inline_markdown,
    render_markdown_fragment as _core_render_markdown_fragment,
    strip_markdown_formatting as _core_strip_markdown_formatting,
)
from app.core.types import ComparePolicy, DiffItem, DiffResult, DocumentIR, Paragraph, Section, Sentence
from app.db import document_repo
from app.ui.app_context import AppContext
from app.ui.theme import Theme

logger = logging.getLogger(__name__)

# ── Diff type → (CSS class, hex color) mapping ────────────────────────────────

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
        "none":   Theme.TEXT_SECONDARY,
    }

_RISK_LABELS: dict[str, str] = {
    "high":   "高风险",
    "medium": "中风险",
    "low":    "低风险",
    "none":   "无风险",
}

_ALL_SECTIONS_KEY = "__all_sections__"

_TABLE_SEPARATOR_CELL_RE = re.compile(r":?-{3,}:?")


def _split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_markdown_table_separator(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_markdown_table_row(line)
    return bool(cells) and all(_TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells)


def _render_inline_markdown(text: str) -> str:
    """Render a small, escaped inline Markdown subset."""
    return _core_render_inline_markdown(text)


def _render_markdown_fragment(markdown_text: str) -> str:
    """Convert stored Markdown text into safe, readable HTML for the diff panes."""
    return _core_render_markdown_fragment(markdown_text)


def _strip_markdown_formatting(markdown_text: str) -> str:
    """Remove common Markdown markers for compact QLabel summaries."""
    return _core_strip_markdown_formatting(markdown_text)


def _normalize_diff_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _ir_from_dict(data: dict) -> DocumentIR:
    sections: list[Section] = []
    for sec in data.get("sections", []):
        paragraphs: list[Paragraph] = []
        for para in sec.get("paragraphs", []):
            sentences = [
                Sentence(text=sent.get("text", ""))
                for sent in para.get("sentences", [])
                if sent.get("text", "")
            ]
            paragraphs.append(
                Paragraph(
                    paragraph_id=para.get("paragraph_id", ""),
                    text=para.get("text", ""),
                    sentences=sentences,
                )
            )
        sections.append(
            Section(
                section_id=sec.get("section_id", ""),
                title=sec.get("title", ""),
                level=int(sec.get("level", 1) or 1),
                paragraphs=paragraphs,
            )
        )
    return DocumentIR(
        doc_id=data.get("doc_id", ""),
        title=data.get("title", ""),
        file_hash=data.get("file_hash", ""),
        sections=sections,
        plain_text=data.get("plain_text", ""),
    )


def _render_changed_inline(text: str, other_text: str) -> str:
    """Escape text and wrap changed character spans for fine-grained visual focus."""
    text = _strip_markdown_formatting(text)
    other_text = _strip_markdown_formatting(other_text)
    if not other_text:
        return html.escape(text, quote=False)

    changed_ranges: list[tuple[int, int]] = []
    matcher = SequenceMatcher(None, text, other_text)
    for tag, start, end, _other_start, _other_end in matcher.get_opcodes():
        if tag == "equal" or start == end:
            continue
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        while end < len(text) and not text[end].isspace():
            end += 1
        changed_ranges.append((start, end))

    if not changed_ranges:
        return html.escape(text, quote=False)

    parts: list[str] = []
    cursor = 0
    merged_ranges: list[tuple[int, int]] = []
    for start, end in sorted(changed_ranges):
        if not merged_ranges or start > merged_ranges[-1][1]:
            merged_ranges.append((start, end))
        else:
            merged_start, merged_end = merged_ranges[-1]
            merged_ranges[-1] = (merged_start, max(merged_end, end))

    for start, end in merged_ranges:
        if cursor < start:
            parts.append(html.escape(text[cursor:start], quote=False))
        piece = html.escape(text[start:end], quote=False)
        if piece:
            parts.append(f'<mark class="diff-token">{piece}</mark>')
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:], quote=False))
    return "".join(parts)


# ── Background worker ──────────────────────────────────────────────────────────

class _CompareWorker(QObject):
    """Run compare_service.run_compare in a background thread."""

    result_ready = Signal(object)   # emits DiffResult
    error = Signal(str)

    def __init__(
        self,
        data_dir: str,
        baseline_version_id: str,
        target_version_id: str,
        embedder,
        provider,
        policy: ComparePolicy,
        task_id: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._data_dir = data_dir
        self._baseline_version_id = baseline_version_id
        self._target_version_id = target_version_id
        self._embedder = embedder
        self._provider = provider
        self._policy = policy
        self._task_id = task_id

    def run(self) -> None:
        try:
            from app.agent.compare_graph import compare_graph
            from app.db.schema import open_db

            conn = open_db(self._data_dir)
            try:
                state = {
                    "data_dir": self._data_dir,
                    "baseline_version_id": self._baseline_version_id,
                    "target_version_id": self._target_version_id,
                    "provider": self._provider,
                    "embedder": self._embedder,
                    "conn": conn,
                }
                if self._task_id:
                    state["task_id"] = self._task_id
                result = compare_graph.invoke(state)
            finally:
                conn.close()

            if result.get("error"):
                self.error.emit(result["error"])
            else:
                self.result_ready.emit(result["result"])
        except Exception as exc:
            logger.exception("Compare worker failed")
            self.error.emit(str(exc))


# ── JS → Python bridge ────────────────────────────────────────────────────────

class _WebBridge(QObject):
    """Object registered with QWebChannel so JS can call back into Python."""

    diff_clicked = Signal(str)   # emitted with diff_id when a span is clicked

    def __init__(self, parent=None):
        super().__init__(parent)

    @Slot(str)
    def onDiffClick(self, diff_id: str) -> None:   # noqa: N802 — name must match JS
        """Called from JavaScript when user clicks a highlighted diff span."""
        self.diff_clicked.emit(diff_id)


# ── Main page ─────────────────────────────────────────────────────────────────

class ComparePage(QWidget):
    """Three-panel document comparison page."""

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._current_result: Optional[DiffResult] = None
        self._diff_items_by_id: dict[str, DiffItem] = {}
        self._visible_diff_items: list[DiffItem] = []
        self._recover_task_id: str | None = None
        self._thread: QThread | None = None
        self._threads: set[QThread] = set()
        self._build_ui()
        from app.ui.theme_manager import ThemeManager
        ThemeManager.instance().theme_changed.connect(self._apply_theme)
        self.refresh_versions()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")

        # ── Top bar: version selectors ─────────────────────────────────────────
        top_group = QGroupBox()
        top_layout = QHBoxLayout(top_group)
        top_layout.setSpacing(10)

        tmp_label = QLabel("基准版本：")
        tmp_label.setStyleSheet(Theme.form_label_large())
        top_layout.addWidget(tmp_label)
        self._baseline_combo = QComboBox()
        self._baseline_combo.setMinimumWidth(200)
        self._baseline_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_layout.addWidget(self._baseline_combo)

        tmp_label = QLabel("目标版本：")
        tmp_label.setStyleSheet(Theme.form_label_large())
        top_layout.addWidget(tmp_label)
        self._target_combo = QComboBox()
        self._target_combo.setMinimumWidth(200)
        self._target_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_layout.addWidget(self._target_combo)

        self._run_btn = QPushButton("▶ 开始对比")
        self._run_btn.setStyleSheet(Theme.btn_primary())
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_compare)
        top_layout.addWidget(self._run_btn)

        self._loading_label = QLabel("")
        self._loading_label.setStyleSheet(Theme.label_secondary())
        top_layout.addWidget(self._loading_label)

        self._export_btn = QPushButton("导出报告")
        self._export_btn.setStyleSheet(
            f"background-color:transparent;color:{Theme.TEXT_PRIMARY};"
            f"border:1px solid {Theme.TEXT_PRIMARY};padding:6px 14px;"
            f"border-radius:{Theme.CARD_RADIUS}px;font-size:13px;"
        )
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_report)
        top_layout.addWidget(self._export_btn)

        root.addWidget(top_group)

        # ── Overview bar: diff-type counts ─────────────────────────────────────
        overview_group = QGroupBox()
        overview_layout = QHBoxLayout(overview_group)
        overview_layout.setSpacing(8)
        self._overview_labels: dict[str, QLabel] = {}
        for diff_type, (_, color) in _diff_css().items():
            _color = QColor(color)
            _color.setAlpha(30)
            lbl = QLabel(f"{diff_type}: 0")
            lbl.setStyleSheet(
                f"background:{_color.name(QColor.NameFormat.HexArgb)};border:1px solid {color};"
                f"color:{Theme.TEXT_PRIMARY};border-radius:4px;padding:3px 8px;font-size:12px;"
            )
            overview_layout.addWidget(lbl)
            self._overview_labels[diff_type] = lbl
        overview_layout.addStretch()
        root.addWidget(overview_group)

        # ── 3-panel horizontal splitter ────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # Left: chapter navigation tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["章节", "差异数"])
        self._tree.setColumnWidth(0, 150)
        self._tree.setMinimumWidth(160)
        self._tree.setMaximumWidth(300)
        self._tree.itemClicked.connect(self._on_tree_item_clicked)
        splitter.addWidget(self._tree)
        splitter.setStretchFactor(0, 0)

        # Center: WebEngineView
        self._web_view = QWebEngineView()
        self._channel = QWebChannel()
        self._bridge = _WebBridge(self)
        self._bridge.diff_clicked.connect(self._on_diff_clicked)
        self._channel.registerObject("bridge", self._bridge)
        self._web_view.page().setWebChannel(self._channel)
        template_path = (
            Path(__file__).parent.parent.parent.parent / "assets" / "diff_template.html"
        )
        self._web_view.load(QUrl.fromLocalFile(str(template_path)))
        self._web_view.loadFinished.connect(lambda _: self._apply_webview_theme())
        splitter.addWidget(self._web_view)
        splitter.setStretchFactor(1, 1)

        # Right: details panel
        right_widget = self._build_details_panel()
        right_widget.setMinimumWidth(240)
        right_widget.setMaximumWidth(380)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(2, 0)

        root.addWidget(splitter, 1)

        # Wire combo changes to run-button state
        self._baseline_combo.currentIndexChanged.connect(self._update_run_btn_state)
        self._target_combo.currentIndexChanged.connect(self._update_run_btn_state)

    def _build_details_panel(self) -> QWidget:
        """Build the right-side diff details panel with filter bar."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Filter bar
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(6)
        tmp_label = QLabel("筛选：")
        tmp_label.setStyleSheet(Theme.form_label_large())
        filter_bar.addWidget(tmp_label)
        
        self._filter_type_combo = QComboBox()
        self._filter_type_combo.addItem("全部类型", None)
        for diff_type in _diff_css():
            self._filter_type_combo.addItem(diff_type, diff_type)
        self._filter_type_combo.currentIndexChanged.connect(self._apply_filters)
        filter_bar.addWidget(self._filter_type_combo)

        self._filter_risk_combo = QComboBox()
        self._filter_risk_combo.addItem("全部风险", None)
        for risk_key, risk_label in _RISK_LABELS.items():
            self._filter_risk_combo.addItem(risk_label, risk_key)
        self._filter_risk_combo.currentIndexChanged.connect(self._apply_filters)
        filter_bar.addWidget(self._filter_risk_combo)
        filter_bar.addStretch()
        layout.addLayout(filter_bar)

        # Scrollable diff card list
        self._detail_scroll = QScrollArea()
        self._detail_scroll.setObjectName("detail_scroll")
        self._detail_scroll.setWidgetResizable(True)
        self._detail_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._detail_scroll.viewport().setStyleSheet("background: transparent;")
        self._detail_content = QWidget()
        self._detail_layout = QVBoxLayout(self._detail_content)
        self._detail_layout.setSpacing(6)
        self._detail_layout.addStretch()

        self._detail_scroll.setWidget(self._detail_content)
        layout.addWidget(self._detail_scroll, 1)

        return widget

    # ── Public API ─────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        self.refresh_versions()

    def load_task(self, task_id: str) -> None:
        """Load an existing compare task into the page."""
        from app.db import compare_repo

        task = compare_repo.get_task_by_id(self.ctx.conn, task_id)
        if task is None:
            QMessageBox.warning(self, "任务不存在", f"未找到对比任务：{task_id}")
            return

        self.refresh_versions()
        self._select_combo_value(self._baseline_combo, task["baseline_version_id"])
        self._select_combo_value(self._target_combo, task["target_version_id"])

        if task["status"] == "completed":
            result = compare_repo.get_task_result(self.ctx.conn, task_id)
            self._recover_task_id = None
            self._run_btn.setText("▶ 开始对比")
            self._display_result(result, f"已加载任务结果：{len(result.items)} 处差异。")
            return

        status_text = {
            "pending": "等待中",
            "running": "进行中",
            "failed": "失败",
        }.get(task["status"], task["status"])
        self._current_result = None
        self._diff_items_by_id = {}
        self._export_btn.setEnabled(False)
        self._tree.clear()
        self._show_diff_list([])
        self._update_overview(DiffResult(task_id, task["baseline_version_id"], task["target_version_id"], []))
        if task_id in self.ctx.active_compare_task_ids:
            self._recover_task_id = None
            self._run_btn.setText("对比中…")
            self._run_btn.setEnabled(False)
            self._loading_label.setText(f"任务正在运行（{status_text}），完成后会自动展示结果。")
            return

        self._recover_task_id = task_id
        self._run_btn.setText("恢复对比")
        self._run_btn.setEnabled(True)
        self._loading_label.setText(
            f"任务未完成（{status_text}）。点击“恢复对比”将重新执行该任务。"
        )

    def recover_task(self, task_id: str) -> None:
        """Recover an unfinished task by re-running it with the same task id."""
        from app.db import compare_repo

        task = compare_repo.get_task_by_id(self.ctx.conn, task_id)
        if task is None:
            QMessageBox.warning(self, "任务不存在", f"未找到对比任务：{task_id}")
            return
        if self.ctx.provider is None or self.ctx.embedder is None:
            QMessageBox.warning(
                self,
                "配置缺失",
                "请先在设置页面配置模型 API 和 Embedding。",
            )
            return
        self.refresh_versions()
        self._select_combo_value(self._baseline_combo, task["baseline_version_id"])
        self._select_combo_value(self._target_combo, task["target_version_id"])
        self._recover_task_id = task_id
        self._start_compare(task["baseline_version_id"], task["target_version_id"], task_id)
        
    def refresh_versions(self) -> None:
        """Repopulate baseline/target combos from the database."""
        self._baseline_combo.blockSignals(True)
        self._target_combo.blockSignals(True)
        try:
            self._baseline_combo.clear()
            self._target_combo.clear()
            docs = document_repo.list_documents(self.ctx.conn)
            for doc in docs:
                versions = document_repo.list_versions(self.ctx.conn, doc["id"])
                for ver in versions:
                    label = f"{doc['doc_name']} — v{ver['version_no']}"
                    if ver["version_label"]:
                        label += f"  ({ver['version_label']})"
                    self._baseline_combo.addItem(label, ver["id"])
                    self._target_combo.addItem(label, ver["id"])
        except Exception as exc:
            logger.warning("refresh_versions failed: %s", exc)
        finally:
            self._baseline_combo.blockSignals(False)
            self._target_combo.blockSignals(False)
        self._update_run_btn_state()

    def _select_combo_value(self, combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    # ── Slot helpers ───────────────────────────────────────────────────────────

    def _update_run_btn_state(self) -> None:
        enabled = (
            self._baseline_combo.count() > 0
            and self._target_combo.count() > 0
            and self._baseline_combo.currentData() is not None
            and self._target_combo.currentData() is not None
        )
        self._run_btn.setEnabled(enabled)

    def _run_compare(self) -> None:
        if self.ctx.provider is None or self.ctx.embedder is None:
            QMessageBox.warning(
                self,
                "配置缺失",
                "请先在设置页面配置模型 API 和 Embedding。",
            )
            return

        baseline_version_id = self._baseline_combo.currentData()
        target_version_id = self._target_combo.currentData()
        if not baseline_version_id or not target_version_id:
            return

        task_id = self._recover_task_id
        if task_id is None:
            from app.db import compare_repo
            task_id = compare_repo.create_compare_task(
                self.ctx.conn,
                baseline_version_id=baseline_version_id,
                target_version_id=target_version_id,
            )
        self._start_compare(baseline_version_id, target_version_id, task_id)

    def _start_compare(
        self,
        baseline_version_id: str,
        target_version_id: str,
        task_id: str,
    ) -> None:
        """Start or recover a compare task in a background thread."""
        self._run_btn.setEnabled(False)
        self._loading_label.setText("对比中，请稍候…")
        self._recover_task_id = task_id
        self.ctx.active_compare_task_ids.add(task_id)

        self._thread = QThread()
        self._worker = _CompareWorker(
            self.ctx.data_dir,
            baseline_version_id,
            target_version_id,
            self.ctx.embedder,
            self.ctx.provider,
            ComparePolicy(),
            task_id=task_id,
        )
        worker = self._worker
        thread = self._thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result_ready.connect(self._on_compare_done)
        worker.result_ready.connect(thread.quit)
        worker.error.connect(self._on_compare_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda tid=task_id: self.ctx.active_compare_task_ids.discard(tid))
        thread.finished.connect(lambda th=thread: self._threads.discard(th))
        self._threads.add(thread)
        thread.start()

    def _on_compare_done(self, result: DiffResult) -> None:
        self._display_result(result, f"完成！发现 {len(result.items)} 处差异。")
        self._recover_task_id = None
        self._run_btn.setText("▶ 开始对比")

    def _display_result(self, result: DiffResult, message: str) -> None:
        self._current_result = result
        self._diff_items_by_id = {item.diff_id: item for item in result.items}
        self._loading_label.setText(message)
        self._run_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._update_overview(result)
        self._populate_tree(result)
        self._render_diff(result)
        self._show_diff_list(result.items)

    def _on_compare_error(self, msg: str) -> None:
        self._loading_label.setText("")
        self._run_btn.setEnabled(True)
        self._run_btn.setText("恢复对比" if self._recover_task_id else "▶ 开始对比")
        QMessageBox.critical(self, "对比失败", msg)

    def _export_report(self) -> None:
        """Open save dialog and export the current diff result."""
        if self._current_result is None:
            return
        from PySide6.QtWidgets import QFileDialog
        default_name = f"report_{self._current_result.task_id[:8]}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出报告",
            default_name,
            "Word 文件 (*.docx);;HTML 文件 (*.html)",
        )
        if not path:
            return
        try:
            from app.services.report_service import export_docx, export_html
            if path.lower().endswith(".html"):
                export_html(self._current_result, path)
            else:
                if not path.lower().endswith(".docx"):
                    path += ".docx"
                export_docx(self._current_result, path)
            QMessageBox.information(self, "导出成功", f"报告已保存至：\n{path}")
        except Exception as exc:
            logger.exception("Export failed")
            QMessageBox.critical(self, "导出失败", str(exc))

    def _on_diff_clicked(self, diff_id: str) -> None:
        """Highlight the clicked diff in the right panel."""
        item = self._diff_items_by_id.get(diff_id)
        if item:
            self._sync_filter_controls(item)
            self._show_diff_list([item])

    def _on_diff_card_clicked(self, diff_id: str) -> None:
        """Focus the middle panes on the clicked diff card."""
        item = self._diff_items_by_id.get(diff_id)
        if item:
            self._sync_filter_controls(item)
        self._focus_diff_in_webview(diff_id)

    def _sync_filter_controls(self, item: DiffItem) -> None:
        """Update filter combos without triggering a filter rebuild."""
        self._filter_type_combo.blockSignals(True)
        self._filter_risk_combo.blockSignals(True)
        try:
            self._select_combo_value(self._filter_type_combo, item.diff_type)
            self._select_combo_value(self._filter_risk_combo, item.risk_level)
        finally:
            self._filter_type_combo.blockSignals(False)
            self._filter_risk_combo.blockSignals(False)

    def _clear_filter_controls(self) -> None:
        """Reset filter combos without rebuilding the current card list."""
        self._filter_type_combo.blockSignals(True)
        self._filter_risk_combo.blockSignals(True)
        try:
            self._filter_type_combo.setCurrentIndex(0)
            self._filter_risk_combo.setCurrentIndex(0)
        finally:
            self._filter_type_combo.blockSignals(False)
            self._filter_risk_combo.blockSignals(False)

    def _focus_diff_in_webview(self, diff_id: str) -> None:
        js = f"focusDiff({json.dumps(diff_id, ensure_ascii=False)});"
        self._web_view.page().runJavaScript(js)

    def _on_tree_item_clicked(self, tree_item: QTreeWidgetItem, _column: int) -> None:
        if self._current_result is None:
            return
        if tree_item is None:
            return
        section_path = tree_item.data(0, Qt.UserRole)
        self._clear_filter_controls()
        if section_path == _ALL_SECTIONS_KEY:
            self._show_diff_list(self._current_result.items)
            return
        items = [i for i in self._current_result.items if i.section_path == section_path]
        self._show_diff_list(items)

    def _apply_filters(self) -> None:
        if self._current_result is None:
            return
        filter_type = self._filter_type_combo.currentData()
        filter_risk = self._filter_risk_combo.currentData()
        items = self._current_result.items
        if filter_type:
            items = [i for i in items if i.diff_type == filter_type]
        if filter_risk:
            items = [i for i in items if i.risk_level == filter_risk]
        self._show_diff_list(items)

    # ── Render helpers ─────────────────────────────────────────────────────────

    def _update_overview(self, result: DiffResult) -> None:
        counts = Counter(item.diff_type for item in result.items)
        for diff_type, lbl in self._overview_labels.items():
            lbl.setText(f"{diff_type}: {counts.get(diff_type, 0)}")

    def _populate_tree(self, result: DiffResult) -> None:
        self._tree.clear()
        all_node = QTreeWidgetItem(["全部差异", str(len(result.items))])
        all_node.setData(0, Qt.UserRole, _ALL_SECTIONS_KEY)
        self._tree.addTopLevelItem(all_node)

        sections: dict[str, list[DiffItem]] = defaultdict(list)
        for item in result.items:
            sections[item.section_path].append(item)
        for section_path, items in sorted(sections.items()):
            display_path = _strip_markdown_formatting(section_path) or section_path
            node = QTreeWidgetItem([display_path, str(len(items))])
            node.setData(0, Qt.UserRole, section_path)
            self._tree.addTopLevelItem(node)
        self._tree.expandAll()

    def _load_version_ir(self, version_id: str) -> DocumentIR | None:
        """Load parsed DocumentIR for a version when available."""
        try:
            version = document_repo.get_version_by_id(self.ctx.conn, version_id)
            if version is None or not version["parsed_json_path"]:
                return None
            path = Path(version["parsed_json_path"])
            if not path.is_absolute():
                path = Path(self.ctx.data_dir) / path
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return _ir_from_dict(data)
        except Exception as exc:
            logger.warning("load parsed IR failed for version %s: %s", version_id, exc)
            return None

    def _build_diff_lookup(self, result: DiffResult, side: str) -> dict[str, DiffItem]:
        lookup: dict[str, DiffItem] = {}
        for item in result.items:
            text = item.baseline_text if side == "baseline" else item.target_text
            key = _normalize_diff_text(text)
            if key:
                lookup[key] = item
        return lookup

    def _render_full_document(self, ir: DocumentIR, lookup: dict[str, DiffItem], side: str) -> str:
        parts: list[str] = []
        for section in ir.sections:
            title = _render_inline_markdown(section.title or "正文")
            parts.append(f"<h3>{title}</h3>")
            for para in section.paragraphs:
                if self._paragraph_looks_like_table(para):
                    parts.append(self._render_table_paragraph(para, lookup, side))
                elif len(para.text) > 500 and len(para.sentences) > 1:
                    for sent in para.sentences:
                        if sent.text.strip():
                            parts.append(self._render_text_unit(sent.text.strip(), lookup, side, line=True))
                else:
                    parts.append(self._render_text_unit(para.text, lookup, side, line=False))
        return "".join(parts)

    def _paragraph_looks_like_table(self, para: Paragraph) -> bool:
        rows = para.sentences or [Sentence(text=line) for line in para.text.splitlines()]
        return sum(1 for row in rows if "|" in row.text) >= 2

    def _render_text_unit(
        self,
        text: str,
        lookup: dict[str, DiffItem],
        side: str,
        *,
        line: bool,
    ) -> str:
        key = _normalize_diff_text(text)
        item = lookup.get(key)
        if item is not None:
            css_cls, _ = _diff_css().get(item.diff_type, ("format", Theme.DIFF_FORMAT))
            other_text = item.target_text if side == "baseline" else item.baseline_text
            content = _render_changed_inline(text, other_text)
            return (
                f'<div class="doc-line diff-item diff-block {css_cls}" '
                f'data-diff-id="{html.escape(item.diff_id, quote=True)}">{content}</div>'
            )
        if line:
            return f'<div class="doc-line">{html.escape(_strip_markdown_formatting(text), quote=False)}</div>'
        return f'<div class="doc-block">{_render_markdown_fragment(text)}</div>'

    def _render_table_paragraph(self, para: Paragraph, lookup: dict[str, DiffItem], side: str) -> str:
        rows = [sent.text.strip() for sent in para.sentences if sent.text.strip()]
        if not rows:
            rows = [line.strip() for line in para.text.splitlines() if line.strip()]

        rendered_rows: list[str] = []
        for index, row in enumerate(rows):
            if _is_markdown_table_separator(row):
                continue

            is_header = index + 1 < len(rows) and _is_markdown_table_separator(rows[index + 1])
            cells = _split_markdown_table_row(row) if "|" in row else [row]
            item = lookup.get(_normalize_diff_text(row))
            row_attrs = 'class="doc-row"'
            other_cells: list[str] = []

            if item is not None:
                css_cls, _ = _diff_css().get(item.diff_type, ("format", Theme.DIFF_FORMAT))
                other_text = item.target_text if side == "baseline" else item.baseline_text
                other_cells = _split_markdown_table_row(other_text) if "|" in other_text else [other_text]
                row_attrs = (
                    f'class="doc-row diff-item diff-row {css_cls}" '
                    f'data-diff-id="{html.escape(item.diff_id, quote=True)}"'
                )

            cell_tag = "th" if is_header else "td"
            cell_html: list[str] = []
            for cell_index, cell in enumerate(cells):
                if item is not None:
                    other_cell = other_cells[cell_index] if cell_index < len(other_cells) else ""
                    if _normalize_diff_text(cell) == _normalize_diff_text(other_cell):
                        content = html.escape(_strip_markdown_formatting(cell), quote=False)
                    else:
                        content = _render_changed_inline(cell, other_cell)
                else:
                    content = _render_inline_markdown(cell)
                cell_html.append(f"<{cell_tag}>{content}</{cell_tag}>")
            rendered_rows.append(f"<tr {row_attrs}>{''.join(cell_html)}</tr>")

        if not rendered_rows:
            return ""
        return f'<div class="doc-table-wrap"><table>{"".join(rendered_rows)}</table></div>'

    def _render_diff(self, result: DiffResult) -> None:
        """Build highlighted HTML and inject it into the WebEngineView via JS."""
        baseline_ir = self._load_version_ir(result.baseline_version_id)
        target_ir = self._load_version_ir(result.target_version_id)
        if baseline_ir is not None and target_ir is not None:
            baseline_html = self._render_full_document(
                baseline_ir,
                self._build_diff_lookup(result, "baseline"),
                "baseline",
            )
            target_html = self._render_full_document(
                target_ir,
                self._build_diff_lookup(result, "target"),
                "target",
            )
            js = (
                f"document.getElementById('baseline-content').innerHTML = "
                f"{json.dumps(baseline_html, ensure_ascii=False)};\n"
                f"document.getElementById('target-content').innerHTML = "
                f"{json.dumps(target_html, ensure_ascii=False)};\n"
                "attachDiffHandlers();"
            )
            self._web_view.page().runJavaScript(js)
            return

        sections: dict[str, list[DiffItem]] = defaultdict(list)
        for item in result.items:
            sections[item.section_path].append(item)

        baseline_parts: list[str] = []
        target_parts: list[str] = []

        for section_path in sorted(sections.keys()):
            items = sections[section_path]
            section_title = _render_inline_markdown(_strip_markdown_formatting(section_path) or section_path)
            baseline_parts.append(f"<h3>{section_title}</h3>")
            target_parts.append(f"<h3>{section_title}</h3>")

            for item in items:
                css_cls, _ = _diff_css().get(item.diff_type, ("format", Theme.DIFF_FORMAT))
                did = html.escape(item.diff_id, quote=True)

                if item.baseline_text:
                    rendered = _render_markdown_fragment(item.baseline_text)
                    baseline_parts.append(
                        f'<div class="diff-item diff-block {css_cls}" data-diff-id="{did}">'
                        f"{rendered}</div>"
                    )

                if item.target_text:
                    rendered = _render_markdown_fragment(item.target_text)
                    target_parts.append(
                        f'<div class="diff-item diff-block {css_cls}" data-diff-id="{did}">'
                        f"{rendered}</div>"
                    )

        baseline_html = "".join(baseline_parts)
        target_html = "".join(target_parts)

        js = (
            f"document.getElementById('baseline-content').innerHTML = "
            f"{json.dumps(baseline_html, ensure_ascii=False)};\n"
            f"document.getElementById('target-content').innerHTML = "
            f"{json.dumps(target_html, ensure_ascii=False)};\n"
            "attachDiffHandlers();"
        )
        self._web_view.page().runJavaScript(js)

    def _show_diff_list(self, items: list[DiffItem]) -> None:
        """Rebuild right-panel cards for the given diff items."""
        self._visible_diff_items = list(items)
        # Remove all cards (keep the stretch at the end)
        while self._detail_layout.count() > 1:
            child = self._detail_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for item in items:
            card = self._make_diff_card(item)
            self._detail_layout.insertWidget(self._detail_layout.count() - 1, card)

    def _make_diff_card(self, item: DiffItem) -> QWidget:
        """Build a compact info card for one DiffItem."""
        css_cls, color = _diff_css().get(item.diff_type, ("format", Theme.DIFF_FORMAT))  # noqa: F841
        risk_color = _risk_colors().get(item.risk_level, Theme.TEXT_SECONDARY)
        card = QWidget()
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setStyleSheet(
            f"background:{self._card_surface_color()};border:1px solid {Theme.BORDER};"
            f"border-left:4px solid {color};"
            "border-radius:8px;padding:8px;"
        )
        card.setProperty("diff_id", item.diff_id)
        card.mousePressEvent = lambda _event, diff_id=item.diff_id: self._on_diff_card_clicked(diff_id)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(4)
        card_layout.setContentsMargins(8, 8, 8, 8)

        # Header: type badge + risk badge
        header_row = QHBoxLayout()
        header_row.setSpacing(6)

        type_badge = QLabel(item.diff_type)
        type_badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        type_badge.setStyleSheet(
            f"background:{Theme.BG_HEADER};color:{color};border:1px solid {color};border-radius:4px;"
            "padding:2px 7px;font-size:11px;font-weight:bold;"
        )
        header_row.addWidget(type_badge)

        risk_lbl = QLabel(_RISK_LABELS.get(item.risk_level, item.risk_level))
        risk_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        risk_lbl.setStyleSheet(
            f"background:{Theme.BG_HEADER};color:{risk_color};border:1px solid {Theme.BORDER};"
            "border-radius:4px;padding:2px 7px;font-size:11px;"
        )
        header_row.addWidget(risk_lbl)
        header_row.addStretch()
        card_layout.addLayout(header_row)

        # Section path
        section_path = _strip_markdown_formatting(item.section_path) or item.section_path
        section_lbl = QLabel(f"章节：{section_path}")
        section_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        section_lbl.setStyleSheet(Theme.label_secondary())
        section_lbl.setWordWrap(True)
        card_layout.addWidget(section_lbl)

        # Baseline text (truncated)
        if item.baseline_text:
            b_text = _strip_markdown_formatting(item.baseline_text)
            display = b_text[:120] + ("…" if len(b_text) > 120 else "")
            b_lbl = QLabel(f"基准：{display}")
            b_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            b_lbl.setStyleSheet(
                f"color:{Theme.TEXT_PRIMARY};font-size:12px;"
                f"background:{Theme.BG_HEADER};border:1px solid {Theme.BORDER};"
                "border-radius:3px;padding:3px 5px;"
            )
            b_lbl.setWordWrap(True)
            card_layout.addWidget(b_lbl)

        # Target text (truncated)
        if item.target_text:
            t_text = _strip_markdown_formatting(item.target_text)
            display = t_text[:120] + ("…" if len(t_text) > 120 else "")
            t_lbl = QLabel(f"目标：{display}")
            t_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            t_lbl.setStyleSheet(
                f"color:{Theme.TEXT_PRIMARY};font-size:12px;"
                f"background:{Theme.BG_HEADER};border:1px solid {Theme.BORDER};"
                "border-radius:3px;padding:3px 5px;"
            )
            t_lbl.setWordWrap(True)
            card_layout.addWidget(t_lbl)

        # Similarity score
        sim_lbl = QLabel(f"相似度：{item.similarity_score:.3f}")
        sim_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        sim_lbl.setStyleSheet(Theme.label_secondary())
        card_layout.addWidget(sim_lbl)

        # AI explanation (truncated)
        if item.explanation:
            exp_text = item.explanation
            display = exp_text[:200] + ("…" if len(exp_text) > 200 else "")
            exp_lbl = QLabel(f"解释：{display}")
            exp_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            exp_lbl.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:12px;")
            exp_lbl.setWordWrap(True)
            card_layout.addWidget(exp_lbl)

        return card

    def _card_surface_color(self) -> str:
        return Theme.BG_CARD

    # ── Theme handling ─────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        """Re-apply all inline stylesheets that reference Theme values."""
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
        self._run_btn.setStyleSheet(Theme.btn_primary())
        self._loading_label.setStyleSheet(Theme.label_secondary())
        self._export_btn.setStyleSheet(
            f"background-color:transparent;color:{Theme.TEXT_PRIMARY};"
            f"border:1px solid {Theme.TEXT_PRIMARY};padding:6px 14px;"
            f"border-radius:{Theme.CARD_RADIUS}px;font-size:13px;"
        )
        for diff_type, lbl in self._overview_labels.items():
            _, color = _diff_css()[diff_type]
            _c = QColor(color)
            _c.setAlpha(30)
            lbl.setStyleSheet(
                f"background:{_c.name(QColor.NameFormat.HexArgb)};border:1px solid {color};"
                f"color:{Theme.TEXT_PRIMARY};border-radius:4px;padding:3px 8px;font-size:12px;"
            )
        self._show_diff_list(self._visible_diff_items)
        self._apply_webview_theme()

    def _apply_webview_theme(self) -> None:
        from app.ui.theme_manager import ThemeManager, ThemeMode
        is_dark = ThemeManager.instance().mode() == ThemeMode.DARK
        js = f"document.body.classList.{'add' if is_dark else 'remove'}('dark');"
        self._web_view.page().runJavaScript(js)
