"""Compare page — side-by-side document diff view with WebEngine rendering."""
from __future__ import annotations

import html
import json
import logging
from collections import Counter, defaultdict
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

from app.core.types import ComparePolicy, DiffItem, DiffResult
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
    }

_RISK_LABELS: dict[str, str] = {
    "high":   "高风险",
    "medium": "中风险",
    "low":    "低风险",
}


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
        parent=None,
    ):
        super().__init__(parent)
        self._data_dir = data_dir
        self._baseline_version_id = baseline_version_id
        self._target_version_id = target_version_id
        self._embedder = embedder
        self._provider = provider
        self._policy = policy

    def run(self) -> None:
        try:
            from app.agent.compare_graph import compare_graph
            from app.db.schema import open_db

            conn = open_db(self._data_dir)
            try:
                result = compare_graph.invoke({
                    "data_dir": self._data_dir,
                    "baseline_version_id": self._baseline_version_id,
                    "target_version_id": self._target_version_id,
                    "provider": self._provider,
                    "embedder": self._embedder,
                    "conn": conn,
                })
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

        top_layout.addWidget(QLabel("基准版本："))
        self._baseline_combo = QComboBox()
        self._baseline_combo.setMinimumWidth(200)
        self._baseline_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_layout.addWidget(self._baseline_combo)

        top_layout.addWidget(QLabel("目标版本："))
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
        overview_group = QGroupBox("差异概览")
        overview_layout = QHBoxLayout(overview_group)
        overview_layout.setSpacing(8)
        self._overview_labels: dict[str, QLabel] = {}
        for diff_type, (_, color) in _diff_css().items():
            _color = QColor(color)
            _color.setAlpha(30)
            lbl = QLabel(f"{diff_type}: 0")
            lbl.setStyleSheet(
                f"background:{_color.name(QColor.NameFormat.HexArgb)};border:1px solid {color};"
                "border-radius:4px;padding:3px 8px;font-size:12px;"
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
        filter_bar.addWidget(QLabel("筛选："))

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
        self._detail_content = QWidget()
        self._detail_layout = QVBoxLayout(self._detail_content)
        self._detail_layout.setSpacing(6)
        self._detail_layout.addStretch()

        self._detail_scroll.setWidget(self._detail_content)
        layout.addWidget(self._detail_scroll, 1)

        return widget

    # ── Public API ─────────────────────────────────────────────────────────────

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

        self._run_btn.setEnabled(False)
        self._loading_label.setText("对比中，请稍候…")

        thread = QThread()
        worker = _CompareWorker(
            self.ctx.data_dir,
            baseline_version_id,
            target_version_id,
            self.ctx.embedder,
            self.ctx.provider,
            ComparePolicy(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result_ready.connect(self._on_compare_done)
        worker.result_ready.connect(thread.quit)
        worker.error.connect(self._on_compare_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._threads.discard(thread))
        self._threads.add(thread)
        thread.start()

    def _on_compare_done(self, result: DiffResult) -> None:
        self._current_result = result
        self._diff_items_by_id = {item.diff_id: item for item in result.items}
        self._loading_label.setText(f"完成！发现 {len(result.items)} 处差异。")
        self._run_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._update_overview(result)
        self._populate_tree(result)
        self._render_diff(result)
        self._show_diff_list(result.items)

    def _on_compare_error(self, msg: str) -> None:
        self._loading_label.setText("")
        self._run_btn.setEnabled(True)
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
            self._show_diff_list([item])

    def _on_tree_item_clicked(self, tree_item: QTreeWidgetItem, _column: int) -> None:
        if self._current_result is None:
            return
        section_path = tree_item.data(0, Qt.UserRole)
        if section_path:
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
        sections: dict[str, list[DiffItem]] = defaultdict(list)
        for item in result.items:
            sections[item.section_path].append(item)
        for section_path, items in sorted(sections.items()):
            node = QTreeWidgetItem([section_path, str(len(items))])
            node.setData(0, Qt.UserRole, section_path)
            self._tree.addTopLevelItem(node)
        self._tree.expandAll()

    def _render_diff(self, result: DiffResult) -> None:
        """Build highlighted HTML and inject it into the WebEngineView via JS."""
        sections: dict[str, list[DiffItem]] = defaultdict(list)
        for item in result.items:
            sections[item.section_path].append(item)

        baseline_parts: list[str] = []
        target_parts: list[str] = []

        for section_path in sorted(sections.keys()):
            items = sections[section_path]
            section_title = html.escape(section_path)
            baseline_parts.append(f"<h3>{section_title}</h3>")
            target_parts.append(f"<h3>{section_title}</h3>")

            for item in items:
                css_cls, _ = _diff_css().get(item.diff_type, ("format", Theme.DIFF_FORMAT))
                did = html.escape(item.diff_id)

                if item.baseline_text:
                    baseline_parts.append(
                        f'<span class="diff-item {css_cls}" data-diff-id="{did}">'
                        f"{html.escape(item.baseline_text)}</span>"
                    )

                if item.target_text:
                    target_parts.append(
                        f'<span class="diff-item {css_cls}" data-diff-id="{did}">'
                        f"{html.escape(item.target_text)}</span>"
                    )

        baseline_html = "".join(baseline_parts)
        target_html = "".join(target_parts)

        js = (
            f"document.getElementById('baseline-content').innerHTML = "
            f"{json.dumps(baseline_html)};\n"
            f"document.getElementById('target-content').innerHTML = "
            f"{json.dumps(target_html)};\n"
            "attachDiffHandlers();"
        )
        self._web_view.page().runJavaScript(js)

    def _show_diff_list(self, items: list[DiffItem]) -> None:
        """Rebuild right-panel cards for the given diff items."""
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
        _color = QColor(color)
        _color.setAlpha(20)
        _risk_color = QColor(risk_color)
        _risk_color.setAlpha(20)
        card = QWidget()
        card.setStyleSheet(
            f"background:{_color.name(QColor.NameFormat.HexArgb)};border:1px solid {color};"
            "border-radius:8px;padding:8px;"
        )
        card.setProperty("diff_id", item.diff_id)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(4)
        card_layout.setContentsMargins(8, 8, 8, 8)

        # Header: type badge + risk badge
        header_row = QHBoxLayout()
        header_row.setSpacing(6)

        type_badge = QLabel(item.diff_type)
        type_badge.setStyleSheet(
            f"background:{_color.name(QColor.NameFormat.HexArgb)};color:white;border-radius:4px;"
            "padding:2px 7px;font-size:11px;font-weight:bold;"
        )
        header_row.addWidget(type_badge)

        risk_lbl = QLabel(_RISK_LABELS.get(item.risk_level, item.risk_level))
        risk_lbl.setStyleSheet(
            f"background:{_risk_color.name(QColor.NameFormat.HexArgb)};color:{risk_color};"
            "border-radius:4px;padding:2px 7px;font-size:11px;"
        )
        header_row.addWidget(risk_lbl)
        header_row.addStretch()
        card_layout.addLayout(header_row)

        # Section path
        section_lbl = QLabel(f"章节：{item.section_path}")
        section_lbl.setStyleSheet(Theme.label_secondary())
        section_lbl.setWordWrap(True)
        card_layout.addWidget(section_lbl)

        # Baseline text (truncated)
        if item.baseline_text:
            b_text = item.baseline_text
            display = b_text[:120] + ("…" if len(b_text) > 120 else "")
            b_lbl = QLabel(f"基准：{display}")
            b_lbl.setStyleSheet(
                f"color:{Theme.TEXT_PRIMARY};font-size:12px;"
                f"background:{Theme.DIFF_DELETED}20;border-radius:3px;padding:3px 5px;"
            )
            b_lbl.setWordWrap(True)
            card_layout.addWidget(b_lbl)

        # Target text (truncated)
        if item.target_text:
            t_text = item.target_text
            display = t_text[:120] + ("…" if len(t_text) > 120 else "")
            t_lbl = QLabel(f"目标：{display}")
            t_lbl.setStyleSheet(
                f"color:{Theme.TEXT_PRIMARY};font-size:12px;"
                f"background:{Theme.DIFF_ADDED}20;border-radius:3px;padding:3px 5px;"
            )
            t_lbl.setWordWrap(True)
            card_layout.addWidget(t_lbl)

        # Similarity score
        sim_lbl = QLabel(f"相似度：{item.similarity_score:.3f}")
        sim_lbl.setStyleSheet(Theme.label_secondary())
        card_layout.addWidget(sim_lbl)

        # AI explanation (truncated)
        if item.explanation:
            exp_text = item.explanation
            display = exp_text[:200] + ("…" if len(exp_text) > 200 else "")
            exp_lbl = QLabel(f"解释：{display}")
            exp_lbl.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:12px;")
            exp_lbl.setWordWrap(True)
            card_layout.addWidget(exp_lbl)

        return card

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
