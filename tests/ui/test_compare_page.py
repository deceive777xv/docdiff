"""Tests for app/ui/pages/compare_page.py."""
from __future__ import annotations

import sqlite3
import json
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from app.config.settings import AppSettings
from app.db import compare_repo, document_repo
from app.db.schema import DDL
from app.ui.app_context import AppContext


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def mem_conn():
    """In-memory SQLite connection with schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture()
def ctx(mem_conn):
    """AppContext backed by in-memory DB, no provider/embedder configured."""
    settings = AppSettings()
    return AppContext(
        settings=settings,
        conn=mem_conn,
        data_dir="/tmp/test_compare_page",
        provider=None,
        embedder=None,
    )


class _FakeWebView(QWidget):
    """Minimal QWidget stand-in for QWebEngineView (no display required)."""

    auto_finish_load = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loadFinished = _FakeSignal()
        self.loaded_urls = []

    def page(self):
        if not hasattr(self, "_mock_page"):
            self._mock_page = MagicMock()
        return self._mock_page

    def load(self, *args):
        if args:
            self.loaded_urls.append(args[0])
        if self.auto_finish_load:
            self.loadFinished.emit(True)


class _DelayedFakeWebView(_FakeWebView):
    auto_finish_load = False


class _FakeSignal:
    def __init__(self):
        self.connections = []

    def connect(self, slot):
        self.connections.append(slot)

    def emit(self, *args):
        for slot in list(self.connections):
            slot(*args)


def test_render_markdown_fragment_formats_tables_and_escapes_html():
    """Markdown fragments should become readable HTML without trusting raw HTML."""
    from app.ui.pages.compare_page import _render_markdown_fragment

    rendered = _render_markdown_fragment(
        "| 项目 | 取值 |\n"
        "| --- | --- |\n"
        "| 付款周期 | <script>alert(1)</script> |\n"
    )

    assert "<table" in rendered
    assert "<th>项目</th>" in rendered
    assert "<td>付款周期</td>" in rendered
    assert "<script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered


def test_render_markdown_fragment_normalizes_literal_breaks_and_emphasis():
    """Stored Markdown/HTML break markers should not appear as visible syntax."""
    from app.ui.pages.compare_page import _render_markdown_fragment

    rendered = _render_markdown_fragment("第一行<br>**重点条款**")

    assert "第一行" in rendered
    assert "<strong>重点条款</strong>" in rendered
    assert "**" not in rendered
    assert "&lt;br&gt;" not in rendered
    assert "<br>" not in rendered


def test_strip_markdown_formatting_removes_common_markers():
    """Detail-card summaries should keep content while dropping Markdown syntax noise."""
    from app.ui.pages.compare_page import _strip_markdown_formatting

    stripped = _strip_markdown_formatting(
        "# 合同条款\n\n"
        "- **付款周期**：`60天`\n"
        "| 项目 | 取值 |\n"
        "| --- | --- |\n"
        "| 期限 | 60天 |\n"
        "备注<br><b>仅展示文字</b>"
    )

    assert "合同条款" in stripped
    assert "付款周期：60天" in stripped
    assert "项目 取值" in stripped
    assert "备注 仅展示文字" in stripped
    assert "| --- | --- |" not in stripped
    assert "**" not in stripped
    assert "`" not in stripped
    assert "<br>" not in stripped
    assert "<b>" not in stripped


def test_render_changed_inline_strips_markdown_before_marking_tokens():
    """Changed inline spans should focus on content words, not Markdown markers."""
    from app.ui.pages.compare_page import _render_changed_inline

    rendered = _render_changed_inline("**付款周期**<br>30天", "**付款周期**<br>60天")

    assert "付款周期" in rendered
    assert "30天" in rendered
    assert "diff-token" in rendered
    assert "**" not in rendered
    assert "&lt;br&gt;" not in rendered


def test_render_inline_markdown_cleans_table_cell_breaks_and_markers():
    """Inline table-cell rendering should not expose source Markdown markers."""
    from app.ui.pages.compare_page import _render_inline_markdown

    rendered = _render_inline_markdown("区<br>域 **销售代表**<br>表")

    assert "区域" in rendered
    assert "<strong>销售代表</strong>" in rendered
    assert "**" not in rendered
    assert "&lt;br" not in rendered
    assert "<br" not in rendered


@pytest.fixture()
def compare_page(qtbot, ctx):
    """ComparePage with QWebEngineView replaced by a plain QWidget mock."""
    with patch("app.ui.pages.compare_page.QWebEngineView", _FakeWebView):
        from app.ui.pages.compare_page import ComparePage

        page = ComparePage(ctx)
        qtbot.addWidget(page)
        yield page


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_compare_page_instantiates(qtbot, ctx):
    """ComparePage must instantiate without raising an exception."""
    with patch("app.ui.pages.compare_page.QWebEngineView", _FakeWebView):
        from app.ui.pages.compare_page import ComparePage

        page = ComparePage(ctx)
        qtbot.addWidget(page)

    assert page is not None
    # Verify key sub-widgets were created
    assert page._baseline_combo is not None
    assert page._target_combo is not None
    assert page._run_btn is not None
    assert page._tree is not None
    assert "无风险" in [page._filter_risk_combo.itemText(i) for i in range(page._filter_risk_combo.count())]


def test_refresh_versions_populates_combos(qtbot, ctx, mem_conn):
    """refresh_versions() should add one entry per document version to each combo."""
    # Start with an empty DB — combos should be empty
    with patch("app.ui.pages.compare_page.QWebEngineView", _FakeWebView):
        from app.ui.pages.compare_page import ComparePage

        page = ComparePage(ctx)
        qtbot.addWidget(page)

    assert page._baseline_combo.count() == 0
    assert page._target_combo.count() == 0

    # Insert a document with two versions
    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="Standards Manual",
        doc_type="pdf",
        file_path="/docs/manual.pdf",
        file_hash="sha256abc",
        source_type="standard",
    )
    document_repo.insert_version(
        mem_conn, document_id=doc_id, version_no=1, version_label="v1"
    )
    document_repo.insert_version(
        mem_conn, document_id=doc_id, version_no=2, version_label="v2"
    )

    page.refresh_versions()

    assert page._baseline_combo.count() == 2
    assert page._target_combo.count() == 2

    # ComboBox item data should be version UUIDs (non-empty strings)
    baseline_data = page._baseline_combo.itemData(0)
    assert isinstance(baseline_data, str) and len(baseline_data) > 0


def test_refresh_versions_run_btn_disabled_when_empty(qtbot, ctx):
    """Run button must stay disabled when no versions are available."""
    with patch("app.ui.pages.compare_page.QWebEngineView", _FakeWebView):
        from app.ui.pages.compare_page import ComparePage

        page = ComparePage(ctx)
        qtbot.addWidget(page)

    assert not page._run_btn.isEnabled()


def test_web_bridge_on_diff_click_emits_signal(qtbot):
    """_WebBridge.onDiffClick must emit diff_clicked with the given diff_id."""
    from app.ui.pages.compare_page import _WebBridge

    bridge = _WebBridge()
    received: list[str] = []
    bridge.diff_clicked.connect(received.append)

    bridge.onDiffClick("diff-001")
    bridge.onDiffClick("diff-002")

    assert received == ["diff-001", "diff-002"]


def test_web_bridge_on_diff_click_no_args(qtbot):
    """_WebBridge.onDiffClick with an empty string should still emit the signal."""
    from app.ui.pages.compare_page import _WebBridge

    bridge = _WebBridge()
    received: list[str] = []
    bridge.diff_clicked.connect(received.append)

    bridge.onDiffClick("")

    assert received == [""]


def test_show_diff_list_renders_cards(qtbot, ctx, mem_conn, compare_page):
    """After _show_diff_list is called, cards appear in the detail panel."""
    from app.core.types import DiffItem

    items = [
        DiffItem(
            diff_id="d1",
            section_path="1.概述",
            diff_type="实质修改",
            risk_level="high",
            baseline_text="原文内容 A",
            target_text="修订内容 A",
            similarity_score=0.62,
            explanation="段落语义发生重大变化。",
        ),
        DiffItem(
            diff_id="d2",
            section_path="2.范围",
            diff_type="新增",
            risk_level="medium",
            baseline_text="",
            target_text="新增内容 B",
            similarity_score=0.0,
            explanation="目标文档新增段落。",
        ),
    ]

    compare_page._show_diff_list(items)

    # detail_layout has N cards + 1 stretch item at the end
    assert compare_page._detail_layout.count() == len(items) + 1


def test_on_diff_clicked_syncs_filter_dropdowns(compare_page):
    """Clicking a middle-pane diff syncs the right-panel filter controls."""
    from app.core.types import DiffItem

    item = DiffItem(
        diff_id="d-sync",
        section_path="1.概述",
        diff_type="实质修改",
        risk_level="high",
        baseline_text="原文",
        target_text="新文",
        similarity_score=0.5,
        explanation="说明",
    )
    compare_page._diff_items_by_id = {item.diff_id: item}

    compare_page._on_diff_clicked(item.diff_id)

    assert compare_page._filter_type_combo.currentData() == "实质修改"
    assert compare_page._filter_risk_combo.currentData() == "high"
    assert compare_page._detail_layout.count() == 2


def test_diff_card_click_focuses_middle_panes(qtbot, compare_page):
    """Clicking a diff card scrolls the middle panes to the matching diff block."""
    from app.core.types import DiffItem

    item = DiffItem(
        diff_id="d-card",
        section_path="1.概述",
        diff_type="新增",
        risk_level="low",
        baseline_text="",
        target_text="新增内容",
        similarity_score=0.0,
        explanation="",
    )
    card = compare_page._make_diff_card(item)
    qtbot.addWidget(card)

    qtbot.mouseClick(card, Qt.MouseButton.LeftButton)

    js = compare_page._web_view.page().runJavaScript.call_args.args[0]
    assert 'focusDiff("d-card")' in js


def test_diff_card_uses_neutral_surface_with_accent_border(compare_page):
    """Cards should use a calm surface background with diff color as an accent border."""
    from app.core.types import DiffItem

    item = DiffItem(
        diff_id="d-style",
        section_path="1.概述",
        diff_type="重写",
        risk_level="medium",
        baseline_text="旧内容",
        target_text="新内容",
        similarity_score=0.2,
        explanation="",
    )

    card = compare_page._make_diff_card(item)

    assert f"background:{compare_page._card_surface_color()}" in card.styleSheet()
    assert "border-left:4px solid" in card.styleSheet()
    assert "HexArgb" not in card.styleSheet()


def test_diff_card_strips_markdown_from_section_path(compare_page):
    """Detail card section labels should be plain readable text."""
    from app.core.types import DiffItem

    item = DiffItem(
        diff_id="d-section-md",
        section_path="**员工考勤摘要 （** **2025 年**3`月） **",
        diff_type="实质修改",
        risk_level="high",
        baseline_text="旧内容",
        target_text="新内容",
        similarity_score=0.5,
        explanation="",
    )

    card = compare_page._make_diff_card(item)
    labels = [label.text() for label in card.findChildren(QLabel)]
    section_labels = [text for text in labels if text.startswith("章节：")]

    assert section_labels
    assert "员工考勤摘要" in section_labels[0]
    assert "2025 年3月" in section_labels[0]
    assert "**" not in section_labels[0]
    assert "`" not in section_labels[0]


def test_theme_change_restyles_visible_diff_cards(compare_page, monkeypatch):
    """Existing diff cards should immediately pick up current Theme colors."""
    from app.core.types import DiffItem
    from app.ui.theme import Theme

    item = DiffItem(
        diff_id="d-theme",
        section_path="1.概述",
        diff_type="新增",
        risk_level="low",
        baseline_text="旧内容",
        target_text="新内容",
        similarity_score=0.8,
        explanation="说明",
    )
    compare_page._show_diff_list([item])

    monkeypatch.setattr(Theme, "BG_CARD", "#123456")
    monkeypatch.setattr(Theme, "BG_HEADER", "#654321")
    compare_page._apply_theme()

    card = compare_page._detail_layout.itemAt(0).widget()
    assert "background:#123456" in card.styleSheet()


def test_tree_has_all_diffs_node_and_can_restore_full_list(compare_page):
    """The chapter tree should offer a way back to all file diffs."""
    from app.core.types import DiffItem, DiffResult

    items = [
        DiffItem(
            diff_id="d1",
            section_path="第一章",
            diff_type="新增",
            risk_level="low",
            baseline_text="",
            target_text="新增内容",
            similarity_score=0.0,
            explanation="",
        ),
        DiffItem(
            diff_id="d2",
            section_path="第二章",
            diff_type="删减",
            risk_level="high",
            baseline_text="删除内容",
            target_text="",
            similarity_score=0.0,
            explanation="",
        ),
    ]
    result = DiffResult("task-1", "b", "t", items)
    compare_page._current_result = result
    compare_page._populate_tree(result)

    all_node = compare_page._tree.topLevelItem(0)
    section_node = compare_page._tree.topLevelItem(1)

    assert all_node.text(0) == "全部差异"
    assert all_node.text(1) == "2"

    compare_page._on_tree_item_clicked(section_node, 0)
    assert compare_page._detail_layout.count() == 2

    compare_page._on_tree_item_clicked(all_node, 0)
    assert compare_page._detail_layout.count() == 3


def test_tree_selection_clears_filter_dropdowns(compare_page):
    """Tree navigation should not leave stale type/risk filters visible."""
    from app.core.types import DiffItem, DiffResult

    item = DiffItem(
        diff_id="d1",
        section_path="第一章",
        diff_type="新增",
        risk_level="low",
        baseline_text="",
        target_text="新增内容",
        similarity_score=0.0,
        explanation="",
    )
    result = DiffResult("task-1", "b", "t", [item])
    compare_page._current_result = result
    compare_page._populate_tree(result)
    compare_page._select_combo_value(compare_page._filter_type_combo, "新增")
    compare_page._select_combo_value(compare_page._filter_risk_combo, "low")

    compare_page._on_tree_item_clicked(compare_page._tree.topLevelItem(1), 0)

    assert compare_page._filter_type_combo.currentData() is None
    assert compare_page._filter_risk_combo.currentData() is None


def test_render_diff_injects_markdown_html_into_middle_panes(compare_page):
    """The center panes should receive rendered Markdown inside clickable diff blocks."""
    from app.core.types import DiffItem, DiffResult

    result = DiffResult(
        task_id="task-1",
        baseline_version_id="b",
        target_version_id="t",
        items=[
            DiffItem(
                diff_id="d1",
                section_path="合同条款",
                diff_type="新增",
                risk_level="medium",
                baseline_text="| 项目 | 取值 |\n| --- | --- |\n| 周期 | 30天 |",
                target_text="# 新条款\n\n- 付款周期调整为60天",
                similarity_score=0.3,
                explanation="",
            )
        ],
    )

    compare_page._render_diff(result)

    js = compare_page._web_view.page().runJavaScript.call_args.args[0]
    assert "<table" in js
    assert "<th>项目</th>" in js
    assert "<h4>新条款</h4>" in js
    assert "<li>付款周期调整为60天</li>" in js
    assert "diff-item diff-block added" in js


def test_render_diff_waits_for_web_template_load(qtbot, ctx):
    """Compare results should be injected after the WebEngine template is ready."""
    from app.core.types import DiffItem, DiffResult
    from app.ui.pages.compare_page import ComparePage

    with patch("app.ui.pages.compare_page.QWebEngineView", _DelayedFakeWebView):
        page = ComparePage(ctx)
        qtbot.addWidget(page)

    result = DiffResult(
        task_id="task-delayed-web",
        baseline_version_id="b",
        target_version_id="t",
        items=[
            DiffItem(
                diff_id="d-delayed",
                section_path="第一章",
                diff_type="新增",
                risk_level="low",
                baseline_text="",
                target_text="新增内容",
                similarity_score=0.0,
                explanation="",
            )
        ],
    )

    page._render_diff(result)

    assert not page._web_view.page().runJavaScript.called

    page._web_view.loadFinished.emit(True)

    assert page._web_view.page().runJavaScript.called
    js = page._web_view.page().runJavaScript.call_args.args[0]
    assert "新增内容" in js
    assert "target-content" in js


def test_render_diff_uses_full_document_with_row_and_token_marks(
    compare_page,
    mem_conn,
    tmp_path,
):
    """When parsed JSON exists, middle panes show full docs and mark the changed row/cell."""
    from app.core.types import DiffItem, DiffResult, DocumentIR, Paragraph, Section, Sentence

    def write_ir(name: str, rows: list[str]) -> str:
        para = Paragraph(
            paragraph_id=f"{name}-p1",
            text="\n".join(rows),
            sentences=[Sentence(text=row) for row in rows],
        )
        ir = DocumentIR(
            doc_id=name,
            title=name,
            file_hash=name,
            sections=[Section(section_id=f"{name}-s1", title="费用表", level=1, paragraphs=[para])],
            plain_text=para.text,
        )
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(asdict(ir), ensure_ascii=False), encoding="utf-8")
        return str(path)

    baseline_rows = [
        "| 项目 | 取值 |",
        "| --- | --- |",
        "| 付款周期 | 30天 |",
        "| 保留条款 | 内容不变 |",
    ]
    target_rows = [
        "| 项目 | 取值 |",
        "| --- | --- |",
        "| 付款周期 | 60天 |",
        "| 保留条款 | 内容不变 |",
    ]
    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="合同",
        doc_type="docx",
        file_path="/docs/contract.docx",
        file_hash="full-doc-contract",
        source_type="standard",
    )
    baseline_id = document_repo.insert_version(
        mem_conn,
        document_id=doc_id,
        version_no=1,
        parsed_json_path=write_ir("baseline", baseline_rows),
    )
    target_id = document_repo.insert_version(
        mem_conn,
        document_id=doc_id,
        version_no=2,
        parsed_json_path=write_ir("target", target_rows),
    )
    result = DiffResult(
        task_id="task-full-doc",
        baseline_version_id=baseline_id,
        target_version_id=target_id,
        items=[
            DiffItem(
                diff_id="diff-row",
                section_path="费用表",
                diff_type="实质修改",
                risk_level="high",
                baseline_text="| 付款周期 | 30天 |",
                target_text="| 付款周期 | 60天 |",
                similarity_score=0.8,
                explanation="付款周期变化",
            )
        ],
    )

    compare_page._render_diff(result)

    js = compare_page._web_view.page().runJavaScript.call_args.args[0]
    assert "保留条款" in js
    assert 'data-diff-id=\\"diff-row\\"' in js or 'data-diff-id="diff-row"' in js
    assert "diff-token" in js
    assert "30天" in js
    assert "60天" in js


def test_diff_template_exposes_focus_diff_function():
    """The WebView template exposes a JS function used by card clicks."""
    from pathlib import Path

    html = Path("assets/diff_template.html").read_text(encoding="utf-8")

    assert "function focusDiff(diffId)" in html
    assert "scrollIntoView" in html


def test_diff_template_syncs_panes_by_relative_scroll_progress():
    """Either document pane should drive proportional scrolling in the other pane."""
    from pathlib import Path

    template = Path("assets/diff_template.html").read_text(encoding="utf-8")

    assert "function maximumScrollTop(pane)" in template
    assert "function syncPaneScroll(sourcePane, targetPane)" in template
    assert "sourcePane.scrollTop / sourceMaximum" in template
    assert "progress * targetMaximum" in template
    assert "baselinePane.addEventListener('scroll'" in template
    assert "targetPane.addEventListener('scroll'" in template
    assert "scrollSyncInProgress" in template


def test_diff_template_click_centers_matching_items_and_pauses_sync_feedback():
    """Clicking a document diff should center both matches without scroll fighting."""
    from pathlib import Path

    template = Path("assets/diff_template.html").read_text(encoding="utf-8")

    assert "function pausePaneScrollSync()" in template
    assert "function schedulePaneScrollSyncResume()" in template
    assert "pausePaneScrollSync();" in template
    assert "schedulePaneScrollSyncResume();" in template
    assert "target.scrollIntoView({ behavior: 'smooth', block: 'center' });" in template
    assert "focusDiff(diffId);" in template


def test_render_diff_resets_both_panes_without_auto_focusing(compare_page):
    """Injecting a result should show document tops without selecting a diff."""
    from app.core.types import DiffItem, DiffResult

    result = DiffResult(
        task_id="task-top",
        baseline_version_id="baseline-top",
        target_version_id="target-top",
        items=[
            DiffItem(
                diff_id="diff-top",
                section_path="第一章",
                diff_type="实质修改",
                risk_level="medium",
                baseline_text="旧内容",
                target_text="新内容",
                similarity_score=0.5,
                explanation="",
            )
        ],
    )

    compare_page._render_diff(result)

    script = compare_page._web_view.page().runJavaScript.call_args.args[0]
    assert "resetDiffPaneScroll();" in script
    assert "focusDiff(" not in script


def test_load_task_result_populates_compare_page(qtbot, ctx, mem_conn):
    """A completed task can be opened from the home page and rendered in ComparePage."""
    from app.core.types import DiffItem

    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="合同",
        doc_type="docx",
        file_path="/docs/contract.docx",
        file_hash="compare-load-contract",
        source_type="standard",
    )
    baseline_id = document_repo.insert_version(
        mem_conn, document_id=doc_id, version_no=1, version_label="初稿"
    )
    target_id = document_repo.insert_version(
        mem_conn, document_id=doc_id, version_no=2, version_label="终稿"
    )
    task_id = compare_repo.create_compare_task(
        mem_conn,
        baseline_version_id=baseline_id,
        target_version_id=target_id,
    )
    compare_repo.insert_diff_items(
        mem_conn,
        task_id,
        [
            DiffItem(
                diff_id="d1",
                section_path="第一章",
                diff_type="新增",
                risk_level="high",
                baseline_text="旧内容",
                target_text="新内容",
                similarity_score=0.5,
                explanation="说明",
            )
        ],
    )
    compare_repo.update_task_status(mem_conn, task_id, "completed", "/tmp/result.json")

    with patch("app.ui.pages.compare_page.QWebEngineView", _FakeWebView):
        from app.ui.pages.compare_page import ComparePage

        page = ComparePage(ctx)
        qtbot.addWidget(page)

    page.load_task(task_id)

    assert page._current_result is not None
    assert page._current_result.task_id == task_id
    assert page._loading_label.text() == "已加载任务结果：1 处差异。"
    assert page._export_btn.isEnabled()
    assert page._web_view.page().runJavaScript.called


def test_refresh_keeps_selectors_aligned_with_displayed_result(qtbot, ctx, mem_conn):
    """Refreshing the page should not replace displayed result versions with newest docs."""
    from app.core.types import DiffItem

    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="当前对比文档",
        doc_type="docx",
        file_path="/docs/current.docx",
        file_hash="compare-current-doc",
        source_type="standard",
    )
    baseline_id = document_repo.insert_version(
        mem_conn, document_id=doc_id, version_no=1, version_label="基准"
    )
    target_id = document_repo.insert_version(
        mem_conn, document_id=doc_id, version_no=2, version_label="目标"
    )
    task_id = compare_repo.create_compare_task(
        mem_conn,
        baseline_version_id=baseline_id,
        target_version_id=target_id,
    )
    compare_repo.insert_diff_items(
        mem_conn,
        task_id,
        [
            DiffItem(
                diff_id="d-current",
                section_path="第一章",
                diff_type="新增",
                risk_level="low",
                baseline_text="旧内容",
                target_text="新内容",
                similarity_score=0.5,
                explanation="说明",
            )
        ],
    )
    compare_repo.update_task_status(mem_conn, task_id, "completed", "/tmp/result.json")

    with patch("app.ui.pages.compare_page.QWebEngineView", _FakeWebView):
        from app.ui.pages.compare_page import ComparePage

        page = ComparePage(ctx)
        qtbot.addWidget(page)

    page.load_task(task_id)

    newest_doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="最新导入文档",
        doc_type="docx",
        file_path="/docs/newest.docx",
        file_hash="compare-newest-doc",
        source_type="standard",
    )
    newest_version_id = document_repo.insert_version(
        mem_conn, document_id=newest_doc_id, version_no=1, version_label="最新"
    )

    page.refresh()

    assert page._baseline_combo.currentData() == baseline_id
    assert page._target_combo.currentData() == target_id
    assert page._baseline_combo.currentData() != newest_version_id
    assert page._target_combo.currentData() != newest_version_id


def test_clear_webview_reloads_empty_template(compare_page):
    """Clearing result content should reset the WebEngine document itself."""
    previous_load_count = len(compare_page._web_view.loaded_urls)

    compare_page._clear_webview()

    assert len(compare_page._web_view.loaded_urls) == previous_load_count + 1


def test_clear_task_if_displayed_resets_current_compare_page(compare_page):
    """Deleting the task currently shown in ComparePage should clear all result UI."""
    from app.core.types import DiffItem, DiffResult

    result = DiffResult(
        task_id="task-delete-current",
        baseline_version_id="baseline",
        target_version_id="target",
        items=[
            DiffItem(
                diff_id="d-delete-current",
                section_path="第一章",
                diff_type="新增",
                risk_level="high",
                baseline_text="旧内容",
                target_text="新内容",
                similarity_score=0.5,
                explanation="说明",
            )
        ],
    )
    compare_page._display_result(result, "已加载任务结果：1 处差异。")
    previous_load_count = len(compare_page._web_view.loaded_urls)

    compare_page.clear_task_if_displayed("task-delete-current")

    assert compare_page._current_result is None
    assert compare_page._diff_items_by_id == {}
    assert compare_page._visible_diff_items == []
    assert not compare_page._export_btn.isEnabled()
    assert compare_page._tree.topLevelItemCount() == 0
    assert compare_page._detail_layout.count() == 1
    assert compare_page._loading_label.text() == "当前对比任务已删除。"
    assert len(compare_page._web_view.loaded_urls) == previous_load_count + 1


def test_start_compare_clears_previous_result_before_running(compare_page):
    """Starting a new compare task should remove stale result content immediately."""
    from app.core.types import DiffItem, DiffResult

    result = DiffResult(
        task_id="task-old",
        baseline_version_id="baseline-old",
        target_version_id="target-old",
        items=[
            DiffItem(
                diff_id="d-old",
                section_path="旧章节",
                diff_type="新增",
                risk_level="low",
                baseline_text="",
                target_text="旧结果",
                similarity_score=0.0,
                explanation="",
            )
        ],
    )
    compare_page._display_result(result, "旧任务结果")
    previous_load_count = len(compare_page._web_view.loaded_urls)

    with patch("app.ui.pages.compare_page.QThread") as thread_cls, patch(
        "app.ui.pages.compare_page._CompareWorker"
    ) as worker_cls:
        thread = MagicMock()
        worker = MagicMock()
        thread_cls.return_value = thread
        worker_cls.return_value = worker

        compare_page._start_compare("baseline-new", "target-new", "task-new")

    assert compare_page._current_result is None
    assert compare_page._diff_items_by_id == {}
    assert compare_page._visible_diff_items == []
    assert not compare_page._export_btn.isEnabled()
    assert compare_page._tree.topLevelItemCount() == 0
    assert compare_page._detail_layout.count() == 1
    assert compare_page._loading_label.text() == "对比中，请稍候…"
    assert compare_page._recover_task_id == "task-new"
    assert len(compare_page._web_view.loaded_urls) == previous_load_count + 1
    thread.start.assert_called_once()


def test_load_unfinished_task_sets_recovery_state(qtbot, ctx, mem_conn):
    """Opening an unfinished task selects its versions and prepares a recovery run."""
    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="合同",
        doc_type="docx",
        file_path="/docs/contract-recover.docx",
        file_hash="compare-recover-contract",
        source_type="standard",
    )
    baseline_id = document_repo.insert_version(mem_conn, document_id=doc_id, version_no=1)
    target_id = document_repo.insert_version(mem_conn, document_id=doc_id, version_no=2)
    task_id = compare_repo.create_compare_task(
        mem_conn,
        baseline_version_id=baseline_id,
        target_version_id=target_id,
    )
    compare_repo.update_task_status(mem_conn, task_id, "running")

    with patch("app.ui.pages.compare_page.QWebEngineView", _FakeWebView):
        from app.ui.pages.compare_page import ComparePage

        page = ComparePage(ctx)
        qtbot.addWidget(page)

    page.load_task(task_id)

    assert page._recover_task_id == task_id
    assert page._baseline_combo.currentData() == baseline_id
    assert page._target_combo.currentData() == target_id
    assert page._run_btn.text() == "恢复对比"
    assert "未完成" in page._loading_label.text()


def test_load_active_running_task_does_not_prepare_recovery(qtbot, ctx, mem_conn):
    """An active in-session task is displayed as running rather than recoverable."""
    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="合同",
        doc_type="docx",
        file_path="/docs/contract-active.docx",
        file_hash="compare-active-contract",
        source_type="standard",
    )
    baseline_id = document_repo.insert_version(mem_conn, document_id=doc_id, version_no=1)
    target_id = document_repo.insert_version(mem_conn, document_id=doc_id, version_no=2)
    task_id = compare_repo.create_compare_task(
        mem_conn,
        baseline_version_id=baseline_id,
        target_version_id=target_id,
    )
    compare_repo.update_task_status(mem_conn, task_id, "running")
    ctx.active_compare_task_ids.add(task_id)

    with patch("app.ui.pages.compare_page.QWebEngineView", _FakeWebView):
        from app.ui.pages.compare_page import ComparePage

        page = ComparePage(ctx)
        qtbot.addWidget(page)

    page.load_task(task_id)

    assert page._recover_task_id is None
    assert page._run_btn.text() == "对比中…"
    assert not page._run_btn.isEnabled()
    assert "正在运行" in page._loading_label.text()


def test_recover_task_starts_compare_with_existing_task_id(qtbot, ctx, mem_conn):
    """Recovering a task reuses its id instead of creating a new compare task."""
    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="合同",
        doc_type="docx",
        file_path="/docs/contract-retry.docx",
        file_hash="compare-retry-contract",
        source_type="standard",
    )
    baseline_id = document_repo.insert_version(mem_conn, document_id=doc_id, version_no=1)
    target_id = document_repo.insert_version(mem_conn, document_id=doc_id, version_no=2)
    task_id = compare_repo.create_compare_task(
        mem_conn,
        baseline_version_id=baseline_id,
        target_version_id=target_id,
    )
    compare_repo.update_task_status(mem_conn, task_id, "running")
    ctx.provider = object()
    ctx.embedder = object()

    with patch("app.ui.pages.compare_page.QWebEngineView", _FakeWebView):
        from app.ui.pages.compare_page import ComparePage

        page = ComparePage(ctx)
        qtbot.addWidget(page)

    with patch.object(page, "_start_compare", autospec=True) as start_compare:
        page.recover_task(task_id)

    start_compare.assert_called_once_with(baseline_id, target_id, task_id)


def test_compare_page_provider_check(qtbot, ctx):
    """_run_compare should show a warning when provider is None instead of crashing."""
    with patch("app.ui.pages.compare_page.QWebEngineView", _FakeWebView):
        from app.ui.pages.compare_page import ComparePage

        page = ComparePage(ctx)
        qtbot.addWidget(page)

    # Ensure provider is None
    assert ctx.provider is None

    with patch.object(page, "QMessageBox", create=True):
        # Patch QMessageBox.warning so the dialog doesn't block the test
        with patch("app.ui.pages.compare_page.QMessageBox") as mock_mb:
            page._run_compare()
            mock_mb.warning.assert_called_once()
