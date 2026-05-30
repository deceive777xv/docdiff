"""Tests for app/ui/pages/compare_page.py."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loadFinished = MagicMock()

    def page(self):
        if not hasattr(self, "_mock_page"):
            self._mock_page = MagicMock()
        return self._mock_page

    def load(self, *_args):
        pass


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


def test_strip_markdown_formatting_removes_common_markers():
    """Detail-card summaries should keep content while dropping Markdown syntax noise."""
    from app.ui.pages.compare_page import _strip_markdown_formatting

    stripped = _strip_markdown_formatting(
        "# 合同条款\n\n"
        "- **付款周期**：`60天`\n"
        "| 项目 | 取值 |\n"
        "| --- | --- |\n"
        "| 期限 | 60天 |"
    )

    assert "合同条款" in stripped
    assert "付款周期：60天" in stripped
    assert "项目 取值" in stripped
    assert "| --- | --- |" not in stripped
    assert "**" not in stripped
    assert "`" not in stripped


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


def test_diff_template_exposes_focus_diff_function():
    """The WebView template exposes a JS function used by card clicks."""
    from pathlib import Path

    html = Path("assets/diff_template.html").read_text(encoding="utf-8")

    assert "function focusDiff(diffId)" in html
    assert "scrollIntoView" in html


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
