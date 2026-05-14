"""Tests for app/ui/pages/compare_page.py."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QWidget

from app.config.settings import AppSettings
from app.db import document_repo
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

    def page(self):
        if not hasattr(self, "_mock_page"):
            self._mock_page = MagicMock()
        return self._mock_page

    def load(self, *_args):
        pass


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
