"""Tests for app/ui/pages/qa_page.py."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from app.config.settings import AppSettings
from app.core.types import Chunk, ChunkHit
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
        data_dir="/tmp/test_qa_page",
        provider=None,
        embedder=None,
    )


@pytest.fixture()
def qa_page(qtbot, ctx):
    from app.ui.pages.qa_page import QaPage

    page = QaPage(ctx)
    qtbot.addWidget(page)
    yield page


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_qa_page_instantiates(qtbot, ctx):
    """QaPage must instantiate without raising an exception."""
    from app.ui.pages.qa_page import QaPage

    page = QaPage(ctx)
    qtbot.addWidget(page)

    assert page is not None
    assert page._scope_combo is not None
    assert page._doc_combo is not None
    assert page._input is not None
    assert page._chat_layout is not None


def test_send_question_no_provider(qtbot, qa_page):
    """When ctx.provider is None, sending a question adds an error bubble to chat."""
    assert qa_page.ctx.provider is None

    # layout starts with one stretch item
    initial_count = qa_page._chat_layout.count()
    qa_page._input.setPlainText("什么是安全规范？")
    qa_page.send_question()

    # one assistant error message widget must have been inserted
    assert qa_page._chat_layout.count() > initial_count


def test_add_message_user(qtbot, qa_page):
    """_add_message('user', ...) inserts exactly one widget into the chat layout."""
    before = qa_page._chat_layout.count()
    qa_page._add_message("user", "hello")
    assert qa_page._chat_layout.count() == before + 1


def test_add_message_assistant_with_citations(qtbot, qa_page):
    """Assistant message inserts one bubble; _on_citations inserts a second widget."""
    before = qa_page._chat_layout.count()

    chunk = Chunk(
        id="c1",
        version_id="v1",
        chunk_no=0,
        section_path="1.概述",
        page_no=5,
        text="示例文本",
    )
    hits = [ChunkHit(chunk=chunk, score=0.9)]

    # _add_message inserts the bubble (1 item)
    qa_page._add_message("assistant", "这是回答")
    assert qa_page._chat_layout.count() == before + 1

    # _on_citations inserts the citation row (1 more item)
    qa_page._on_citations(hits)
    assert qa_page._chat_layout.count() == before + 2


def test_refresh_documents(qtbot, qa_page, mem_conn):
    """refresh_documents() populates combo after inserting a document + version."""
    assert qa_page._doc_combo.count() == 0

    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="Test Doc",
        doc_type="pdf",
        file_path="/test/doc.pdf",
        file_hash="hash123",
        source_type="standard",
    )
    document_repo.insert_version(
        mem_conn, document_id=doc_id, version_no=1, version_label="v1"
    )

    qa_page.refresh_documents()

    assert qa_page._doc_combo.count() == 1
    ver_id = qa_page._doc_combo.itemData(0)
    assert isinstance(ver_id, str) and len(ver_id) > 0
