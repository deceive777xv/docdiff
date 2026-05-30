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

class _FakeSignal:
    def __init__(self):
        self.connections = []

    def connect(self, slot):
        self.connections.append(slot)

    def emit(self, *args):
        for slot in list(self.connections):
            slot(*args)


class _FakeQThread:
    instances = []

    def __init__(self):
        self.started = _FakeSignal()
        self.finished = _FakeSignal()
        self.started_called = False
        self.quit_called = False
        self.deleted = False
        _FakeQThread.instances.append(self)

    def start(self):
        self.started_called = True

    def quit(self):
        self.quit_called = True

    def deleteLater(self):
        self.deleted = True


class _FakeQaWorker:
    instances = []

    def __init__(self, **_kwargs):
        self.kwargs = _kwargs
        self.token_received = _FakeSignal()
        self.citations_ready = _FakeSignal()
        self.error = _FakeSignal()
        self.done = _FakeSignal()
        self.thread = None
        self.deleted = False
        _FakeQaWorker.instances.append(self)

    def moveToThread(self, thread):
        self.thread = thread

    def run(self):
        pass

    def deleteLater(self):
        self.deleted = True

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


def test_send_question_connects_worker_to_created_thread(qtbot, qa_page):
    """Sending a question must use the created QThread, not QWidget.thread()."""
    _FakeQThread.instances.clear()
    _FakeQaWorker.instances.clear()
    qa_page.ctx.embedder = object()
    qa_page.ctx.lc_model = object()
    qa_page._input.setPlainText("什么是安全规范？")

    with patch("app.ui.pages.qa_page.QThread", _FakeQThread):
        with patch("app.ui.pages.qa_page._QaWorker", _FakeQaWorker):
            qa_page.send_question()

    thread = _FakeQThread.instances[-1]
    worker = _FakeQaWorker.instances[-1]

    assert worker.thread is thread
    assert thread.started.connections[0].__self__ is worker
    assert thread.started.connections[0].__name__ == "run"
    assert thread.started_called


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


def test_theme_change_restyles_existing_reply_bubbles(qtbot, qa_page, monkeypatch):
    """Existing assistant replies should immediately pick up current Theme colors."""
    from app.ui.theme import Theme

    bubble, _ = qa_page._add_message("assistant", "这是回答")
    monkeypatch.setattr(Theme, "BG_CARD", "#123456")
    monkeypatch.setattr(Theme, "BORDER", "#654321")

    qa_page._apply_theme()

    assert "background:#123456" in bubble.styleSheet()
    assert "border:1px solid #654321" in bubble.styleSheet()


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


def test_send_question_passes_compare_task_id_to_worker(qtbot, qa_page):
    """Compare-scope questions need the task id so QA can include diff results."""
    _FakeQThread.instances.clear()
    _FakeQaWorker.instances.clear()
    qa_page.ctx.embedder = object()
    qa_page.ctx.lc_model = object()
    qa_page._scope_combo.setCurrentText("对比文档")
    qa_page._compare_task_combo.addItem(
        "合同 v1 ↔ 合同 v2",
        {
            "task_id": "task-compare-1",
            "version_ids": ("baseline-v1", "target-v1"),
        },
    )
    qa_page._compare_task_combo.setCurrentIndex(0)
    qa_page._input.setPlainText("两者有什么差异？")

    with patch("app.ui.pages.qa_page.QThread", _FakeQThread):
        with patch("app.ui.pages.qa_page._QaWorker", _FakeQaWorker):
            qa_page.send_question()

    worker = _FakeQaWorker.instances[-1]
    assert worker.kwargs["compare_task_id"] == "task-compare-1"
    assert worker.kwargs["current_version_ids"] == ["baseline-v1", "target-v1"]
