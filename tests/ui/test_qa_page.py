"""Tests for app/ui/pages/qa_page.py."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMessageBox

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
    assert page._session_combo is not None


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


def test_send_button_disabled_until_stream_finishes(qtbot, qa_page):
    """A streaming answer in progress should block duplicate sends."""
    _FakeQThread.instances.clear()
    _FakeQaWorker.instances.clear()
    qa_page.ctx.embedder = object()
    qa_page.ctx.lc_model = object()
    qa_page._input.setPlainText("什么是安全规范？")

    with patch("app.ui.pages.qa_page.QThread", _FakeQThread):
        with patch("app.ui.pages.qa_page._QaWorker", _FakeQaWorker):
            qa_page.send_question()

    worker = _FakeQaWorker.instances[-1]
    assert not qa_page._send_btn.isEnabled()

    worker.done.emit()

    assert qa_page._send_btn.isEnabled()


def test_send_question_creates_persisted_session_and_user_message(qtbot, qa_page):
    """Sending the first question stores a QA session and the user message."""
    from app.db import qa_repo

    _FakeQThread.instances.clear()
    _FakeQaWorker.instances.clear()
    qa_page.ctx.embedder = object()
    qa_page.ctx.lc_model = object()
    qa_page._input.setPlainText("两份合同有什么差异？")

    with patch("app.ui.pages.qa_page.QThread", _FakeQThread):
        with patch("app.ui.pages.qa_page._QaWorker", _FakeQaWorker):
            qa_page.send_question()

    sessions = qa_repo.list_sessions(qa_page.ctx.conn)
    messages = qa_repo.list_messages(qa_page.ctx.conn, sessions[0]["id"])
    worker = _FakeQaWorker.instances[-1]

    assert sessions[0]["title"] == "两份合同有什么差异？"
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "两份合同有什么差异？"
    assert worker.kwargs["thread_id"] == sessions[0]["id"]


def test_qa_page_loads_existing_session_messages(qtbot, ctx, mem_conn):
    """Selecting a previous session restores its messages in the chat pane."""
    from app.db import qa_repo
    from app.ui.pages.qa_page import QaPage

    session_id = qa_repo.create_session(mem_conn, title="历史会话", scope="all")
    qa_repo.add_message(mem_conn, session_id, "user", "第一问")
    qa_repo.add_message(mem_conn, session_id, "assistant", "第一答")

    page = QaPage(ctx)
    qtbot.addWidget(page)

    labels = [
        (label.property("qa_role"), label.text())
        for label in page._chat_content.findChildren(QLabel)
        if label.property("qa_role") in ("user", "assistant")
    ]
    assert page._session_combo.currentData() == session_id
    assert labels[0] == ("user", "第一问")
    assert labels[1][0] == "assistant"
    assert "第一答" in labels[1][1]


def test_qa_page_deletes_selected_session(qtbot, qa_page, monkeypatch):
    """The session delete action removes the selected persisted conversation."""
    from app.db import qa_repo

    session_id = qa_repo.create_session(qa_page.ctx.conn, title="待删除", scope="all")
    qa_repo.add_message(qa_page.ctx.conn, session_id, "user", "旧问题")
    qa_page.refresh_sessions(select_session_id=session_id)
    monkeypatch.setattr(
        "app.ui.pages.qa_page.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    qa_page._delete_session()

    assert qa_repo.get_session(qa_page.ctx.conn, session_id) is None
    assert qa_repo.list_messages(qa_page.ctx.conn, session_id) == []


def test_on_done_persists_streamed_assistant_reply(qtbot, qa_page):
    """A completed streamed answer is stored in the current QA session."""
    from app.db import qa_repo

    session_id = qa_repo.create_session(qa_page.ctx.conn, title="问答", scope="all")
    qa_page._thread_id = session_id
    qa_page._session_persisted = True
    qa_page._accumulated = "这是模型回答。"

    qa_page._on_done()

    messages = qa_repo.list_messages(qa_page.ctx.conn, session_id)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "这是模型回答。"


def test_add_message_user(qtbot, qa_page):
    """_add_message('user', ...) inserts exactly one widget into the chat layout."""
    before = qa_page._chat_layout.count()
    qa_page._add_message("user", "hello")
    assert qa_page._chat_layout.count() == before + 1


def test_add_message_assistant_renders_markdown(qtbot, qa_page):
    """Assistant Markdown replies should be rendered as safe rich text."""
    bubble, _ = qa_page._add_message("assistant", "**重点**\n- 条目")

    assert bubble.textFormat() == Qt.TextFormat.RichText
    assert "<strong>重点</strong>" in bubble.text()
    assert "<li>条目</li>" in bubble.text()
    assert "**" not in bubble.text()


def test_streamed_assistant_reply_rerenders_markdown(qtbot, qa_page):
    """Streaming should keep persisted text raw while updating the visible rich text."""
    bubble, _ = qa_page._add_message("assistant", "")
    qa_page._current_bubble = bubble

    qa_page._on_token("**重点**")

    assert qa_page._accumulated == "**重点**"
    assert "<strong>重点</strong>" in bubble.text()
    assert "**" not in bubble.text()


def test_on_citations_does_not_insert_redundant_source_row(qtbot, qa_page):
    """Raw retrieval citations are kept out of the chat UI to avoid noisy source rows."""
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

    qa_page._add_message("assistant", "这是回答")
    assert qa_page._chat_layout.count() == before + 1

    qa_page._on_citations(hits)
    citation_labels = [
        label
        for label in qa_page._chat_content.findChildren(QLabel)
        if label.property("qa_role") == "citation"
    ]
    assert qa_page._chat_layout.count() == before + 1
    assert citation_labels == []


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


def test_scope_change_syncs_selector_labels(qtbot, qa_page):
    """Scope changes should keep the following selector label in sync."""
    qa_page._scope_combo.setCurrentText("当前文档")
    assert not qa_page._doc_label.isHidden()
    assert qa_page._doc_label.text() == "文档："
    assert not qa_page._doc_combo.isHidden()
    assert qa_page._compare_task_label.isHidden()
    assert qa_page._compare_task_combo.isHidden()

    qa_page._scope_combo.setCurrentText("对比文档")
    assert qa_page._doc_label.isHidden()
    assert qa_page._doc_combo.isHidden()
    assert not qa_page._compare_task_label.isHidden()
    assert qa_page._compare_task_label.text() == "对比任务："
    assert not qa_page._compare_task_combo.isHidden()

    qa_page._scope_combo.setCurrentText("文档库")
    assert qa_page._doc_label.isHidden()
    assert qa_page._doc_combo.isHidden()
    assert qa_page._compare_task_label.isHidden()
    assert qa_page._compare_task_combo.isHidden()

    qa_page._scope_combo.setCurrentText("全部")
    assert qa_page._doc_label.isHidden()
    assert qa_page._doc_combo.isHidden()
    assert qa_page._compare_task_label.isHidden()
    assert qa_page._compare_task_combo.isHidden()


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
