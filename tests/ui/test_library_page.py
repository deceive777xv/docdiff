"""Tests for app/ui/pages/library_page.py."""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from app.config.settings import AppSettings
from app.db import document_repo
from app.db.schema import DDL
from app.ui.app_context import AppContext


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


class _FakeIngestWorker:
    instances = []

    def __init__(self, ctx, file_path, document_id=None):
        self.ctx = ctx
        self.file_path = file_path
        self.document_id = document_id
        self.finished = _FakeSignal()
        self.error = _FakeSignal()
        self.refresh_needed = _FakeSignal()
        self.thread = None
        self.deleted = False
        _FakeIngestWorker.instances.append(self)

    def moveToThread(self, thread):
        self.thread = thread

    def run(self):
        pass

    def deleteLater(self):
        self.deleted = True


@pytest.fixture()
def mem_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture()
def ctx(mem_conn):
    return AppContext(
        settings=AppSettings(),
        conn=mem_conn,
        data_dir="/tmp/test_library_page",
        provider=None,
        embedder=None,
    )


@pytest.fixture()
def library_page(qtbot, ctx):
    from app.ui.pages.library_page import LibraryPage

    page = LibraryPage(ctx)
    qtbot.addWidget(page)
    yield page


def test_run_ingest_connects_ui_updates_to_page_slots(library_page):
    """Worker signals that touch UI should connect to LibraryPage slots."""
    _FakeQThread.instances.clear()
    _FakeIngestWorker.instances.clear()

    with patch("app.ui.pages.library_page.QThread", _FakeQThread):
        with patch("app.ui.pages.library_page._IngestWorker", _FakeIngestWorker):
            library_page._run_ingest("C:/docs/example.pdf")

    worker = _FakeIngestWorker.instances[-1]

    assert worker.refresh_needed.connections[0].__self__ is library_page
    assert worker.refresh_needed.connections[0].__name__ == "_on_ingest_done"
    assert worker.error.connections[0].__self__ is library_page
    assert worker.error.connections[0].__name__ == "_on_ingest_error"


def test_run_ingest_finished_discards_the_finished_thread(library_page):
    """Finishing an earlier import must not remove a newer active thread."""
    _FakeQThread.instances.clear()
    _FakeIngestWorker.instances.clear()

    with patch("app.ui.pages.library_page.QThread", _FakeQThread):
        with patch("app.ui.pages.library_page._IngestWorker", _FakeIngestWorker):
            library_page._run_ingest("C:/docs/one.pdf")
            first_thread = _FakeQThread.instances[-1]
            library_page._run_ingest("C:/docs/two.pdf")
            second_thread = _FakeQThread.instances[-1]

    assert first_thread in library_page._threads
    assert second_thread in library_page._threads

    first_thread.finished.emit()

    assert first_thread not in library_page._threads
    assert second_thread in library_page._threads


def test_refresh_shows_latest_version_and_version_count(library_page, mem_conn):
    """The library table should make newly added versions visible to users."""
    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="合同",
        doc_type="docx",
        file_path="/docs/contract.docx",
        file_hash="library-version-hash",
        source_type="standard",
    )
    document_repo.insert_version(
        mem_conn,
        document_id=doc_id,
        version_no=1,
        version_label="初稿",
    )
    document_repo.insert_version(
        mem_conn,
        document_id=doc_id,
        version_no=2,
        version_label="终稿",
    )

    library_page.refresh()

    assert library_page._table.columnCount() == 4
    assert library_page._table.horizontalHeaderItem(2).text() == "版本"
    assert library_page._table.item(0, 2).text() == "v2（终稿） · 共 2 版"
    assert library_page._status.text() == "共 1 份文档，2 个版本"
