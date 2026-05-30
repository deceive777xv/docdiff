"""Tests for app/ui/pages/library_page.py."""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from app.config.settings import AppSettings
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
