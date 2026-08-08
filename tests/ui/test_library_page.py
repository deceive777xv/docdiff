"""Tests for app/ui/pages/library_page.py."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QMessageBox

from app.config.settings import AppSettings
from app.db import document_repo
from app.db.schema import DDL
from app.ui.app_context import AppContext


def test_document_file_filter_covers_every_supported_extension():
    from app.core.parser.router import SUPPORTED_EXTENSIONS
    from app.ui.pages.library_page import DOCUMENT_FILE_FILTER

    assert all(f"*{extension}" in DOCUMENT_FILE_FILTER for extension in SUPPORTED_EXTENSIONS)


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

    def quit(self, *_args):
        self.quit_called = True

    def deleteLater(self):
        self.deleted = True


class _FakeIngestWorker:
    instances = []

    def __init__(self, ctx, file_path, document_id=None, normalization_depth="standard"):
        self.ctx = ctx
        self.file_path = file_path
        self.document_id = document_id
        self.normalization_depth = normalization_depth
        self.finished = _FakeSignal()
        self.error = _FakeSignal()
        self.refresh_needed = _FakeSignal()
        self.progress = _FakeSignal()
        self.thread = None
        self.deleted = False
        _FakeIngestWorker.instances.append(self)

    def moveToThread(self, thread):
        self.thread = thread

    def run(self):
        pass

    def deleteLater(self):
        self.deleted = True


class _FakeProgressDialog:
    instances = []

    def __init__(self, paths, parent=None):
        self.paths = list(paths)
        self.parent = parent
        self.updates = []
        self.shown = False
        self.hidden = False
        _FakeProgressDialog.instances.append(self)

    def show(self):
        self.shown = True

    def raise_(self):
        pass

    def activateWindow(self):
        pass

    def update_file(self, file_path, percent, stage, state="running"):
        self.updates.append((file_path, percent, stage, state))

    def finish(self):
        self.hidden = True


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
    assert worker.progress.connections


def test_import_thinking_depth_defaults_to_low_and_reaches_workers(library_page):
    _FakeQThread.instances.clear()
    _FakeIngestWorker.instances.clear()

    assert library_page._normalization_depth.currentData() == "off"
    assert [
        library_page._normalization_depth.itemData(index)
        for index in range(library_page._normalization_depth.count())
    ] == ["off", "standard", "review"]
    library_page._normalization_depth.setCurrentIndex(2)

    with (
        patch("app.ui.pages.library_page.QThread", _FakeQThread),
        patch("app.ui.pages.library_page._IngestWorker", _FakeIngestWorker),
    ):
        library_page._start_ingest_batch(["C:/docs/example.docx"])

    assert _FakeIngestWorker.instances[-1].normalization_depth == "review"


def test_run_ingest_finished_discards_the_finished_thread(library_page):
    """Finishing an earlier import must not remove a newer active thread."""
    _FakeQThread.instances.clear()
    _FakeIngestWorker.instances.clear()

    with patch("app.ui.pages.library_page.QThread", _FakeQThread):
        with patch("app.ui.pages.library_page._IngestWorker", _FakeIngestWorker):
            library_page._start_ingest_batch(
                ["C:/docs/one.pdf", "C:/docs/two.pdf"]
            )
            first_thread, second_thread = _FakeQThread.instances

    assert first_thread in library_page._threads
    assert second_thread in library_page._threads

    library_page._on_ingest_thread_finished(first_thread)

    assert first_thread not in library_page._threads
    assert second_thread in library_page._threads


def test_refresh_shows_each_document_version_as_its_own_row(library_page, mem_conn):
    """The library table should expose every version as an independently selectable row."""
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
    assert library_page._table.rowCount() == 2
    assert library_page._table.item(0, 0).text() == "合同"
    assert library_page._table.item(0, 2).text() == "v2（终稿）"
    assert library_page._table.item(1, 0).text() == "合同"
    assert library_page._table.item(1, 2).text() == "v1（初稿）"
    assert library_page._status.text() == "共 1 份文档，2 个版本"


def test_delete_button_removes_only_the_selected_version(library_page, mem_conn):
    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="合同",
        doc_type="docx",
        file_path="/docs/contract.docx",
        file_hash="delete-selected-version-hash",
        source_type="standard",
    )
    v1_id = document_repo.insert_version(
        mem_conn,
        document_id=doc_id,
        version_no=1,
        version_label="初稿",
    )
    v2_id = document_repo.insert_version(
        mem_conn,
        document_id=doc_id,
        version_no=2,
        version_label="终稿",
    )
    library_page.refresh()
    library_page._table.selectRow(0)
    assert library_page._delete_btn.text() == "删除版本"

    with patch(
        "app.ui.pages.library_page.QMessageBox.question",
        return_value=QMessageBox.StandardButton.Yes,
    ) as question:
        library_page._delete_btn.click()

    assert "v2（终稿）" in question.call_args.args[2]
    assert document_repo.get_document_by_id(mem_conn, doc_id) is not None
    assert document_repo.get_version_by_id(mem_conn, v1_id) is not None
    assert document_repo.get_version_by_id(mem_conn, v2_id) is None
    assert library_page._table.rowCount() == 1
    assert library_page._table.item(0, 2).text() == "v1（初稿）"


def test_last_version_confirmation_explains_parent_document_deletion(
    library_page,
    mem_conn,
):
    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="合同",
        doc_type="docx",
        file_path="/docs/contract.docx",
        file_hash="last-version-confirmation-hash",
        source_type="standard",
    )
    version_id = document_repo.insert_version(
        mem_conn,
        document_id=doc_id,
        version_no=1,
    )
    library_page.refresh()
    library_page._table.selectRow(0)

    with patch(
        "app.ui.pages.library_page.QMessageBox.question",
        return_value=QMessageBox.StandardButton.No,
    ) as question:
        library_page._delete_btn.click()

    assert "最后一个版本" in question.call_args.args[2]
    assert "文档记录也会一并删除" in question.call_args.args[2]
    assert document_repo.get_document_by_id(mem_conn, doc_id) is not None
    assert document_repo.get_version_by_id(mem_conn, version_id) is not None


def test_empty_document_is_visible_and_can_be_deleted(library_page, mem_conn):
    doc_id = document_repo.insert_document(
        mem_conn,
        doc_name="空文档",
        doc_type="pdf",
        file_path="/docs/empty.pdf",
        file_hash="empty-document-hash",
        source_type="standard",
    )
    library_page.refresh()

    assert library_page._table.rowCount() == 1
    assert library_page._table.item(0, 2).text() == "暂无版本"
    library_page._table.selectRow(0)
    assert library_page._delete_btn.text() == "删除文档"

    with patch(
        "app.ui.pages.library_page.QMessageBox.question",
        return_value=QMessageBox.StandardButton.Yes,
    ):
        library_page._delete_btn.click()

    assert document_repo.get_document_by_id(mem_conn, doc_id) is None
    assert library_page._table.rowCount() == 0


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ({"file_check": {"status": "file_checked"}}, (10, "正在解析文档")),
        ({"parse_doc": {"status": "parsed"}}, (35, "正在规范化段落与跨页表格")),
        ({"save_document": {"status": "saved"}}, (85, "正在构建检索索引")),
        ({"build_embeddings": {"status": "completed"}}, (100, "导入完成")),
        ({"parse_doc": {"status": "failed"}}, None),
        ({"parse_doc": {"status": "parsed", "error": "parse failed"}}, None),
        ({"unknown_node": {"status": "completed"}}, None),
    ],
)
def test_ingest_graph_updates_map_to_monotonic_progress(update, expected):
    from app.ui.pages.library_page import _progress_from_graph_update

    assert _progress_from_graph_update(update) == expected


def test_ingest_worker_streams_graph_stage_progress(ctx):
    from app.ui.pages.library_page import _IngestWorker

    graph = MagicMock()
    graph.stream.return_value = iter(
        [
            {"file_check": {"status": "file_checked"}},
            {"parse_doc": {"status": "parsed"}},
            {"save_document": {"status": "saved"}},
            {"build_embeddings": {"status": "completed"}},
        ]
    )
    conn = MagicMock()
    worker = _IngestWorker(
        ctx,
        "C:/docs/example.pdf",
        normalization_depth="review",
    )
    progress = []
    completed = []
    worker.progress.connect(
        lambda path, percent, stage: progress.append((path, percent, stage))
    )
    worker.refresh_needed.connect(completed.append)

    with (
        patch("app.agent.ingest_graph.ingest_graph", graph),
        patch("app.db.schema.open_db", return_value=conn),
    ):
        worker.run()

    assert [percent for _, percent, _ in progress] == [0, 10, 35, 85, 100]
    assert completed == ["C:/docs/example.pdf"]
    graph.stream.assert_called_once()
    assert graph.stream.call_args.kwargs == {"stream_mode": "updates"}
    assert graph.stream.call_args.args[0]["normalization_depth"] == "review"
    conn.close.assert_called_once_with()


def test_progress_dialog_averages_file_progress_and_counts_completion(qtbot):
    from app.ui.pages.library_page import _ImportProgressDialog

    dialog = _ImportProgressDialog(["C:/docs/a.pdf", "C:/docs/b.pdf"])
    qtbot.addWidget(dialog)

    dialog.update_file("C:/docs/a.pdf", 100, "导入完成", state="success")
    dialog.update_file("C:/docs/b.pdf", 35, "正在规范化段落与跨页表格")

    assert dialog._progress.value() == 67
    assert dialog._summary.text() == "已完成 1/2"
    assert "a.pdf" in dialog._items["C:/docs/a.pdf"].text()
    assert "正在规范化段落与跨页表格" in dialog._items["C:/docs/b.pdf"].text()


def test_open_progress_dialog_refreshes_inline_styles_when_theme_changes(qtbot):
    from PySide6.QtWidgets import QApplication, QLabel

    from app.ui.pages.library_page import _ImportProgressDialog
    from app.ui.theme import LATTE, MOCHA, Theme
    from app.ui.theme_manager import ThemeManager

    app = QApplication.instance()
    assert app is not None
    previous_manager = ThemeManager._instance
    previous_stylesheet = app.styleSheet()
    previous_theme = {
        name: getattr(Theme, name)
        for name in LATTE
        if hasattr(Theme, name)
    }
    dialog = None
    try:
        ThemeManager._instance = None
        manager = ThemeManager.instance()
        with patch("app.config.settings.save"):
            manager.setup(SimpleNamespace(theme="light"), app)
            dialog = _ImportProgressDialog(["C:/docs/a.pdf"])
            qtbot.addWidget(dialog)

            labels = {
                label.text(): label
                for label in dialog.findChildren(QLabel)
            }
            title = labels["正在导入并规范化文档"]
            summary = labels["已完成 0/1"]
            hint = labels["关闭此窗口不会停止后台导入。"]

            assert LATTE["BG_PAGE"] in dialog.styleSheet()
            assert LATTE["TEXT_PRIMARY"] in title.styleSheet()
            assert LATTE["TEXT_SECONDARY"] in summary.styleSheet()
            assert LATTE["TEXT_SECONDARY"] in hint.styleSheet()

            manager.toggle()

            assert MOCHA["BG_PAGE"] in dialog.styleSheet()
            assert MOCHA["TEXT_PRIMARY"] in title.styleSheet()
            assert MOCHA["TEXT_SECONDARY"] in summary.styleSheet()
            assert MOCHA["TEXT_SECONDARY"] in hint.styleSheet()

            manager.toggle()

            assert LATTE["BG_PAGE"] in dialog.styleSheet()
            assert LATTE["TEXT_PRIMARY"] in title.styleSheet()
            assert LATTE["TEXT_SECONDARY"] in summary.styleSheet()
            assert LATTE["TEXT_SECONDARY"] in hint.styleSheet()
    finally:
        if dialog is not None:
            dialog.close()
        app.setStyleSheet(previous_stylesheet)
        for name, value in previous_theme.items():
            setattr(Theme, name, value)
        ThemeManager._instance = previous_manager


def test_closing_progress_dialog_only_hides_it(qtbot):
    from app.ui.pages.library_page import _ImportProgressDialog

    dialog = _ImportProgressDialog(["C:/docs/a.pdf"])
    qtbot.addWidget(dialog)
    event = MagicMock()

    dialog.show()
    dialog.closeEvent(event)

    assert dialog.isHidden()
    event.ignore.assert_called_once_with()


def test_multi_file_batch_uses_one_dialog_and_one_completion_message(
    library_page,
):
    _FakeQThread.instances.clear()
    _FakeIngestWorker.instances.clear()
    _FakeProgressDialog.instances.clear()

    with (
        patch("app.ui.pages.library_page.QThread", _FakeQThread),
        patch("app.ui.pages.library_page._IngestWorker", _FakeIngestWorker),
        patch("app.ui.pages.library_page._ImportProgressDialog", _FakeProgressDialog),
        patch.object(library_page, "refresh") as refresh,
        patch("app.ui.pages.library_page.QMessageBox.information") as information,
    ):
        library_page._delete_btn.setEnabled(True)
        library_page._start_ingest_batch(
            ["C:/docs/a.pdf", "C:/docs/b.pdf"]
        )

        assert len(_FakeProgressDialog.instances) == 1
        assert len(_FakeIngestWorker.instances) == 2
        assert not library_page._import_btn.isEnabled()
        assert not library_page._add_version_btn.isEnabled()
        assert not library_page._delete_btn.isEnabled()

        first, second = _FakeIngestWorker.instances
        first.progress.emit("C:/docs/a.pdf", 35, "正在规范化段落与跨页表格")
        first.refresh_needed.emit("C:/docs/a.pdf")
        library_page._on_ingest_thread_finished(first.thread)

        assert refresh.call_count == 0
        assert information.call_count == 0

        second.refresh_needed.emit("C:/docs/b.pdf")

        assert refresh.call_count == 0
        assert information.call_count == 0

        library_page._on_ingest_thread_finished(second.thread)

        assert refresh.call_count == 1
        assert information.call_count == 1
        assert library_page._import_btn.isEnabled()
        assert _FakeProgressDialog.instances[0].hidden


def test_batch_reports_partial_failure_after_other_files_continue(library_page):
    _FakeQThread.instances.clear()
    _FakeIngestWorker.instances.clear()
    _FakeProgressDialog.instances.clear()

    with (
        patch("app.ui.pages.library_page.QThread", _FakeQThread),
        patch("app.ui.pages.library_page._IngestWorker", _FakeIngestWorker),
        patch("app.ui.pages.library_page._ImportProgressDialog", _FakeProgressDialog),
        patch.object(library_page, "refresh") as refresh,
        patch("app.ui.pages.library_page.QMessageBox.warning") as warning,
    ):
        library_page._start_ingest_batch(
            ["C:/docs/good.pdf", "C:/docs/bad.pdf"]
        )
        good, bad = _FakeIngestWorker.instances

        bad.error.emit("C:/docs/bad.pdf", "parse failed")
        library_page._on_ingest_thread_finished(bad.thread)
        assert good.thread.quit_called is False
        assert warning.call_count == 0

        good.refresh_needed.emit("C:/docs/good.pdf")

        assert refresh.call_count == 0
        assert warning.call_count == 0

        library_page._on_ingest_thread_finished(good.thread)

        assert refresh.call_count == 1
        assert warning.call_count == 1
        message = warning.call_args.args[2]
        assert "成功 1 个" in message
        assert "失败 1 个" in message
        assert "bad.pdf" in message
