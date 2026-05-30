"""Tests for app/ui/pages/home_page.py."""
from __future__ import annotations

import sqlite3

import pytest
from PySide6.QtWidgets import QMessageBox, QHeaderView, QPushButton

from app.config.settings import AppSettings
from app.core.types import DiffItem
from app.db import compare_repo, document_repo
from app.db.schema import DDL
from app.ui.app_context import AppContext


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
        data_dir="/tmp/test_home_page",
        provider=None,
        embedder=None,
    )


def _insert_versions(conn: sqlite3.Connection) -> tuple[str, str]:
    doc_id = document_repo.insert_document(
        conn,
        doc_name="合同",
        doc_type="docx",
        file_path="/docs/contract.docx",
        file_hash="hash-home-contract",
        source_type="standard",
    )
    baseline_id = document_repo.insert_version(
        conn, document_id=doc_id, version_no=1, version_label="初稿"
    )
    target_id = document_repo.insert_version(
        conn, document_id=doc_id, version_no=2, version_label="终稿"
    )
    return baseline_id, target_id


def _diff_item(diff_id: str, risk_level: str = "high") -> DiffItem:
    return DiffItem(
        diff_id=diff_id,
        section_path="第一章",
        diff_type="新增",
        risk_level=risk_level,
        baseline_text="旧内容",
        target_text="新内容",
        similarity_score=0.5,
        explanation="说明",
    )


@pytest.fixture()
def home_page(qtbot, ctx):
    from app.ui.pages.home_page import HomePage

    page = HomePage(ctx)
    qtbot.addWidget(page)
    yield page


def _action_button(page, row: int, text: str) -> QPushButton:
    action = page._tasks_table.cellWidget(row, 5)
    buttons = action.findChildren(QPushButton)
    return next(button for button in buttons if button.text() == text)


def test_home_page_refresh_shows_versions_result_summary_and_open_action(home_page, mem_conn):
    """Recent completed tasks show version names, result counts, and an open button."""
    baseline_id, target_id = _insert_versions(mem_conn)
    task_id = compare_repo.create_compare_task(
        mem_conn,
        baseline_version_id=baseline_id,
        target_version_id=target_id,
    )
    compare_repo.insert_diff_items(
        mem_conn,
        task_id,
        [_diff_item("d1", "high"), _diff_item("d2", "medium"), _diff_item("d3", "none")],
    )
    compare_repo.update_task_status(mem_conn, task_id, "completed", "/tmp/result.json")

    home_page.refresh()

    assert home_page._tasks_table.columnCount() == 6
    assert home_page._tasks_table.item(0, 1).text() == "合同 v1(初稿) → 合同 v2(终稿)"
    assert home_page._tasks_table.item(0, 3).text() == "3处差异 / 高1 中1 低0 无1"
    action = home_page._tasks_table.cellWidget(0, 5)
    assert [button.text() for button in action.findChildren(QPushButton)] == ["打开", "删除"]


def test_home_page_version_column_does_not_take_stretch_space(home_page):
    """Recent-task version column should stay compact instead of owning free width."""
    header = home_page._tasks_table.horizontalHeader()

    assert header.sectionResizeMode(1) != QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(3) == QHeaderView.ResizeMode.Stretch
    assert home_page._tasks_table.columnWidth(1) <= 320


def test_home_page_open_button_emits_task_id(home_page, mem_conn, qtbot):
    """Clicking a completed task action emits the task id for the main window."""
    baseline_id, target_id = _insert_versions(mem_conn)
    task_id = compare_repo.create_compare_task(
        mem_conn,
        baseline_version_id=baseline_id,
        target_version_id=target_id,
    )
    compare_repo.update_task_status(mem_conn, task_id, "completed", "/tmp/result.json")
    home_page.refresh()

    with qtbot.waitSignal(home_page.compare_task_open_requested) as blocker:
        _action_button(home_page, 0, "打开").click()

    assert blocker.args == [task_id]


def test_home_page_running_task_offers_recovery_when_not_active(home_page, mem_conn, qtbot):
    """A stale running task exposes a recover action."""
    baseline_id, target_id = _insert_versions(mem_conn)
    task_id = compare_repo.create_compare_task(
        mem_conn,
        baseline_version_id=baseline_id,
        target_version_id=target_id,
    )
    compare_repo.update_task_status(mem_conn, task_id, "running")

    home_page.refresh()

    action = _action_button(home_page, 0, "恢复")
    with qtbot.waitSignal(home_page.compare_task_recover_requested) as blocker:
        action.click()
    assert blocker.args == [task_id]


def test_home_page_active_running_task_disables_recovery(home_page, mem_conn):
    """A task running in the current session is shown as active, not recoverable."""
    baseline_id, target_id = _insert_versions(mem_conn)
    task_id = compare_repo.create_compare_task(
        mem_conn,
        baseline_version_id=baseline_id,
        target_version_id=target_id,
    )
    compare_repo.update_task_status(mem_conn, task_id, "running")
    home_page.ctx.active_compare_task_ids.add(task_id)

    home_page.refresh()

    action = _action_button(home_page, 0, "进行中")
    delete_action = _action_button(home_page, 0, "删除")
    assert not action.isEnabled()
    assert not delete_action.isEnabled()


def test_home_page_delete_button_removes_task_after_confirmation(
    home_page, mem_conn, qtbot, monkeypatch
):
    """Clicking delete removes the task record and refreshes recent tasks."""
    baseline_id, target_id = _insert_versions(mem_conn)
    task_id = compare_repo.create_compare_task(
        mem_conn,
        baseline_version_id=baseline_id,
        target_version_id=target_id,
    )
    compare_repo.insert_diff_items(mem_conn, task_id, [_diff_item("d-delete")])
    compare_repo.update_task_status(mem_conn, task_id, "completed", "/tmp/result.json")
    monkeypatch.setattr(
        "app.ui.pages.home_page.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    home_page.refresh()
    _action_button(home_page, 0, "删除").click()
    qtbot.waitUntil(lambda: home_page._tasks_table.rowCount() == 0)

    assert compare_repo.get_task_by_id(mem_conn, task_id) is None
    assert compare_repo.get_diff_items(mem_conn, task_id) == []
