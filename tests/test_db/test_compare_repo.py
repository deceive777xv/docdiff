"""Tests for app/db/compare_repo.py — CRUD on compare_tasks and diff_items."""
from __future__ import annotations

import time
import uuid

import pytest

from app.core.types import DiffItem
from app.db.schema import init_db
from app.db import compare_repo


@pytest.fixture
def db_conn(tmp_path):
    conn = init_db(str(tmp_path))
    yield conn
    conn.close()


def make_diff_item(diff_type="新增", risk_level="high"):
    return DiffItem(
        diff_id=str(uuid.uuid4()),
        section_path="第一章/第1条",
        diff_type=diff_type,
        risk_level=risk_level,
        baseline_text="原文",
        target_text="新文",
        similarity_score=0.8,
        explanation="说明",
    )


def test_create_task_pending(db_conn):
    """create_compare_task creates a task with status 'pending'."""
    task_id = compare_repo.create_compare_task(
        db_conn,
        baseline_version_id="bv-001",
        target_version_id="tv-001",
    )
    row = compare_repo.get_task_by_id(db_conn, task_id)
    assert row is not None
    assert row["status"] == "pending"
    assert row["baseline_version_id"] == "bv-001"
    assert row["target_version_id"] == "tv-001"


def test_update_task_completed(db_conn):
    """update_task_status sets finished_at and result_json_path on completion."""
    task_id = compare_repo.create_compare_task(
        db_conn,
        baseline_version_id="bv-002",
        target_version_id="tv-002",
    )
    compare_repo.update_task_status(
        db_conn, task_id, "completed", result_json_path="/data/result.json"
    )
    row = compare_repo.get_task_by_id(db_conn, task_id)
    assert row["status"] == "completed"
    assert row["result_json_path"] == "/data/result.json"
    assert row["finished_at"] is not None


def test_list_tasks_ordered(db_conn):
    """list_tasks returns tasks ordered by created_at DESC (most recent first)."""
    id1 = compare_repo.create_compare_task(
        db_conn,
        baseline_version_id="bv-003",
        target_version_id="tv-003",
    )
    # Small sleep to ensure different created_at timestamps
    time.sleep(0.01)
    id2 = compare_repo.create_compare_task(
        db_conn,
        baseline_version_id="bv-004",
        target_version_id="tv-004",
    )
    tasks = compare_repo.list_tasks(db_conn)
    assert len(tasks) == 2
    # Most recent first
    assert tasks[0]["id"] == id2
    assert tasks[1]["id"] == id1


def test_insert_and_get_diff_items(db_conn):
    """insert_diff_items stores items; get_diff_items returns all of them."""
    task_id = compare_repo.create_compare_task(
        db_conn,
        baseline_version_id="bv-005",
        target_version_id="tv-005",
    )
    items = [make_diff_item() for _ in range(3)]
    compare_repo.insert_diff_items(db_conn, task_id, items)
    rows = compare_repo.get_diff_items(db_conn, task_id)
    assert len(rows) == 3


def test_get_diff_items_filtered(db_conn):
    """get_diff_items with diff_type filter returns only matching rows."""
    task_id = compare_repo.create_compare_task(
        db_conn,
        baseline_version_id="bv-006",
        target_version_id="tv-006",
    )
    items = [
        make_diff_item(diff_type="新增"),
        make_diff_item(diff_type="新增"),
        make_diff_item(diff_type="删减"),
    ]
    compare_repo.insert_diff_items(db_conn, task_id, items)
    rows = compare_repo.get_diff_items(db_conn, task_id, diff_type="新增")
    assert len(rows) == 2
    assert all(row["diff_type"] == "新增" for row in rows)
