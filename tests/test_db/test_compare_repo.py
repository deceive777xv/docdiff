"""Tests for app/db/compare_repo.py — CRUD on compare_tasks and diff_items."""
from __future__ import annotations

import time
import uuid

import pytest

from app.core.types import DiffItem
from app.db.schema import init_db
from app.db import compare_repo, document_repo


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


def test_list_recent_task_summaries_includes_versions_and_result_counts(db_conn):
    """Recent task summaries include version labels and diff/risk counts."""
    doc_id = document_repo.insert_document(
        db_conn,
        doc_name="合同",
        doc_type="docx",
        file_path="/docs/contract.docx",
        file_hash="hash-contract",
        source_type="standard",
    )
    baseline_id = document_repo.insert_version(
        db_conn,
        document_id=doc_id,
        version_no=1,
        version_label="初稿",
    )
    target_id = document_repo.insert_version(
        db_conn,
        document_id=doc_id,
        version_no=2,
        version_label="终稿",
    )
    task_id = compare_repo.create_compare_task(
        db_conn,
        baseline_version_id=baseline_id,
        target_version_id=target_id,
    )
    compare_repo.insert_diff_items(
        db_conn,
        task_id,
        [
            make_diff_item(diff_type="新增", risk_level="high"),
            make_diff_item(diff_type="删减", risk_level="medium"),
            make_diff_item(diff_type="微调", risk_level="low"),
        ],
    )
    compare_repo.update_task_status(db_conn, task_id, "completed", "/tmp/result.json")

    summaries = compare_repo.list_recent_task_summaries(db_conn, limit=5)

    assert summaries[0]["id"] == task_id
    assert summaries[0]["baseline_doc_name"] == "合同"
    assert summaries[0]["baseline_version_no"] == 1
    assert summaries[0]["baseline_version_label"] == "初稿"
    assert summaries[0]["target_version_no"] == 2
    assert summaries[0]["target_version_label"] == "终稿"
    assert summaries[0]["diff_count"] == 3
    assert summaries[0]["high_count"] == 1
    assert summaries[0]["medium_count"] == 1
    assert summaries[0]["low_count"] == 1


def test_get_task_result_builds_diff_result(db_conn):
    """get_task_result rebuilds a DiffResult from compare_tasks and diff_items."""
    task_id = compare_repo.create_compare_task(
        db_conn,
        baseline_version_id="bv-007",
        target_version_id="tv-007",
    )
    compare_repo.insert_diff_items(db_conn, task_id, [make_diff_item()])

    result = compare_repo.get_task_result(db_conn, task_id)

    assert result.task_id == task_id
    assert result.baseline_version_id == "bv-007"
    assert result.target_version_id == "tv-007"
    assert len(result.items) == 1
    assert result.items[0].baseline_text == "原文"


def test_prepare_task_for_rerun_clears_old_items_and_marks_running(db_conn):
    """Recovering an interrupted task clears stale items and sets status to running."""
    task_id = compare_repo.create_compare_task(
        db_conn,
        baseline_version_id="bv-008",
        target_version_id="tv-008",
    )
    compare_repo.insert_diff_items(db_conn, task_id, [make_diff_item()])
    compare_repo.update_task_status(db_conn, task_id, "failed")

    compare_repo.prepare_task_for_rerun(db_conn, task_id)

    row = compare_repo.get_task_by_id(db_conn, task_id)
    assert row["status"] == "running"
    assert row["finished_at"] is None
    assert row["result_json_path"] == ""
    assert compare_repo.get_diff_items(db_conn, task_id) == []
