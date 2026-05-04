"""CRUD for compare_tasks and diff_items tables."""
from __future__ import annotations
import sqlite3
import uuid
from datetime import datetime, timezone

from app.core.types import DiffItem


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_compare_task(
    conn: sqlite3.Connection,
    *,
    baseline_version_id: str,
    target_version_id: str,
) -> str:
    """Create a new compare_task in 'pending' status. Returns task id."""
    task_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO compare_tasks
           (id, baseline_version_id, target_version_id, status, created_at)
           VALUES (?,?,?,?,?)""",
        (task_id, baseline_version_id, target_version_id, "pending", _now()),
    )
    conn.commit()
    return task_id


def update_task_status(
    conn: sqlite3.Connection,
    task_id: str,
    status: str,
    result_json_path: str = "",
) -> None:
    finished_at = _now() if status in ("completed", "failed") else None
    conn.execute(
        """UPDATE compare_tasks
           SET status = ?, result_json_path = ?, finished_at = ?
           WHERE id = ?""",
        (status, result_json_path, finished_at, task_id),
    )
    conn.commit()


def get_task_by_id(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM compare_tasks WHERE id = ?", (task_id,)
    ).fetchone()


def list_tasks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM compare_tasks ORDER BY created_at DESC"
    ).fetchall()


def insert_diff_items(conn: sqlite3.Connection, task_id: str, items: list[DiffItem]) -> None:
    """Bulk-insert diff items for a completed compare task."""
    rows = [
        (
            item.diff_id, task_id, item.section_path, item.diff_type,
            item.risk_level, item.baseline_text, item.target_text,
            item.similarity_score, item.explanation,
            item.baseline_page, item.target_page,
        )
        for item in items
    ]
    conn.executemany(
        """INSERT INTO diff_items
           (id, compare_task_id, section_path, diff_type, risk_level,
            baseline_text, target_text, similarity_score, explanation,
            baseline_page, target_page)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()


def get_diff_items(
    conn: sqlite3.Connection,
    task_id: str,
    diff_type: str | None = None,
    risk_level: str | None = None,
) -> list[sqlite3.Row]:
    """Fetch diff items with optional filters."""
    sql = "SELECT * FROM diff_items WHERE compare_task_id = ?"
    params: list = [task_id]
    if diff_type:
        sql += " AND diff_type = ?"
        params.append(diff_type)
    if risk_level:
        sql += " AND risk_level = ?"
        params.append(risk_level)
    sql += " ORDER BY section_path"
    return conn.execute(sql, params).fetchall()
