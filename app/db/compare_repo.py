"""CRUD for compare_tasks and diff_items tables."""
from __future__ import annotations
import sqlite3
import uuid
from datetime import datetime, timezone

from app.core.types import DiffItem, DiffResult


def _now() -> str:
    return datetime.now().isoformat()


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


def prepare_task_for_rerun(conn: sqlite3.Connection, task_id: str) -> None:
    """Clear stale output and mark an existing task as running for recovery."""
    if get_task_by_id(conn, task_id) is None:
        raise ValueError(f"Compare task not found: {task_id}")
    conn.execute("DELETE FROM diff_items WHERE compare_task_id = ?", (task_id,))
    conn.execute(
        """UPDATE compare_tasks
           SET status = 'running', result_json_path = '', finished_at = NULL
           WHERE id = ?""",
        (task_id,),
    )
    conn.commit()


def delete_compare_task(conn: sqlite3.Connection, task_id: str) -> None:
    """Delete a compare task and its persisted diff items."""
    conn.execute("DELETE FROM diff_items WHERE compare_task_id = ?", (task_id,))
    conn.execute("DELETE FROM compare_tasks WHERE id = ?", (task_id,))
    conn.commit()


def get_task_by_id(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM compare_tasks WHERE id = ?", (task_id,)
    ).fetchone()


def list_tasks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM compare_tasks ORDER BY created_at DESC"
    ).fetchall()


def list_recent_task_summaries(
    conn: sqlite3.Connection,
    limit: int = 10,
) -> list[sqlite3.Row]:
    """Return recent compare tasks with version labels and diff/risk counts."""
    return conn.execute(
        """
        SELECT
            t.*,
            bd.doc_name AS baseline_doc_name,
            bv.version_no AS baseline_version_no,
            bv.version_label AS baseline_version_label,
            td.doc_name AS target_doc_name,
            tv.version_no AS target_version_no,
            tv.version_label AS target_version_label,
            COUNT(di.id) AS diff_count,
            COALESCE(SUM(CASE WHEN di.risk_level = 'high' THEN 1 ELSE 0 END), 0) AS high_count,
            COALESCE(SUM(CASE WHEN di.risk_level = 'medium' THEN 1 ELSE 0 END), 0) AS medium_count,
            COALESCE(SUM(CASE WHEN di.risk_level = 'low' THEN 1 ELSE 0 END), 0) AS low_count,
            COALESCE(SUM(CASE WHEN di.risk_level = 'none' THEN 1 ELSE 0 END), 0) AS none_count
        FROM compare_tasks t
        LEFT JOIN document_versions bv ON bv.id = t.baseline_version_id
        LEFT JOIN documents bd ON bd.id = bv.document_id
        LEFT JOIN document_versions tv ON tv.id = t.target_version_id
        LEFT JOIN documents td ON td.id = tv.document_id
        LEFT JOIN diff_items di ON di.compare_task_id = t.id
        GROUP BY t.id
        ORDER BY t.created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def insert_diff_items(conn: sqlite3.Connection, task_id: str, items: list[DiffItem]) -> None:
    """Bulk-insert diff items for a completed compare task."""
    rows = [
        (
            item.diff_id, task_id, item.section_path, item.diff_type,
            item.risk_level, item.baseline_text, item.target_text,
            item.similarity_score, item.explanation,
            0, 0,
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


def get_task_result(conn: sqlite3.Connection, task_id: str) -> DiffResult:
    """Rebuild a DiffResult from persisted task and diff item rows."""
    task = get_task_by_id(conn, task_id)
    if task is None:
        raise ValueError(f"Compare task not found: {task_id}")

    items = [
        DiffItem(
            diff_id=row["id"],
            section_path=row["section_path"] or "",
            diff_type=row["diff_type"],
            risk_level=row["risk_level"],
            baseline_text=row["baseline_text"] or "",
            target_text=row["target_text"] or "",
            similarity_score=float(row["similarity_score"] or 0.0),
            explanation=row["explanation"] or "",
        )
        for row in get_diff_items(conn, task_id)
    ]
    return DiffResult(
        task_id=task_id,
        baseline_version_id=task["baseline_version_id"],
        target_version_id=task["target_version_id"],
        items=items,
    )
