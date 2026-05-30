"""Tests for app/db/schema.py — SQLite schema DDL and database initialization."""
from __future__ import annotations

import sqlite3

import pytest

from app.db.schema import get_db_path, init_db


def test_init_db_creates_file(tmp_path):
    """init_db creates the app.db file inside data_dir."""
    conn = init_db(str(tmp_path))
    conn.close()
    assert (tmp_path / "app.db").exists()


def test_all_tables_created(tmp_path):
    """After init_db, all expected tables are present in sqlite_master."""
    conn = init_db(str(tmp_path))
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    conn.close()
    table_names = {row["name"] for row in rows}
    expected = {
        "documents",
        "document_versions",
        "chunks",
        "compare_tasks",
        "diff_items",
        "qa_sessions",
        "qa_messages",
        "qa_checkpoints",
        "qa_checkpoint_writes",
        "qa_checkpoint_blobs",
    }
    assert expected == table_names


def test_idempotent(tmp_path):
    """Calling init_db twice on the same directory does not raise."""
    conn1 = init_db(str(tmp_path))
    conn1.close()
    conn2 = init_db(str(tmp_path))
    conn2.close()


def test_foreign_keys_enabled(tmp_path):
    """PRAGMA foreign_keys returns 1 after init_db."""
    conn = init_db(str(tmp_path))
    result = conn.execute("PRAGMA foreign_keys").fetchone()
    conn.close()
    assert result[0] == 1


def test_wal_mode(tmp_path):
    """PRAGMA journal_mode returns 'wal' after init_db."""
    conn = init_db(str(tmp_path))
    result = conn.execute("PRAGMA journal_mode").fetchone()
    conn.close()
    assert result[0] == "wal"


def test_diff_items_accept_none_risk_level(tmp_path):
    """Fresh schemas allow persisted no-risk diff items."""
    conn = init_db(str(tmp_path))
    conn.execute(
        """INSERT INTO compare_tasks
           (id, baseline_version_id, target_version_id, status, created_at)
           VALUES (?,?,?,?,?)""",
        ("task-x", "b", "t", "completed", "now"),
    )
    conn.execute(
        """INSERT INTO diff_items
           (id, compare_task_id, section_path, diff_type, risk_level,
            baseline_text, target_text, similarity_score, explanation,
            baseline_page, target_page)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        ("d-none", "task-x", "章节", "格式变化", "none", "a", "b", 1.0, "", 0, 0),
    )
    conn.commit()
    row = conn.execute("SELECT risk_level FROM diff_items WHERE id = 'd-none'").fetchone()
    conn.close()
    assert row["risk_level"] == "none"


def test_init_db_migrates_old_diff_item_risk_check_to_none(tmp_path):
    """Existing DBs with the old high/medium/low CHECK are migrated in place."""
    db_path = get_db_path(str(tmp_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE diff_items (
            id                TEXT PRIMARY KEY,
            compare_task_id   TEXT NOT NULL,
            section_path      TEXT,
            diff_type         TEXT NOT NULL,
            risk_level        TEXT NOT NULL CHECK(risk_level IN ('high','medium','low')),
            baseline_text     TEXT,
            target_text       TEXT,
            similarity_score  REAL,
            explanation       TEXT,
            baseline_page     INTEGER,
            target_page       INTEGER
        );
        INSERT INTO diff_items
           (id, compare_task_id, section_path, diff_type, risk_level,
            baseline_text, target_text, similarity_score, explanation,
            baseline_page, target_page)
        VALUES ('legacy', 'task-old', '章节', '微调', 'low', 'a', 'b', 0.9, '', 0, 0);
        """
    )
    conn.commit()
    conn.close()

    migrated = init_db(str(tmp_path))
    migrated.execute(
        """INSERT INTO compare_tasks
           (id, baseline_version_id, target_version_id, status, created_at)
           VALUES (?,?,?,?,?)""",
        ("task-x", "b", "t", "completed", "now"),
    )
    migrated.execute(
        """INSERT INTO diff_items
           (id, compare_task_id, section_path, diff_type, risk_level,
            baseline_text, target_text, similarity_score, explanation,
            baseline_page, target_page)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        ("d-none", "task-x", "章节", "格式变化", "none", "a", "b", 1.0, "", 0, 0),
    )
    migrated.commit()
    risks = [
        row["risk_level"]
        for row in migrated.execute("SELECT risk_level FROM diff_items ORDER BY id").fetchall()
    ]
    migrated.close()

    assert risks == ["none", "low"]
