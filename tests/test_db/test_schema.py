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
    """After init_db, all 5 expected tables are present in sqlite_master."""
    conn = init_db(str(tmp_path))
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    conn.close()
    table_names = {row["name"] for row in rows}
    expected = {"documents", "document_versions", "chunks", "compare_tasks", "diff_items"}
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
