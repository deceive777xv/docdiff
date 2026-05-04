"""SQLite schema DDL and database initialization."""
from __future__ import annotations
from pathlib import Path
import sqlite3


DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    doc_name        TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    file_hash       TEXT UNIQUE NOT NULL,
    source_type     TEXT NOT NULL CHECK(source_type IN ('standard','uploaded')),
    business_category TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_versions (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id),
    version_no      INTEGER NOT NULL,
    version_label   TEXT,
    status          TEXT NOT NULL CHECK(status IN ('active','archived','needs_review')),
    parsed_json_path TEXT,
    summary         TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    version_id      TEXT NOT NULL REFERENCES document_versions(id),
    chunk_no        INTEGER NOT NULL,
    section_path    TEXT,
    page_no         INTEGER,
    text            TEXT NOT NULL,
    faiss_index_id  INTEGER DEFAULT -1
);

CREATE TABLE IF NOT EXISTS compare_tasks (
    id                      TEXT PRIMARY KEY,
    baseline_version_id     TEXT NOT NULL,
    target_version_id       TEXT NOT NULL,
    status                  TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
    result_json_path        TEXT,
    created_at              TEXT NOT NULL,
    finished_at             TEXT
);

CREATE TABLE IF NOT EXISTS diff_items (
    id                TEXT PRIMARY KEY,
    compare_task_id   TEXT NOT NULL REFERENCES compare_tasks(id),
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
"""


def get_db_path(data_dir: str) -> Path:
    return Path(data_dir) / "app.db"


def open_db(data_dir: str) -> sqlite3.Connection:
    """Open an existing database for use within one thread (no DDL applied)."""
    db_path = get_db_path(data_dir)
    conn = sqlite3.connect(str(db_path), check_same_thread=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(data_dir: str) -> sqlite3.Connection:
    """Initialize the database, create tables if missing, return connection."""
    db_path = get_db_path(data_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()
    return conn
