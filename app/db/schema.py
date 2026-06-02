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
    risk_level        TEXT NOT NULL CHECK(risk_level IN ('high','medium','low','none')),
    baseline_text     TEXT,
    target_text       TEXT,
    similarity_score  REAL,
    explanation       TEXT,
    baseline_page     INTEGER,
    target_page       INTEGER
);

CREATE TABLE IF NOT EXISTS qa_sessions (
    id                         TEXT PRIMARY KEY,
    title                      TEXT NOT NULL,
    scope                      TEXT NOT NULL,
    current_version_ids_json   TEXT NOT NULL DEFAULT '[]',
    compare_task_id            TEXT,
    created_at                 TEXT NOT NULL,
    updated_at                 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS qa_messages (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES qa_sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS qa_checkpoints (
    thread_id               TEXT NOT NULL,
    checkpoint_ns           TEXT NOT NULL DEFAULT '',
    checkpoint_id           TEXT NOT NULL,
    checkpoint_type         TEXT NOT NULL,
    checkpoint_blob         BLOB NOT NULL,
    metadata_type           TEXT NOT NULL,
    metadata_blob           BLOB NOT NULL,
    parent_checkpoint_id    TEXT,
    created_at              TEXT NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS qa_checkpoint_writes (
    thread_id        TEXT NOT NULL,
    checkpoint_ns    TEXT NOT NULL DEFAULT '',
    checkpoint_id    TEXT NOT NULL,
    task_id          TEXT NOT NULL,
    idx              INTEGER NOT NULL,
    channel          TEXT NOT NULL,
    value_type       TEXT NOT NULL,
    value_blob       BLOB NOT NULL,
    task_path        TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE TABLE IF NOT EXISTS qa_checkpoint_blobs (
    thread_id        TEXT NOT NULL,
    checkpoint_ns    TEXT NOT NULL DEFAULT '',
    channel          TEXT NOT NULL,
    version          TEXT NOT NULL,
    value_type       TEXT NOT NULL,
    value_blob       BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE INDEX IF NOT EXISTS idx_documents_source_created
ON documents(source_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_document_versions_document_version
ON document_versions(document_id, version_no DESC);

CREATE INDEX IF NOT EXISTS idx_chunks_version_chunk_no
ON chunks(version_id, chunk_no);

CREATE INDEX IF NOT EXISTS idx_chunks_version_faiss_id
ON chunks(version_id, faiss_index_id);

CREATE INDEX IF NOT EXISTS idx_compare_tasks_created
ON compare_tasks(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_compare_tasks_status_created
ON compare_tasks(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_diff_items_task_section
ON diff_items(compare_task_id, section_path);

CREATE INDEX IF NOT EXISTS idx_qa_sessions_updated
ON qa_sessions(updated_at DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_qa_messages_session_rowid
ON qa_messages(session_id, created_at);
"""

_DIFF_ITEM_COLUMNS = """
    id,
    compare_task_id,
    section_path,
    diff_type,
    risk_level,
    baseline_text,
    target_text,
    similarity_score,
    explanation,
    baseline_page,
    target_page
"""

_CREATE_DIFF_ITEMS_SQL = """
CREATE TABLE diff_items (
    id                TEXT PRIMARY KEY,
    compare_task_id   TEXT NOT NULL REFERENCES compare_tasks(id),
    section_path      TEXT,
    diff_type         TEXT NOT NULL,
    risk_level        TEXT NOT NULL CHECK(risk_level IN ('high','medium','low','none')),
    baseline_text     TEXT,
    target_text       TEXT,
    similarity_score  REAL,
    explanation       TEXT,
    baseline_page     INTEGER,
    target_page       INTEGER
);
"""


def _migrate_diff_items_risk_level(conn: sqlite3.Connection) -> None:
    """Rebuild legacy diff_items tables whose risk CHECK lacks 'none'."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='diff_items'"
    ).fetchone()
    if row is None or row["sql"] is None or "'none'" in row["sql"]:
        return

    conn.execute("PRAGMA foreign_keys=OFF;")
    conn.execute("ALTER TABLE diff_items RENAME TO diff_items_legacy;")
    conn.executescript(_CREATE_DIFF_ITEMS_SQL)
    conn.execute(
        f"""INSERT INTO diff_items ({_DIFF_ITEM_COLUMNS})
            SELECT {_DIFF_ITEM_COLUMNS} FROM diff_items_legacy"""
    )
    conn.execute("DROP TABLE diff_items_legacy;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.commit()


def get_db_path(data_dir: str) -> Path:
    return Path(data_dir) / "app.db"


def open_db(data_dir: str) -> sqlite3.Connection:
    """Open an existing database for use within one thread (no DDL applied)."""
    db_path = get_db_path(data_dir)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
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
    _migrate_diff_items_risk_level(conn)
    conn.commit()
    return conn
