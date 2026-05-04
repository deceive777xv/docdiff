"""CRUD for documents and document_versions tables."""
from __future__ import annotations
import sqlite3
import uuid
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_document(
    conn: sqlite3.Connection,
    *,
    doc_name: str,
    doc_type: str,
    file_path: str,
    file_hash: str,
    source_type: str,
    business_category: str = "",
) -> str:
    """Insert a document row. Returns new document id."""
    doc_id = str(uuid.uuid4())
    now = _now()
    conn.execute(
        """INSERT INTO documents
           (id, doc_name, doc_type, file_path, file_hash, source_type, business_category, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (doc_id, doc_name, doc_type, file_path, file_hash, source_type, business_category, now, now),
    )
    conn.commit()
    return doc_id


def get_document_by_hash(conn: sqlite3.Connection, file_hash: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
    ).fetchone()


def get_document_by_id(conn: sqlite3.Connection, doc_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()


def list_documents(
    conn: sqlite3.Connection,
    source_type: str | None = None,
) -> list[sqlite3.Row]:
    if source_type:
        return conn.execute(
            "SELECT * FROM documents WHERE source_type = ? ORDER BY created_at DESC",
            (source_type,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM documents ORDER BY created_at DESC"
    ).fetchall()


def insert_version(
    conn: sqlite3.Connection,
    *,
    document_id: str,
    version_no: int,
    version_label: str = "",
    status: str = "active",
    parsed_json_path: str = "",
    summary: str = "",
) -> str:
    """Insert a document_versions row. Returns new version id."""
    version_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO document_versions
           (id, document_id, version_no, version_label, status, parsed_json_path, summary, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (version_id, document_id, version_no, version_label, status, parsed_json_path, summary, _now()),
    )
    conn.commit()
    return version_id


def get_version_by_id(conn: sqlite3.Connection, version_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM document_versions WHERE id = ?", (version_id,)
    ).fetchone()


def list_versions(conn: sqlite3.Connection, document_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM document_versions WHERE document_id = ? ORDER BY version_no DESC",
        (document_id,),
    ).fetchall()


def update_version_status(conn: sqlite3.Connection, version_id: str, status: str) -> None:
    conn.execute(
        "UPDATE document_versions SET status = ? WHERE id = ?", (status, version_id)
    )
    conn.commit()
