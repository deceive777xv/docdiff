"""CRUD for documents and document_versions tables."""
from __future__ import annotations

import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class DeleteVersionResult:
    document_deleted: bool
    cleanup_failures: tuple[Path, ...] = ()


def _now() -> str:
    return datetime.now().isoformat()


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


def list_library_entries(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return every document version as a row, retaining documents without versions."""
    return conn.execute(
        """
        SELECT
            d.id AS document_id,
            d.doc_name,
            d.doc_type,
            d.created_at AS document_created_at,
            v.id AS version_id,
            v.version_no,
            v.version_label,
            v.created_at AS version_created_at
        FROM documents d
        LEFT JOIN document_versions v ON v.document_id = d.id
        ORDER BY d.created_at DESC, v.version_no DESC
        """
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


def list_latest_versions(
    conn: sqlite3.Connection,
    source_type: str | None = None,
) -> list[sqlite3.Row]:
    """Return the latest version row for each document, optionally filtered by source."""
    params: list[str] = []
    where = ""
    if source_type:
        where = "WHERE d.source_type = ?"
        params.append(source_type)

    return conn.execute(
        f"""
        SELECT v.*
        FROM document_versions v
        JOIN documents d ON d.id = v.document_id
        JOIN (
            SELECT document_id, MAX(version_no) AS max_version_no
            FROM document_versions
            GROUP BY document_id
        ) latest
          ON latest.document_id = v.document_id
         AND latest.max_version_no = v.version_no
        {where}
        ORDER BY d.created_at DESC
        """,
        params,
    ).fetchall()


def update_version_status(conn: sqlite3.Connection, version_id: str, status: str) -> None:
    conn.execute(
        "UPDATE document_versions SET status = ? WHERE id = ?", (status, version_id)
    )
    conn.commit()


def _validate_import_artifacts(
    versions: list[sqlite3.Row],
    data_dir: str | None,
) -> None:
    if not data_dir:
        return
    from app.core.structure_repair.storage import import_artifact_paths

    for version in versions:
        if version["parsed_json_path"]:
            import_artifact_paths(data_dir, version["parsed_json_path"])


def _delete_version_rows(
    conn: sqlite3.Connection,
    version_ids: list[str],
) -> None:
    if not version_ids:
        return
    placeholders = ",".join("?" * len(version_ids))
    task_ids = [
        row[0]
        for row in conn.execute(
            f"SELECT id FROM compare_tasks WHERE baseline_version_id IN ({placeholders})"
            f" OR target_version_id IN ({placeholders})",
            version_ids + version_ids,
        ).fetchall()
    ]
    if task_ids:
        task_placeholders = ",".join("?" * len(task_ids))
        conn.execute(
            f"DELETE FROM diff_items WHERE compare_task_id IN ({task_placeholders})",
            task_ids,
        )
        conn.execute(
            f"DELETE FROM compare_tasks WHERE id IN ({task_placeholders})",
            task_ids,
        )
    conn.execute(
        f"DELETE FROM chunks WHERE version_id IN ({placeholders})",
        version_ids,
    )
    conn.execute(
        f"DELETE FROM document_versions WHERE id IN ({placeholders})",
        version_ids,
    )


def _cleanup_version_files(
    versions: list[sqlite3.Row],
    data_dir: str | None,
) -> tuple[Path, ...]:
    if not data_dir:
        return ()
    from app.core.structure_repair.storage import delete_import_artifacts

    cleanup_failures: list[Path] = []
    for version in versions:
        faiss_dir = Path(data_dir) / "faiss" / version["id"]
        try:
            if faiss_dir.exists():
                shutil.rmtree(faiss_dir)
        except OSError:
            cleanup_failures.append(faiss_dir)
        if version["parsed_json_path"]:
            cleanup_failures.extend(
                delete_import_artifacts(data_dir, version["parsed_json_path"])
            )
    return tuple(cleanup_failures)


def delete_version(
    conn: sqlite3.Connection,
    version_id: str,
    data_dir: str | None = None,
) -> DeleteVersionResult:
    """Delete one version and its dependent data, removing an empty parent document."""
    version = conn.execute(
        "SELECT id, document_id, parsed_json_path FROM document_versions WHERE id = ?",
        (version_id,),
    ).fetchone()
    if version is None:
        raise ValueError(f"Version not found: {version_id}")

    versions = [version]
    _validate_import_artifacts(versions, data_dir)
    document_id = str(version["document_id"])
    with conn:
        _delete_version_rows(conn, [version_id])
        has_versions = conn.execute(
            "SELECT 1 FROM document_versions WHERE document_id = ? LIMIT 1",
            (document_id,),
        ).fetchone()
        document_deleted = has_versions is None
        if document_deleted:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    return DeleteVersionResult(
        document_deleted=document_deleted,
        cleanup_failures=_cleanup_version_files(versions, data_dir),
    )


def delete_document(
    conn: sqlite3.Connection,
    doc_id: str,
    data_dir: str | None = None,
) -> tuple[Path, ...]:
    """Delete a document and all associated data (versions, chunks, compare tasks, diff items).

    If data_dir is provided, also removes each version's FAISS and import artifacts.
    """
    versions = conn.execute(
        "SELECT id, parsed_json_path FROM document_versions WHERE document_id = ?",
        (doc_id,),
    ).fetchall()
    version_ids = [v["id"] for v in versions]

    _validate_import_artifacts(versions, data_dir)

    with conn:
        _delete_version_rows(conn, version_ids)
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    return _cleanup_version_files(versions, data_dir)
