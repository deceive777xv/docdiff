"""Tests for app/db/document_repo.py — CRUD for documents and document_versions."""
from __future__ import annotations

import pytest

from app.db.schema import init_db
from app.db import document_repo


@pytest.fixture
def db_conn(tmp_path):
    conn = init_db(str(tmp_path))
    yield conn
    conn.close()


# ── Helper ─────────────────────────────────────────────────────────────────────

def _insert_doc(conn, *, doc_name="Contract A", doc_type="pdf",
                file_path="/tmp/a.pdf", file_hash="hash_abc",
                source_type="standard", business_category="legal"):
    return document_repo.insert_document(
        conn,
        doc_name=doc_name,
        doc_type=doc_type,
        file_path=file_path,
        file_hash=file_hash,
        source_type=source_type,
        business_category=business_category,
    )


def _write_import_artifacts(data_dir, artifact_id):
    paths = (
        data_dir / "parsed" / f"{artifact_id}.json",
        data_dir / "parsed" / "raw" / f"{artifact_id}.json",
        data_dir / "parsed" / "traces" / f"{artifact_id}.structure.json",
        data_dir / "parsed" / "traces" / f"{artifact_id}.normalization.json",
        data_dir / "parsed" / "profiles" / f"{artifact_id}.boundary.json",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    return paths


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_insert_and_get_by_hash(db_conn):
    """insert_document returns an id; get_document_by_hash returns the same row."""
    doc_id = _insert_doc(db_conn)

    row = document_repo.get_document_by_hash(db_conn, "hash_abc")

    assert row is not None
    assert row["id"] == doc_id
    assert row["doc_name"] == "Contract A"
    assert row["doc_type"] == "pdf"
    assert row["file_path"] == "/tmp/a.pdf"
    assert row["file_hash"] == "hash_abc"
    assert row["source_type"] == "standard"
    assert row["business_category"] == "legal"


def test_get_by_id(db_conn):
    """get_document_by_id returns the row inserted by insert_document."""
    doc_id = _insert_doc(db_conn)

    row = document_repo.get_document_by_id(db_conn, doc_id)

    assert row is not None
    assert row["id"] == doc_id
    assert row["file_hash"] == "hash_abc"


def test_list_documents_filtered(db_conn):
    """list_documents with source_type filter returns only matching rows."""
    _insert_doc(db_conn, doc_name="Standard Doc", file_hash="hash_std",
                source_type="standard", file_path="/s.pdf")
    _insert_doc(db_conn, doc_name="Uploaded Doc", file_hash="hash_upl",
                source_type="uploaded", file_path="/u.pdf")

    standards = document_repo.list_documents(db_conn, source_type="standard")
    uploaded = document_repo.list_documents(db_conn, source_type="uploaded")
    all_docs = document_repo.list_documents(db_conn)

    assert len(standards) == 1
    assert standards[0]["source_type"] == "standard"

    assert len(uploaded) == 1
    assert uploaded[0]["source_type"] == "uploaded"

    assert len(all_docs) == 2


def test_insert_version_and_list(db_conn):
    """insert_version + list_versions returns both versions ordered by version_no DESC."""
    doc_id = _insert_doc(db_conn)

    v1_id = document_repo.insert_version(
        db_conn, document_id=doc_id, version_no=1, version_label="v1"
    )
    v2_id = document_repo.insert_version(
        db_conn, document_id=doc_id, version_no=2, version_label="v2"
    )

    versions = document_repo.list_versions(db_conn, doc_id)

    assert len(versions) == 2
    # First row should be the highest version_no (DESC order)
    assert versions[0]["version_no"] == 2
    assert versions[0]["id"] == v2_id
    assert versions[1]["version_no"] == 1
    assert versions[1]["id"] == v1_id


def test_list_library_entries_returns_every_version_and_empty_document(db_conn):
    versioned_doc = _insert_doc(
        db_conn,
        doc_name="Versioned",
        file_hash="hash_versioned",
    )
    v1_id = document_repo.insert_version(
        db_conn,
        document_id=versioned_doc,
        version_no=1,
    )
    v2_id = document_repo.insert_version(
        db_conn,
        document_id=versioned_doc,
        version_no=2,
    )
    empty_doc = _insert_doc(
        db_conn,
        doc_name="Empty",
        file_hash="hash_empty",
        file_path="/tmp/empty.pdf",
    )
    db_conn.execute(
        "UPDATE documents SET created_at = ? WHERE id = ?",
        ("2026-01-01T00:00:00", versioned_doc),
    )
    db_conn.execute(
        "UPDATE documents SET created_at = ? WHERE id = ?",
        ("2026-01-02T00:00:00", empty_doc),
    )
    db_conn.commit()

    entries = document_repo.list_library_entries(db_conn)

    assert [
        (entry["document_id"], entry["version_id"])
        for entry in entries
    ] == [
        (empty_doc, None),
        (versioned_doc, v2_id),
        (versioned_doc, v1_id),
    ]
    assert entries[0]["doc_name"] == "Empty"
    assert entries[1]["version_no"] == 2


def test_list_latest_versions_filters_by_source_type(db_conn):
    """list_latest_versions returns one latest version per document."""
    std_doc = _insert_doc(db_conn, doc_name="Standard", file_hash="hash_std")
    std_v1 = document_repo.insert_version(db_conn, document_id=std_doc, version_no=1)
    std_v2 = document_repo.insert_version(db_conn, document_id=std_doc, version_no=2)
    uploaded_doc = _insert_doc(
        db_conn,
        doc_name="Uploaded",
        file_hash="hash_uploaded",
        source_type="uploaded",
        file_path="/uploaded.pdf",
    )
    uploaded_v3 = document_repo.insert_version(
        db_conn,
        document_id=uploaded_doc,
        version_no=3,
    )

    standards = document_repo.list_latest_versions(db_conn, source_type="standard")
    all_latest = document_repo.list_latest_versions(db_conn)

    assert [row["id"] for row in standards] == [std_v2]
    assert std_v1 not in [row["id"] for row in all_latest]
    assert {row["id"] for row in all_latest} == {std_v2, uploaded_v3}


def test_update_version_status(db_conn):
    """update_version_status changes the status field; get_version_by_id reflects it."""
    doc_id = _insert_doc(db_conn)
    version_id = document_repo.insert_version(
        db_conn, document_id=doc_id, version_no=1, status="active"
    )

    row_before = document_repo.get_version_by_id(db_conn, version_id)
    assert row_before["status"] == "active"

    document_repo.update_version_status(db_conn, version_id, "archived")

    row_after = document_repo.get_version_by_id(db_conn, version_id)
    assert row_after["status"] == "archived"


def test_delete_version_removes_only_selected_versions_data(db_conn, tmp_path):
    doc_id = _insert_doc(db_conn)
    sibling_paths = _write_import_artifacts(tmp_path, "sibling-artifact")
    selected_paths = _write_import_artifacts(tmp_path, "selected-artifact")
    sibling_id = document_repo.insert_version(
        db_conn,
        document_id=doc_id,
        version_no=1,
        parsed_json_path=str(sibling_paths[0]),
    )
    selected_id = document_repo.insert_version(
        db_conn,
        document_id=doc_id,
        version_no=2,
        parsed_json_path=str(selected_paths[0]),
    )
    db_conn.execute(
        """INSERT INTO chunks
           (id, version_id, chunk_no, section_path, page_no, text, faiss_index_id)
           VALUES (?,?,?,?,?,?,?)""",
        ("selected-chunk", selected_id, 0, "", 1, "text", -1),
    )
    db_conn.execute(
        """INSERT INTO compare_tasks
           (id, baseline_version_id, target_version_id, status, created_at)
           VALUES (?,?,?,?,?)""",
        ("selected-task", sibling_id, selected_id, "completed", "2026-01-01"),
    )
    db_conn.execute(
        """INSERT INTO diff_items
           (id, compare_task_id, diff_type, risk_level)
           VALUES (?,?,?,?)""",
        ("selected-diff", "selected-task", "modified", "low"),
    )
    db_conn.commit()
    selected_faiss = tmp_path / "faiss" / selected_id
    selected_faiss.mkdir(parents=True)
    (selected_faiss / "index.faiss").write_bytes(b"index")

    result = document_repo.delete_version(
        db_conn,
        selected_id,
        data_dir=str(tmp_path),
    )

    assert result.document_deleted is False
    assert result.cleanup_failures == ()
    assert document_repo.get_document_by_id(db_conn, doc_id) is not None
    assert document_repo.get_version_by_id(db_conn, sibling_id) is not None
    assert document_repo.get_version_by_id(db_conn, selected_id) is None
    assert db_conn.execute(
        "SELECT 1 FROM chunks WHERE version_id = ?", (selected_id,)
    ).fetchone() is None
    assert db_conn.execute(
        "SELECT 1 FROM compare_tasks WHERE id = 'selected-task'"
    ).fetchone() is None
    assert db_conn.execute(
        "SELECT 1 FROM diff_items WHERE id = 'selected-diff'"
    ).fetchone() is None
    assert not selected_faiss.exists()
    assert all(not path.exists() for path in selected_paths)
    assert all(path.exists() for path in sibling_paths)


def test_delete_version_removes_parent_when_it_was_the_last_version(
    db_conn,
    tmp_path,
):
    doc_id = _insert_doc(db_conn)
    artifact_paths = _write_import_artifacts(tmp_path, "last-artifact")
    version_id = document_repo.insert_version(
        db_conn,
        document_id=doc_id,
        version_no=1,
        parsed_json_path=str(artifact_paths[0]),
    )

    result = document_repo.delete_version(
        db_conn,
        version_id,
        data_dir=str(tmp_path),
    )

    assert result.document_deleted is True
    assert result.cleanup_failures == ()
    assert document_repo.get_document_by_id(db_conn, doc_id) is None
    assert all(not path.exists() for path in artifact_paths)


def test_delete_version_commits_database_and_reports_artifact_cleanup_failure(
    db_conn,
    tmp_path,
    monkeypatch,
):
    doc_id = _insert_doc(db_conn)
    artifact_paths = _write_import_artifacts(tmp_path, "locked-artifact")
    version_id = document_repo.insert_version(
        db_conn,
        document_id=doc_id,
        version_no=1,
        parsed_json_path=str(artifact_paths[0]),
    )
    locked_path = artifact_paths[1].resolve()
    path_type = type(locked_path)
    original_unlink = path_type.unlink

    def fail_locked(path, *args, **kwargs):
        if path == locked_path:
            raise PermissionError("locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "unlink", fail_locked)

    result = document_repo.delete_version(
        db_conn,
        version_id,
        data_dir=str(tmp_path),
    )

    assert result.cleanup_failures == (locked_path,)
    assert document_repo.get_version_by_id(db_conn, version_id) is None
    assert document_repo.get_document_by_id(db_conn, doc_id) is None
    assert locked_path.exists()


def test_delete_missing_version_changes_nothing(db_conn, tmp_path):
    doc_id = _insert_doc(db_conn)

    with pytest.raises(ValueError, match="Version not found"):
        document_repo.delete_version(
            db_conn,
            "missing-version",
            data_dir=str(tmp_path),
        )

    assert document_repo.get_document_by_id(db_conn, doc_id) is not None


def test_delete_document_removes_every_versions_complete_artifact_set(
    db_conn,
    tmp_path,
):
    doc_id = _insert_doc(db_conn)
    first_paths = _write_import_artifacts(tmp_path, "first-artifact")
    second_paths = _write_import_artifacts(tmp_path, "second-artifact")
    document_repo.insert_version(
        db_conn,
        document_id=doc_id,
        version_no=1,
        parsed_json_path=str(first_paths[0]),
    )
    document_repo.insert_version(
        db_conn,
        document_id=doc_id,
        version_no=2,
        parsed_json_path=str(second_paths[0]),
    )

    document_repo.delete_document(db_conn, doc_id, data_dir=str(tmp_path))

    assert all(not path.exists() for path in first_paths)
    assert all(not path.exists() for path in second_paths)
