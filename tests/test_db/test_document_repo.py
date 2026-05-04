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
