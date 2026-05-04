"""Tests for app/db/chunk_repo.py — CRUD for chunks table."""
from __future__ import annotations

import uuid

import pytest

from app.db.schema import init_db
from app.db import document_repo, chunk_repo
from app.core.types import Chunk


@pytest.fixture
def db_conn(tmp_path):
    conn = init_db(str(tmp_path))
    yield conn
    conn.close()


# ── Helper ─────────────────────────────────────────────────────────────────────

def _make_version(conn) -> str:
    """Insert a minimal document + version, return version_id."""
    doc_id = document_repo.insert_document(
        conn,
        doc_name="Test Doc",
        doc_type="pdf",
        file_path="/tmp/test.pdf",
        file_hash=str(uuid.uuid4()),
        source_type="standard",
    )
    return document_repo.insert_version(conn, document_id=doc_id, version_no=1)


def _make_chunks(version_id: str, count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            id=str(uuid.uuid4()),
            version_id=version_id,
            chunk_no=i,
            section_path=f"section/{i}",
            page_no=i + 1,
            text=f"chunk text {i}",
            faiss_index_id=-1,
        )
        for i in range(count)
    ]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_insert_and_get_chunks(db_conn):
    """insert_chunks then get_chunks_by_version returns all chunks ordered by chunk_no."""
    version_id = _make_version(db_conn)
    chunks = _make_chunks(version_id, count=3)

    chunk_repo.insert_chunks(db_conn, chunks)

    rows = chunk_repo.get_chunks_by_version(db_conn, version_id)

    assert len(rows) == 3
    for i, row in enumerate(rows):
        assert row["chunk_no"] == i
        assert row["text"] == f"chunk text {i}"
        assert row["version_id"] == version_id


def test_update_faiss_ids(db_conn):
    """update_faiss_ids sets faiss_index_id on targeted chunks."""
    version_id = _make_version(db_conn)
    chunks = _make_chunks(version_id, count=3)
    chunk_repo.insert_chunks(db_conn, chunks)

    # Build a mapping from chunk_id -> faiss_id
    mapping = {chunks[0].id: 100, chunks[1].id: 101, chunks[2].id: 102}
    chunk_repo.update_faiss_ids(db_conn, mapping)

    rows = chunk_repo.get_chunks_by_version(db_conn, version_id)
    id_to_faiss = {row["id"]: row["faiss_index_id"] for row in rows}

    assert id_to_faiss[chunks[0].id] == 100
    assert id_to_faiss[chunks[1].id] == 101
    assert id_to_faiss[chunks[2].id] == 102


def test_get_chunk_by_faiss_id(db_conn):
    """After setting a faiss_id, get_chunk_by_faiss_id retrieves the correct chunk."""
    version_id = _make_version(db_conn)
    chunks = _make_chunks(version_id, count=2)
    chunk_repo.insert_chunks(db_conn, chunks)

    chunk_repo.update_faiss_ids(db_conn, {chunks[1].id: 999})

    row = chunk_repo.get_chunk_by_faiss_id(db_conn, version_id, 999)

    assert row is not None
    assert row["id"] == chunks[1].id
    assert row["chunk_no"] == 1

    # Non-existent faiss id should return None
    missing = chunk_repo.get_chunk_by_faiss_id(db_conn, version_id, 42)
    assert missing is None
