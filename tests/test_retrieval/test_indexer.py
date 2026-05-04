"""Tests for app/core/retrieval/indexer.py"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.retrieval.indexer import build_index
from app.core.types import Chunk
from app.db import chunk_repo, faiss_store
from app.db.document_repo import insert_document, insert_version
from app.db.schema import init_db


@pytest.fixture
def setup(tmp_path):
    conn = init_db(str(tmp_path))
    doc_id = insert_document(
        conn,
        doc_name="test",
        doc_type="pdf",
        file_path="x.pdf",
        file_hash="abc123",
        source_type="standard",
    )
    version_id = insert_version(conn, document_id=doc_id, version_no=1)
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            version_id=version_id,
            chunk_no=i,
            section_path="sec1",
            page_no=1,
            text=f"text {i}",
        )
        for i in range(5)
    ]
    insert_chunks_helper = chunk_repo.insert_chunks
    insert_chunks_helper(conn, chunks)
    return conn, str(tmp_path), version_id, chunks


def _make_embedder(n: int, dim: int = 8) -> MagicMock:
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [np.random.rand(dim).tolist() for _ in range(n)]
    return mock_embedder


def test_build_index_creates_faiss_file(setup):
    """build_index should create a FAISS index file on disk."""
    conn, data_dir, version_id, chunks = setup
    embedder = _make_embedder(len(chunks))

    build_index(data_dir, conn, version_id, chunks, embedder)

    assert faiss_store.index_exists(data_dir, version_id)


def test_build_index_updates_faiss_ids(setup):
    """After build_index, all chunks in the DB should have faiss_index_id >= 0."""
    conn, data_dir, version_id, chunks = setup
    embedder = _make_embedder(len(chunks))

    build_index(data_dir, conn, version_id, chunks, embedder)

    rows = chunk_repo.get_chunks_by_version(conn, version_id)
    assert len(rows) == len(chunks)
    for row in rows:
        assert row["faiss_index_id"] >= 0


def test_build_index_empty_chunks_no_error(setup):
    """build_index with an empty list should not raise any exception."""
    conn, data_dir, version_id, _ = setup
    embedder = _make_embedder(0)

    # Should complete silently without raising
    build_index(data_dir, conn, version_id, [], embedder)
