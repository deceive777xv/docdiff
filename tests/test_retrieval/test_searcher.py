"""Tests for app/core/retrieval/searcher.py"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.retrieval.indexer import build_index
from app.core.retrieval.searcher import search
from app.core.types import Chunk, ChunkHit
from app.db import chunk_repo
from app.db.document_repo import insert_document, insert_version
from app.db.schema import init_db


_DIM = 8


def _make_chunks(version_id: str, count: int) -> list[Chunk]:
    return [
        Chunk(
            id=str(uuid.uuid4()),
            version_id=version_id,
            chunk_no=i,
            section_path="sec1",
            page_no=1,
            text=f"text {i}",
        )
        for i in range(count)
    ]


def _make_embedder(n: int, dim: int = _DIM) -> MagicMock:
    """Return an embedder mock that always produces consistent n-vector responses."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [np.random.rand(dim).tolist() for _ in range(n)]
    return mock_embedder


@pytest.fixture
def indexed_setup(tmp_path):
    """Build DB + FAISS index with 10 chunks; return (conn, data_dir, version_id)."""
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
    chunks = _make_chunks(version_id, count=10)
    chunk_repo.insert_chunks(conn, chunks)

    embedder = _make_embedder(10)
    build_index(str(tmp_path), conn, version_id, chunks, embedder)

    return conn, str(tmp_path), version_id


def test_search_returns_top_k(indexed_setup):
    """search returns at most top_k results."""
    conn, data_dir, version_id = indexed_setup
    top_k = 3

    # Embedder for the query (single vector)
    query_embedder = MagicMock()
    query_embedder.embed.return_value = [np.random.rand(_DIM).tolist()]

    results = search(data_dir, conn, "query text", query_embedder, [version_id], top_k=top_k)

    assert len(results) <= top_k


def test_search_returns_chunk_hits(indexed_setup):
    """Each search result is a ChunkHit with a Chunk and a float score."""
    conn, data_dir, version_id = indexed_setup

    query_embedder = MagicMock()
    query_embedder.embed.return_value = [np.random.rand(_DIM).tolist()]

    results = search(data_dir, conn, "query text", query_embedder, [version_id], top_k=5)

    assert len(results) > 0
    for hit in results:
        assert isinstance(hit, ChunkHit)
        assert isinstance(hit.chunk, Chunk)
        assert isinstance(hit.score, float)


def test_search_no_index_returns_empty(tmp_path):
    """Searching a version that has no FAISS index returns an empty list."""
    conn = init_db(str(tmp_path))
    doc_id = insert_document(
        conn,
        doc_name="test",
        doc_type="pdf",
        file_path="x.pdf",
        file_hash="def456",
        source_type="standard",
    )
    version_id = insert_version(conn, document_id=doc_id, version_no=1)

    query_embedder = MagicMock()
    query_embedder.embed.return_value = [np.random.rand(_DIM).tolist()]

    results = search(str(tmp_path), conn, "query", query_embedder, [version_id], top_k=5)

    assert results == []
