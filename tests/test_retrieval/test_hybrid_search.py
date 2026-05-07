"""Tests for hybrid BM25+FAISS retrieval with RRF merging."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from app.core.types import Chunk


def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE chunks (
            id TEXT, version_id TEXT, chunk_no INTEGER,
            section_path TEXT, page_no INTEGER, text TEXT, faiss_index_id INTEGER
        )
    """)
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", [
        ("c1", "v1", 0, "", 1, "付款周期三十天", 0),
        ("c2", "v1", 1, "", 2, "违约金计算方式", 1),
        ("c3", "v1", 2, "", 3, "交货期六十天", 2),
    ])
    conn.commit()
    return conn


def test_rrf_doc_in_both_ranks_higher_than_faiss_only():
    """A doc in both FAISS and BM25 must rank above a doc in only FAISS."""
    from app.core.retrieval.searcher import search

    conn = _make_conn()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 4]

    # FAISS returns c1 (faiss_id=0, rank 0) and c2 (faiss_id=1, rank 1)
    # BM25 query "付款" → c1 ranks highest (text "付款周期三十天"), c3 ranks next
    # c1 appears in BOTH → highest RRF; c2 only in FAISS; c3 only in BM25
    with patch("app.core.retrieval.searcher.faiss_store.index_exists", return_value=True), \
         patch("app.core.retrieval.searcher.faiss_store.search",
               return_value=[(0, 0.1), (1, 0.2)]):
        hits = search("/tmp", conn, "付款", mock_embedder, ["v1"], top_k=3)

    assert len(hits) > 0
    # c1 must be the top result (in both lists)
    assert hits[0].chunk.id == "c1"
    conn.close()


def test_rrf_score_is_higher_better():
    """RRF scores should be in descending order (higher = better)."""
    from app.core.retrieval.searcher import search

    conn = _make_conn()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 4]

    with patch("app.core.retrieval.searcher.faiss_store.index_exists", return_value=True), \
         patch("app.core.retrieval.searcher.faiss_store.search",
               return_value=[(0, 0.1), (1, 0.2)]):
        hits = search("/tmp", conn, "付款", mock_embedder, ["v1"], top_k=3)

    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    conn.close()


def test_no_faiss_index_uses_bm25_only():
    """When FAISS index is absent, results still come from BM25 alone."""
    from app.core.retrieval.searcher import search

    conn = _make_conn()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 4]

    with patch("app.core.retrieval.searcher.faiss_store.index_exists", return_value=False):
        hits = search("/tmp", conn, "付款", mock_embedder, ["v1"], top_k=3)

    assert len(hits) > 0
    assert hits[0].chunk.id == "c1"  # BM25 ranks "付款周期三十天" first
    conn.close()


def test_empty_version_ids_returns_empty():
    from app.core.retrieval.searcher import search

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 4]
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    hits = search("/tmp", conn, "付款", mock_embedder, [], top_k=5)
    assert hits == []
    conn.close()
