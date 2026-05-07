"""Tests for BM25 lexical search."""
from __future__ import annotations

import pytest

from app.core.types import Chunk


def _make_chunk(idx: int, text: str) -> Chunk:
    return Chunk(id=f"c{idx}", version_id="v", chunk_no=idx,
                 section_path="", page_no=idx + 1, text=text)


def test_bm25_search_returns_relevant_chunk():
    from app.core.retrieval.bm25_searcher import bm25_search

    chunks = [
        _make_chunk(0, "付款周期为三十天"),
        _make_chunk(1, "违约金按日万分之五计算"),
        _make_chunk(2, "交货期为六十个工作日"),
    ]
    results = bm25_search(chunks, "付款", top_k=2)

    assert len(results) == 2
    top_idx, top_score = results[0]
    assert top_idx == 0  # "付款周期" should rank highest for "付款" query
    assert top_score > 0


def test_bm25_search_empty_chunks_returns_empty():
    from app.core.retrieval.bm25_searcher import bm25_search

    results = bm25_search([], "付款", top_k=5)
    assert results == []


def test_bm25_search_top_k_limits_results():
    from app.core.retrieval.bm25_searcher import bm25_search

    chunks = [_make_chunk(i, f"文本内容{i}付款") for i in range(10)]
    results = bm25_search(chunks, "付款", top_k=3)
    assert len(results) <= 3


def test_bm25_search_returns_chunk_index_not_id():
    """Return values are (chunk_index_in_list, score), not chunk ids."""
    from app.core.retrieval.bm25_searcher import bm25_search

    # Three docs so IDF is non-zero (BM25 IDF = 0 when df == N/2)
    chunks = [
        _make_chunk(0, "完全无关的内容"),
        _make_chunk(1, "付款方式和条款"),
        _make_chunk(2, "交货期六十天合同"),
    ]
    results = bm25_search(chunks, "付款", top_k=3)
    # index 1 should score higher
    assert results[0][0] == 1
