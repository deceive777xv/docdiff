"""Tests for app/db/faiss_store.py — FAISS flat L2 index wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from app.db.faiss_store import build_and_save, index_exists, load_index, search

DIM = 8
N = 10
VERSION_ID = "v_test_001"

np.random.seed(42)
_EMBEDDINGS = np.random.rand(N, DIM).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. build_and_save creates the index file on disk
# ---------------------------------------------------------------------------

def test_build_and_save_creates_file(tmp_path):
    """build_and_save must write faiss/{version_id}/index.faiss."""
    build_and_save(str(tmp_path), VERSION_ID, _EMBEDDINGS.copy())
    expected = tmp_path / "faiss" / VERSION_ID / "index.faiss"
    assert expected.exists(), f"Expected file not found: {expected}"


# ---------------------------------------------------------------------------
# 2. search returns exactly top_k results
# ---------------------------------------------------------------------------

def test_search_returns_correct_count(tmp_path):
    """search with top_k=3 must return exactly 3 (idx, dist) pairs."""
    build_and_save(str(tmp_path), VERSION_ID, _EMBEDDINGS.copy())
    query = _EMBEDDINGS[0:1].copy()
    results = search(str(tmp_path), VERSION_ID, query, top_k=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# 3. nearest neighbour of a stored vector is itself (distance ≈ 0)
# ---------------------------------------------------------------------------

def test_search_nearest_is_self(tmp_path):
    """Searching with row-0 vector should return index 0 with distance ~0."""
    build_and_save(str(tmp_path), VERSION_ID, _EMBEDDINGS.copy())
    query = _EMBEDDINGS[0].copy()   # shape (dim,) — test 1-D input path too
    results = search(str(tmp_path), VERSION_ID, query, top_k=1)
    assert len(results) == 1
    nearest_idx, nearest_dist = results[0]
    assert nearest_idx == 0
    assert nearest_dist == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. index_exists reflects actual disk state
# ---------------------------------------------------------------------------

def test_index_exists(tmp_path):
    """index_exists is False before build and True after."""
    assert not index_exists(str(tmp_path), VERSION_ID)
    build_and_save(str(tmp_path), VERSION_ID, _EMBEDDINGS.copy())
    assert index_exists(str(tmp_path), VERSION_ID)


# ---------------------------------------------------------------------------
# 5. load_index + search after reload produces identical results
# ---------------------------------------------------------------------------

def test_load_and_search_after_reload(tmp_path):
    """Results from a freshly loaded index must match the original."""
    build_and_save(str(tmp_path), VERSION_ID, _EMBEDDINGS.copy())
    query = _EMBEDDINGS[3:4].copy()

    results_direct = search(str(tmp_path), VERSION_ID, query, top_k=5)

    # Reload index manually and search
    reloaded = load_index(str(tmp_path), VERSION_ID)
    q = np.atleast_2d(query).astype(np.float32)
    distances, indices = reloaded.search(q, 5)
    results_reloaded = [
        (int(idx), float(dist))
        for dist, idx in zip(distances[0], indices[0])
        if idx >= 0
    ]

    assert results_direct == results_reloaded
