"""Search FAISS indexes and return ranked ChunkHits."""
from __future__ import annotations
import logging
import sqlite3
from typing import Sequence

import numpy as np

from app.core.model.base_provider import BaseProvider
from app.core.types import Chunk, ChunkHit, RetrievalScope
from app.db import chunk_repo, faiss_store

logger = logging.getLogger(__name__)


def _row_to_chunk(row) -> Chunk:
    return Chunk(
        id=row["id"],
        version_id=row["version_id"],
        chunk_no=row["chunk_no"],
        section_path=row["section_path"] or "",
        page_no=row["page_no"] or 0,
        text=row["text"],
        faiss_index_id=row["faiss_index_id"],
    )


def _search_version(
    data_dir: str,
    conn: sqlite3.Connection,
    version_id: str,
    query_vec: np.ndarray,
    top_k: int,
) -> list[ChunkHit]:
    """Search one version index, return ChunkHits."""
    if not faiss_store.index_exists(data_dir, version_id):
        logger.warning("No FAISS index for version %s — skipping", version_id)
        return []

    hits = faiss_store.search(data_dir, version_id, query_vec, top_k)
    results: list[ChunkHit] = []
    for faiss_id, distance in hits:
        row = chunk_repo.get_chunk_by_faiss_id(conn, version_id, faiss_id)
        if row:
            results.append(ChunkHit(chunk=_row_to_chunk(row), score=float(distance)))
    return results


def search(
    data_dir: str,
    conn: sqlite3.Connection,
    query: str,
    embedder: BaseProvider,
    version_ids: list[str],
    top_k: int = 5,
) -> list[ChunkHit]:
    """
    Embed query and search across multiple version indexes.
    Returns top_k hits merged and sorted by ascending distance (lower = better).
    """
    query_embedding = embedder.embed([query])[0]
    query_vec = np.array(query_embedding, dtype=np.float32)

    all_hits: list[ChunkHit] = []
    for vid in version_ids:
        all_hits.extend(_search_version(data_dir, conn, vid, query_vec, top_k))

    all_hits.sort(key=lambda h: h.score)
    return all_hits[:top_k]
