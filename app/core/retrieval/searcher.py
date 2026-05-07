"""Hybrid BM25+FAISS retrieval with Reciprocal Rank Fusion."""
from __future__ import annotations

import logging
import sqlite3

import numpy as np

from app.core.model.base_provider import BaseProvider
from app.core.retrieval.bm25_searcher import bm25_search
from app.core.types import Chunk, ChunkHit
from app.db import chunk_repo, faiss_store

logger = logging.getLogger(__name__)

_RRF_K = 60


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


def search(
    data_dir: str,
    conn: sqlite3.Connection,
    query: str,
    embedder: BaseProvider,
    version_ids: list[str],
    top_k: int = 5,
) -> list[ChunkHit]:
    """Hybrid BM25+FAISS search with RRF merge.

    Returns top_k hits sorted by RRF score descending (higher = better).
    """
    if not version_ids:
        return []

    query_embedding = embedder.embed([query])[0]
    query_vec = np.array(query_embedding, dtype=np.float32)

    # chunk_id → {"faiss": rank, "bm25": rank}
    ranks: dict[str, dict[str, int]] = {}
    chunk_map: dict[str, Chunk] = {}

    for vid in version_ids:
        all_rows = chunk_repo.get_chunks_by_version(conn, vid)
        if not all_rows:
            continue
        all_chunks = [_row_to_chunk(r) for r in all_rows]
        for c in all_chunks:
            chunk_map[c.id] = c

        # FAISS branch
        if faiss_store.index_exists(data_dir, vid):
            faiss_hits = faiss_store.search(data_dir, vid, query_vec, top_k)
            for rank, (faiss_id, _dist) in enumerate(faiss_hits):
                row = chunk_repo.get_chunk_by_faiss_id(conn, vid, faiss_id)
                if row:
                    cid = row["id"]
                    ranks.setdefault(cid, {})["faiss"] = rank

        # BM25 branch
        bm25_hits = bm25_search(all_chunks, query, top_k)
        for rank, (chunk_idx, _score) in enumerate(bm25_hits):
            cid = all_chunks[chunk_idx].id
            ranks.setdefault(cid, {})["bm25"] = rank

    def _rrf(rank_dict: dict[str, int]) -> float:
        score = 0.0
        if "faiss" in rank_dict:
            score += 1.0 / (_RRF_K + rank_dict["faiss"])
        if "bm25" in rank_dict:
            score += 1.0 / (_RRF_K + rank_dict["bm25"])
        return score

    scored = [(cid, _rrf(r)) for cid, r in ranks.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [
        ChunkHit(chunk=chunk_map[cid], score=score)
        for cid, score in scored[:top_k]
        if cid in chunk_map
    ]
