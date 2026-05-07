"""BM25 lexical search over a list of Chunk objects."""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from app.core.types import Chunk


def bm25_search(chunks: list[Chunk], query: str, top_k: int) -> list[tuple[int, float]]:
    """Return (chunk_index, bm25_score) pairs sorted by score descending.

    chunk_index is the position in the input list, not the chunk's own id.
    Character-level tokenization — effective for Chinese text.
    """
    if not chunks:
        return []

    tokenized_corpus = [list(c.text.replace(" ", "")) for c in chunks]
    tokenized_query = list(query.replace(" ", ""))

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in ranked[:top_k]]
