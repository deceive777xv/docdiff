"""BM25 lexical search over a list of Chunk objects."""
from __future__ import annotations

from collections import OrderedDict

from rank_bm25 import BM25Okapi

from app.core.types import Chunk

_CACHE_MAX_SIZE = 8
_CACHE_MAX_CHUNKS = 2000
_BM25_CACHE: OrderedDict[tuple[tuple[str, str], ...], BM25Okapi] = OrderedDict()


def _cache_key(chunks: list[Chunk]) -> tuple[tuple[str, str], ...]:
    return tuple((c.id, c.text) for c in chunks)


def _get_bm25(chunks: list[Chunk]) -> BM25Okapi:
    if len(chunks) > _CACHE_MAX_CHUNKS:
        tokenized_corpus = [list(c.text.replace(" ", "")) for c in chunks]
        return BM25Okapi(tokenized_corpus)

    key = _cache_key(chunks)
    cached = _BM25_CACHE.get(key)
    if cached is not None:
        _BM25_CACHE.move_to_end(key)
        return cached

    tokenized_corpus = [list(c.text.replace(" ", "")) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    _BM25_CACHE[key] = bm25
    if len(_BM25_CACHE) > _CACHE_MAX_SIZE:
        _BM25_CACHE.popitem(last=False)
    return bm25


def bm25_search(chunks: list[Chunk], query: str, top_k: int) -> list[tuple[int, float]]:
    """Return (chunk_index, bm25_score) pairs sorted by score descending.

    chunk_index is the position in the input list, not the chunk's own id.
    Character-level tokenization — effective for Chinese text.
    """
    if not chunks:
        return []

    tokenized_query = list(query.replace(" ", ""))

    bm25 = _get_bm25(chunks)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in ranked[:top_k]]
