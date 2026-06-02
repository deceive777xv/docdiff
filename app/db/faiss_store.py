"""FAISS vector store — one flat L2 index per document version."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import threading
import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss-cpu is required: pip install faiss-cpu") from e

_INDEX_CACHE_MAX_SIZE = 2
_INDEX_CACHE: OrderedDict[tuple[str, str], tuple[int, faiss.Index]] = OrderedDict()
_INDEX_CACHE_LOCK = threading.Lock()


def _index_dir(data_dir: str, version_id: str) -> Path:
    return Path(data_dir) / "faiss" / version_id


def _index_path(data_dir: str, version_id: str) -> Path:
    return _index_dir(data_dir, version_id) / "index.faiss"


def clear_index_cache() -> None:
    """Clear cached FAISS indexes, mainly for tests and low-memory recovery."""
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.clear()


def _cache_key(data_dir: str, version_id: str) -> tuple[str, str]:
    return (str(Path(data_dir).resolve()), version_id)


def _drop_cached_index(data_dir: str, version_id: str) -> None:
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.pop(_cache_key(data_dir, version_id), None)


def build_and_save(
    data_dir: str,
    version_id: str,
    embeddings: np.ndarray,          # shape (n, dim), float32
) -> dict[int, int]:
    """
    Build a flat L2 FAISS index from embeddings, save to disk.
    Returns {faiss_internal_id: faiss_internal_id} — i.e. 0..n-1 mapping
    since flat index IDs equal row index.
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    idx_dir = _index_dir(data_dir, version_id)
    idx_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_dir / "index.faiss"))
    _drop_cached_index(data_dir, version_id)

    # Return {row_position: faiss_id} — for flat index these are identical
    return {i: i for i in range(len(embeddings))}


def load_index(data_dir: str, version_id: str) -> faiss.Index:
    """Load a previously saved FAISS index for a version."""
    idx_path = _index_path(data_dir, version_id)
    if not idx_path.exists():
        raise FileNotFoundError(f"No FAISS index for version {version_id}")

    key = _cache_key(data_dir, version_id)
    mtime_ns = idx_path.stat().st_mtime_ns
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is not None and cached[0] == mtime_ns:
            _INDEX_CACHE.move_to_end(key)
            return cached[1]

    index = faiss.read_index(str(idx_path))
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE[key] = (mtime_ns, index)
        if len(_INDEX_CACHE) > _INDEX_CACHE_MAX_SIZE:
            _INDEX_CACHE.popitem(last=False)
    return index


def search(
    data_dir: str,
    version_id: str,
    query_embedding: np.ndarray,     # shape (1, dim) or (dim,), float32
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """
    Search the index for a version.
    Returns list of (faiss_index_id, distance) sorted by ascending distance.
    """
    index = load_index(data_dir, version_id)
    q = np.atleast_2d(query_embedding).astype(np.float32)
    distances, indices = index.search(q, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= 0:   # FAISS returns -1 for unfilled slots
            results.append((int(idx), float(dist)))
    return results


def index_exists(data_dir: str, version_id: str) -> bool:
    return _index_path(data_dir, version_id).exists()
