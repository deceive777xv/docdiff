"""FAISS vector store — one flat L2 index per document version."""
from __future__ import annotations
from pathlib import Path
import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss-cpu is required: pip install faiss-cpu") from e


def _index_dir(data_dir: str, version_id: str) -> Path:
    return Path(data_dir) / "faiss" / version_id


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

    # Return {row_position: faiss_id} — for flat index these are identical
    return {i: i for i in range(len(embeddings))}


def load_index(data_dir: str, version_id: str) -> faiss.Index:
    """Load a previously saved FAISS index for a version."""
    idx_path = _index_dir(data_dir, version_id) / "index.faiss"
    if not idx_path.exists():
        raise FileNotFoundError(f"No FAISS index for version {version_id}")
    return faiss.read_index(str(idx_path))


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
    return (_index_dir(data_dir, version_id) / "index.faiss").exists()
