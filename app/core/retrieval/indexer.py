"""Build and persist FAISS indexes for document version chunks."""
from __future__ import annotations
import logging
import sqlite3

import numpy as np

from app.core.model.base_provider import BaseProvider
from app.core.types import Chunk
from app.db import chunk_repo, faiss_store

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64


def build_index(
    data_dir: str,
    conn: sqlite3.Connection,
    version_id: str,
    chunks: list[Chunk],
    embedder: BaseProvider,
) -> None:
    """
    Embed all chunks and build a FAISS index for this version.
    Updates faiss_index_id in the chunks table.
    """
    if not chunks:
        logger.warning("No chunks to index for version %s", version_id)
        return

    texts = [c.text for c in chunks]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        batch_embeddings = embedder.embed(batch)
        all_embeddings.extend(batch_embeddings)
        logger.debug("Embedded %d/%d chunks", min(i + _BATCH_SIZE, len(texts)), len(texts))

    embeddings_np = np.array(all_embeddings, dtype=np.float32)
    row_to_faiss = faiss_store.build_and_save(data_dir, version_id, embeddings_np)

    # Map chunk id → faiss index id
    chunk_id_to_faiss: dict[str, int] = {
        chunks[row].id: faiss_id
        for row, faiss_id in row_to_faiss.items()
    }
    chunk_repo.update_faiss_ids(conn, chunk_id_to_faiss)
    logger.info("Built FAISS index for version %s (%d vectors)", version_id, len(chunks))
