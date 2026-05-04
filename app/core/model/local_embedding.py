"""Local embedding provider using sentence-transformers."""
from __future__ import annotations
import logging
from pathlib import Path

from app.core.model.base_provider import BaseProvider

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment,misc]


class LocalEmbeddingProvider(BaseProvider):
    """Loads a sentence-transformers model from a local path for embeddings only."""

    def __init__(self, model_path: str) -> None:
        if SentenceTransformer is None:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        self._model_path = model_path
        self._model = SentenceTransformer(model_path)
        logger.info("Loaded local embedding model from %s", model_path)

    def chat(self, messages: list[dict], **kwargs) -> str:
        raise NotImplementedError("LocalEmbeddingProvider does not support chat.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]

    def health_check(self) -> bool:
        try:
            self._model.encode(["ping"])
            return True
        except Exception as e:
            logger.warning("Local embedding health check failed: %s", e)
            return False
