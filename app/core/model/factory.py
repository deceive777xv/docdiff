"""Build a provider from AppSettings."""
from __future__ import annotations
from pathlib import Path

from app.config.settings import AppSettings, ProviderConfig, get_active_provider
from app.core.model.base_provider import BaseProvider
from app.core.model.openai_compatible import OpenAICompatibleProvider


def build_provider(config: ProviderConfig) -> BaseProvider:
    if config.type in ("openai_compatible",):
        return OpenAICompatibleProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            chat_model=config.chat_model,
            embed_model=config.embed_model,
        )
    if config.type == "azure":
        raise NotImplementedError(
            "Azure OpenAI provider is planned for Phase 2. "
            "Please use an OpenAI-compatible provider for now."
        )
    raise ValueError(f"Unknown provider type: {config.type!r}")


def get_provider(settings: AppSettings) -> BaseProvider:
    """Get the active chat+embed provider from settings."""
    config = get_active_provider(settings)
    if config is None:
        raise RuntimeError("No provider configured. Please add one in Settings.")
    return build_provider(config)


def get_embedder(settings: AppSettings) -> BaseProvider:
    """
    Return the appropriate embedder:
    - Local sentence-transformers model if enabled and path exists
    - Otherwise, the active API provider
    """
    le = settings.local_embedding
    if le.enabled and le.model_path and Path(le.model_path).exists():
        from app.core.model.local_embedding import LocalEmbeddingProvider
        return LocalEmbeddingProvider(le.model_path)
    return get_provider(settings)
