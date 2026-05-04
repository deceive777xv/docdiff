"""Azure OpenAI provider — Phase 2 implementation."""
from __future__ import annotations
from app.core.model.base_provider import BaseProvider


class AzureOpenAIProvider(BaseProvider):
    """Not yet implemented — placeholder for Phase 2."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Azure provider will be implemented in Phase 2.")

    def chat(self, messages: list[dict], **kwargs) -> str:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def health_check(self) -> bool:
        raise NotImplementedError
