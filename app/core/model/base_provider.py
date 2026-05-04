"""Abstract base class for all model providers."""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseProvider(ABC):

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send messages to chat model, return text response."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, return list of float vectors."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the provider is reachable and configured."""
