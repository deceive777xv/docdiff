"""OpenAI-compatible provider (works with DeepSeek, Moonshot, Qwen, etc.)."""
from __future__ import annotations
import logging

from openai import OpenAI, APIError

from app.core.model.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseProvider):

    def __init__(
        self,
        api_key: str,
        base_url: str,
        chat_model: str,
        embed_model: str,
        timeout: int = 60,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )
        self.chat_model = chat_model
        self.embed_model = embed_model

    def chat(self, messages: list[dict], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def health_check(self) -> bool:
        try:
            self._client.models.list()
            return True
        except APIError as e:
            logger.warning("Provider health check failed: %s", e)
            return False
