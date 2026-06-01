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
        embed_api_key: str = "",
        embed_base_url: str = "",
        timeout: int = 60,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )
        self._uses_dedicated_embed_client = bool(embed_api_key or embed_base_url)
        if self._uses_dedicated_embed_client:
            self._embed_client = OpenAI(
                api_key=embed_api_key or api_key,
                base_url=embed_base_url or base_url or None,
                timeout=timeout,
            )
        else:
            self._embed_client = self._client
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.api_key = api_key
        self.base_url = base_url
        self.embed_api_key = embed_api_key
        self.embed_base_url = embed_base_url

    def chat(self, messages: list[dict], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        embed_client = self._embed_client if self._uses_dedicated_embed_client else self._client
        response = embed_client.embeddings.create(
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
