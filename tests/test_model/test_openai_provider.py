"""Tests for OpenAI-compatible provider and factory."""
from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch

from openai import APIError

from app.config.settings import ProviderConfig
from app.core.model.factory import build_provider
from app.core.model.openai_compatible import OpenAICompatibleProvider


def _make_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://example.com/v1",
        chat_model="test-model",
        embed_model="test-embed",
    )


class TestChatReturnsString(unittest.TestCase):
    def test_chat_returns_string(self):
        provider = _make_provider()
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "hello"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        result = provider.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "hello")


class TestEmbedReturnsListOfVectors(unittest.TestCase):
    def test_embed_returns_list_of_vectors(self):
        provider = _make_provider()
        mock_client = MagicMock()
        item_a = MagicMock()
        item_a.embedding = [0.1, 0.2, 0.3]
        item_b = MagicMock()
        item_b.embedding = [0.4, 0.5, 0.6]
        mock_response = MagicMock()
        mock_response.data = [item_a, item_b]
        mock_client.embeddings.create.return_value = mock_response
        provider._client = mock_client

        result = provider.embed(["a", "b"])
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[1], list)

    def test_embed_uses_dedicated_embedding_client(self):
        provider = OpenAICompatibleProvider(
            api_key="chat-key",
            base_url="https://chat.example.com/v1",
            chat_model="chat-model",
            embed_model="embed-model",
            embed_api_key="embed-key",
            embed_base_url="https://embed.example.com/v1",
        )
        chat_client = MagicMock()
        embed_client = MagicMock()
        item = MagicMock()
        item.embedding = [0.1, 0.2]
        response = MagicMock()
        response.data = [item]
        embed_client.embeddings.create.return_value = response
        provider._client = chat_client
        provider._embed_client = embed_client

        result = provider.embed(["hello"])

        self.assertEqual(result, [[0.1, 0.2]])
        embed_client.embeddings.create.assert_called_once_with(
            model="embed-model",
            input=["hello"],
        )
        chat_client.embeddings.create.assert_not_called()


class TestHealthCheckTrue(unittest.TestCase):
    def test_health_check_true(self):
        provider = _make_provider()
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock()
        provider._client = mock_client

        self.assertTrue(provider.health_check())


class TestHealthCheckFalse(unittest.TestCase):
    def test_health_check_false(self):
        provider = _make_provider()
        mock_client = MagicMock()
        mock_client.models.list.side_effect = APIError(
            message="connection error",
            request=MagicMock(),
            body=None,
        )
        provider._client = mock_client

        self.assertFalse(provider.health_check())


class TestBuildProviderOpenAICompatible(unittest.TestCase):
    def test_build_provider_openai_compatible(self):
        config = ProviderConfig(
            name="x",
            type="openai_compatible",
            api_key="k",
            base_url="",
            chat_model="m",
            embed_api_key="embed-k",
            embed_base_url="https://embed.example.com/v1",
            embed_model="e",
        )
        result = build_provider(config)
        self.assertIsInstance(result, OpenAICompatibleProvider)
        self.assertEqual(result.embed_api_key, "embed-k")
        self.assertEqual(result.embed_base_url, "https://embed.example.com/v1")


if __name__ == "__main__":
    unittest.main()
