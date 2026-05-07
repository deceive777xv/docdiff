"""Tests for LangChain model factory."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config.settings import AppSettings, ProviderConfig


def test_get_chat_model_returns_none_when_no_providers():
    from app.core.model.lc_factory import get_chat_model

    settings = AppSettings(providers=[], active_provider="")
    assert get_chat_model(settings) is None


def test_get_chat_model_returns_none_when_providers_empty_list():
    from app.core.model.lc_factory import get_chat_model

    settings = AppSettings(providers=[], active_provider="default")
    assert get_chat_model(settings) is None


def test_get_chat_model_returns_model_when_configured():
    from app.core.model.lc_factory import get_chat_model

    settings = AppSettings(
        providers=[
            ProviderConfig(
                name="default",
                api_key="sk-test-key",
                base_url="https://api.example.com/v1",
                chat_model="gpt-4o",
            )
        ],
        active_provider="default",
    )
    model = get_chat_model(settings)
    assert model is not None


def test_get_chat_model_uses_active_provider_chat_model():
    from app.core.model.lc_factory import get_chat_model

    settings = AppSettings(
        providers=[
            ProviderConfig(
                name="default",
                api_key="sk-key",
                base_url="https://api.example.com/v1",
                chat_model="deepseek-chat",
            )
        ],
        active_provider="default",
    )
    model = get_chat_model(settings)
    assert model is not None
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
    assert model_name == "deepseek-chat"
