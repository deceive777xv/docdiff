"""LangChain ChatOpenAI factory for streaming QA generation."""
from __future__ import annotations

from app.config.settings import AppSettings, get_active_provider


def get_chat_model(settings: AppSettings):
    """Create a LangChain ChatOpenAI from active provider config, or None.

    Returns None when no provider is configured. Called at startup and on
    provider_changed signal.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    config = get_active_provider(settings)
    if config is None:
        return None

    return ChatOpenAI(
        model=config.chat_model,
        api_key=config.api_key,
        base_url=config.base_url or None,
        streaming=True,
    )
