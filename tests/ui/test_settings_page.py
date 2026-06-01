from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from app.config.settings import AppSettings, ProviderConfig
from app.ui.app_context import AppContext


@pytest.fixture()
def ctx(tmp_path):
    conn = sqlite3.connect(":memory:")
    settings = AppSettings(
        providers=[
            ProviderConfig(
                name="default",
                api_key="chat-key",
                base_url="https://chat.example.com/v1",
                chat_model="chat-model",
                embed_api_key="embed-key",
                embed_base_url="https://embed.example.com/v1",
                embed_model="embed-model",
            )
        ],
        active_provider="default",
        data_dir=str(tmp_path),
    )
    yield AppContext(settings=settings, conn=conn, data_dir=str(tmp_path))
    conn.close()


def test_settings_dialog_loads_and_saves_embedding_api_fields(qtbot, ctx, monkeypatch):
    """Settings should allow embedding API URL/key to differ from chat API config."""
    from app.ui.pages.settings_page import SettingsDialog

    saved_settings = []
    monkeypatch.setattr(
        "app.ui.pages.settings_page.settings_mod.save",
        lambda settings: saved_settings.append(settings),
    )

    dialog = SettingsDialog(ctx)
    qtbot.addWidget(dialog)

    assert dialog._embed_base_url.text() == "https://embed.example.com/v1"
    assert dialog._embed_api_key.text() == "embed-key"

    dialog._embed_base_url.setText("https://other-embed.example.com/v1")
    dialog._embed_api_key.setText("other-embed-key")

    with patch("app.core.model.factory.build_provider", return_value=MagicMock()):
        with patch("app.core.model.factory.get_embedder", return_value=MagicMock()):
            with patch("openai.OpenAI", return_value=MagicMock()):
                dialog._save()

    assert saved_settings
    provider = ctx.settings.providers[0]
    assert provider.embed_base_url == "https://other-embed.example.com/v1"
    assert provider.embed_api_key == "other-embed-key"
