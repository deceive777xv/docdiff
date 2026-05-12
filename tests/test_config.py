"""Tests for app/config/crypto.py and app/config/settings.py."""
from __future__ import annotations

import pytest

from app.config.crypto import decrypt, encrypt
from app.config.settings import (
    AppSettings,
    LocalEmbeddingConfig,
    ProviderConfig,
    get_active_provider,
    load,
    save,
)


# ── crypto tests ──────────────────────────────────────────────────────────────

def test_encrypt_differs_from_plaintext():
    """encrypt(plaintext) produces a ciphertext that differs from the input."""
    plaintext = "sk-test-api-key-12345"
    ciphertext = encrypt(plaintext)
    assert ciphertext != plaintext


def test_encrypt_decrypt_round_trip():
    """decrypt(encrypt(x)) == x for arbitrary plaintext."""
    plaintext = "super-secret-key-abc"
    assert decrypt(encrypt(plaintext)) == plaintext


# ── settings tests ────────────────────────────────────────────────────────────

def test_load_defaults_when_no_file(tmp_path, monkeypatch):
    """load() returns AppSettings with empty providers when config.json is absent."""
    monkeypatch.setattr(
        "app.config.settings._config_path",
        lambda: tmp_path / "config.json",
    )
    settings = load()
    assert isinstance(settings, AppSettings)
    assert settings.providers == []


def test_save_load_round_trip(tmp_path, monkeypatch):
    """save() + load() preserves provider data including the API key."""
    monkeypatch.setattr(
        "app.config.settings._config_path",
        lambda: tmp_path / "config.json",
    )
    provider = ProviderConfig(
        name="deepseek",
        type="openai_compatible",
        api_key="sk-round-trip-key-xyz",
        base_url="https://api.deepseek.com",
        chat_model="deepseek-chat",
        embed_model="text-embedding-ada-002",
    )
    original = AppSettings(
        providers=[provider],
        local_embedding=LocalEmbeddingConfig(enabled=False, model_path=""),
        active_provider="deepseek",
        data_dir=str(tmp_path),
    )
    save(original)
    loaded = load()

    assert len(loaded.providers) == 1
    p = loaded.providers[0]
    assert p.name == "deepseek"
    assert p.api_key == "sk-round-trip-key-xyz"
    assert p.base_url == "https://api.deepseek.com"
    assert loaded.active_provider == "deepseek"


def test_get_active_provider_by_name():
    """get_active_provider returns the provider matching active_provider name."""
    p1 = ProviderConfig(name="alpha", api_key="k1")
    p2 = ProviderConfig(name="beta", api_key="k2")
    settings = AppSettings(providers=[p1, p2], active_provider="beta")
    result = get_active_provider(settings)
    assert result is p2


def test_get_active_provider_fallback_to_first():
    """get_active_provider returns first provider when name does not match."""
    p1 = ProviderConfig(name="alpha", api_key="k1")
    p2 = ProviderConfig(name="beta", api_key="k2")
    settings = AppSettings(providers=[p1, p2], active_provider="nonexistent")
    result = get_active_provider(settings)
    assert result is p1


def test_settings_theme_defaults_to_light():
    """AppSettings defaults theme to 'light'."""
    s = AppSettings()
    assert s.theme == "light"


def test_settings_theme_persists_round_trip(tmp_path, monkeypatch):
    """theme value survives a save/load cycle."""
    monkeypatch.setattr("app.config.settings._config_path", lambda: tmp_path / "config.json")
    s = AppSettings(theme="dark")
    save(s)
    s2 = load()
    assert s2.theme == "dark"


def test_settings_invalid_theme_resets_to_light(tmp_path, monkeypatch):
    """An invalid theme value in config.json is normalised to 'light'."""
    import json
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"theme": "purple"}))
    monkeypatch.setattr("app.config.settings._config_path", lambda: config_path)
    s = load()
    assert s.theme == "light"
