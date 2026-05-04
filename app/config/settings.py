"""Application settings — read/write config.json from APPDATA."""
from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from app.config.crypto import decrypt, encrypt


def _default_data_dir() -> str:
    appdata = os.environ.get("APPDATA", str(Path.home()))
    return str(Path(appdata) / "DocDiffAgent")


def _config_path() -> Path:
    return Path(_default_data_dir()) / "config.json"


@dataclass
class ProviderConfig:
    name: str
    type: str = "openai_compatible"   # openai_compatible | azure
    api_key: str = ""                 # stored encrypted in JSON
    base_url: str = ""
    chat_model: str = "deepseek-chat"
    embed_model: str = "text-embedding-ada-002"


@dataclass
class LocalEmbeddingConfig:
    enabled: bool = False
    model_path: str = ""


@dataclass
class AppSettings:
    providers: list[ProviderConfig] = field(default_factory=list)
    local_embedding: LocalEmbeddingConfig = field(default_factory=LocalEmbeddingConfig)
    active_provider: str = ""
    data_dir: str = field(default_factory=_default_data_dir)


def load() -> AppSettings:
    """Load settings from config.json. Returns defaults if file missing."""
    path = _config_path()
    if not path.exists():
        return AppSettings()
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    providers = []
    for p in raw.get("providers", []):
        pc = ProviderConfig(
            name=p.get("name", ""),
            type=p.get("type", "openai_compatible"),
            api_key=decrypt(p["api_key"]) if p.get("api_key") else "",
            base_url=p.get("base_url", ""),
            chat_model=p.get("chat_model", "deepseek-chat"),
            embed_model=p.get("embed_model", "text-embedding-ada-002"),
        )
        providers.append(pc)
    le_raw = raw.get("local_embedding", {})
    return AppSettings(
        providers=providers,
        local_embedding=LocalEmbeddingConfig(
            enabled=le_raw.get("enabled", False),
            model_path=le_raw.get("model_path", ""),
        ),
        active_provider=raw.get("active_provider", ""),
        data_dir=raw.get("data_dir", "") or _default_data_dir(),
    )


def save(settings: AppSettings) -> None:
    """Save settings to config.json, encrypting API keys."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    providers_raw = []
    for p in settings.providers:
        providers_raw.append({
            "name": p.name,
            "type": p.type,
            "api_key": encrypt(p.api_key) if p.api_key else "",
            "base_url": p.base_url,
            "chat_model": p.chat_model,
            "embed_model": p.embed_model,
        })
    raw = {
        "providers": providers_raw,
        "local_embedding": {
            "enabled": settings.local_embedding.enabled,
            "model_path": settings.local_embedding.model_path,
        },
        "active_provider": settings.active_provider,
        "data_dir": settings.data_dir,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


def get_active_provider(settings: AppSettings) -> Optional[ProviderConfig]:
    """Return the active ProviderConfig, or None if not configured."""
    for p in settings.providers:
        if p.name == settings.active_provider:
            return p
    return settings.providers[0] if settings.providers else None
