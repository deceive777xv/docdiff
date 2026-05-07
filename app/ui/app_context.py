"""Application context — shared state for all UI pages."""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass

from app.config.settings import AppSettings
from app.core.model.base_provider import BaseProvider


@dataclass
class AppContext:
    settings: AppSettings
    conn: sqlite3.Connection
    data_dir: str
    provider: BaseProvider | None = None
    embedder: BaseProvider | None = None
    lc_model: object | None = None  # BaseChatModel, typed as object to avoid hard dep
