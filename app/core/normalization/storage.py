"""Persistence helpers for document-level normalization artifacts."""

from __future__ import annotations

from pathlib import Path


def normalization_trace_path(data_dir: str | Path, doc_id: str) -> Path:
    return Path(data_dir) / "parsed" / "traces" / f"{doc_id}.normalization.json"


def boundary_profile_path(data_dir: str | Path, doc_id: str) -> Path:
    return Path(data_dir) / "parsed" / "profiles" / f"{doc_id}.boundary.json"
