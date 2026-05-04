"""Shared utility functions."""
from __future__ import annotations
import hashlib
from pathlib import Path


def file_hash(path: Path) -> str:
    """Return SHA-256 hex digest of file contents."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()
