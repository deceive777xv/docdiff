"""Persistence for final compare results."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from app.core.normalization.table_trace import write_json_atomic
from app.core.types import DiffItem


def persist_compare_result(
    data_dir: str | Path,
    task_id: str,
    items: list[DiffItem],
) -> Path:
    """Atomically publish a compare result without normalization sidecars."""
    result_path = Path(data_dir) / "exports" / f"{task_id}.json"
    write_json_atomic(result_path, [asdict(item) for item in items])
    return result_path
