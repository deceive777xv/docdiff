"""Persistence helpers for document-level normalization artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.diff.reconstruction_trace import SourceRowRef
from app.core.types import DocumentIR

from .models import DeferredTableCandidate


def normalization_trace_path(data_dir: str | Path, doc_id: str) -> Path:
    return Path(data_dir) / "parsed" / "traces" / f"{doc_id}.normalization.json"


def boundary_profile_path(data_dir: str | Path, doc_id: str) -> Path:
    return Path(data_dir) / "parsed" / "profiles" / f"{doc_id}.boundary.json"


def load_deferred_table_candidates(
    data_dir: str | Path,
    document: DocumentIR,
) -> list[DeferredTableCandidate] | None:
    """Load import-deferred table candidates; return None for legacy documents."""
    path = normalization_trace_path(data_dir, document.doc_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("scope") != "document":
        raise ValueError("normalization trace must be a document-scoped object")
    refs = payload.get("source_document_refs")
    if (
        not isinstance(refs, list)
        or len(refs) != 1
        or not isinstance(refs[0], dict)
        or refs[0].get("doc_id") != document.doc_id
        or refs[0].get("file_hash") != document.file_hash
    ):
        raise ValueError("normalization trace does not match loaded document")
    raw_candidates = payload.get("deferred_table_candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("deferred_table_candidates must be a list")
    candidates: list[DeferredTableCandidate] = []
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"deferred_table_candidates[{index}] must be an object")
        raw_rows = raw_candidate.get("source_rows")
        raw_mapping = raw_candidate.get("column_mapping")
        if not isinstance(raw_rows, list) or len(raw_rows) != 2:
            raise ValueError(f"deferred_table_candidates[{index}].source_rows is invalid")
        if not isinstance(raw_mapping, dict):
            raise ValueError(f"deferred_table_candidates[{index}].column_mapping is invalid")
        rows = []
        for row_index, raw_row in enumerate(raw_rows):
            if (
                not isinstance(raw_row, dict)
                or not isinstance(raw_row.get("section_id"), str)
                or not isinstance(raw_row.get("paragraph_id"), str)
                or isinstance(raw_row.get("sentence_index"), bool)
                or not isinstance(raw_row.get("sentence_index"), int)
            ):
                raise ValueError(
                    f"deferred_table_candidates[{index}].source_rows[{row_index}] is invalid"
                )
            rows.append(
                SourceRowRef(
                    raw_row["section_id"],
                    raw_row["paragraph_id"],
                    raw_row["sentence_index"],
                )
            )
        try:
            mapping = {int(key): int(value) for key, value in raw_mapping.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"deferred_table_candidates[{index}].column_mapping is invalid"
            ) from exc
        candidate_id = raw_candidate.get("candidate_id")
        failure_code = raw_candidate.get("failure_code")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"deferred_table_candidates[{index}].candidate_id is invalid")
        if not isinstance(failure_code, str) or not failure_code:
            raise ValueError(f"deferred_table_candidates[{index}].failure_code is invalid")
        candidates.append(
            DeferredTableCandidate(candidate_id, rows, mapping, failure_code)
        )
    return candidates
