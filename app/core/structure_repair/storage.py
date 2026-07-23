"""Atomic persistence and safe fallback for import-time structure repair."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from app.core.diff.reconstruction_trace import write_json_atomic
from app.core.document_ir_codec import document_ir_to_dict
from app.core.types import DocumentIR

from .models import StructureRepairResult, StructureRepairTrace
from .pipeline import ALGORITHM_VERSION, SCHEMA_VERSION, repair_document


@dataclass(frozen=True)
class ImportStructureArtifacts:
    document: DocumentIR
    raw_path: Path
    normalized_path: Path
    trace_path: Path
    trace: StructureRepairTrace


def _hash(document: DocumentIR) -> str:
    payload = json.dumps(
        document_ir_to_dict(document),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_import_ir(
    data_dir: str | Path,
    raw_document: DocumentIR,
    *,
    provider: object | None = None,
    model: str = "",
) -> ImportStructureArtifacts:
    """Persist immutable raw IR, repair a copy, and atomically publish artifacts."""
    parsed_dir = Path(data_dir) / "parsed"
    raw_path = parsed_dir / "raw" / f"{raw_document.doc_id}.json"
    normalized_path = parsed_dir / f"{raw_document.doc_id}.json"
    trace_path = parsed_dir / "traces" / f"{raw_document.doc_id}.structure.json"

    write_json_atomic(raw_path, document_ir_to_dict(raw_document))
    try:
        result = repair_document(
            raw_document,
            provider=provider,
            model=model,
        )
    except Exception as exc:
        fallback = deepcopy(raw_document)
        document_hash = _hash(fallback)
        warning = f"{type(exc).__name__}: {exc}"
        trace = StructureRepairTrace(
            schema_version=SCHEMA_VERSION,
            algorithm_version=ALGORITHM_VERSION,
            doc_id=raw_document.doc_id,
            raw_hash=document_hash,
            normalized_hash=document_hash,
            status="fallback",
            operations=[],
            decisions=[],
            rejected=[],
            warnings=[warning],
        )
        result = StructureRepairResult(
            document=fallback,
            trace=trace,
            status="fallback",
            warnings=[warning],
        )

    write_json_atomic(normalized_path, document_ir_to_dict(result.document))
    write_json_atomic(trace_path, asdict(result.trace))
    return ImportStructureArtifacts(
        document=result.document,
        raw_path=raw_path,
        normalized_path=normalized_path,
        trace_path=trace_path,
        trace=result.trace,
    )
