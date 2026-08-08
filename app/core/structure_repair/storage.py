"""Atomic persistence and safe fallback for import-time structure repair."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from app.core.normalization.table_trace import (
    ALGORITHM_VERSION as TABLE_ALGORITHM_VERSION,
    SCHEMA_VERSION as TABLE_SCHEMA_VERSION,
    DocumentTraceRef,
    ReconstructionTrace,
    write_json_atomic,
)
from app.core.document_ir_codec import document_ir_to_dict
from app.core.normalization import (
    ALGORITHM_VERSION as NORMALIZATION_ALGORITHM_VERSION,
    SCHEMA_VERSION as NORMALIZATION_SCHEMA_VERSION,
    DocumentBoundaryProfile,
    DocumentNormalizationResult,
    DocumentNormalizationTrace,
    NormalizationDepth,
    normalize_document,
)
from app.core.types import DocumentIR

from .models import StructureRepairResult, StructureRepairTrace
from .pipeline import ALGORITHM_VERSION, SCHEMA_VERSION


@dataclass(frozen=True)
class ImportStructureArtifacts:
    document: DocumentIR
    raw_path: Path
    normalized_path: Path
    trace_path: Path
    trace: StructureRepairTrace
    normalization_trace_path: Path
    boundary_profile_path: Path
    normalization_trace: DocumentNormalizationTrace
    boundary_profile: DocumentBoundaryProfile


def import_artifact_paths(
    data_dir: str | Path,
    parsed_json_path: str | Path,
) -> tuple[Path, ...]:
    """Return the exact import artifacts associated with one normalized IR."""
    parsed_dir = (Path(data_dir) / "parsed").resolve()
    normalized_path = Path(parsed_json_path).resolve()
    if (
        normalized_path.parent != parsed_dir
        or normalized_path.suffix != ".json"
        or normalized_path.name != f"{normalized_path.stem}.json"
    ):
        raise ValueError(
            f"Parsed JSON path is outside the expected parsed root: {parsed_json_path}"
        )

    artifact_id = normalized_path.stem
    candidates = (
        normalized_path,
        parsed_dir / "raw" / f"{artifact_id}.json",
        parsed_dir / "traces" / f"{artifact_id}.structure.json",
        parsed_dir / "traces" / f"{artifact_id}.normalization.json",
        parsed_dir / "profiles" / f"{artifact_id}.boundary.json",
    )
    resolved: list[Path] = []
    for candidate in candidates:
        path = candidate.resolve()
        try:
            path.relative_to(parsed_dir)
        except ValueError as exc:
            raise ValueError(
                f"Import artifact path is outside the expected parsed root: {path}"
            ) from exc
        resolved.append(path)
    return tuple(resolved)


def delete_import_artifacts(
    data_dir: str | Path,
    parsed_json_path: str | Path,
) -> list[Path]:
    """Delete one version's import artifacts and return paths that could not be removed."""
    failures: list[Path] = []
    for path in import_artifact_paths(data_dir, parsed_json_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            failures.append(path)
    return failures


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
    depth: NormalizationDepth = NormalizationDepth.OFF,
) -> ImportStructureArtifacts:
    """Persist immutable raw IR, repair a copy, and atomically publish artifacts."""
    parsed_dir = Path(data_dir) / "parsed"
    raw_path = parsed_dir / "raw" / f"{raw_document.doc_id}.json"
    normalized_path = parsed_dir / f"{raw_document.doc_id}.json"
    trace_path = parsed_dir / "traces" / f"{raw_document.doc_id}.structure.json"
    normalization_trace_path = (
        parsed_dir / "traces" / f"{raw_document.doc_id}.normalization.json"
    )
    boundary_profile_path = (
        parsed_dir / "profiles" / f"{raw_document.doc_id}.boundary.json"
    )

    write_json_atomic(raw_path, document_ir_to_dict(raw_document))
    try:
        result = normalize_document(
            raw_document,
            provider=provider,
            model=model,
            depth=depth,
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
        structure_result = StructureRepairResult(
            document=fallback,
            trace=trace,
            status="fallback",
            warnings=[warning],
        )
        table_trace = ReconstructionTrace(
            schema_version=TABLE_SCHEMA_VERSION,
            algorithm_version=TABLE_ALGORITHM_VERSION,
            baseline=DocumentTraceRef(raw_document.doc_id, raw_document.file_hash),
            target=DocumentTraceRef(
                f"{raw_document.doc_id}:normalization-empty-peer",
                "0" * 64,
            ),
            decisions=[],
            operations=[],
        )
        normalization_trace = DocumentNormalizationTrace(
            scope="document",
            schema_version=NORMALIZATION_SCHEMA_VERSION,
            algorithm_version=NORMALIZATION_ALGORITHM_VERSION,
            source_document_refs=[
                DocumentTraceRef(raw_document.doc_id, raw_document.file_hash)
            ],
            status="fallback",
            structure_trace=trace,
            table_trace=table_trace,
            warnings=[warning],
            normalization_depth=NormalizationDepth(depth).value,
        )
        boundary_profile = DocumentBoundaryProfile(
            schema_version=NORMALIZATION_SCHEMA_VERSION,
            algorithm_version=NORMALIZATION_ALGORITHM_VERSION,
            doc_id=raw_document.doc_id,
            file_hash=raw_document.file_hash,
            table_candidate_count=0,
        )
        result = DocumentNormalizationResult(
            document=structure_result.document,
            structure_trace=trace,
            table_trace=table_trace,
            trace=normalization_trace,
            boundary_profile=boundary_profile,
            status="fallback",
            warnings=[warning],
        )

    write_json_atomic(normalized_path, document_ir_to_dict(result.document))
    write_json_atomic(trace_path, asdict(result.structure_trace))
    write_json_atomic(normalization_trace_path, asdict(result.trace))
    write_json_atomic(boundary_profile_path, asdict(result.boundary_profile))
    return ImportStructureArtifacts(
        document=result.document,
        raw_path=raw_path,
        normalized_path=normalized_path,
        trace_path=trace_path,
        trace=result.structure_trace,
        normalization_trace_path=normalization_trace_path,
        boundary_profile_path=boundary_profile_path,
        normalization_trace=result.trace,
        boundary_profile=result.boundary_profile,
    )
