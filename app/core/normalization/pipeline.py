"""Unified, single-document normalization orchestration."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from app.core.document_ir_codec import document_ir_to_dict
from app.core.normalization.table_trace import (
    ALGORITHM_VERSION as TABLE_ALGORITHM_VERSION,
    SCHEMA_VERSION as TABLE_SCHEMA_VERSION,
    DocumentTraceRef,
    ReconstructionTrace,
)
from app.core.normalization.table_pipeline import (
    reconstruct_document_tables,
)
from app.core.model.base_provider import BaseProvider
from app.core.structure_repair.pipeline import repair_document
from app.core.structure_repair.models import StructureRepairTrace
from app.core.types import DocumentIR

from .models import (
    DocumentBoundaryProfile,
    DocumentNormalizationResult,
    DocumentNormalizationTrace,
    NormalizationDepth,
)


SCHEMA_VERSION = 3
ALGORITHM_VERSION = "unified-document-normalization-v8"
def _empty_document_table_trace(document: DocumentIR) -> ReconstructionTrace:
    return ReconstructionTrace(
        schema_version=TABLE_SCHEMA_VERSION,
        algorithm_version=TABLE_ALGORITHM_VERSION,
        baseline=DocumentTraceRef(document.doc_id, document.file_hash),
        target=DocumentTraceRef(
            f"{document.doc_id}:normalization-empty-peer",
            "0" * 64,
        ),
        decisions=[],
        operations=[],
    )
def _document_hash(document: DocumentIR) -> str:
    payload = json.dumps(
        document_ir_to_dict(document),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _skipped_result(document: DocumentIR) -> DocumentNormalizationResult:
    skipped = deepcopy(document)
    content_hash = _document_hash(skipped)
    structure_trace = StructureRepairTrace(
        schema_version=0,
        algorithm_version="normalization-skipped",
        doc_id=document.doc_id,
        raw_hash=content_hash,
        normalized_hash=content_hash,
        status="unchanged",
    )
    table_trace = _empty_document_table_trace(skipped)
    document_ref = DocumentTraceRef(document.doc_id, document.file_hash)
    trace = DocumentNormalizationTrace(
        scope="document",
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        source_document_refs=[document_ref],
        status="skipped",
        structure_trace=structure_trace,
        table_trace=table_trace,
        normalization_depth=NormalizationDepth.OFF.value,
    )
    boundary_profile = DocumentBoundaryProfile(
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        doc_id=document.doc_id,
        file_hash=document.file_hash,
        table_candidate_count=0,
    )
    return DocumentNormalizationResult(
        document=skipped,
        structure_trace=structure_trace,
        table_trace=table_trace,
        trace=trace,
        boundary_profile=boundary_profile,
        status="skipped",
    )


def normalize_document(
    document: DocumentIR,
    *,
    provider: BaseProvider | None = None,
    model: str = "",
    depth: NormalizationDepth = NormalizationDepth.OFF,
) -> DocumentNormalizationResult:
    """Normalize paragraphs and cross-page tables on copies of one document."""
    depth = NormalizationDepth(depth)
    if depth is NormalizationDepth.OFF:
        return _skipped_result(document)
    structure_result = repair_document(
        document,
        provider=provider,
        model=model,
        review_changes=depth is NormalizationDepth.REVIEW,
    )
    warnings = list(structure_result.warnings)
    if structure_result.status == "fallback":
        normalized_document = structure_result.document
        table_trace = _empty_document_table_trace(normalized_document)
        status = "fallback"
    else:
        try:
            table_result = reconstruct_document_tables(
                structure_result.document,
                provider,
                review_changes=depth is NormalizationDepth.REVIEW,
            )
        except Exception as exc:
            normalized_document = structure_result.document
            table_trace = _empty_document_table_trace(normalized_document)
            status = "fallback"
            warnings.append(
                f"table_reconstruction_failed: {type(exc).__name__}: {exc}"
            )
        else:
            normalized_document = table_result.document
            table_trace = table_result.trace
            status = (
                "repaired"
                if structure_result.status == "repaired" or table_trace.operations
                else "unchanged"
            )
    document_ref = DocumentTraceRef(document.doc_id, document.file_hash)
    trace = DocumentNormalizationTrace(
        scope="document",
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        source_document_refs=[document_ref],
        status=status,
        structure_trace=structure_result.trace,
        table_trace=table_trace,
        warnings=warnings,
        normalization_depth=depth.value,
    )
    boundary_profile = DocumentBoundaryProfile(
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        doc_id=document.doc_id,
        file_hash=document.file_hash,
        table_candidate_count=len(table_trace.decisions),
    )
    return DocumentNormalizationResult(
        document=normalized_document,
        structure_trace=structure_result.trace,
        table_trace=table_trace,
        trace=trace,
        boundary_profile=boundary_profile,
        status=status,
        warnings=warnings,
    )
