"""Unified, single-document normalization orchestration."""

from __future__ import annotations

from app.core.diff.reconstruction_trace import (
    ALGORITHM_VERSION as TABLE_ALGORITHM_VERSION,
    SCHEMA_VERSION as TABLE_SCHEMA_VERSION,
    DocumentTraceRef,
    ReconstructionTrace,
    SourceRowRef,
)
from app.core.diff.structure_aligner import align_sections
from app.core.diff.table_reconstruction_pipeline import (
    ReconstructionResult,
    reconstruct_document_tables,
    reconstruct_table_pairs,
)
from app.core.model.base_provider import BaseProvider
from app.core.structure_repair.pipeline import repair_document
from app.core.types import DocumentIR

from .models import (
    DeferredTableCandidate,
    DocumentBoundaryProfile,
    DocumentNormalizationResult,
    DocumentNormalizationTrace,
)


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "unified-document-normalization-v1"
_HARD_TABLE_CONFLICTS = {
    "new_key_value",
    "header_or_separator",
    "incompatible_schema",
    "new_section_or_table",
    "crosses_real_body_row",
    "conflicting_key_cells",
    "unsafe_fragment_projection",
}


def _deferred_table_candidates(table_trace) -> list[DeferredTableCandidate]:
    deferred: list[DeferredTableCandidate] = []
    for decision in table_trace.decisions:
        if decision.final_action != "keep_separate":
            continue
        if _HARD_TABLE_CONFLICTS.intersection(decision.rule_conflicts):
            continue
        judgment = decision.llm
        if judgment is not None and judgment.decision == "keep_separate":
            continue
        failure_code = (
            "llm_unavailable"
            if judgment is None
            else "llm_below_merge_threshold"
        )
        deferred.append(
            DeferredTableCandidate(
                candidate_id=decision.candidate_id,
                source_rows=list(decision.source_rows),
                column_mapping=dict(decision.column_mapping),
                failure_code=failure_code,
            )
        )
    return deferred


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


def normalize_document(
    document: DocumentIR,
    *,
    provider: BaseProvider | None = None,
    model: str = "",
) -> DocumentNormalizationResult:
    """Normalize paragraphs and cross-page tables on copies of one document."""
    structure_result = repair_document(
        document,
        provider=provider,
        model=model,
    )
    if structure_result.status == "fallback":
        normalized_document = structure_result.document
        table_trace = _empty_document_table_trace(normalized_document)
        status = "fallback"
    else:
        table_result = reconstruct_document_tables(
            structure_result.document,
            provider,
        )
        normalized_document = table_result.document
        table_trace = table_result.trace
        status = (
            "repaired"
            if structure_result.status == "repaired" or table_trace.operations
            else "unchanged"
        )
    deferred = _deferred_table_candidates(table_trace)
    warnings = list(structure_result.warnings)
    document_ref = DocumentTraceRef(document.doc_id, document.file_hash)
    trace = DocumentNormalizationTrace(
        scope="document",
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        source_document_refs=[document_ref],
        status=status,
        structure_trace=structure_result.trace,
        table_trace=table_trace,
        deferred_table_candidates=deferred,
        warnings=warnings,
    )
    boundary_profile = DocumentBoundaryProfile(
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        doc_id=document.doc_id,
        file_hash=document.file_hash,
        table_candidate_count=len(table_trace.decisions),
        deferred_table_candidates=deferred,
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


def _deferred_source_pairs(
    candidates: list[DeferredTableCandidate] | None,
) -> set[tuple[SourceRowRef, SourceRowRef]] | None:
    if candidates is None:
        return None
    return {
        (candidate.source_rows[0], candidate.source_rows[1])
        for candidate in candidates
        if len(candidate.source_rows) == 2
    }


def normalize_pair(
    baseline: DocumentIR,
    target: DocumentIR,
    *,
    provider: BaseProvider | None = None,
    baseline_deferred: list[DeferredTableCandidate] | None = None,
    target_deferred: list[DeferredTableCandidate] | None = None,
    section_pairs=None,
) -> ReconstructionResult:
    """Recheck only import-deferred table boundaries with pair evidence.

    ``None`` denotes a legacy document without a normalization artifact and keeps
    the former full-analysis behavior.  An empty list is an explicit statement
    that import normalization resolved every table candidate.
    """
    return reconstruct_table_pairs(
        section_pairs if section_pairs is not None else align_sections(baseline, target),
        baseline,
        target,
        provider,
        candidate_source_filter={
            "baseline": _deferred_source_pairs(baseline_deferred),
            "target": _deferred_source_pairs(target_deferred),
        },
    )
