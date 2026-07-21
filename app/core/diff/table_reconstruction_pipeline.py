"""Shared orchestration for deterministic cross-page table reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence, cast

from app.core.diff.reconstruction_trace import (
    ALGORITHM_VERSION,
    SCHEMA_VERSION,
    DocumentTraceRef,
    LLMJudgment,
    ReconstructionDecision,
    ReconstructionTrace,
    SourceRowRef,
    validate_trace_documents,
)
from app.core.diff.structure_aligner import SectionPair, align_sections
from app.core.diff.table_reconstruction import (
    CandidateAssessment,
    ContinuationCandidate,
    TableFragment,
    apply_reconstruction_operations,
    assess_candidate,
    build_reconstruction_operations,
    classify_repeated_boundary_regions,
    collect_table_fragments,
    generate_continuation_candidates,
    infer_active_columns,
    infer_monotonic_column_mapping,
    infer_regions,
)
from app.core.diff.table_reconstruction_llm import adjudicate_continuation
from app.core.model.base_provider import BaseProvider
from app.core.types import DocumentIR, Section


_LLM_MERGE_THRESHOLD = 0.75


@dataclass(frozen=True)
class ReconstructionResult:
    baseline_ir: DocumentIR
    target_ir: DocumentIR
    section_pairs: list[SectionPair]
    trace: ReconstructionTrace


def _jointly_infer_fragments(
    fragments: Sequence[TableFragment],
    cross_version_fragments: Sequence[TableFragment],
) -> tuple[TableFragment, ...]:
    peers = tuple(fragments) + tuple(cross_version_fragments)
    with_regions = tuple(
        replace(
            fragment,
            regions=infer_regions(
                fragment,
                tuple(peer for peer in peers if peer is not fragment),
            ),
        )
        for fragment in fragments
    )
    with_body_regions = tuple(
        replace(
            fragment,
            body_region_indexes=tuple(
                index
                for index, region in enumerate(fragment.regions)
                if region.role == "body"
            ),
        )
        for fragment in with_regions
    )
    peers_with_regions = with_body_regions + tuple(cross_version_fragments)
    return tuple(
        replace(
            fragment,
            active_columns=infer_active_columns(
                fragment,
                tuple(peer for peer in peers_with_regions if peer is not fragment),
            ),
        )
        for fragment in with_body_regions
    )


def _section_fragments(section: Section | None) -> tuple[TableFragment, ...]:
    if section is None:
        return ()
    return tuple(collect_table_fragments(section))


def _paragraph_boundaries(
    fragments: Sequence[TableFragment],
    boundary_rows: set[SourceRowRef],
) -> set[str]:
    return {
        fragment.paragraph_id
        for fragment in fragments
        if fragment.rows and all(row.source in boundary_rows for row in fragment.rows)
    }


def _side_candidates(
    fragments: Sequence[TableFragment],
    cross_version_fragments: Sequence[TableFragment],
    boundary_rows: set[SourceRowRef],
    side: Literal["baseline", "target"],
) -> list[ContinuationCandidate]:
    candidates: list[ContinuationCandidate] = []
    ordered = sorted(
        (
            fragment
            for fragment in fragments
            if any(row.source not in boundary_rows for row in fragment.rows)
        ),
        key=lambda fragment: (fragment.paragraph_index, fragment.paragraph_id),
    )
    for left, right in zip(ordered, ordered[1:]):
        mapping = infer_monotonic_column_mapping(left, right, cross_version_fragments)
        if mapping is None:
            continue
        candidates.extend(
            generate_continuation_candidates(
                left,
                right,
                mapping,
                boundary_rows,
                cross_version_fragments,
                side,
                allow_non_table_gap=True,
            )
        )
    return candidates


def _collect_pair_analysis(
    section_pair: SectionPair,
) -> tuple[
    list[ContinuationCandidate],
    dict[str, set[SourceRowRef]],
    dict[str, set[str]],
]:
    baseline_initial = _section_fragments(section_pair.baseline_section)
    target_initial = _section_fragments(section_pair.target_section)
    baseline_fragments = _jointly_infer_fragments(baseline_initial, target_initial)
    target_fragments = _jointly_infer_fragments(target_initial, baseline_fragments)
    baseline_rows = classify_repeated_boundary_regions(baseline_fragments)
    target_rows = classify_repeated_boundary_regions(target_fragments)
    candidates = [
        *_side_candidates(
            baseline_fragments,
            target_fragments,
            baseline_rows,
            "baseline",
        ),
        *_side_candidates(
            target_fragments,
            baseline_fragments,
            target_rows,
            "target",
        ),
    ]
    return (
        candidates,
        {"baseline": baseline_rows, "target": target_rows},
        {
            "baseline": _paragraph_boundaries(baseline_fragments, baseline_rows),
            "target": _paragraph_boundaries(target_fragments, target_rows),
        },
    )


def _resolve_assessment(
    assessment: CandidateAssessment,
    provider: BaseProvider | None,
) -> tuple[CandidateAssessment, LLMJudgment | None]:
    if assessment.final_action != "needs_llm" or provider is None:
        final_action = (
            assessment.final_action
            if assessment.final_action != "needs_llm"
            else "keep_separate"
        )
        return replace(assessment, final_action=final_action), None
    judgment = adjudicate_continuation(assessment.candidate, provider)
    final_action = (
        "merge"
        if judgment is not None
        and judgment.decision == "merge"
        and judgment.confidence >= _LLM_MERGE_THRESHOLD
        else "keep_separate"
    )
    return replace(assessment, final_action=final_action), judgment


def _merge_sets(
    destination: dict[str, set],
    source: Mapping[str, set],
) -> None:
    for side in ("baseline", "target"):
        destination[side].update(source.get(side, set()))


def _decision_sort_key(decision: ReconstructionDecision) -> tuple[str, str]:
    return decision.side, decision.candidate_id


def _trace_decisions(
    assessments: Sequence[CandidateAssessment],
    judgments: Mapping[tuple[str, str], LLMJudgment | None],
    generated_row_ids: Mapping[str, str],
) -> list[ReconstructionDecision]:
    decisions = []
    for assessment in assessments:
        candidate = assessment.candidate
        decisions.append(
            ReconstructionDecision(
                candidate_id=candidate.candidate_id,
                side=candidate.side,
                source_rows=[
                    candidate.previous_row.source,
                    candidate.continuation_row.source,
                ],
                column_mapping=dict(sorted(candidate.mapping.logical_by_physical.items())),
                rule_confidence=assessment.rule_confidence,
                rule_evidence=list(candidate.evidence),
                rule_conflicts=list(
                    dict.fromkeys((*candidate.conflicts, *candidate.vetoes))
                ),
                llm=judgments[(candidate.side, candidate.candidate_id)],
                final_action=cast(
                    Literal["merge", "keep_separate"], assessment.final_action
                ),
                generated_row_id=generated_row_ids.get(candidate.candidate_id, ""),
            )
        )
    return sorted(decisions, key=_decision_sort_key)


def replay_reconstruction(
    baseline_ir: DocumentIR,
    target_ir: DocumentIR,
    trace: ReconstructionTrace,
) -> tuple[DocumentIR, DocumentIR]:
    """Validate trace provenance and replay its operations on document copies."""
    validate_trace_documents(trace, baseline_ir, target_ir)
    return apply_reconstruction_operations(baseline_ir, target_ir, trace.operations)


def reconstruct_table_pairs(
    section_pairs: Sequence[SectionPair],
    baseline_ir: DocumentIR,
    target_ir: DocumentIR,
    provider: BaseProvider | None,
) -> ReconstructionResult:
    """Analyze aligned versions jointly and emit normalized IR plus replay trace."""
    candidates: dict[tuple[str, str], ContinuationCandidate] = {}
    boundary_rows: dict[str, set[SourceRowRef]] = {"baseline": set(), "target": set()}
    boundary_paragraphs: dict[str, set[str]] = {"baseline": set(), "target": set()}
    for section_pair in section_pairs:
        pair_candidates, pair_rows, pair_paragraphs = _collect_pair_analysis(section_pair)
        for candidate in pair_candidates:
            candidates.setdefault((candidate.side, candidate.candidate_id), candidate)
        _merge_sets(boundary_rows, pair_rows)
        _merge_sets(boundary_paragraphs, pair_paragraphs)

    resolved: list[CandidateAssessment] = []
    judgments: dict[tuple[str, str], LLMJudgment | None] = {}
    for key, candidate in sorted(candidates.items()):
        assessment, judgment = _resolve_assessment(assess_candidate(candidate), provider)
        resolved.append(assessment)
        judgments[key] = judgment

    operations = build_reconstruction_operations(
        resolved,
        boundary_rows,
        boundary_paragraphs,
    )
    generated_row_ids = {
        operation.decision_id: operation.generated_row_id
        for operation in operations
        if operation.type == "merge_rows" and operation.decision_id
    }
    trace = ReconstructionTrace(
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        baseline=DocumentTraceRef(baseline_ir.doc_id, baseline_ir.file_hash),
        target=DocumentTraceRef(target_ir.doc_id, target_ir.file_hash),
        decisions=_trace_decisions(resolved, judgments, generated_row_ids),
        operations=operations,
    )
    normalized_baseline, normalized_target = replay_reconstruction(
        baseline_ir,
        target_ir,
        trace,
    )
    normalized_pairs = align_sections(normalized_baseline, normalized_target)
    return ReconstructionResult(
        normalized_baseline,
        normalized_target,
        normalized_pairs,
        trace,
    )
