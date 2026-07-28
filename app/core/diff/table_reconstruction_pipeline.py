"""Shared orchestration for deterministic cross-page table reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import re
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
    ColumnMapping,
    ContinuationCandidate,
    TableFragment,
    apply_reconstruction_operations,
    assess_candidate,
    build_reconstruction_operations,
    classify_repeated_boundary_regions,
    collect_table_fragments,
    corresponding_peer_fragments,
    generate_continuation_candidates,
    infer_active_columns,
    infer_monotonic_column_mapping,
    infer_regions,
)
from app.core.diff.table_reconstruction_llm import adjudicate_continuation
from app.core.diff.table_boundary_context import (
    TableBoundaryContext,
    locate_table_boundary_context,
)
from app.core.model.base_provider import BaseProvider
from app.core.types import DocumentIR, Paragraph, Section


_LLM_MERGE_THRESHOLD = 0.75
_LLM_CHOICE_MARGIN = Decimal("0.05")
_MAX_BOUNDARY_PARAGRAPH_NORMALIZED_LENGTH = 160
_UNSAFE_FRAGMENT_PROJECTION = "unsafe_fragment_projection"
_AMBIGUOUS_CONTINUATION_CHOICES = "ambiguous_continuation_choices"
_LOWER_CONFIDENCE_CONTINUATION_CHOICE = "lower_confidence_continuation_choice"


@dataclass(frozen=True)
class ReconstructionResult:
    baseline_ir: DocumentIR
    target_ir: DocumentIR
    section_pairs: list[SectionPair]
    trace: ReconstructionTrace


def _occupied_region_columns(fragment: TableFragment, region_index: int) -> set[int]:
    return {
        physical_index
        for row in fragment.regions[region_index].rows
        for physical_index, occupied in enumerate(row.occupied)
        if occupied
    }


def _keyed_body_region_columns(
    fragment: TableFragment,
) -> tuple[set[int], set[int]]:
    body_indexes: set[int] = set()
    active_columns: set[int] = set()
    for region_index, region in enumerate(fragment.regions):
        keyed_rows = [
            row
            for row in region.rows
            if row.kind == "content"
            and sum(row.occupied) >= 2
            and any(
                value_type in {"integer", "hierarchical_number"}
                for value_type in row.value_types
            )
        ]
        if not keyed_rows:
            continue
        body_indexes.add(region_index)
        active_columns.update(
            physical_index
            for row in keyed_rows
            for physical_index, occupied in enumerate(row.occupied)
            if occupied
        )
    return body_indexes, active_columns


def _corresponding_peer_active_columns(
    fragment: TableFragment,
    peer_fragments: Sequence[TableFragment],
) -> tuple[int, ...]:
    width = max((len(row.raw_cells) for row in fragment.rows), default=0)
    candidate_columns = {
        tuple(peer.active_columns)
        for peer in corresponding_peer_fragments(fragment, peer_fragments)
        if peer.body_region_indexes
        and peer.active_columns
        and max((len(row.raw_cells) for row in peer.rows), default=0) == width
    }
    if len(candidate_columns) != 1:
        return ()
    return next(iter(candidate_columns))


def _rescue_sparse_boundary_fragment(
    fragment: TableFragment,
    peer_fragments: Sequence[TableFragment],
) -> TableFragment:
    if fragment.body_region_indexes:
        return fragment

    body_indexes, local_active_columns = _keyed_body_region_columns(fragment)
    active_columns = tuple(sorted(local_active_columns))
    if not body_indexes:
        active_columns = _corresponding_peer_active_columns(
            fragment,
            peer_fragments,
        )
        if not active_columns:
            return fragment
        last_separator = max(
            (
                index
                for index, region in enumerate(fragment.regions)
                if any(row.kind == "separator" for row in region.rows)
            ),
            default=-1,
        )
        compatible_regions = [
            index
            for index in range(last_separator + 1, len(fragment.regions))
            if fragment.regions[index].rows
            and all(row.kind == "content" for row in fragment.regions[index].rows)
            and (
                occupied := _occupied_region_columns(fragment, index)
            )
            and occupied.issubset(active_columns)
        ]
        if not compatible_regions:
            return fragment
        body_indexes.add(compatible_regions[-1])

    first_body_index = min(body_indexes)
    preceding_index = first_body_index - 1
    while preceding_index >= 0:
        region = fragment.regions[preceding_index]
        occupied = _occupied_region_columns(fragment, preceding_index)
        if (
            not region.rows
            or not all(row.kind == "content" for row in region.rows)
            or not occupied
            or not occupied.issubset(active_columns)
        ):
            break
        body_indexes.add(preceding_index)
        preceding_index -= 1

    first_body_index = min(body_indexes)
    regions = tuple(
        replace(
            region,
            role=(
                "boundary"
                if any(row.kind == "separator" for row in region.rows)
                else "body"
                if index in body_indexes
                else "header"
                if index < first_body_index
                else "unknown"
            ),
        )
        for index, region in enumerate(fragment.regions)
    )
    return replace(
        fragment,
        regions=regions,
        body_region_indexes=tuple(sorted(body_indexes)),
        active_columns=active_columns,
    )


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
            active_columns=(),
        )
        for fragment in with_regions
    )
    rescued_fragments = tuple(
        _rescue_sparse_boundary_fragment(
            fragment,
            tuple(
                peer
                for peer in (*with_body_regions, *cross_version_fragments)
                if peer is not fragment
            ),
        )
        for fragment in with_body_regions
    )
    peers_with_regions = rescued_fragments + tuple(cross_version_fragments)
    return tuple(
        replace(
            fragment,
            active_columns=(
                fragment.active_columns
                or infer_active_columns(
                    fragment,
                    tuple(
                        peer
                        for peer in peers_with_regions
                        if peer is not fragment
                    ),
                )
            ),
        )
        for fragment in rescued_fragments
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


def _ordered_retained_fragments(
    fragments: Sequence[TableFragment],
    boundary_rows: set[SourceRowRef],
) -> tuple[TableFragment, ...]:
    return tuple(
        sorted(
            (
                fragment
                for fragment in fragments
                if any(row.source not in boundary_rows for row in fragment.rows)
            ),
            key=lambda fragment: (fragment.paragraph_index, fragment.paragraph_id),
        )
    )


def _compatible_fragment_pairs(
    fragments: Sequence[TableFragment],
    cross_version_fragments: Sequence[TableFragment],
    boundary_rows: set[SourceRowRef],
) -> tuple[tuple[TableFragment, TableFragment, ColumnMapping], ...]:
    ordered = _ordered_retained_fragments(fragments, boundary_rows)
    compatible: list[tuple[TableFragment, TableFragment, ColumnMapping]] = []
    for left, right in zip(ordered, ordered[1:]):
        mapping = infer_monotonic_column_mapping(left, right, cross_version_fragments)
        if mapping is not None:
            compatible.append((left, right, mapping))
    return tuple(compatible)


def _normalized_paragraph_content(paragraph: Paragraph) -> str:
    text = paragraph.text or "\n".join(sentence.text for sentence in paragraph.sentences)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _classify_repeated_boundary_paragraphs(
    section: Section | None,
    fragments: Sequence[TableFragment],
    compatible_pairs: Sequence[tuple[TableFragment, TableFragment, ColumnMapping]],
) -> set[str]:
    """Confirm repeated short ordinary paragraphs at multiple table boundaries."""
    if section is None or not compatible_pairs:
        return set()
    table_paragraph_ids = {fragment.paragraph_id for fragment in fragments}
    boundary_positions: dict[int, set[tuple[int, int, int]]] = {}
    for pair_index, (left, right, _) in enumerate(compatible_pairs):
        if left.section_id != section.section_id or right.section_id != section.section_id:
            continue
        for paragraph_index in range(left.paragraph_index + 1, right.paragraph_index):
            if 0 <= paragraph_index < len(section.paragraphs):
                boundary_positions.setdefault(paragraph_index, set()).add(
                    (
                        pair_index,
                        paragraph_index - left.paragraph_index,
                        right.paragraph_index - paragraph_index,
                    )
                )

    occurrences: dict[str, list[int]] = {}
    for paragraph_index, paragraph in enumerate(section.paragraphs):
        if paragraph.paragraph_id in table_paragraph_ids:
            continue
        normalized = _normalized_paragraph_content(paragraph)
        occurrences.setdefault(normalized, []).append(paragraph_index)

    confirmed: set[str] = set()
    for normalized, paragraph_indexes in occurrences.items():
        placements = [
            tuple(boundary_positions.get(paragraph_index, set()))
            for paragraph_index in paragraph_indexes
        ]
        supporting_boundaries = {
            placement[0]
            for paragraph_placements in placements
            for placement in paragraph_placements
        }
        position_signatures = {
            placement[1:]
            for paragraph_placements in placements
            for placement in paragraph_placements
        }
        if (
            normalized
            and len(normalized) <= _MAX_BOUNDARY_PARAGRAPH_NORMALIZED_LENGTH
            and len(paragraph_indexes) >= 2
            and all(len(paragraph_placements) == 1 for paragraph_placements in placements)
            and len(supporting_boundaries) >= 2
            and len(position_signatures) == 1
        ):
            confirmed.update(
                section.paragraphs[paragraph_index].paragraph_id
                for paragraph_index in paragraph_indexes
            )
    return confirmed


def _side_candidates(
    section: Section | None,
    fragments: Sequence[TableFragment],
    cross_version_fragments: Sequence[TableFragment],
    boundary_rows: set[SourceRowRef],
    boundary_paragraph_ids: set[str],
    side: Literal["baseline", "target"],
) -> tuple[list[ContinuationCandidate], set[str]]:
    candidates: list[ContinuationCandidate] = []
    compatible_pairs = _compatible_fragment_pairs(
        fragments,
        cross_version_fragments,
        boundary_rows,
    )
    ordinary_boundary_paragraphs = _classify_repeated_boundary_paragraphs(
        section,
        fragments,
        compatible_pairs,
    )
    confirmed_boundary_paragraphs = (
        set(boundary_paragraph_ids) | ordinary_boundary_paragraphs
    )
    for left, right, mapping in compatible_pairs:
        intervening_paragraph_ids = (
            [
                section.paragraphs[index].paragraph_id
                for index in range(left.paragraph_index + 1, right.paragraph_index)
            ]
            if section is not None
            and 0 <= left.paragraph_index < right.paragraph_index < len(section.paragraphs)
            else None
        )
        allow_confirmed_non_table_gap = bool(intervening_paragraph_ids) and all(
            paragraph_id in confirmed_boundary_paragraphs
            for paragraph_id in intervening_paragraph_ids
        )
        candidates.extend(
            generate_continuation_candidates(
                left,
                right,
                mapping,
                boundary_rows,
                cross_version_fragments,
                side,
                allow_non_table_gap=allow_confirmed_non_table_gap,
            )
        )
    return candidates, ordinary_boundary_paragraphs


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
    baseline_table_paragraphs = _paragraph_boundaries(
        baseline_fragments,
        baseline_rows,
    )
    target_table_paragraphs = _paragraph_boundaries(
        target_fragments,
        target_rows,
    )
    baseline_candidates, baseline_ordinary_paragraphs = _side_candidates(
        section_pair.baseline_section,
        baseline_fragments,
        target_fragments,
        baseline_rows,
        baseline_table_paragraphs,
        "baseline",
    )
    target_candidates, target_ordinary_paragraphs = _side_candidates(
        section_pair.target_section,
        target_fragments,
        baseline_fragments,
        target_rows,
        target_table_paragraphs,
        "target",
    )
    candidates = [
        *baseline_candidates,
        *target_candidates,
    ]
    return (
        candidates,
        {"baseline": baseline_rows, "target": target_rows},
        {
            "baseline": baseline_table_paragraphs | baseline_ordinary_paragraphs,
            "target": target_table_paragraphs | target_ordinary_paragraphs,
        },
    )


def _resolve_assessment(
    assessment: CandidateAssessment,
    provider: BaseProvider | None,
    context: TableBoundaryContext | None = None,
) -> tuple[CandidateAssessment, LLMJudgment | None]:
    candidate = assessment.candidate
    requires_semantic_decision = (
        assessment.rule_confidence in {"high", "medium"}
        and not candidate.vetoes
    )
    if not requires_semantic_decision or provider is None:
        return replace(assessment, final_action="keep_separate"), None
    judgment = adjudicate_continuation(candidate, provider, context)
    final_action = (
        "merge"
        if judgment is not None
        and judgment.decision == "merge"
        and judgment.confidence >= _LLM_MERGE_THRESHOLD
        else "keep_separate"
    )
    return replace(assessment, final_action=final_action), judgment


def _downgrade_with_conflict(
    assessment: CandidateAssessment,
    conflict: str,
) -> CandidateAssessment:
    candidate = replace(
        assessment.candidate,
        conflicts=tuple(
            dict.fromkeys((*assessment.candidate.conflicts, conflict))
        ),
    )
    return replace(
        assessment,
        candidate=candidate,
        final_action="keep_separate",
    )


def _validate_resolved_assessments(
    assessments: Sequence[CandidateAssessment],
    judgments: Mapping[tuple[str, str], LLMJudgment | None],
    boundary_rows: Mapping[str, set[SourceRowRef]],
    boundary_paragraphs: Mapping[str, set[str]],
) -> list[CandidateAssessment]:
    def judgment_confidence(choice: CandidateAssessment) -> float:
        judgment = judgments.get(
            (choice.candidate.side, choice.candidate.candidate_id)
        )
        return judgment.confidence if judgment is not None else -1.0

    validated: list[CandidateAssessment] = []
    accepted_by_previous: dict[
        tuple[str, SourceRowRef],
        list[CandidateAssessment],
    ] = {}
    for assessment in assessments:
        if assessment.final_action == "merge":
            key = (
                assessment.candidate.side,
                assessment.candidate.previous_row.source,
            )
            accepted_by_previous.setdefault(key, []).append(assessment)
    ambiguous_previous_rows: set[tuple[str, SourceRowRef]] = set()
    lower_confidence_choices: set[tuple[str, str]] = set()
    for previous_key, choices in accepted_by_previous.items():
        if len(choices) <= 1:
            continue
        ranked = sorted(
            choices,
            key=lambda choice: (
                -judgment_confidence(choice),
                choice.candidate.candidate_id,
            ),
        )
        winner_judgment = judgments.get(
            (ranked[0].candidate.side, ranked[0].candidate.candidate_id)
        )
        runner_up_judgment = judgments.get(
            (ranked[1].candidate.side, ranked[1].candidate.candidate_id)
        )
        if (
            winner_judgment is None
            or runner_up_judgment is None
            or Decimal(str(winner_judgment.confidence))
            - Decimal(str(runner_up_judgment.confidence))
            < _LLM_CHOICE_MARGIN
        ):
            ambiguous_previous_rows.add(previous_key)
            continue
        lower_confidence_choices.update(
            (choice.candidate.side, choice.candidate.candidate_id)
            for choice in ranked[1:]
        )
    build_reconstruction_operations([], boundary_rows, boundary_paragraphs)
    for assessment in assessments:
        if assessment.final_action != "merge":
            validated.append(assessment)
            continue
        candidate_key = (
            assessment.candidate.side,
            assessment.candidate.previous_row.source,
        )
        decision_key = (
            assessment.candidate.side,
            assessment.candidate.candidate_id,
        )
        if decision_key in lower_confidence_choices:
            validated.append(
                _downgrade_with_conflict(
                    assessment,
                    _LOWER_CONFIDENCE_CONTINUATION_CHOICE,
                )
            )
            continue
        if candidate_key in ambiguous_previous_rows:
            validated.append(
                _downgrade_with_conflict(
                    assessment,
                    _AMBIGUOUS_CONTINUATION_CHOICES,
                )
            )
            continue
        try:
            build_reconstruction_operations(
                [*validated, assessment],
                boundary_rows,
                boundary_paragraphs,
            )
        except ValueError:
            validated.append(
                _downgrade_with_conflict(
                    assessment,
                    _UNSAFE_FRAGMENT_PROJECTION,
                )
            )
        else:
            validated.append(assessment)
    return validated


def _merge_sets(
    destination: dict[str, set],
    source: Mapping[str, set],
) -> None:
    for side in ("baseline", "target"):
        destination[side].update(source.get(side, set()))


def _decision_sort_key(decision: ReconstructionDecision) -> tuple[str, str]:
    return decision.side, decision.candidate_id


def _source_page_no(
    document: DocumentIR,
    source: SourceRowRef,
) -> int | None:
    for section in document.sections:
        if section.section_id != source.section_id:
            continue
        for paragraph in section.paragraphs:
            if paragraph.paragraph_id == source.paragraph_id:
                return paragraph.page_no
    return None


def _has_invalid_known_page_boundary(
    document: DocumentIR,
    candidate: ContinuationCandidate,
) -> bool:
    previous_page = _source_page_no(document, candidate.previous_row.source)
    continuation_page = _source_page_no(
        document,
        candidate.continuation_row.source,
    )
    return (
        previous_page is not None
        and continuation_page is not None
        and continuation_page != previous_page + 1
    )


def _trace_decisions(
    assessments: Sequence[CandidateAssessment],
    judgments: Mapping[tuple[str, str], LLMJudgment | None],
    generated_row_ids: Mapping[str, str],
    contexts: Mapping[tuple[str, str], TableBoundaryContext | None],
) -> list[ReconstructionDecision]:
    decisions = []
    for assessment in assessments:
        candidate = assessment.candidate
        context = contexts.get((candidate.side, candidate.candidate_id))
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
                boundary_id=context.boundary_id if context is not None else candidate.candidate_id,
                previous_page_no=(
                    context.previous_page_no if context is not None else None
                ),
                next_page_no=context.next_page_no if context is not None else None,
                context_refs=(
                    [item.item_id for item in context.items]
                    if context is not None
                    else []
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
    contexts: dict[tuple[str, str], TableBoundaryContext | None] = {}
    for key, candidate in sorted(candidates.items()):
        document = baseline_ir if candidate.side == "baseline" else target_ir
        context = locate_table_boundary_context(
            document,
            candidate.side,
            candidate.previous_row.source,
            candidate.continuation_row.source,
        )
        contexts[key] = context
        assessment, judgment = _resolve_assessment(
            assess_candidate(candidate),
            (
                None
                if _has_invalid_known_page_boundary(document, candidate)
                else provider
            ),
            context,
        )
        resolved.append(assessment)
        judgments[key] = judgment

    resolved = _validate_resolved_assessments(
        resolved,
        judgments,
        boundary_rows,
        boundary_paragraphs,
    )
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
        decisions=_trace_decisions(
            resolved,
            judgments,
            generated_row_ids,
            contexts,
        ),
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
