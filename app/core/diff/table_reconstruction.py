"""Deterministic structural inference for fragmented Markdown tables.

The module deliberately works from row shape and value categories.  It does not
depend on document-specific labels, row numbers, or physical column positions.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from math import exp
from statistics import median
import re
from typing import Literal, Mapping, Sequence

from app.core.diff.reconstruction_trace import (
    ALGORITHM_VERSION,
    ReconstructionOperation,
    SourceRowRef,
)
from app.core.types import DocumentIR, Paragraph, Section, Sentence


RowKind = Literal["content", "separator", "empty"]
RegionRole = Literal["body", "header", "boundary", "unknown"]


@dataclass(frozen=True)
class NormalizationConfig:
    long_text_min_length: int = 32
    separator_min_run: int = 3


@dataclass(frozen=True)
class RegionInferenceConfig:
    width_change_weight: float = 0.35
    occupancy_change_weight: float = 0.35
    type_change_weight: float = 0.20
    separator_change_weight: float = 0.10
    break_threshold: float = 0.28
    stable_pattern_similarity: float = 0.82
    unstable_predecessor_similarity: float = 0.60
    stable_pattern_min_rows: int = 2
    schema_compatibility_threshold: float = 0.70
    candidate_stability_weight: float = 0.65
    candidate_body_structure_weight: float = 0.35
    body_key_role_weight: float = 0.35
    body_column_stability_weight: float = 0.20
    body_value_diversity_weight: float = 0.20
    body_low_row_repetition_weight: float = 0.25


@dataclass(frozen=True)
class ColumnInferenceConfig:
    non_empty_weight: float = 0.35
    stable_type_weight: float = 0.20
    median_length_weight: float = 0.10
    non_repetition_weight: float = 0.10
    adjacency_weight: float = 0.15
    cross_version_role_weight: float = 0.10
    minimum_non_empty_ratio: float = 0.50
    active_score_threshold: float = 0.43
    type_similarity_weight: float = 0.45
    occupancy_similarity_weight: float = 0.15
    length_similarity_weight: float = 0.15
    repetition_similarity_weight: float = 0.10
    rank_similarity_weight: float = 0.10
    peer_role_similarity_weight: float = 0.05
    gap_penalty: float = 0.12
    mapping_compatibility_threshold: float = 0.58
    minimum_mapping_coverage: float = 0.66
    key_type_ratio_threshold: float = 0.60
    key_max_repetition_ratio: float = 0.50
    text_key_min_rows: int = 3
    text_key_min_logical_columns: int = 2
    text_key_non_empty_ratio: float = 0.90
    text_key_short_text_ratio: float = 0.90
    text_key_max_median_length: float = 24.0
    text_key_max_repetition_ratio: float = 0.10
    text_key_max_logical_rank: float = 0.25


@dataclass(frozen=True)
class BoundaryInferenceConfig:
    minimum_fragment_repetition_ratio: float = 0.67
    edge_row_count: int = 1
    width_difference_threshold: float = 0.25
    occupancy_similarity_threshold: float = 0.60
    type_similarity_threshold: float = 0.65
    within_row_repetition_threshold: float = 0.35
    minimum_signal_families: int = 4


NORMALIZATION_CONFIG = NormalizationConfig()
REGION_CONFIG = RegionInferenceConfig()
COLUMN_CONFIG = ColumnInferenceConfig()
BOUNDARY_CONFIG = BoundaryInferenceConfig()


@dataclass(frozen=True)
class TableRowMatrix:
    source: SourceRowRef
    raw_text: str
    raw_cells: tuple[str, ...]
    normalized_cells: tuple[str, ...]
    occupied: tuple[bool, ...]
    value_types: tuple[str, ...]
    kind: RowKind


@dataclass(frozen=True)
class ColumnProfile:
    physical_index: int
    non_empty_ratio: float
    type_ratios: dict[str, float]
    median_length: float
    repetition_ratio: float


@dataclass(frozen=True)
class TableRegion:
    rows: tuple[TableRowMatrix, ...]
    start_index: int
    end_index: int
    role: RegionRole


@dataclass(frozen=True)
class TableFragment:
    section_id: str
    paragraph_id: str
    paragraph_index: int
    rows: tuple[TableRowMatrix, ...]
    regions: tuple[TableRegion, ...]
    body_region_indexes: tuple[int, ...]
    active_columns: tuple[int, ...]


@dataclass(frozen=True)
class ColumnMapping:
    source_columns: tuple[int, ...]
    logical_by_physical: dict[int, int]
    score: float


EvidenceCode = Literal[
    "blank_key_cells",
    "next_row_restores_key_pattern",
    "complementary_content_cells",
    "textual_continuity",
    "boundary_artifacts_only",
    "cross_version_support",
]
VetoCode = Literal[
    "new_key_value",
    "header_or_separator",
    "incompatible_schema",
    "new_section_or_table",
    "crosses_real_body_row",
    "conflicting_key_cells",
]


@dataclass(frozen=True)
class ContinuationCandidate:
    candidate_id: str
    side: Literal["baseline", "target"]
    previous_row: TableRowMatrix
    continuation_row: TableRowMatrix
    next_full_row: TableRowMatrix | None
    mapping: ColumnMapping
    evidence: tuple[EvidenceCode, ...]
    conflicts: tuple[str, ...]
    vetoes: tuple[VetoCode, ...]
    cross_version_rows: tuple[TableRowMatrix, ...]


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: ContinuationCandidate
    rule_confidence: Literal["high", "medium", "low"]
    final_action: Literal["merge", "keep_separate", "needs_llm"]


def generate_continuation_candidates(
    left: TableFragment,
    right: TableFragment,
    mapping: ColumnMapping,
    boundary_rows: set[SourceRowRef],
    cross_version_fragments: Sequence[TableFragment],
    side: Literal["baseline", "target"],
) -> list[ContinuationCandidate]:
    left_body_rows = tuple(row for row in _body_rows(left) if row.source not in boundary_rows)
    right_body_rows = tuple(row for row in _body_rows(right) if row.source not in boundary_rows)
    if not left_body_rows or not right_body_rows:
        return []

    previous_row = left_body_rows[-1]
    continuation_row = right_body_rows[0]
    key_logical_columns, key_profile_sufficient = _key_logical_columns(
        left,
        right,
        mapping,
        left_body_rows,
        right_body_rows,
    )
    next_full_row = next(
        (
            row
            for row in right_body_rows[1:]
            if _is_complete_logical_row(row, mapping, key_logical_columns)
        ),
        None,
    )
    vetoes = _candidate_vetoes(
        left,
        right,
        previous_row,
        continuation_row,
        mapping,
        boundary_rows,
        key_logical_columns,
    )
    evidence: tuple[EvidenceCode, ...] = ()
    conflicts = () if key_profile_sufficient or vetoes else ("insufficient_key_profile",)
    cross_version_rows: tuple[TableRowMatrix, ...] = ()
    if not vetoes and not conflicts:
        evidence, cross_version_rows = _candidate_evidence(
            left,
            right,
            previous_row,
            continuation_row,
            next_full_row,
            mapping,
            boundary_rows,
            cross_version_fragments,
            key_logical_columns,
        )
    candidate = ContinuationCandidate(
        candidate_id=_candidate_id(side, previous_row.source, continuation_row.source, mapping),
        side=side,
        previous_row=previous_row,
        continuation_row=continuation_row,
        next_full_row=next_full_row,
        mapping=mapping,
        evidence=evidence,
        conflicts=conflicts,
        vetoes=vetoes,
        cross_version_rows=cross_version_rows,
    )
    return [candidate]


def assess_candidate(candidate: ContinuationCandidate) -> CandidateAssessment:
    if candidate.vetoes or candidate.conflicts:
        return CandidateAssessment(candidate, "low", "keep_separate")
    evidence_count = len(set(candidate.evidence))
    if evidence_count >= 4:
        return CandidateAssessment(candidate, "high", "merge")
    if evidence_count >= 2:
        return CandidateAssessment(candidate, "medium", "needs_llm")
    return CandidateAssessment(candidate, "low", "keep_separate")


_OPERATION_PRECEDENCE = {
    "project_columns": 0,
    "drop_boundary_rows": 1,
    "drop_boundary_paragraphs": 2,
    "merge_rows": 3,
    "merge_fragments": 4,
}


def _source_row_key(source: SourceRowRef) -> tuple[str, str, int]:
    return source.section_id, source.paragraph_id, source.sentence_index


def _operation_payload(operation: ReconstructionOperation) -> dict[str, object]:
    return {
        "side": operation.side,
        "type": operation.type,
        "source_rows": [
            [source.section_id, source.paragraph_id, source.sentence_index]
            for source in operation.source_rows
        ],
        "source_paragraph_ids": list(operation.source_paragraph_ids),
        "column_mapping": sorted(operation.column_mapping.items()),
        "decision_id": operation.decision_id,
    }


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _operation_id(operation: ReconstructionOperation) -> str:
    return f"operation-{_stable_digest(_operation_payload(operation))}"


def derived_row_id(operation: ReconstructionOperation) -> str:
    """Return the stable row ID implied by an operation's source provenance."""
    payload = {
        "side": operation.side,
        "type": operation.type,
        "source_rows": [
            [source.section_id, source.paragraph_id, source.sentence_index]
            for source in operation.source_rows
        ],
    }
    return f"reconstructed-row-{_stable_digest(payload)}"


def derived_paragraph_id(operation: ReconstructionOperation) -> str:
    """Return the stable paragraph ID implied by an operation's sources."""
    payload = {
        "side": operation.side,
        "type": operation.type,
        "source_paragraph_ids": list(operation.source_paragraph_ids),
    }
    return f"reconstructed-table-{_stable_digest(payload)}"


def _with_stable_identity(operation: ReconstructionOperation) -> ReconstructionOperation:
    operation = replace(operation, operation_id=_operation_id(operation))
    if operation.type == "merge_rows":
        operation = replace(operation, generated_row_id=derived_row_id(operation))
    if operation.type == "merge_fragments":
        operation = replace(operation, generated_paragraph_id=derived_paragraph_id(operation))
    return operation


def _operation_sort_key(operation: ReconstructionOperation) -> tuple[object, ...]:
    source_key: tuple[object, ...]
    if operation.source_rows:
        source_key = min(_source_row_key(source) for source in operation.source_rows)
    elif operation.source_paragraph_ids:
        source_key = ("", min(operation.source_paragraph_ids), -1)
    else:
        source_key = ("", "", -1)
    return (
        0 if operation.side == "baseline" else 1,
        _OPERATION_PRECEDENCE[operation.type],
        source_key,
        operation.operation_id,
    )


def build_reconstruction_operations(
    analyses: Sequence[CandidateAssessment],
    boundary_rows: Mapping[str, set[SourceRowRef]],
    boundary_paragraph_ids: Mapping[str, set[str]],
) -> list[ReconstructionOperation]:
    """Build a stable, explicit transformation trace from accepted assessments."""
    operations: list[ReconstructionOperation] = []
    for side in ("baseline", "target"):
        side_rows = sorted(boundary_rows.get(side, set()), key=_source_row_key)
        if side_rows:
            operations.append(
                ReconstructionOperation("", side, "drop_boundary_rows", side_rows)
            )
        side_paragraphs = sorted(boundary_paragraph_ids.get(side, set()))
        if side_paragraphs:
            operations.append(
                ReconstructionOperation(
                    "",
                    side,
                    "drop_boundary_paragraphs",
                    source_paragraph_ids=side_paragraphs,
                )
            )

    for assessment in analyses:
        if assessment.final_action != "merge":
            continue
        candidate = assessment.candidate
        sources = [candidate.previous_row.source, candidate.continuation_row.source]
        source_paragraph_ids = list(
            dict.fromkeys(source.paragraph_id for source in sources)
        )
        mapping = dict(sorted(candidate.mapping.logical_by_physical.items()))
        operations.extend(
            (
                ReconstructionOperation(
                    "",
                    candidate.side,
                    "project_columns",
                    [candidate.continuation_row.source],
                    column_mapping=mapping,
                    decision_id=candidate.candidate_id,
                ),
                ReconstructionOperation(
                    "",
                    candidate.side,
                    "merge_rows",
                    sources,
                    decision_id=candidate.candidate_id,
                ),
            )
        )
        if len(source_paragraph_ids) > 1:
            operations.append(
                ReconstructionOperation(
                    "",
                    candidate.side,
                    "merge_fragments",
                    source_paragraph_ids=source_paragraph_ids,
                    decision_id=candidate.candidate_id,
                )
            )

    unique: dict[str, ReconstructionOperation] = {}
    for operation in operations:
        identified = _with_stable_identity(operation)
        unique.setdefault(identified.operation_id, identified)
    return sorted(unique.values(), key=_operation_sort_key)


def _project_raw_cells(
    row: TableRowMatrix,
    mapping: Mapping[int, int],
) -> tuple[str, ...]:
    if not mapping:
        return row.raw_cells
    logical_width = max(mapping.values(), default=-1) + 1
    projected = [""] * logical_width
    assigned: set[int] = set()
    for physical_index, logical_index in sorted(mapping.items()):
        if physical_index < 0 or logical_index < 0:
            raise ValueError("column mapping indexes must be non-negative")
        if logical_index in assigned:
            raise ValueError("column mapping contains duplicate logical columns")
        assigned.add(logical_index)
        if physical_index < len(row.raw_cells):
            projected[logical_index] = row.raw_cells[physical_index]
    return tuple(projected)


_BREAK_AT_END_RE = re.compile(r"<br\s*/?>\s*$", re.IGNORECASE)
_BREAK_AT_START_RE = re.compile(r"^\s*<br\s*/?>", re.IGNORECASE)


def merge_logical_rows(
    previous: TableRowMatrix,
    continuation: TableRowMatrix,
    mapping: ColumnMapping,
    key_logical_columns: frozenset[int],
) -> tuple[str, ...]:
    """Merge raw logical cells without inventing content or punctuation."""
    previous_cells = previous.raw_cells
    continuation_cells = _project_raw_cells(
        continuation,
        mapping.logical_by_physical,
    )
    width = max(len(previous_cells), len(continuation_cells))
    result: list[str] = []
    for logical_index in range(width):
        left = previous_cells[logical_index] if logical_index < len(previous_cells) else ""
        right = (
            continuation_cells[logical_index]
            if logical_index < len(continuation_cells)
            else ""
        )
        if logical_index in key_logical_columns and left and right:
            if _normalize_cell(left).casefold() != _normalize_cell(right).casefold():
                raise ValueError(f"key column conflict at logical column {logical_index}")
            result.append(left)
        elif not left or not right:
            result.append(left or right)
        elif _BREAK_AT_END_RE.search(left) or _BREAK_AT_START_RE.search(right):
            result.append(f"{left}{right}")
        else:
            result.append(f"{left}<br>{right}")
    return tuple(result)


_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_INTEGER_RE = re.compile(r"[+-]?\d+")
_DECIMAL_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+)")
_HIERARCHICAL_NUMBER_RE = re.compile(r"\d+(?:\.\d+){2,}")
_PLACEHOLDERS = frozenset({"-", "--", "—", "–", "/", "\\", "n/a", "na", "none", "null"})


def _strip_emphasis(value: str) -> str:
    result = value.strip()
    for marker in ("**", "__", "*", "_"):
        if result.startswith(marker) and result.endswith(marker) and len(result) >= 2 * len(marker):
            result = result[len(marker) : -len(marker)].strip()
            break
    return result


def _normalize_cell(value: str) -> str:
    without_breaks = _BREAK_RE.sub(" ", value)
    return _WHITESPACE_RE.sub(" ", _strip_emphasis(without_breaks)).strip()


def _is_separator(value: str, config: NormalizationConfig = NORMALIZATION_CONFIG) -> bool:
    return re.fullmatch(rf":?[-=]{{{config.separator_min_run},}}:?", value) is not None


def _value_type(value: str, config: NormalizationConfig = NORMALIZATION_CONFIG) -> str:
    if not value:
        return "empty"
    if _is_separator(value, config):
        return "separator"
    folded = value.casefold()
    if folded in _PLACEHOLDERS:
        return "placeholder"
    compact = value.replace(",", "")
    if _HIERARCHICAL_NUMBER_RE.fullmatch(compact):
        return "hierarchical_number"
    if _INTEGER_RE.fullmatch(compact):
        return "integer"
    if _DECIMAL_RE.fullmatch(compact):
        return "decimal"
    if len(value) >= config.long_text_min_length:
        return "long_text"
    return "short_text"


def split_markdown_table_row(text: str, source: SourceRowRef) -> TableRowMatrix | None:
    """Parse a pipe-delimited row without discarding meaningful empty cells."""
    stripped = text.strip()
    if "|" not in stripped:
        return None
    parts = re.split(r"(?<!\\)\|", stripped)
    if stripped.startswith("|"):
        parts = parts[1:]
    if stripped.endswith("|"):
        parts = parts[:-1]
    if len(parts) < 2:
        return None
    raw_cells = tuple(part.strip() for part in parts)
    normalized_cells = tuple(_normalize_cell(cell) for cell in raw_cells)
    occupied = tuple(bool(cell) for cell in normalized_cells)
    value_types = tuple(_value_type(cell) for cell in normalized_cells)
    if not any(occupied):
        kind: RowKind = "empty"
    elif all(value_type in {"empty", "separator"} for value_type in value_types):
        kind = "separator"
    else:
        kind = "content"
    return TableRowMatrix(source, text, raw_cells, normalized_cells, occupied, value_types, kind)


def _type_distribution(rows: Sequence[TableRowMatrix]) -> dict[str, float]:
    counts = Counter(
        value_type
        for row in rows
        for value_type, occupied in zip(row.value_types, row.occupied)
        if occupied
    )
    total = sum(counts.values())
    if not total:
        return {}
    return {value_type: count / total for value_type, count in counts.items()}


def _distribution_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left and not right:
        return 1.0
    keys = left.keys() | right.keys()
    return max(0.0, 1.0 - sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys) / 2.0)


def _occupied_indexes(row: TableRowMatrix) -> set[int]:
    return {index for index, occupied in enumerate(row.occupied) if occupied}


def _set_similarity(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _row_shape_similarity(left: TableRowMatrix, right: TableRowMatrix) -> float:
    width_similarity = 1.0 - abs(len(left.raw_cells) - len(right.raw_cells)) / max(
        len(left.raw_cells), len(right.raw_cells), 1
    )
    occupancy_similarity = _set_similarity(_occupied_indexes(left), _occupied_indexes(right))
    type_similarity = _distribution_similarity(_type_distribution((left,)), _type_distribution((right,)))
    return 0.35 * width_similarity + 0.40 * occupancy_similarity + 0.25 * type_similarity


def _region_break_score(
    left: TableRowMatrix,
    right: TableRowMatrix,
    config: RegionInferenceConfig,
) -> float:
    max_width = max(len(left.raw_cells), len(right.raw_cells), 1)
    width_change = abs(len(left.raw_cells) - len(right.raw_cells)) / max_width
    occupancy_change = 1.0 - _set_similarity(_occupied_indexes(left), _occupied_indexes(right))
    type_change = 1.0 - _distribution_similarity(
        _type_distribution((left,)), _type_distribution((right,))
    )
    separator_change = float(left.kind == "separator" or right.kind == "separator")
    return (
        config.width_change_weight * width_change
        + config.occupancy_change_weight * occupancy_change
        + config.type_change_weight * type_change
        + config.separator_change_weight * separator_change
    )


def _segment_rows(
    rows: tuple[TableRowMatrix, ...],
    config: RegionInferenceConfig,
) -> list[tuple[int, int]]:
    if not rows:
        return []
    breaks = [0]
    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]
        starts_stable_pattern = (
            index + 1 < len(rows)
            and _row_shape_similarity(current, rows[index + 1]) >= config.stable_pattern_similarity
            and _row_shape_similarity(previous, current) < config.unstable_predecessor_similarity
        )
        if (
            previous.kind == "separator"
            or current.kind == "separator"
            or starts_stable_pattern
            or _region_break_score(previous, current, config) >= config.break_threshold
        ):
            breaks.append(index)
    breaks.append(len(rows))
    return [(start, end) for start, end in zip(breaks, breaks[1:]) if start < end]


def _segment_stability(rows: Sequence[TableRowMatrix]) -> float:
    if len(rows) < 2:
        return 0.0
    return sum(_row_shape_similarity(left, right) for left, right in zip(rows, rows[1:])) / (
        len(rows) - 1
    )


def _body_structure_score(
    rows: Sequence[TableRowMatrix],
    config: RegionInferenceConfig,
) -> float:
    if not rows:
        return 0.0
    width = max(len(row.raw_cells) for row in rows)
    key_role_strength = 0.0
    column_stabilities: list[float] = []
    column_diversities: list[float] = []
    for physical_index in range(width):
        values = [
            row.normalized_cells[physical_index]
            for row in rows
            if physical_index < len(row.normalized_cells) and row.occupied[physical_index]
        ]
        value_types = [
            row.value_types[physical_index]
            for row in rows
            if physical_index < len(row.value_types) and row.occupied[physical_index]
        ]
        if not values:
            continue
        type_counts = Counter(value_types)
        key_type_ratio = (
            type_counts.get("integer", 0) + type_counts.get("hierarchical_number", 0)
        ) / len(value_types)
        distinct_ratio = len(set(values)) / len(values)
        key_role_strength = max(key_role_strength, key_type_ratio * distinct_ratio)
        column_stabilities.append(max(type_counts.values()) / len(value_types))
        column_diversities.append(distinct_ratio)

    row_repetition_ratios = []
    for row in rows:
        values = [cell for cell in row.normalized_cells if cell]
        row_repetition_ratios.append(1.0 - len(set(values)) / len(values) if values else 0.0)
    column_stability = sum(column_stabilities) / len(column_stabilities) if column_stabilities else 0.0
    value_diversity = sum(column_diversities) / len(column_diversities) if column_diversities else 0.0
    low_row_repetition = 1.0 - sum(row_repetition_ratios) / len(row_repetition_ratios)
    return (
        config.body_key_role_weight * key_role_strength
        + config.body_column_stability_weight * column_stability
        + config.body_value_diversity_weight * value_diversity
        + config.body_low_row_repetition_weight * low_row_repetition
    )


def _segment_schema_similarity(
    rows: Sequence[TableRowMatrix], body_rows: Sequence[TableRowMatrix]
) -> float:
    row_width = median([len(row.raw_cells) for row in rows]) if rows else 0.0
    body_width = median([len(row.raw_cells) for row in body_rows]) if body_rows else 0.0
    width_similarity = min(row_width, body_width) / max(row_width, body_width, 1.0)
    occupancy_similarity = _set_similarity(
        set().union(*(_occupied_indexes(row) for row in rows)),
        set().union(*(_occupied_indexes(row) for row in body_rows)),
    )
    type_similarity = _distribution_similarity(_type_distribution(rows), _type_distribution(body_rows))
    return 0.35 * width_similarity + 0.40 * occupancy_similarity + 0.25 * type_similarity


def infer_regions(
    fragment: TableFragment,
    peer_fragments: Sequence[TableFragment],
) -> tuple[TableRegion, ...]:
    """Segment a fragment and identify its stable body-shaped region(s)."""
    del peer_fragments  # reserved for Task 3 evidence; Task 2 remains deterministic and local
    segments = _segment_rows(fragment.rows, REGION_CONFIG)
    if not segments:
        return ()

    candidates: list[tuple[float, int]] = []
    for segment_index, (start, end) in enumerate(segments):
        rows = fragment.rows[start:end]
        if (
            len(rows) >= REGION_CONFIG.stable_pattern_min_rows
            and all(row.kind == "content" for row in rows)
        ):
            evidence_score = len(rows) * (
                REGION_CONFIG.candidate_stability_weight * _segment_stability(rows)
                + REGION_CONFIG.candidate_body_structure_weight
                * _body_structure_score(rows, REGION_CONFIG)
            )
            candidates.append((evidence_score, -segment_index))

    body_segment_index = -max(candidates)[1] if candidates else None
    body_rows = (
        fragment.rows[segments[body_segment_index][0] : segments[body_segment_index][1]]
        if body_segment_index is not None
        else ()
    )
    regions: list[TableRegion] = []
    for segment_index, (start, end) in enumerate(segments):
        rows = fragment.rows[start:end]
        at_edge = start == 0 or end == len(fragment.rows)
        if any(row.kind == "separator" for row in rows):
            role: RegionRole = "boundary"
        elif body_segment_index is not None and (
            segment_index == body_segment_index
            or (
                len(rows) >= REGION_CONFIG.stable_pattern_min_rows
                and _segment_schema_similarity(rows, body_rows)
                >= REGION_CONFIG.schema_compatibility_threshold
            )
        ):
            role = "body"
        elif body_rows and at_edge and _segment_schema_similarity(rows, body_rows) < REGION_CONFIG.schema_compatibility_threshold:
            role = "boundary"
        elif body_rows and end <= segments[body_segment_index][0]:
            role = "header"
        else:
            role = "unknown"
        regions.append(TableRegion(rows, start, end, role))
    return tuple(regions)


def _body_rows(fragment: TableFragment) -> tuple[TableRowMatrix, ...]:
    indexed_rows = tuple(
        row
        for index in fragment.body_region_indexes
        if 0 <= index < len(fragment.regions)
        for row in fragment.regions[index].rows
    )
    if indexed_rows:
        return indexed_rows
    role_rows = tuple(row for region in fragment.regions if region.role == "body" for row in region.rows)
    if role_rows:
        return role_rows
    return tuple(row for row in fragment.rows if row.kind == "content")


def _column_profiles(rows: Sequence[TableRowMatrix]) -> dict[int, ColumnProfile]:
    if not rows:
        return {}
    width = max((len(row.raw_cells) for row in rows), default=0)
    profiles: dict[int, ColumnProfile] = {}
    for physical_index in range(width):
        values = [
            row.normalized_cells[physical_index]
            for row in rows
            if physical_index < len(row.normalized_cells) and row.occupied[physical_index]
        ]
        types = [
            row.value_types[physical_index]
            for row in rows
            if physical_index < len(row.value_types) and row.occupied[physical_index]
        ]
        counts = Counter(types)
        profiles[physical_index] = ColumnProfile(
            physical_index=physical_index,
            non_empty_ratio=len(values) / len(rows),
            type_ratios={value_type: count / len(types) for value_type, count in counts.items()} if types else {},
            median_length=float(median(map(len, values))) if values else 0.0,
            repetition_ratio=(1.0 - len(set(values)) / len(values)) if values else 0.0,
        )
    return profiles


def _adjacency_cooccurrence(rows: Sequence[TableRowMatrix], physical_index: int) -> float:
    if not rows:
        return 0.0
    occurrences = 0
    for row in rows:
        if physical_index >= len(row.occupied) or not row.occupied[physical_index]:
            continue
        neighbor_indexes = (physical_index - 1, physical_index + 1)
        if any(0 <= neighbor < len(row.occupied) and row.occupied[neighbor] for neighbor in neighbor_indexes):
            occurrences += 1
    return occurrences / len(rows)


def _peer_role_similarity(
    profile: ColumnProfile,
    peer_fragments: Sequence[TableFragment],
) -> float:
    similarities = [
        _distribution_similarity(profile.type_ratios, peer_profile.type_ratios)
        for peer in peer_fragments
        for peer_profile in _column_profiles(_body_rows(peer)).values()
        if peer_profile.non_empty_ratio >= COLUMN_CONFIG.minimum_non_empty_ratio
    ]
    return max(similarities, default=0.5)


def infer_active_columns(
    fragment: TableFragment,
    peer_fragments: Sequence[TableFragment],
) -> tuple[int, ...]:
    rows = _body_rows(fragment)
    profiles = _column_profiles(rows)
    active: list[int] = []
    for physical_index, profile in profiles.items():
        stable_type_role = max(profile.type_ratios.values(), default=0.0)
        normalized_length = 1.0 - exp(-profile.median_length / 8.0)
        score = (
            COLUMN_CONFIG.non_empty_weight * profile.non_empty_ratio
            + COLUMN_CONFIG.stable_type_weight * stable_type_role
            + COLUMN_CONFIG.median_length_weight * normalized_length
            + COLUMN_CONFIG.non_repetition_weight * (1.0 - profile.repetition_ratio)
            + COLUMN_CONFIG.adjacency_weight * _adjacency_cooccurrence(rows, physical_index)
            + COLUMN_CONFIG.cross_version_role_weight * _peer_role_similarity(profile, peer_fragments)
        )
        if (
            profile.non_empty_ratio >= COLUMN_CONFIG.minimum_non_empty_ratio
            and score >= COLUMN_CONFIG.active_score_threshold
        ):
            active.append(physical_index)
    return tuple(active)


def collect_table_fragments(section: Section) -> list[TableFragment]:
    """Collect paragraph-local tables, then infer regions and active columns."""
    fragments: list[TableFragment] = []
    for paragraph_index, paragraph in enumerate(section.paragraphs):
        texts = [sentence.text for sentence in paragraph.sentences]
        if not texts:
            texts = paragraph.text.splitlines()
        rows = tuple(
            row
            for sentence_index, text in enumerate(texts)
            if (
                row := split_markdown_table_row(
                    text,
                    SourceRowRef(section.section_id, paragraph.paragraph_id, sentence_index),
                )
            )
            is not None
        )
        if rows:
            fragments.append(
                TableFragment(section.section_id, paragraph.paragraph_id, paragraph_index, rows, (), (), ())
            )

    with_regions = [
        replace(fragment, regions=infer_regions(fragment, tuple(peer for peer in fragments if peer is not fragment)))
        for fragment in fragments
    ]
    with_region_indexes = [
        replace(
            fragment,
            body_region_indexes=tuple(
                index for index, region in enumerate(fragment.regions) if region.role == "body"
            ),
        )
        for fragment in with_regions
    ]
    return [
        replace(
            fragment,
            active_columns=infer_active_columns(
                fragment,
                tuple(peer for peer in with_region_indexes if peer is not fragment),
            ),
        )
        for fragment in with_region_indexes
    ]


def _length_similarity(left: float, right: float) -> float:
    if left == 0.0 and right == 0.0:
        return 1.0
    return min(left, right) / max(left, right)


def _profile_similarity(
    left: ColumnProfile,
    right: ColumnProfile,
    left_rank: float,
    right_rank: float,
    peer_similarity: float,
) -> float:
    return (
        COLUMN_CONFIG.type_similarity_weight
        * _distribution_similarity(left.type_ratios, right.type_ratios)
        + COLUMN_CONFIG.occupancy_similarity_weight
        * (1.0 - abs(left.non_empty_ratio - right.non_empty_ratio))
        + COLUMN_CONFIG.length_similarity_weight
        * _length_similarity(left.median_length, right.median_length)
        + COLUMN_CONFIG.repetition_similarity_weight
        * (1.0 - abs(left.repetition_ratio - right.repetition_ratio))
        + COLUMN_CONFIG.rank_similarity_weight * (1.0 - abs(left_rank - right_rank))
        + COLUMN_CONFIG.peer_role_similarity_weight * peer_similarity
    )


def _is_key_profile(profile: ColumnProfile) -> bool:
    key_type_ratio = profile.type_ratios.get("integer", 0.0) + profile.type_ratios.get(
        "hierarchical_number", 0.0
    )
    return (
        key_type_ratio >= COLUMN_CONFIG.key_type_ratio_threshold
        and profile.non_empty_ratio >= COLUMN_CONFIG.minimum_non_empty_ratio
        and profile.repetition_ratio <= COLUMN_CONFIG.key_max_repetition_ratio
    )


def _mapping_peer_similarity(
    left_profile: ColumnProfile,
    right_profile: ColumnProfile,
    cross_version_fragments: Sequence[TableFragment],
) -> float:
    peer_profiles = [
        profile
        for fragment in cross_version_fragments
        for profile in _column_profiles(_body_rows(fragment)).values()
        if profile.non_empty_ratio >= COLUMN_CONFIG.minimum_non_empty_ratio
    ]
    if not peer_profiles:
        return _distribution_similarity(left_profile.type_ratios, right_profile.type_ratios)
    return max(
        (
            _distribution_similarity(left_profile.type_ratios, peer.type_ratios)
            + _distribution_similarity(right_profile.type_ratios, peer.type_ratios)
        )
        / 2.0
        for peer in peer_profiles
    )


def infer_monotonic_column_mapping(
    left: TableFragment,
    right: TableFragment,
    cross_version_fragments: Sequence[TableFragment],
) -> ColumnMapping | None:
    """Align active columns using maximum-scoring order-preserving DP."""
    left_columns = left.active_columns or infer_active_columns(left, cross_version_fragments)
    right_columns = right.active_columns or infer_active_columns(right, cross_version_fragments)
    if not left_columns or not right_columns:
        return None
    left_profiles = _column_profiles(_body_rows(left))
    right_profiles = _column_profiles(_body_rows(right))
    if any(column not in left_profiles for column in left_columns) or any(
        column not in right_profiles for column in right_columns
    ):
        return None

    left_has_key = any(_is_key_profile(left_profiles[column]) for column in left_columns)
    right_has_key = any(_is_key_profile(right_profiles[column]) for column in right_columns)
    if left_has_key != right_has_key:
        return None

    left_count = len(left_columns)
    right_count = len(right_columns)
    scores = [[float("-inf")] * (right_count + 1) for _ in range(left_count + 1)]
    pairs: list[list[tuple[tuple[int, int], ...]]] = [
        [() for _ in range(right_count + 1)] for _ in range(left_count + 1)
    ]
    scores[0][0] = 0.0
    for left_index in range(left_count + 1):
        for right_index in range(right_count + 1):
            current_score = scores[left_index][right_index]
            if current_score == float("-inf"):
                continue
            if left_index < left_count:
                skipped = current_score - COLUMN_CONFIG.gap_penalty
                if skipped > scores[left_index + 1][right_index]:
                    scores[left_index + 1][right_index] = skipped
                    pairs[left_index + 1][right_index] = pairs[left_index][right_index]
            if right_index < right_count:
                skipped = current_score - COLUMN_CONFIG.gap_penalty
                if skipped > scores[left_index][right_index + 1]:
                    scores[left_index][right_index + 1] = skipped
                    pairs[left_index][right_index + 1] = pairs[left_index][right_index]
            if left_index < left_count and right_index < right_count:
                left_column = left_columns[left_index]
                right_column = right_columns[right_index]
                left_profile = left_profiles[left_column]
                right_profile = right_profiles[right_column]
                if _is_key_profile(left_profile) != _is_key_profile(right_profile):
                    continue
                left_rank = left_index / max(left_count - 1, 1)
                right_rank = right_index / max(right_count - 1, 1)
                match_score = _profile_similarity(
                    left_profile,
                    right_profile,
                    left_rank,
                    right_rank,
                    _mapping_peer_similarity(
                        left_profile, right_profile, cross_version_fragments
                    ),
                )
                matched = current_score + match_score
                if matched > scores[left_index + 1][right_index + 1]:
                    scores[left_index + 1][right_index + 1] = matched
                    pairs[left_index + 1][right_index + 1] = pairs[left_index][right_index] + (
                        (left_index, right_index),
                    )

    aligned_pairs = pairs[left_count][right_count]
    key_pair_aligned = any(
        _is_key_profile(left_profiles[left_columns[left_index]])
        and _is_key_profile(right_profiles[right_columns[right_index]])
        for left_index, right_index in aligned_pairs
    )
    if left_has_key and right_has_key and not key_pair_aligned:
        return None
    coverage = len(aligned_pairs) / max(left_count, right_count)
    normalized_score = max(0.0, scores[left_count][right_count] / max(left_count, right_count))
    if (
        coverage < COLUMN_CONFIG.minimum_mapping_coverage
        or normalized_score < COLUMN_CONFIG.mapping_compatibility_threshold
    ):
        return None
    logical_by_physical = {
        right_columns[right_index]: left_index for left_index, right_index in aligned_pairs
    }
    return ColumnMapping(tuple(logical_by_physical), logical_by_physical, normalized_score)


def _row_fingerprint(row: TableRowMatrix) -> tuple[str, ...]:
    return row.normalized_cells


def _region_fingerprint(region: TableRegion) -> tuple[tuple[str, ...], ...]:
    return tuple(_row_fingerprint(row) for row in region.rows)


def _within_row_repetition_ratio(row: TableRowMatrix) -> float:
    values = [cell for cell in row.normalized_cells if cell]
    if not values:
        return 0.0
    return 1.0 - len(set(values)) / len(values)


def _body_schema_incompatible(row: TableRowMatrix, fragment: TableFragment) -> bool:
    body_rows = _body_rows(fragment)
    if not body_rows:
        return False
    body_width = float(median([len(body_row.raw_cells) for body_row in body_rows]))
    width_difference = abs(len(row.raw_cells) - body_width) / max(len(row.raw_cells), body_width, 1.0)
    active = set(fragment.active_columns) or set().union(*(_occupied_indexes(body_row) for body_row in body_rows))
    occupancy_similarity = _set_similarity(_occupied_indexes(row), active)
    type_similarity = _distribution_similarity(_type_distribution((row,)), _type_distribution(body_rows))
    shape_incompatible = (
        width_difference >= BOUNDARY_CONFIG.width_difference_threshold
        or occupancy_similarity < BOUNDARY_CONFIG.occupancy_similarity_threshold
    )
    return shape_incompatible and type_similarity < BOUNDARY_CONFIG.type_similarity_threshold


def _key_discontinuity(row: TableRowMatrix, fragment: TableFragment) -> bool:
    body_profiles = _column_profiles(_body_rows(fragment))
    key_columns = [
        column
        for column in fragment.active_columns
        if column in body_profiles and _is_key_profile(body_profiles[column])
    ]
    if not key_columns:
        return False
    key_column = key_columns[0]
    if key_column >= len(row.value_types):
        return True
    return row.value_types[key_column] not in {"integer", "hierarchical_number"}


def classify_repeated_boundary_regions(
    fragments: Sequence[TableFragment],
) -> set[SourceRowRef]:
    """Return every row in repeated edge regions when independent signals agree."""
    if not fragments:
        return set()
    edge_regions_by_fragment: list[tuple[TableRegion, ...]] = []
    fingerprint_fragments: dict[tuple[tuple[str, ...], ...], set[int]] = {}
    for fragment_index, fragment in enumerate(fragments):
        edge_regions = tuple(
            region
            for region in fragment.regions
            if region.role == "boundary"
            and (
                region.start_index < BOUNDARY_CONFIG.edge_row_count
                or region.end_index > len(fragment.rows) - BOUNDARY_CONFIG.edge_row_count
            )
        )
        edge_regions_by_fragment.append(edge_regions)
        for region in edge_regions:
            fingerprint_fragments.setdefault(_region_fingerprint(region), set()).add(fragment_index)

    minimum_fragments = max(
        2,
        int(len(fragments) * BOUNDARY_CONFIG.minimum_fragment_repetition_ratio + 0.999999),
    )
    repeated_fingerprints = {
        fingerprint
        for fingerprint, fragment_indexes in fingerprint_fragments.items()
        if len(fragment_indexes) >= minimum_fragments
    }
    dropped: set[SourceRowRef] = set()
    for fragment, edge_regions in zip(fragments, edge_regions_by_fragment):
        for region in edge_regions:
            repeated = _region_fingerprint(region) in repeated_fingerprints
            at_boundary = (
                region.start_index < BOUNDARY_CONFIG.edge_row_count
                or region.end_index > len(fragment.rows) - BOUNDARY_CONFIG.edge_row_count
            )
            schema_incompatible = bool(region.rows) and all(
                _body_schema_incompatible(row, fragment) for row in region.rows
            )
            key_discontinuous = bool(region.rows) and all(
                _key_discontinuity(row, fragment) for row in region.rows
            )
            mean_repetition = (
                sum(_within_row_repetition_ratio(row) for row in region.rows) / len(region.rows)
                if region.rows
                else 0.0
            )
            abnormal_repetition = mean_repetition >= BOUNDARY_CONFIG.within_row_repetition_threshold
            signal_count = sum(
                (repeated, at_boundary, schema_incompatible, key_discontinuous, abnormal_repetition)
            )
            if (
                repeated
                and at_boundary
                and schema_incompatible
                and signal_count >= BOUNDARY_CONFIG.minimum_signal_families
            ):
                dropped.update(row.source for row in region.rows)
    return dropped


def _candidate_id(
    side: Literal["baseline", "target"],
    previous: SourceRowRef,
    continuation: SourceRowRef,
    mapping: ColumnMapping,
) -> str:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "side": side,
        "previous": (previous.section_id, previous.paragraph_id, previous.sentence_index),
        "continuation": (
            continuation.section_id,
            continuation.paragraph_id,
            continuation.sentence_index,
        ),
        "mapping": sorted(mapping.logical_by_physical.items()),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"candidate-{digest[:24]}"


def _active_columns(fragment: TableFragment) -> tuple[int, ...]:
    return fragment.active_columns or infer_active_columns(fragment, ())


def _key_logical_columns(
    left: TableFragment,
    right: TableFragment,
    mapping: ColumnMapping,
    left_body_rows: Sequence[TableRowMatrix],
    right_body_rows: Sequence[TableRowMatrix],
) -> tuple[frozenset[int], bool]:
    left_profiles = _column_profiles(left_body_rows)
    active_columns = _active_columns(left)
    numeric_keys = frozenset(
        logical_index
        for logical_index, physical_index in enumerate(active_columns)
        if physical_index in left_profiles and _is_key_profile(left_profiles[physical_index])
    )
    if numeric_keys:
        return numeric_keys, True
    if _mapping_is_incompatible(left, right, mapping):
        return frozenset(), False

    logical_columns = frozenset(mapping.logical_by_physical.values())
    pooled_rows = _pooled_complete_logical_rows(
        left,
        mapping,
        left_body_rows,
        right_body_rows[1:],
        logical_columns,
    )
    profile_sufficient = len(pooled_rows) >= COLUMN_CONFIG.text_key_min_rows
    if not profile_sufficient:
        return frozenset(), False
    profiles = _logical_column_profiles(pooled_rows, logical_columns)
    textual_keys = frozenset(
        logical_index
        for logical_index in logical_columns
        if logical_index in profiles
        and _is_textual_key_profile(
            profiles[logical_index],
            row_count=len(pooled_rows),
            logical_column_count=len(logical_columns),
            logical_rank=logical_index / max(len(logical_columns) - 1, 1),
        )
    )
    return textual_keys, True


def _pooled_complete_logical_rows(
    left: TableFragment,
    mapping: ColumnMapping,
    left_rows: Sequence[TableRowMatrix],
    right_context_rows: Sequence[TableRowMatrix],
    logical_columns: frozenset[int],
) -> tuple[dict[int, str], ...]:
    projected_rows = (
        *(_left_logical_cells(row, left) for row in left_rows if row.kind == "content"),
        *(
            _mapped_logical_cells(row, mapping)
            for row in right_context_rows
            if row.kind == "content"
        ),
    )
    return tuple(
        cells
        for cells in projected_rows
        if logical_columns and all(cells.get(column, "") for column in logical_columns)
    )


def _logical_column_profiles(
    rows: Sequence[dict[int, str]],
    logical_columns: frozenset[int],
) -> dict[int, ColumnProfile]:
    profiles: dict[int, ColumnProfile] = {}
    for logical_index in logical_columns:
        values = [cells.get(logical_index, "") for cells in rows]
        occupied_values = [value for value in values if value]
        value_types = [_value_type(value) for value in occupied_values]
        type_counts = Counter(value_types)
        profiles[logical_index] = ColumnProfile(
            physical_index=logical_index,
            non_empty_ratio=len(occupied_values) / len(rows) if rows else 0.0,
            type_ratios={
                value_type: count / len(value_types)
                for value_type, count in type_counts.items()
            }
            if value_types
            else {},
            median_length=float(median(map(len, occupied_values))) if occupied_values else 0.0,
            repetition_ratio=(
                1.0 - len(set(occupied_values)) / len(occupied_values)
                if occupied_values
                else 0.0
            ),
        )
    return profiles


def _is_textual_key_profile(
    profile: ColumnProfile,
    *,
    row_count: int,
    logical_column_count: int,
    logical_rank: float,
) -> bool:
    return (
        row_count >= COLUMN_CONFIG.text_key_min_rows
        and logical_column_count >= COLUMN_CONFIG.text_key_min_logical_columns
        and logical_rank <= COLUMN_CONFIG.text_key_max_logical_rank
        and profile.non_empty_ratio >= COLUMN_CONFIG.text_key_non_empty_ratio
        and profile.type_ratios.get("short_text", 0.0)
        >= COLUMN_CONFIG.text_key_short_text_ratio
        and 0.0 < profile.median_length <= COLUMN_CONFIG.text_key_max_median_length
        and profile.repetition_ratio <= COLUMN_CONFIG.text_key_max_repetition_ratio
    )


def _left_logical_cells(
    row: TableRowMatrix,
    fragment: TableFragment,
) -> dict[int, str]:
    return {
        logical_index: row.normalized_cells[physical_index]
        if physical_index < len(row.normalized_cells)
        else ""
        for logical_index, physical_index in enumerate(_active_columns(fragment))
    }


def _left_logical_types(
    row: TableRowMatrix,
    fragment: TableFragment,
) -> dict[int, str]:
    return {
        logical_index: row.value_types[physical_index]
        if physical_index < len(row.value_types)
        else "empty"
        for logical_index, physical_index in enumerate(_active_columns(fragment))
    }


def _mapped_logical_cells(
    row: TableRowMatrix,
    mapping: ColumnMapping,
) -> dict[int, str]:
    return {
        logical_index: row.normalized_cells[physical_index]
        if physical_index < len(row.normalized_cells)
        else ""
        for physical_index, logical_index in mapping.logical_by_physical.items()
    }


def _mapped_logical_types(
    row: TableRowMatrix,
    mapping: ColumnMapping,
) -> dict[int, str]:
    return {
        logical_index: row.value_types[physical_index]
        if physical_index < len(row.value_types)
        else "empty"
        for physical_index, logical_index in mapping.logical_by_physical.items()
    }


def _is_complete_logical_row(
    row: TableRowMatrix,
    mapping: ColumnMapping,
    key_logical_columns: frozenset[int],
) -> bool:
    cells = _mapped_logical_cells(row, mapping)
    if key_logical_columns:
        return all(cells.get(column, "") for column in key_logical_columns) and any(
            value for column, value in cells.items() if column not in key_logical_columns
        )
    occupied_count = sum(bool(value) for value in cells.values())
    minimum_occupied = max(1, (2 * len(cells) + 2) // 3)
    return occupied_count >= minimum_occupied


def _row_position(fragment: TableFragment, source: SourceRowRef) -> int:
    return next(
        (index for index, row in enumerate(fragment.rows) if row.source == source),
        -1,
    )


def _intervening_rows(
    left: TableFragment,
    right: TableFragment,
    previous: TableRowMatrix,
    continuation: TableRowMatrix,
) -> tuple[TableRowMatrix, ...]:
    left_index = _row_position(left, previous.source)
    right_index = _row_position(right, continuation.source)
    trailing = left.rows[left_index + 1 :] if left_index >= 0 else ()
    leading = right.rows[:right_index] if right_index >= 0 else ()
    return trailing + leading


def _row_role(fragment: TableFragment, source: SourceRowRef) -> RegionRole:
    return next(
        (
            region.role
            for region in fragment.regions
            if any(row.source == source for row in region.rows)
        ),
        "unknown",
    )


def _mapping_is_incompatible(
    left: TableFragment,
    right: TableFragment,
    mapping: ColumnMapping,
) -> bool:
    left_columns = _active_columns(left)
    right_columns = _active_columns(right)
    mapped_columns = set(mapping.logical_by_physical)
    coverage = len(mapped_columns & set(right_columns)) / max(
        len(left_columns), len(right_columns), 1
    )
    logical_indexes = tuple(mapping.logical_by_physical.values())
    return (
        mapping.score < COLUMN_CONFIG.mapping_compatibility_threshold
        or coverage < COLUMN_CONFIG.minimum_mapping_coverage
        or len(set(logical_indexes)) != len(logical_indexes)
        or any(index < 0 or index >= len(left_columns) for index in logical_indexes)
    )


def _candidate_vetoes(
    left: TableFragment,
    right: TableFragment,
    previous: TableRowMatrix,
    continuation: TableRowMatrix,
    mapping: ColumnMapping,
    boundary_rows: set[SourceRowRef],
    key_logical_columns: frozenset[int],
) -> tuple[VetoCode, ...]:
    previous_cells = _left_logical_cells(previous, left)
    continuation_cells = _mapped_logical_cells(continuation, mapping)
    intervening = _intervening_rows(left, right, previous, continuation)
    vetoes: list[VetoCode] = []
    if any(continuation_cells.get(column, "") for column in key_logical_columns):
        vetoes.append("new_key_value")
    if (
        continuation.kind != "content"
        or continuation.source in boundary_rows
        or _row_role(right, continuation.source) in {"header", "boundary"}
    ):
        vetoes.append("header_or_separator")
    if _mapping_is_incompatible(left, right, mapping):
        vetoes.append("incompatible_schema")
    if (
        left.section_id != right.section_id
        or right.paragraph_index != left.paragraph_index + 1
    ):
        vetoes.append("new_section_or_table")
    if any(row.kind == "content" and row.source not in boundary_rows for row in intervening):
        vetoes.append("crosses_real_body_row")
    if any(
        previous_cells.get(column, "")
        and continuation_cells.get(column, "")
        and previous_cells[column].casefold() != continuation_cells[column].casefold()
        for column in key_logical_columns
    ):
        vetoes.append("conflicting_key_cells")
    return tuple(vetoes)


def _blank_key_cells_evidence(
    continuation_cells: dict[int, str],
    key_logical_columns: frozenset[int],
) -> bool:
    return bool(key_logical_columns) and all(
        not continuation_cells.get(column, "") for column in key_logical_columns
    )


def _key_type_family(value_type: str) -> str:
    if value_type in {"integer", "hierarchical_number"}:
        return "numbered_key"
    return value_type


def _next_row_restores_key_pattern_evidence(
    previous: TableRowMatrix,
    next_full_row: TableRowMatrix | None,
    left: TableFragment,
    mapping: ColumnMapping,
    key_logical_columns: frozenset[int],
) -> bool:
    if next_full_row is None or not key_logical_columns:
        return False
    previous_types = _left_logical_types(previous, left)
    next_types = _mapped_logical_types(next_full_row, mapping)
    return all(
        _key_type_family(previous_types.get(column, "empty"))
        == _key_type_family(next_types.get(column, "empty"))
        != "empty"
        for column in key_logical_columns
    )


def _complementary_content_cells_evidence(
    previous_cells: dict[int, str],
    continuation_cells: dict[int, str],
    key_logical_columns: frozenset[int],
) -> bool:
    content_columns = (previous_cells.keys() | continuation_cells.keys()) - key_logical_columns
    previous_only = any(
        previous_cells.get(column, "") and not continuation_cells.get(column, "")
        for column in content_columns
    )
    continuation_only = any(
        continuation_cells.get(column, "") and not previous_cells.get(column, "")
        for column in content_columns
    )
    return previous_only and continuation_only


_TERMINAL_PUNCTUATION = frozenset(".!?;:。！？；：")


def _starts_like_continuation(value: str) -> bool:
    if not value:
        return False
    first = value[0]
    return first.islower() or "\u3400" <= first <= "\u9fff"


def _textual_continuity_evidence(
    previous_cells: dict[int, str],
    continuation_cells: dict[int, str],
    key_logical_columns: frozenset[int],
) -> bool:
    return any(
        previous_cells.get(column, "")
        and continuation_cells.get(column, "")
        and previous_cells[column][-1] not in _TERMINAL_PUNCTUATION
        and _starts_like_continuation(continuation_cells[column])
        for column in (previous_cells.keys() | continuation_cells.keys()) - key_logical_columns
    )


def _boundary_artifacts_only_evidence(
    left: TableFragment,
    right: TableFragment,
    previous: TableRowMatrix,
    continuation: TableRowMatrix,
    boundary_rows: set[SourceRowRef],
) -> bool:
    if (
        left.section_id != right.section_id
        or right.paragraph_index != left.paragraph_index + 1
    ):
        return False
    intervening = _intervening_rows(left, right, previous, continuation)
    return all(row.source in boundary_rows for row in intervening)


def _candidate_cell_options(previous: str, continuation: str) -> frozenset[str]:
    if previous and continuation:
        return frozenset(
            {
                f"{previous}{continuation}".casefold(),
                f"{previous} {continuation}".casefold(),
            }
        )
    return frozenset({(previous or continuation).casefold()})


def _cross_version_support_rows(
    left: TableFragment,
    previous: TableRowMatrix,
    continuation: TableRowMatrix,
    mapping: ColumnMapping,
    cross_version_fragments: Sequence[TableFragment],
) -> tuple[TableRowMatrix, ...]:
    previous_cells = _left_logical_cells(previous, left)
    continuation_cells = _mapped_logical_cells(continuation, mapping)
    logical_columns = previous_cells.keys() | continuation_cells.keys()
    matches: list[TableRowMatrix] = []
    for fragment in cross_version_fragments:
        peer_mapping = infer_monotonic_column_mapping(left, fragment, ())
        if peer_mapping is None:
            continue
        for row in _body_rows(fragment):
            peer_cells = _mapped_logical_cells(row, peer_mapping)
            if all(
                peer_cells.get(column, "").casefold()
                in _candidate_cell_options(
                    previous_cells.get(column, ""),
                    continuation_cells.get(column, ""),
                )
                for column in logical_columns
            ):
                matches.append(row)
                if len(matches) == 3:
                    return tuple(matches)
    return tuple(matches)


def _candidate_evidence(
    left: TableFragment,
    right: TableFragment,
    previous: TableRowMatrix,
    continuation: TableRowMatrix,
    next_full_row: TableRowMatrix | None,
    mapping: ColumnMapping,
    boundary_rows: set[SourceRowRef],
    cross_version_fragments: Sequence[TableFragment],
    key_logical_columns: frozenset[int],
) -> tuple[tuple[EvidenceCode, ...], tuple[TableRowMatrix, ...]]:
    previous_cells = _left_logical_cells(previous, left)
    continuation_cells = _mapped_logical_cells(continuation, mapping)
    cross_version_rows = _cross_version_support_rows(
        left,
        previous,
        continuation,
        mapping,
        cross_version_fragments,
    )
    evidence: list[EvidenceCode] = []
    if _blank_key_cells_evidence(continuation_cells, key_logical_columns):
        evidence.append("blank_key_cells")
    if _next_row_restores_key_pattern_evidence(
        previous, next_full_row, left, mapping, key_logical_columns
    ):
        evidence.append("next_row_restores_key_pattern")
    if _complementary_content_cells_evidence(
        previous_cells, continuation_cells, key_logical_columns
    ):
        evidence.append("complementary_content_cells")
    if _textual_continuity_evidence(previous_cells, continuation_cells, key_logical_columns):
        evidence.append("textual_continuity")
    if _boundary_artifacts_only_evidence(
        left, right, previous, continuation, boundary_rows
    ):
        evidence.append("boundary_artifacts_only")
    if cross_version_rows:
        evidence.append("cross_version_support")
    return tuple(evidence), cross_version_rows


def _render_table_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _rebuild_paragraph(paragraph: Paragraph) -> None:
    paragraph.text = "\n".join(sentence.text for sentence in paragraph.sentences)


def _rebuild_plain_text(document: DocumentIR) -> None:
    document.plain_text = "\n".join(
        paragraph.text
        for section in document.sections
        for paragraph in section.paragraphs
    )


def _initialize_replay_metadata(document: DocumentIR) -> None:
    if not hasattr(document, "_reconstruction_registry"):
        document._reconstruction_registry = {}
    for section in document.sections:
        for paragraph in section.paragraphs:
            if not hasattr(paragraph, "_reconstruction_source_paragraph_ids"):
                paragraph._reconstruction_source_paragraph_ids = frozenset(
                    {paragraph.paragraph_id}
                )
            for sentence_index, sentence in enumerate(paragraph.sentences):
                if not hasattr(sentence, "_reconstruction_source_rows"):
                    sentence._reconstruction_source_rows = frozenset(
                        {
                            SourceRowRef(
                                section.section_id,
                                paragraph.paragraph_id,
                                sentence_index,
                            )
                        }
                    )
                if not hasattr(sentence, "_reconstruction_operations"):
                    sentence._reconstruction_operations = frozenset()


def _row_locations(
    document: DocumentIR,
    source: SourceRowRef,
) -> list[tuple[Section, Paragraph, int, Sentence]]:
    locations: list[tuple[Section, Paragraph, int, Sentence]] = []
    for section in document.sections:
        for paragraph in section.paragraphs:
            for sentence_index, sentence in enumerate(paragraph.sentences):
                provenance = getattr(sentence, "_reconstruction_source_rows", frozenset())
                if source in provenance:
                    locations.append((section, paragraph, sentence_index, sentence))
    return locations


def _resolve_row(
    document: DocumentIR,
    source: SourceRowRef,
) -> tuple[Section, Paragraph, int, Sentence]:
    locations = _row_locations(document, source)
    if not locations:
        raise ValueError(
            "missing source row "
            f"{source.section_id}/{source.paragraph_id}/{source.sentence_index}"
        )
    if len(locations) != 1:
        raise ValueError(
            "ambiguous source row "
            f"{source.section_id}/{source.paragraph_id}/{source.sentence_index}"
        )
    return locations[0]


def _paragraph_locations(
    document: DocumentIR,
    source_paragraph_id: str,
) -> list[tuple[int, Section, int, Paragraph]]:
    locations: list[tuple[int, Section, int, Paragraph]] = []
    for section_index, section in enumerate(document.sections):
        for paragraph_index, paragraph in enumerate(section.paragraphs):
            provenance = getattr(
                paragraph,
                "_reconstruction_source_paragraph_ids",
                frozenset({paragraph.paragraph_id}),
            )
            if source_paragraph_id in provenance:
                locations.append((section_index, section, paragraph_index, paragraph))
    return locations


def _resolve_paragraph(
    document: DocumentIR,
    source_paragraph_id: str,
) -> tuple[int, Section, int, Paragraph]:
    locations = _paragraph_locations(document, source_paragraph_id)
    if not locations:
        raise ValueError(f"missing source paragraph {source_paragraph_id}")
    if len(locations) != 1:
        raise ValueError(f"ambiguous source paragraph {source_paragraph_id}")
    return locations[0]


def _expected_row_id(operation: ReconstructionOperation) -> str:
    expected = derived_row_id(operation)
    if operation.generated_row_id and operation.generated_row_id != expected:
        raise ValueError("generated row ID does not match operation provenance")
    return expected


def _expected_paragraph_id(operation: ReconstructionOperation) -> str:
    expected = derived_paragraph_id(operation)
    if operation.generated_paragraph_id and operation.generated_paragraph_id != expected:
        raise ValueError("generated paragraph ID does not match operation provenance")
    return expected


def _already_applied(
    document: DocumentIR,
    operation: ReconstructionOperation,
) -> bool:
    registry = document._reconstruction_registry
    recorded = registry.get(operation.operation_id)
    if recorded is None:
        if operation.type == "merge_rows":
            expected_id = _expected_row_id(operation)
            matches = [
                sentence
                for section in document.sections
                for paragraph in section.paragraphs
                for sentence in paragraph.sentences
                if getattr(sentence, "_reconstruction_row_id", "") == expected_id
                and set(operation.source_rows).issubset(
                    getattr(sentence, "_reconstruction_source_rows", frozenset())
                )
            ]
            return len(matches) == 1
        if operation.type == "merge_fragments":
            expected_id = _expected_paragraph_id(operation)
            matches = [
                paragraph
                for section in document.sections
                for paragraph in section.paragraphs
                if paragraph.paragraph_id == expected_id
                and set(operation.source_paragraph_ids).issubset(
                    getattr(
                        paragraph,
                        "_reconstruction_source_paragraph_ids",
                        frozenset(),
                    )
                )
            ]
            return len(matches) == 1
        return False
    if recorded != _operation_payload(operation):
        raise ValueError(f"operation ID collision for {operation.operation_id}")
    if operation.type == "merge_rows":
        expected_id = _expected_row_id(operation)
        return any(
            getattr(sentence, "_reconstruction_row_id", "") == expected_id
            and set(operation.source_rows).issubset(
                getattr(sentence, "_reconstruction_source_rows", frozenset())
            )
            for section in document.sections
            for paragraph in section.paragraphs
            for sentence in paragraph.sentences
        )
    if operation.type == "merge_fragments":
        expected_id = _expected_paragraph_id(operation)
        return any(
            paragraph.paragraph_id == expected_id
            and set(operation.source_paragraph_ids).issubset(
                getattr(paragraph, "_reconstruction_source_paragraph_ids", frozenset())
            )
            for section in document.sections
            for paragraph in section.paragraphs
        )
    return True


def _record_operation(document: DocumentIR, operation: ReconstructionOperation) -> None:
    document._reconstruction_registry[operation.operation_id] = _operation_payload(operation)


def _parse_replay_row(sentence: Sentence, source: SourceRowRef) -> TableRowMatrix:
    row = split_markdown_table_row(sentence.text, source)
    if row is None:
        raise ValueError(
            "source row is not a Markdown table row: "
            f"{source.section_id}/{source.paragraph_id}/{source.sentence_index}"
        )
    return row


def _apply_project_columns(
    document: DocumentIR,
    operation: ReconstructionOperation,
) -> None:
    if not operation.source_rows:
        raise ValueError("project_columns requires source rows")
    locations = [(_resolve_row(document, source), source) for source in operation.source_rows]
    for (_, paragraph, _, sentence), source in locations:
        row = _parse_replay_row(sentence, source)
        sentence.text = _render_table_row(
            _project_raw_cells(row, operation.column_mapping)
        )
        sentence._reconstruction_operations = frozenset(
            set(sentence._reconstruction_operations) | {operation.operation_id}
        )
        _rebuild_paragraph(paragraph)


def _apply_drop_boundary_rows(
    document: DocumentIR,
    operation: ReconstructionOperation,
) -> None:
    if not operation.source_rows:
        raise ValueError("drop_boundary_rows requires source rows")
    locations = [_resolve_row(document, source) for source in operation.source_rows]
    by_paragraph: dict[int, tuple[Paragraph, list[Sentence]]] = {}
    for _, paragraph, _, sentence in locations:
        entry = by_paragraph.setdefault(id(paragraph), (paragraph, []))
        if all(existing is not sentence for existing in entry[1]):
            entry[1].append(sentence)
    for paragraph, sentences in by_paragraph.values():
        sentence_ids = {id(sentence) for sentence in sentences}
        paragraph.sentences[:] = [
            sentence
            for sentence in paragraph.sentences
            if id(sentence) not in sentence_ids
        ]
        _rebuild_paragraph(paragraph)


def _apply_drop_boundary_paragraphs(
    document: DocumentIR,
    operation: ReconstructionOperation,
) -> None:
    if not operation.source_paragraph_ids:
        raise ValueError("drop_boundary_paragraphs requires paragraph IDs")
    locations = [
        _resolve_paragraph(document, paragraph_id)
        for paragraph_id in operation.source_paragraph_ids
    ]
    removals: dict[int, tuple[Section, set[int]]] = {}
    for _, section, _, paragraph in locations:
        entry = removals.setdefault(id(section), (section, set()))
        entry[1].add(id(paragraph))
    for section, paragraph_ids in removals.values():
        section.paragraphs[:] = [
            paragraph
            for paragraph in section.paragraphs
            if id(paragraph) not in paragraph_ids
        ]


def _apply_merge_rows(
    document: DocumentIR,
    operation: ReconstructionOperation,
) -> None:
    if len(operation.source_rows) != 2:
        raise ValueError("merge_rows requires exactly two source rows")
    previous_source, continuation_source = operation.source_rows
    previous_location = _resolve_row(document, previous_source)
    continuation_location = _resolve_row(document, continuation_source)
    if previous_location[3] is continuation_location[3]:
        expected_id = _expected_row_id(operation)
        sentence = previous_location[3]
        provenance = getattr(sentence, "_reconstruction_source_rows", frozenset())
        if (
            getattr(sentence, "_reconstruction_row_id", "") == expected_id
            and set(operation.source_rows).issubset(provenance)
        ):
            return
        raise ValueError("merge_rows source rows resolve to the same non-derived row")

    _, previous_paragraph, previous_index, previous_sentence = previous_location
    _, continuation_paragraph, _, continuation_sentence = continuation_location
    previous_row = _parse_replay_row(previous_sentence, previous_source)
    continuation_row = _parse_replay_row(continuation_sentence, continuation_source)
    identity = ColumnMapping(
        tuple(range(len(continuation_row.raw_cells))),
        {index: index for index in range(len(continuation_row.raw_cells))},
        1.0,
    )
    merged_cells = merge_logical_rows(
        previous_row,
        continuation_row,
        identity,
        frozenset(),
    )
    merged = Sentence(_render_table_row(merged_cells))
    merged._reconstruction_row_id = _expected_row_id(operation)
    merged._reconstruction_source_rows = frozenset(
        set(previous_sentence._reconstruction_source_rows)
        | set(continuation_sentence._reconstruction_source_rows)
    )
    merged._reconstruction_operations = frozenset({operation.operation_id})

    if previous_paragraph is continuation_paragraph:
        continuation_index = next(
            index
            for index, sentence in enumerate(previous_paragraph.sentences)
            if sentence is continuation_sentence
        )
        insert_index = previous_index - int(continuation_index < previous_index)
        removed_ids = {id(previous_sentence), id(continuation_sentence)}
        previous_paragraph.sentences[:] = [
            sentence
            for sentence in previous_paragraph.sentences
            if id(sentence) not in removed_ids
        ]
        previous_paragraph.sentences.insert(insert_index, merged)
        _rebuild_paragraph(previous_paragraph)
        return

    previous_paragraph.sentences[previous_index] = merged
    continuation_paragraph.sentences[:] = [
        sentence
        for sentence in continuation_paragraph.sentences
        if sentence is not continuation_sentence
    ]
    _rebuild_paragraph(previous_paragraph)
    _rebuild_paragraph(continuation_paragraph)


def _apply_merge_fragments(
    document: DocumentIR,
    operation: ReconstructionOperation,
) -> None:
    if len(operation.source_paragraph_ids) < 2:
        raise ValueError("merge_fragments requires at least two paragraph IDs")
    locations = [
        _resolve_paragraph(document, paragraph_id)
        for paragraph_id in operation.source_paragraph_ids
    ]
    unique: dict[int, tuple[int, Section, int, Paragraph]] = {
        id(location[3]): location for location in locations
    }
    if len(unique) == 1:
        paragraph = next(iter(unique.values()))[3]
        expected_id = _expected_paragraph_id(operation)
        provenance = getattr(
            paragraph,
            "_reconstruction_source_paragraph_ids",
            frozenset(),
        )
        if paragraph.paragraph_id == expected_id and set(
            operation.source_paragraph_ids
        ).issubset(provenance):
            return
        raise ValueError("merge_fragments sources resolve to one non-derived paragraph")
    ordered = sorted(unique.values(), key=lambda location: (location[0], location[2]))
    section_ids = {id(location[1]) for location in ordered}
    if len(section_ids) != 1:
        raise ValueError("merge_fragments cannot cross sections")
    _, section, insert_index, _ = ordered[0]
    paragraphs = [location[3] for location in ordered]
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph.sentences]
    merged = Paragraph(
        paragraph_id=_expected_paragraph_id(operation),
        text="",
        sentences=sentences,
    )
    merged._reconstruction_source_paragraph_ids = frozenset(
        source_id
        for paragraph in paragraphs
        for source_id in paragraph._reconstruction_source_paragraph_ids
    )
    _rebuild_paragraph(merged)
    paragraph_ids = {id(paragraph) for paragraph in paragraphs}
    section.paragraphs[:] = [
        paragraph
        for paragraph in section.paragraphs
        if id(paragraph) not in paragraph_ids
    ]
    section.paragraphs.insert(insert_index, merged)


def _apply_operation(
    document: DocumentIR,
    operation: ReconstructionOperation,
) -> None:
    if _already_applied(document, operation):
        _record_operation(document, operation)
        return
    handlers = {
        "project_columns": _apply_project_columns,
        "drop_boundary_rows": _apply_drop_boundary_rows,
        "drop_boundary_paragraphs": _apply_drop_boundary_paragraphs,
        "merge_rows": _apply_merge_rows,
        "merge_fragments": _apply_merge_fragments,
    }
    handlers[operation.type](document, operation)
    _record_operation(document, operation)


def apply_reconstruction_operations(
    baseline_ir: DocumentIR,
    target_ir: DocumentIR,
    operations: Sequence[ReconstructionOperation],
) -> tuple[DocumentIR, DocumentIR]:
    """Replay an explicit reconstruction trace on deep-copied documents."""
    baseline = deepcopy(baseline_ir)
    target = deepcopy(target_ir)
    documents = {"baseline": baseline, "target": target}
    for document in documents.values():
        _initialize_replay_metadata(document)
    for operation in sorted(operations, key=_operation_sort_key):
        _apply_operation(documents[operation.side], operation)
    for document in documents.values():
        _rebuild_plain_text(document)
    return baseline, target
