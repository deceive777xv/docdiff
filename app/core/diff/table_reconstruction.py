"""Deterministic structural inference for fragmented Markdown tables.

The module deliberately works from row shape and value categories.  It does not
depend on document-specific labels, row numbers, or physical column positions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from math import exp
from statistics import median
import re
from typing import Literal, Sequence

from app.core.diff.reconstruction_trace import SourceRowRef
from app.core.types import Section


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
            candidates.append((len(rows) * (0.5 + _segment_stability(rows)), segment_index))

    body_segment_index = max(candidates)[1] if candidates else None
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
    """Return repeated edge rows only when multiple independent signals agree."""
    if not fragments:
        return set()
    fingerprint_fragments: dict[tuple[str, ...], set[int]] = {}
    for fragment_index, fragment in enumerate(fragments):
        for row in fragment.rows:
            fingerprint_fragments.setdefault(_row_fingerprint(row), set()).add(fragment_index)

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
    for fragment in fragments:
        for row_index, row in enumerate(fragment.rows):
            repeated = _row_fingerprint(row) in repeated_fingerprints
            at_boundary = (
                row_index < BOUNDARY_CONFIG.edge_row_count
                or row_index >= len(fragment.rows) - BOUNDARY_CONFIG.edge_row_count
            )
            schema_incompatible = _body_schema_incompatible(row, fragment)
            key_discontinuous = _key_discontinuity(row, fragment)
            abnormal_repetition = (
                _within_row_repetition_ratio(row)
                >= BOUNDARY_CONFIG.within_row_repetition_threshold
            )
            signal_count = sum(
                (repeated, at_boundary, schema_incompatible, key_discontinuous, abnormal_repetition)
            )
            if (
                repeated
                and at_boundary
                and schema_incompatible
                and signal_count >= BOUNDARY_CONFIG.minimum_signal_families
            ):
                dropped.add(row.source)
    return dropped
