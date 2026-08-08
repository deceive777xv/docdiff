"""Build compare-only logical section scopes from read-only document IRs."""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.core.diff.structure_aligner import _title_similarity
from app.core.diff.text_similarity import lexical_similarity, rule_score_delta
from app.core.model.base_provider import BaseProvider
from app.core.types import DocumentIR, Paragraph, Section


_TITLE_MATCH_THRESHOLD = 0.3
_LONG_EXACT_ANCHOR_CHARS = 24
_BODY_COVERAGE_THRESHOLD = 0.5
_SECTION_CANDIDATE_MARGIN = 0.15
_LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+•]\s+)+")
_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[^>\n]+>")
_TABLE_SEPARATOR_CELL_RE = re.compile(r":?-{2,}:?")


@dataclass(frozen=True)
class AlignmentContentRef:
    side: Literal["baseline", "target"]
    paragraph_id: str
    sentence_index: int | None = None
    cell_index: int | None = None


@dataclass(frozen=True)
class SectionAlignmentEvidence:
    kind: Literal[
        "title",
        "body_exact",
        "body_semantic",
        "fake_paragraph",
        "fake_table_boundary_cell",
    ]
    score: float
    baseline_section_id: str | None
    target_section_id: str | None
    content_ref: AlignmentContentRef | None = None


@dataclass(frozen=True)
class SectionScopeGroup:
    group_id: str
    baseline_sections: tuple[Section, ...]
    target_sections: tuple[Section, ...]
    evidence: tuple[SectionAlignmentEvidence, ...] = ()
    baseline_crossable_boundaries: frozenset[tuple[str, str]] = frozenset()
    target_crossable_boundaries: frozenset[tuple[str, str]] = frozenset()


@dataclass(frozen=True)
class SectionAlignmentPlan:
    groups: tuple[SectionScopeGroup, ...]
    baseline_section_order: tuple[str, ...]
    target_section_order: tuple[str, ...]


@dataclass
class _MutableGroup:
    key: int
    baseline_sections: list[Section]
    target_sections: list[Section]
    evidence: list[SectionAlignmentEvidence]
    baseline_crossable_boundaries: set[tuple[str, str]]
    target_crossable_boundaries: set[tuple[str, str]]


@dataclass(frozen=True)
class _ContentCandidate:
    section: Section
    kind: Literal["fake_paragraph", "fake_table_boundary_cell"]
    content_ref: AlignmentContentRef


def _normalize_title_identity(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _normalize_body_text(text: str) -> str:
    without_marker = _LEADING_LIST_MARKER_RE.sub("", text or "")
    return re.sub(r"\s+", "", without_marker).lower()


def normalize_fake_title_evidence(text: str) -> str:
    """Normalize punctuation noise without removing business symbols."""
    normalized = unicodedata.normalize("NFKC", html.unescape(text or ""))
    normalized = _HTML_BREAK_RE.sub("", normalized)
    normalized = _HTML_TAG_RE.sub("", normalized)
    return "".join(
        character.lower()
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    before_trailing_pipe = stripped[:-1] if stripped.endswith("|") else stripped
    trailing_backslashes = len(before_trailing_pipe) - len(
        before_trailing_pipe.rstrip("\\")
    )
    if stripped.endswith("|") and trailing_backslashes % 2 == 0:
        stripped = stripped[:-1]
    cells: list[str] = []
    cell: list[str] = []
    backslashes = 0
    for character in stripped:
        if character == "|" and backslashes % 2 == 0:
            cells.append("".join(cell).strip())
            cell = []
            backslashes = 0
            continue
        cell.append(character)
        if character == "\\":
            backslashes += 1
        else:
            backslashes = 0
    cells.append("".join(cell).strip())
    return [value.replace(r"\|", "|") for value in cells]


def _plain_cell(cell: str) -> str:
    text = html.unescape(cell or "")
    text = _HTML_BREAK_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    return re.sub(r"[`*_~]+", "", text).strip()


def _is_table_separator_row(line: str) -> bool:
    cells = _split_table_row(line)
    return bool(cells) and all(
        _TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells
    )


def _is_empty_table_row(line: str) -> bool:
    cells = _split_table_row(line)
    return bool(cells) and all(not _plain_cell(cell) for cell in cells)


def _looks_like_table(paragraph: Paragraph) -> bool:
    rows = [sentence.text for sentence in paragraph.sentences if "|" in sentence.text]
    if len(rows) >= 2:
        return True
    stripped = paragraph.text.strip()
    return bool(rows) and stripped.startswith("|") and stripped.endswith("|")


def _ordinary_paragraphs(section: Section) -> list[Paragraph]:
    return [
        paragraph
        for paragraph in section.paragraphs
        if not _looks_like_table(paragraph) and "|" not in paragraph.text
    ]


def _table_boundary_candidates(
    section: Section,
    side: Literal["baseline", "target"],
) -> list[tuple[str, _ContentCandidate]]:
    candidates: list[tuple[str, _ContentCandidate]] = []
    for paragraph in section.paragraphs:
        if not _looks_like_table(paragraph):
            continue
        business_rows = [
            (index, sentence.text)
            for index, sentence in enumerate(paragraph.sentences)
            if "|" in sentence.text
            and not _is_table_separator_row(sentence.text)
            and not _is_empty_table_row(sentence.text)
        ]
        if not business_rows:
            continue
        boundary_rows = [business_rows[0]]
        if business_rows[-1][0] != business_rows[0][0]:
            boundary_rows.append(business_rows[-1])
        for sentence_index, row in boundary_rows:
            for cell_index, cell in enumerate(_split_table_row(row)):
                plain = _plain_cell(cell)
                normalized = normalize_fake_title_evidence(plain)
                if not normalized:
                    continue
                candidates.append((
                    normalized,
                    _ContentCandidate(
                        section=section,
                        kind="fake_table_boundary_cell",
                        content_ref=AlignmentContentRef(
                            side=side,
                            paragraph_id=paragraph.paragraph_id,
                            sentence_index=sentence_index,
                            cell_index=cell_index,
                        ),
                    ),
                ))
    return candidates


def _paragraph_candidates(
    section: Section,
    side: Literal["baseline", "target"],
) -> list[tuple[str, _ContentCandidate]]:
    candidates: list[tuple[str, _ContentCandidate]] = []
    for paragraph in _ordinary_paragraphs(section):
        normalized = normalize_fake_title_evidence(paragraph.text)
        if not normalized:
            continue
        candidates.append((
            normalized,
            _ContentCandidate(
                section=section,
                kind="fake_paragraph",
                content_ref=AlignmentContentRef(side, paragraph.paragraph_id),
            ),
        ))
    return candidates


def _cosine(left: list[float], right: list[float]) -> float:
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    if left_vector.shape != right_vector.shape:
        return 0.0
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator <= 1e-9:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)


def _parent_ids(document: DocumentIR) -> dict[str, str | None]:
    parents: dict[str, str | None] = {}
    stack: list[Section] = []
    for section in document.sections:
        while stack and stack[-1].level >= section.level:
            stack.pop()
        parents[section.section_id] = stack[-1].section_id if stack else None
        stack.append(section)
    return parents


def _semantic_section_candidates(
    baseline_sections: list[Section],
    target_sections: list[Section],
    embedder: BaseProvider,
    similarity_threshold: float,
) -> tuple[
    dict[tuple[str, str], float],
    set[tuple[str, str]],
]:
    baseline_paragraphs = {
        section.section_id: [
            paragraph
            for paragraph in _ordinary_paragraphs(section)
            if _normalize_body_text(paragraph.text)
        ]
        for section in baseline_sections
    }
    target_paragraphs = {
        section.section_id: [
            paragraph
            for paragraph in _ordinary_paragraphs(section)
            if _normalize_body_text(paragraph.text)
        ]
        for section in target_sections
    }
    all_paragraphs = [
        paragraph
        for section in baseline_sections
        for paragraph in baseline_paragraphs[section.section_id]
    ] + [
        paragraph
        for section in target_sections
        for paragraph in target_paragraphs[section.section_id]
    ]
    if not all_paragraphs:
        return {}, set()
    try:
        embeddings = embedder.embed([paragraph.text for paragraph in all_paragraphs])
        matrix = np.asarray(embeddings, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != len(all_paragraphs)
            or matrix.shape[1] == 0
            or not np.isfinite(matrix).all()
        ):
            return {}, set()
    except Exception:
        return {}, set()
    embedding_by_paragraph = {
        id(paragraph): embedding.tolist()
        for paragraph, embedding in zip(all_paragraphs, matrix)
    }

    qualities: dict[tuple[str, str], float] = {}
    eligible_pairs: set[tuple[str, str]] = set()
    for baseline_section in baseline_sections:
        baseline_items = baseline_paragraphs[baseline_section.section_id]
        if not baseline_items:
            continue
        for target_section in target_sections:
            if baseline_section.level != target_section.level:
                continue
            target_items = target_paragraphs[target_section.section_id]
            if not target_items:
                continue
            similarities = [
                [
                    max(
                        _cosine(
                            embedding_by_paragraph[id(baseline_paragraph)],
                            embedding_by_paragraph[id(target_paragraph)],
                        ),
                        lexical_similarity(
                            baseline_paragraph.text,
                            target_paragraph.text,
                        ),
                    )
                    - rule_score_delta(
                        baseline_paragraph.text,
                        target_paragraph.text,
                    )
                    for target_paragraph in target_items
                ]
                for baseline_paragraph in baseline_items
            ]
            baseline_best: dict[int, int] = {}
            for baseline_index, row in enumerate(similarities):
                best = max(row)
                indexes = [index for index, value in enumerate(row) if value == best]
                if len(indexes) == 1:
                    baseline_best[baseline_index] = indexes[0]
            target_best: dict[int, int] = {}
            for target_index in range(len(target_items)):
                column = [row[target_index] for row in similarities]
                best = max(column)
                indexes = [index for index, value in enumerate(column) if value == best]
                if len(indexes) == 1:
                    target_best[target_index] = indexes[0]
            mutual = [
                (baseline_index, target_index)
                for baseline_index, target_index in baseline_best.items()
                if target_best.get(target_index) == baseline_index
            ]
            mutual.sort()
            pair_key = (baseline_section.section_id, target_section.section_id)
            if len(baseline_items) == len(target_items) == 1:
                similarity = similarities[0][0]
                qualities[pair_key] = similarity
                if similarity >= similarity_threshold:
                    eligible_pairs.add(pair_key)
                continue

            def mutual_quality(
                candidates: list[tuple[int, int]],
            ) -> float | None:
                if len(candidates) < 2 or any(
                    left[1] >= right[1]
                    for left, right in zip(candidates, candidates[1:])
                ):
                    return None
                baseline_total = sum(
                    len(_normalize_body_text(paragraph.text))
                    for paragraph in baseline_items
                )
                target_total = sum(
                    len(_normalize_body_text(paragraph.text))
                    for paragraph in target_items
                )
                baseline_covered = sum(
                    len(_normalize_body_text(baseline_items[index].text))
                    for index, _ in candidates
                )
                target_covered = sum(
                    len(_normalize_body_text(target_items[index].text))
                    for _, index in candidates
                )
                if (
                    not baseline_total
                    or not target_total
                    or baseline_covered / baseline_total < _BODY_COVERAGE_THRESHOLD
                    or target_covered / target_total < _BODY_COVERAGE_THRESHOLD
                ):
                    return None
                weighted_total = 0.0
                weight = 0
                for baseline_index, target_index in candidates:
                    pair_weight = max(
                        len(_normalize_body_text(baseline_items[baseline_index].text)),
                        len(_normalize_body_text(target_items[target_index].text)),
                    )
                    weighted_total += (
                        similarities[baseline_index][target_index] * pair_weight
                    )
                    weight += pair_weight
                return weighted_total / weight if weight else None

            diagnostic_quality = mutual_quality(mutual)
            threshold_mutual = [
                (baseline_index, target_index)
                for baseline_index, target_index in mutual
                if similarities[baseline_index][target_index]
                >= similarity_threshold
            ]
            eligible_quality = mutual_quality(threshold_mutual)
            if eligible_quality is not None:
                qualities[pair_key] = eligible_quality
                eligible_pairs.add(pair_key)
            elif diagnostic_quality is not None:
                qualities[pair_key] = diagnostic_quality
    return qualities, eligible_pairs


def _reciprocal_margin_matches(
    baseline_sections: list[Section],
    target_sections: list[Section],
    qualities: dict[tuple[str, str], float],
    eligible_pairs: set[tuple[str, str]],
) -> list[tuple[Section, Section, float]]:
    baseline_by_id = {section.section_id: section for section in baseline_sections}
    target_by_id = {section.section_id: section for section in target_sections}

    def best_candidates(
        candidate_ids: list[str],
        scores: list[tuple[str, float]],
    ) -> tuple[str, float] | None:
        ranked = sorted(scores, key=lambda item: (-item[1], candidate_ids.index(item[0])))
        if not ranked:
            return None
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < _SECTION_CANDIDATE_MARGIN:
            return None
        return ranked[0]

    baseline_best: dict[str, tuple[str, float]] = {}
    for baseline in baseline_sections:
        scores = [
            (target.section_id, qualities[(baseline.section_id, target.section_id)])
            for target in target_sections
            if (baseline.section_id, target.section_id) in qualities
        ]
        selected = best_candidates(
            [section.section_id for section in target_sections],
            scores,
        )
        if (
            selected is not None
            and (baseline.section_id, selected[0]) in eligible_pairs
        ):
            baseline_best[baseline.section_id] = selected

    target_best: dict[str, tuple[str, float]] = {}
    for target in target_sections:
        scores = [
            (baseline.section_id, qualities[(baseline.section_id, target.section_id)])
            for baseline in baseline_sections
            if (baseline.section_id, target.section_id) in qualities
        ]
        selected = best_candidates(
            [section.section_id for section in baseline_sections],
            scores,
        )
        if (
            selected is not None
            and (selected[0], target.section_id) in eligible_pairs
        ):
            target_best[target.section_id] = selected

    matches: list[tuple[Section, Section, float]] = []
    for baseline_id, (target_id, score) in baseline_best.items():
        if target_best.get(target_id, (None, 0.0))[0] == baseline_id:
            matches.append((baseline_by_id[baseline_id], target_by_id[target_id], score))
    return matches


def align_compare_scopes(
    baseline: DocumentIR,
    target: DocumentIR,
    embedder: BaseProvider,
    *,
    similarity_threshold: float = 0.75,
) -> SectionAlignmentPlan:
    """Align logical compare scopes without mutating either input document."""
    baseline_sections = list(baseline.sections)
    target_sections = list(target.sections)
    baseline_index = {
        section.section_id: index for index, section in enumerate(baseline_sections)
    }
    target_index = {
        section.section_id: index for index, section in enumerate(target_sections)
    }
    target_by_id = {section.section_id: section for section in target_sections}
    baseline_parents = _parent_ids(baseline)
    target_parents = _parent_ids(target)
    matches: dict[str, str] = {}
    reverse_matches: dict[str, str] = {}
    evidence_by_pair: dict[tuple[str, str], SectionAlignmentEvidence] = {}

    baseline_titles: dict[str, list[Section]] = {}
    target_titles: dict[str, list[Section]] = {}
    for section in baseline_sections:
        baseline_titles.setdefault(_normalize_title_identity(section.title), []).append(section)
    for section in target_sections:
        target_titles.setdefault(_normalize_title_identity(section.title), []).append(section)
    for title, baseline_candidates in baseline_titles.items():
        target_candidates = target_titles.get(title, [])
        if (
            title
            and len(baseline_candidates) == len(target_candidates) == 1
        ):
            baseline_section = baseline_candidates[0]
            target_section = target_candidates[0]
            matches[baseline_section.section_id] = target_section.section_id
            reverse_matches[target_section.section_id] = baseline_section.section_id
            evidence_by_pair[(baseline_section.section_id, target_section.section_id)] = (
                SectionAlignmentEvidence(
                    "title",
                    1.0,
                    baseline_section.section_id,
                    target_section.section_id,
                )
            )

    unresolved_baseline = [
        section for section in baseline_sections if section.section_id not in matches
    ]
    unresolved_target = [
        section for section in target_sections if section.section_id not in reverse_matches
    ]
    weak_by_baseline: dict[str, list[tuple[Section, float]]] = {}
    weak_by_target: dict[str, list[tuple[Section, float]]] = {}
    for baseline_section in unresolved_baseline:
        for target_section in unresolved_target:
            if baseline_section.level != target_section.level:
                continue
            score = _title_similarity(baseline_section.title, target_section.title)
            if score < _TITLE_MATCH_THRESHOLD:
                continue
            weak_by_baseline.setdefault(baseline_section.section_id, []).append(
                (target_section, score)
            )
            weak_by_target.setdefault(target_section.section_id, []).append(
                (baseline_section, score)
            )
    deferred_weak_matches: list[tuple[Section, Section, float]] = []
    for baseline_section in unresolved_baseline:
        candidates = weak_by_baseline.get(baseline_section.section_id, [])
        if len(candidates) != 1:
            continue
        target_section, score = candidates[0]
        reverse = weak_by_target.get(target_section.section_id, [])
        if len(reverse) != 1 or reverse[0][0].section_id != baseline_section.section_id:
            continue
        deferred_weak_matches.append((baseline_section, target_section, score))

    unresolved_baseline = [
        section for section in baseline_sections if section.section_id not in matches
    ]
    unresolved_target = [
        section for section in target_sections if section.section_id not in reverse_matches
    ]
    baseline_occurrences: dict[str, list[tuple[Section, Paragraph]]] = {}
    target_occurrences: dict[str, list[tuple[Section, Paragraph]]] = {}
    for section in baseline_sections:
        for paragraph in _ordinary_paragraphs(section):
            normalized = _normalize_body_text(paragraph.text)
            if normalized:
                baseline_occurrences.setdefault(normalized, []).append((section, paragraph))
    for section in target_sections:
        for paragraph in _ordinary_paragraphs(section):
            normalized = _normalize_body_text(paragraph.text)
            if normalized:
                target_occurrences.setdefault(normalized, []).append((section, paragraph))

    exact_targets: dict[str, list[tuple[str, int]]] = {}
    exact_sources: dict[str, list[tuple[str, int]]] = {}
    for normalized, baseline_items in baseline_occurrences.items():
        target_items = target_occurrences.get(normalized, [])
        if (
            len(normalized) < _LONG_EXACT_ANCHOR_CHARS
            or len(baseline_items) != 1
            or len(target_items) != 1
        ):
            continue
        baseline_section = baseline_items[0][0]
        target_section = target_items[0][0]
        exact_targets.setdefault(baseline_section.section_id, []).append(
            (target_section.section_id, len(normalized))
        )
        exact_sources.setdefault(target_section.section_id, []).append(
            (baseline_section.section_id, len(normalized))
        )
    for baseline_section in unresolved_baseline:
        anchors = exact_targets.get(baseline_section.section_id, [])
        target_ids = {target_id for target_id, _ in anchors}
        if len(target_ids) != 1:
            continue
        target_id = next(iter(target_ids))
        reverse = exact_sources.get(target_id, [])
        if {baseline_id for baseline_id, _ in reverse} != {baseline_section.section_id}:
            continue
        target_section = target_by_id[target_id]
        if (
            target_id in reverse_matches
            or baseline_section.level != target_section.level
        ):
            continue
        anchor_chars = sum(length for _, length in anchors)
        baseline_chars = sum(
            len(_normalize_body_text(paragraph.text))
            for paragraph in _ordinary_paragraphs(baseline_section)
        )
        target_chars = sum(
            len(_normalize_body_text(paragraph.text))
            for paragraph in _ordinary_paragraphs(target_section)
        )
        if not (
            len(anchors) >= 2
            or (
                baseline_chars
                and target_chars
                and anchor_chars / baseline_chars >= _BODY_COVERAGE_THRESHOLD
                and anchor_chars / target_chars >= _BODY_COVERAGE_THRESHOLD
            )
        ):
            continue
        matches[baseline_section.section_id] = target_id
        reverse_matches[target_id] = baseline_section.section_id
        evidence_by_pair[(baseline_section.section_id, target_id)] = (
            SectionAlignmentEvidence(
                "body_exact",
                1.0,
                baseline_section.section_id,
                target_id,
            )
        )

    unresolved_baseline = [
        section for section in baseline_sections if section.section_id not in matches
    ]
    unresolved_target = [
        section for section in target_sections if section.section_id not in reverse_matches
    ]
    semantic_qualities, semantic_eligible_pairs = _semantic_section_candidates(
        unresolved_baseline,
        unresolved_target,
        embedder,
        similarity_threshold,
    )
    for baseline_section, target_section, score in _reciprocal_margin_matches(
        unresolved_baseline,
        unresolved_target,
        semantic_qualities,
        semantic_eligible_pairs,
    ):
        matches[baseline_section.section_id] = target_section.section_id
        reverse_matches[target_section.section_id] = baseline_section.section_id
        evidence_by_pair[(baseline_section.section_id, target_section.section_id)] = (
            SectionAlignmentEvidence(
                "body_semantic",
                score,
                baseline_section.section_id,
                target_section.section_id,
            )
        )

    def weak_parent_compatible(
        baseline_section: Section,
        target_section: Section,
    ) -> bool:
        baseline_parent = baseline_parents[baseline_section.section_id]
        target_parent = target_parents[target_section.section_id]
        if baseline_parent is None or target_parent is None:
            return baseline_parent is target_parent
        return (
            matches.get(baseline_parent) == target_parent
            and reverse_matches.get(target_parent) == baseline_parent
        )

    def weak_crosses_existing(
        baseline_section: Section,
        target_section: Section,
    ) -> bool:
        baseline_parent = baseline_parents[baseline_section.section_id]
        target_parent = target_parents[target_section.section_id]
        for other_baseline_id, other_target_id in matches.items():
            if (
                baseline_parents[other_baseline_id] != baseline_parent
                or target_parents[other_target_id] != target_parent
            ):
                continue
            baseline_delta = (
                baseline_index[baseline_section.section_id]
                - baseline_index[other_baseline_id]
            )
            target_delta = (
                target_index[target_section.section_id]
                - target_index[other_target_id]
            )
            if baseline_delta * target_delta < 0:
                return True
        return False

    for baseline_section, target_section, score in sorted(
        deferred_weak_matches,
        key=lambda item: (
            item[0].level,
            baseline_index[item[0].section_id],
            target_index[item[1].section_id],
        ),
    ):
        if (
            baseline_section.section_id in matches
            or target_section.section_id in reverse_matches
            or not weak_parent_compatible(baseline_section, target_section)
            or weak_crosses_existing(baseline_section, target_section)
        ):
            continue
        matches[baseline_section.section_id] = target_section.section_id
        reverse_matches[target_section.section_id] = baseline_section.section_id
        evidence_by_pair[(baseline_section.section_id, target_section.section_id)] = (
            SectionAlignmentEvidence(
                "title",
                score,
                baseline_section.section_id,
                target_section.section_id,
            )
        )

    groups: list[_MutableGroup] = []
    group_by_baseline: dict[str, _MutableGroup] = {}
    group_by_target: dict[str, _MutableGroup] = {}
    for baseline_section in baseline_sections:
        target_id = matches.get(baseline_section.section_id)
        if target_id is None:
            continue
        target_section = target_by_id[target_id]
        group = _MutableGroup(
            key=len(groups),
            baseline_sections=[baseline_section],
            target_sections=[target_section],
            evidence=[evidence_by_pair[(baseline_section.section_id, target_id)]],
            baseline_crossable_boundaries=set(),
            target_crossable_boundaries=set(),
        )
        groups.append(group)
        group_by_baseline[baseline_section.section_id] = group
        group_by_target[target_id] = group

    def attach_unpaired(
        sections: list[Section],
        parents: dict[str, str | None],
        group_by_section: dict[str, _MutableGroup],
        side: Literal["baseline", "target"],
    ) -> None:
        for section in sections:
            if section.section_id in group_by_section:
                continue
            parent_id = parents.get(section.section_id)
            group = None
            while parent_id is not None:
                group = group_by_section.get(parent_id)
                if group is not None:
                    break
                parent_id = parents.get(parent_id)
            if group is None:
                group = _MutableGroup(
                    key=len(groups),
                    baseline_sections=[],
                    target_sections=[],
                    evidence=[],
                    baseline_crossable_boundaries=set(),
                    target_crossable_boundaries=set(),
                )
                groups.append(group)
            if side == "baseline":
                group.baseline_sections.append(section)
            else:
                group.target_sections.append(section)
            group_by_section[section.section_id] = group

    attach_unpaired(
        baseline_sections,
        baseline_parents,
        group_by_baseline,
        "baseline",
    )
    attach_unpaired(target_sections, target_parents, group_by_target, "target")

    matched_baseline_ids = set(matches)
    matched_target_ids = set(reverse_matches)

    def fake_decisions(
        side: Literal["baseline", "target"],
    ) -> list[tuple[Section, _ContentCandidate, _MutableGroup]]:
        own_sections = baseline_sections if side == "baseline" else target_sections
        other_sections = target_sections if side == "baseline" else baseline_sections
        own_index = baseline_index if side == "baseline" else target_index
        other_index = target_index if side == "baseline" else baseline_index
        own_parents = baseline_parents if side == "baseline" else target_parents
        other_parents = target_parents if side == "baseline" else baseline_parents
        own_matched_ids = matched_baseline_ids if side == "baseline" else matched_target_ids
        match_lookup = matches if side == "baseline" else reverse_matches
        other_groups = group_by_target if side == "baseline" else group_by_baseline
        other_side: Literal["baseline", "target"] = (
            "target" if side == "baseline" else "baseline"
        )
        decisions: list[tuple[Section, _ContentCandidate, _MutableGroup]] = []
        for section in own_sections:
            if section.section_id in own_matched_ids:
                continue
            index = own_index[section.section_id]
            previous = next(
                (
                    candidate
                    for candidate in reversed(own_sections[:index])
                    if candidate.section_id in own_matched_ids
                ),
                None,
            )
            following = next(
                (
                    candidate
                    for candidate in own_sections[index + 1 :]
                    if candidate.section_id in own_matched_ids
                ),
                None,
            )
            candidate_sections: list[Section]
            if previous is not None and following is not None:
                previous_other_id = match_lookup[previous.section_id]
                following_other_id = match_lookup[following.section_id]
                previous_other_index = other_index[previous_other_id]
                following_other_index = other_index[following_other_id]
                if previous_other_index >= following_other_index:
                    continue
                candidate_sections = other_sections[
                    previous_other_index : following_other_index + 1
                ]
            elif previous is not None or following is not None:
                anchor = previous if previous is not None else following
                if anchor is None:
                    continue
                section_parent = own_parents[section.section_id]
                anchor_parent = own_parents[anchor.section_id]
                if section_parent is None:
                    if anchor_parent is not None:
                        continue
                else:
                    if section_parent not in own_matched_ids:
                        continue
                    if (
                        anchor.section_id != section_parent
                        and anchor_parent != section_parent
                    ):
                        continue
                anchor_other_id = match_lookup[anchor.section_id]
                if section_parent is not None:
                    projected_parent = match_lookup[section_parent]
                    if (
                        anchor_other_id != projected_parent
                        and other_parents[anchor_other_id] != projected_parent
                    ):
                        continue
                anchor_group = other_groups[anchor_other_id]
                candidate_sections = (
                    anchor_group.target_sections
                    if side == "baseline"
                    else anchor_group.baseline_sections
                )
            else:
                continue

            normalized_title = normalize_fake_title_evidence(section.title)
            if not normalized_title:
                continue
            candidates: list[_ContentCandidate] = []
            for candidate_section in candidate_sections:
                for normalized, candidate in _paragraph_candidates(
                    candidate_section,
                    other_side,
                ) + _table_boundary_candidates(candidate_section, other_side):
                    if normalized == normalized_title:
                        candidates.append(candidate)
            if len(candidates) != 1:
                continue
            candidate = candidates[0]
            decisions.append((section, candidate, other_groups[candidate.section.section_id]))
        return decisions

    def apply_fake_decisions(
        side: Literal["baseline", "target"],
        decisions: list[tuple[Section, _ContentCandidate, _MutableGroup]],
    ) -> None:
        group_by_section = group_by_baseline if side == "baseline" else group_by_target
        real_section_ids = (
            matched_baseline_ids if side == "baseline" else matched_target_ids
        )
        source_sections = baseline_sections if side == "baseline" else target_sections
        source_index = baseline_index if side == "baseline" else target_index
        for section, candidate, destination in decisions:
            current = group_by_section[section.section_id]
            if current is not destination:
                if side == "baseline":
                    current.baseline_sections.remove(section)
                    destination.baseline_sections.append(section)
                else:
                    current.target_sections.remove(section)
                    destination.target_sections.append(section)
                group_by_section[section.section_id] = destination
            evidence = SectionAlignmentEvidence(
                candidate.kind,
                1.0,
                section.section_id if side == "baseline" else candidate.section.section_id,
                candidate.section.section_id if side == "baseline" else section.section_id,
                candidate.content_ref,
            )
            destination.evidence.append(evidence)
            index = source_index[section.section_id]
            boundaries = (
                destination.baseline_crossable_boundaries
                if side == "baseline"
                else destination.target_crossable_boundaries
            )
            for neighbor_index in (index - 1, index + 1):
                if not (0 <= neighbor_index < len(source_sections)):
                    continue
                neighbor = source_sections[neighbor_index]
                if (
                    neighbor.section_id not in real_section_ids
                    or group_by_section.get(neighbor.section_id) is not destination
                ):
                    continue
                if neighbor_index < index:
                    boundaries.add((neighbor.section_id, section.section_id))
                else:
                    boundaries.add((section.section_id, neighbor.section_id))

    baseline_fake_decisions = fake_decisions("baseline")
    target_fake_decisions = fake_decisions("target")
    claimed_refs: dict[
        tuple[str, str, int | None, int | None],
        int,
    ] = {}
    for _, candidate, _ in baseline_fake_decisions + target_fake_decisions:
        ref = candidate.content_ref
        key = (ref.side, ref.paragraph_id, ref.sentence_index, ref.cell_index)
        claimed_refs[key] = claimed_refs.get(key, 0) + 1
    baseline_fake_decisions = [
        decision
        for decision in baseline_fake_decisions
        if claimed_refs[
            (
                decision[1].content_ref.side,
                decision[1].content_ref.paragraph_id,
                decision[1].content_ref.sentence_index,
                decision[1].content_ref.cell_index,
            )
        ] == 1
    ]
    target_fake_decisions = [
        decision
        for decision in target_fake_decisions
        if claimed_refs[
            (
                decision[1].content_ref.side,
                decision[1].content_ref.paragraph_id,
                decision[1].content_ref.sentence_index,
                decision[1].content_ref.cell_index,
            )
        ] == 1
    ]
    apply_fake_decisions("baseline", baseline_fake_decisions)
    apply_fake_decisions("target", target_fake_decisions)

    groups = [
        group
        for group in groups
        if group.baseline_sections or group.target_sections
    ]
    for group in groups:
        group.baseline_sections.sort(key=lambda section: baseline_index[section.section_id])
        group.target_sections.sort(key=lambda section: target_index[section.section_id])

    def group_sort_key(group: _MutableGroup) -> tuple[int, int, int]:
        baseline_position = min(
            (baseline_index[section.section_id] for section in group.baseline_sections),
            default=len(baseline_sections),
        )
        target_position = min(
            (target_index[section.section_id] for section in group.target_sections),
            default=len(target_sections),
        )
        return baseline_position, target_position, group.key

    frozen_groups = tuple(
        SectionScopeGroup(
            group_id=f"scope:{index:04d}",
            baseline_sections=tuple(group.baseline_sections),
            target_sections=tuple(group.target_sections),
            evidence=tuple(group.evidence),
            baseline_crossable_boundaries=frozenset(
                group.baseline_crossable_boundaries
            ),
            target_crossable_boundaries=frozenset(group.target_crossable_boundaries),
        )
        for index, group in enumerate(sorted(groups, key=group_sort_key))
    )
    plan = SectionAlignmentPlan(
        groups=frozen_groups,
        baseline_section_order=tuple(
            section.section_id for section in baseline_sections
        ),
        target_section_order=tuple(section.section_id for section in target_sections),
    )
    _validate_plan(plan, baseline, target)
    return plan


def _validate_plan(
    plan: SectionAlignmentPlan,
    baseline: DocumentIR,
    target: DocumentIR,
) -> None:
    baseline_sections = {
        section.section_id: section for section in baseline.sections
    }
    target_sections = {section.section_id: section for section in target.sections}
    seen_baseline: list[str] = []
    seen_target: list[str] = []
    baseline_index = {
        section.section_id: index for index, section in enumerate(baseline.sections)
    }
    target_index = {
        section.section_id: index for index, section in enumerate(target.sections)
    }
    paragraph_locations = {
        "baseline": {
            paragraph.paragraph_id: (section, paragraph)
            for section in baseline.sections
            for paragraph in section.paragraphs
        },
        "target": {
            paragraph.paragraph_id: (section, paragraph)
            for section in target.sections
            for paragraph in section.paragraphs
        },
    }
    seen_content_refs: set[tuple[str, str, int | None, int | None]] = set()
    for group in plan.groups:
        if not group.baseline_sections and not group.target_sections:
            raise ValueError("section alignment group cannot be empty")
        seen_baseline.extend(section.section_id for section in group.baseline_sections)
        seen_target.extend(section.section_id for section in group.target_sections)
        fake_section_ids: dict[str, set[str]] = {
            "baseline": set(),
            "target": set(),
        }
        real_section_ids: dict[str, set[str]] = {
            "baseline": set(),
            "target": set(),
        }
        group_baseline_ids = {
            section.section_id for section in group.baseline_sections
        }
        group_target_ids = {
            section.section_id for section in group.target_sections
        }
        for evidence in group.evidence:
            if (
                evidence.baseline_section_id is not None
                and evidence.baseline_section_id not in baseline_sections
            ):
                raise ValueError("alignment evidence references unknown baseline section")
            if (
                evidence.target_section_id is not None
                and evidence.target_section_id not in target_sections
            ):
                raise ValueError("alignment evidence references unknown target section")
            if evidence.kind.startswith("fake_"):
                if evidence.content_ref is None:
                    raise ValueError("fake alignment evidence requires content reference")
                fake_section_id = (
                    evidence.target_section_id
                    if evidence.content_ref.side == "baseline"
                    else evidence.baseline_section_id
                )
                if fake_section_id is None:
                    raise ValueError("fake alignment evidence is missing fake section")
                fake_side = (
                    "target"
                    if evidence.content_ref.side == "baseline"
                    else "baseline"
                )
                fake_section_ids[fake_side].add(fake_section_id)
            else:
                if evidence.baseline_section_id is not None:
                    real_section_ids["baseline"].add(
                        evidence.baseline_section_id
                    )
                if evidence.target_section_id is not None:
                    real_section_ids["target"].add(evidence.target_section_id)
            if (
                evidence.baseline_section_id is not None
                and evidence.baseline_section_id not in group_baseline_ids
            ):
                raise ValueError("alignment evidence leaves its baseline group")
            if (
                evidence.target_section_id is not None
                and evidence.target_section_id not in group_target_ids
            ):
                raise ValueError("alignment evidence leaves its target group")
            if evidence.content_ref is None:
                continue
            content_ref_key = (
                evidence.content_ref.side,
                evidence.content_ref.paragraph_id,
                evidence.content_ref.sentence_index,
                evidence.content_ref.cell_index,
            )
            if content_ref_key in seen_content_refs:
                raise ValueError("alignment content reference is reused")
            seen_content_refs.add(content_ref_key)
            location = paragraph_locations[evidence.content_ref.side].get(
                evidence.content_ref.paragraph_id
            )
            if location is None:
                raise ValueError("alignment content reference paragraph does not exist")
            content_section, content_paragraph = location
            expected_section_id = (
                evidence.baseline_section_id
                if evidence.content_ref.side == "baseline"
                else evidence.target_section_id
            )
            if content_section.section_id != expected_section_id:
                raise ValueError("alignment content reference belongs to another section")
            if evidence.kind == "fake_paragraph":
                if (
                    evidence.content_ref.sentence_index is not None
                    or evidence.content_ref.cell_index is not None
                ):
                    raise ValueError("paragraph evidence cannot reference a table cell")
            elif evidence.kind == "fake_table_boundary_cell":
                sentence_index = evidence.content_ref.sentence_index
                cell_index = evidence.content_ref.cell_index
                if sentence_index is None or not (
                    0 <= sentence_index < len(content_paragraph.sentences)
                ):
                    raise ValueError("table evidence sentence index is invalid")
                cells = _split_table_row(
                    content_paragraph.sentences[sentence_index].text
                )
                if cell_index is None or not (0 <= cell_index < len(cells)):
                    raise ValueError("table evidence cell index is invalid")

        for side, boundaries, indexes, group_sections in (
            (
                "baseline",
                group.baseline_crossable_boundaries,
                baseline_index,
                group.baseline_sections,
            ),
            (
                "target",
                group.target_crossable_boundaries,
                target_index,
                group.target_sections,
            ),
        ):
            group_ids = {section.section_id for section in group_sections}
            for left, right in boundaries:
                if left not in group_ids or right not in group_ids:
                    raise ValueError(f"{side} crossable boundary leaves its group")
                if indexes[right] - indexes[left] != 1:
                    raise ValueError(f"{side} crossable boundary is not adjacent")
                if (
                    (left in fake_section_ids[side])
                    == (right in fake_section_ids[side])
                ):
                    raise ValueError(
                        f"{side} crossable boundary must have one fake endpoint"
                    )
                real_endpoint = right if left in fake_section_ids[side] else left
                if real_endpoint not in real_section_ids[side]:
                    raise ValueError(
                        f"{side} crossable boundary has no real endpoint"
                    )
    if seen_baseline != list(dict.fromkeys(seen_baseline)):
        raise ValueError("baseline section assigned more than once")
    if seen_target != list(dict.fromkeys(seen_target)):
        raise ValueError("target section assigned more than once")
    if set(seen_baseline) != set(baseline_sections):
        raise ValueError("baseline section missing from alignment plan")
    if set(seen_target) != set(target_sections):
        raise ValueError("target section missing from alignment plan")
    if plan.baseline_section_order != tuple(
        section.section_id for section in baseline.sections
    ):
        raise ValueError("baseline section order does not match input")
    if plan.target_section_order != tuple(
        section.section_id for section in target.sections
    ):
        raise ValueError("target section order does not match input")
