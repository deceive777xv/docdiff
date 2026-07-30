"""Conservative import-time normalization for parsed DocumentIR."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Iterable

from app.core.document_ir_codec import document_ir_to_dict
from app.core.types import DocumentIR, Paragraph, Section, Sentence

from .llm import (
    adjudicate_page_noise_batch,
    adjudicate_paragraph_merge,
    adjudicate_section_parent,
    review_page_noise_batch,
)
from .models import (
    RejectedStructureCandidate,
    StructureRepairDecision,
    StructureRepairOperation,
    StructureRepairResult,
    StructureRepairTrace,
)


SCHEMA_VERSION = 2
ALGORITHM_VERSION = "post-parse-structure-v4"

_NUMBERED_TITLE_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)+)\s+(?P<title>\S.{0,100})\s*$"
)
_SECTION_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")
_GENERIC_STRUCTURE_RE = re.compile(
    r"^\s*(?:[A-Za-zＡ-Ｚａ-ｚ]\s*[)）．.、]|[（(][A-Za-zＡ-Ｚａ-ｚ][）)])"
)
_EXPLICIT_PAGE_NUMBER_RE = re.compile(r"^\s*第\s*\d+\s*页\s*$")
_PURE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")
_TERMINAL_RE = re.compile(r"[。！？!?；;：:]\s*$")
_HEADING_END_RE = re.compile(r"[。！？!?；;]\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_JOIN_LIST_MARKER_RE = re.compile(r"^(?P<marker>\s*[-*+•]\s+)(?P<body>.*)$", re.DOTALL)
_IMAGE_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:\*\*)?==>\s*picture\s*\[\s*\d+\s*x\s*\d+\s*]\s*"
    r"intentionally omitted\s*<==\s*$",
    re.IGNORECASE,
)
_DOCUMENT_CODE_RE = re.compile(
    r"^\s*(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)"
    r"[A-Z0-9]+(?:-[A-Z0-9]+){2,}\s*[（(]\d{8}[）)]\s*$"
)
_PUNCTUATION_ONLY_RE = re.compile(
    r"^[\s…·—\-_=,.，。:：;；!?！？、（）()\[\]{}]+$"
)
_CONTINUATION_SUFFIXES = (
    "的",
    "地",
    "得",
    "在",
    "对",
    "与",
    "和",
    "或",
    "及",
    "为",
    "是",
    "当",
    "若",
    "如",
    "到",
    "从",
    "由",
    "将",
    "并",
    "且",
    "则",
    "时",
    "后",
    "前",
    "内",
    "中",
    "未",
    "不",
    "应",
    "可",
    "需",
    "必须",
    "包括",
    "例如",
    "通过",
)
_CONTINUATION_PREFIXES = (
    "并",
    "且",
    "或",
    "以及",
    "同时",
    "时",
    "后",
    "前",
    "则",
    "而",
    "但",
)
def _stable_id(prefix: str, source_ids: Iterable[str]) -> str:
    digest = hashlib.sha256(
        "\x1f".join(source_ids).encode("utf-8")
    ).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _operation_id(operation_type: str, source_ids: list[str]) -> str:
    return _stable_id(operation_type, source_ids)


def _document_hash(document: DocumentIR) -> str:
    payload = json.dumps(
        document_ir_to_dict(document),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_title(title: str) -> str:
    value = re.sub(r"^\s*(?:\*\*|__)(.*?)(?:\*\*|__)\s*$", r"\1", title)
    return re.sub(r"\s+", " ", value).strip()


def _section_number(title: str) -> tuple[int, ...] | None:
    match = _SECTION_NUMBER_RE.match(_normalize_title(title))
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:
        return None


def _strict_heading(paragraph: Paragraph, current: Section) -> tuple[str, tuple[int, ...]] | None:
    if "\n" in paragraph.text or len(paragraph.text.strip()) > 120:
        return None
    if _HEADING_END_RE.search(paragraph.text):
        return None
    match = _NUMBERED_TITLE_RE.match(_normalize_title(paragraph.text))
    if not match:
        return None
    current_number = _section_number(current.title)
    candidate = tuple(int(part) for part in match.group("number").split("."))
    if current_number is None:
        return None
    same_level_next = (
        len(candidate) == len(current_number)
        and candidate[:-1] == current_number[:-1]
        and candidate[-1] == current_number[-1] + 1
    )
    direct_child = (
        len(candidate) == len(current_number) + 1
        and candidate[:-1] == current_number
        and candidate[-1] == 1
    )
    if not (same_level_next or direct_child):
        return None
    return _normalize_title(paragraph.text), candidate


def _normalize_titles(
    document: DocumentIR,
    operations: list[StructureRepairOperation],
) -> None:
    for section in document.sections:
        normalized = _normalize_title(section.title)
        if normalized == section.title:
            continue
        operations.append(
            StructureRepairOperation(
                operation_id=_operation_id("normalize_title", [section.section_id]),
                type="normalize_title",
                source_ids=[section.section_id],
                output_id=section.section_id,
                reason="strip Markdown emphasis and normalize whitespace",
            )
        )
        section.title = normalized


def _demote_generic_sections(
    document: DocumentIR,
    operations: list[StructureRepairOperation],
) -> None:
    repaired: list[Section] = []
    numbered_parent: Section | None = None
    for section in document.sections:
        if _section_number(section.title) is not None:
            repaired.append(section)
            numbered_parent = section
            continue
        if numbered_parent is None or not _GENERIC_STRUCTURE_RE.match(section.title):
            repaired.append(section)
            continue
        title_paragraph_id = _stable_id("paragraph", [section.section_id])
        page_no = section.paragraphs[0].page_no if section.paragraphs else None
        numbered_parent.paragraphs.append(
            Paragraph(
                paragraph_id=title_paragraph_id,
                text=section.title,
                sentences=[Sentence(text=section.title)],
                page_no=page_no,
            )
        )
        numbered_parent.paragraphs.extend(section.paragraphs)
        operations.append(
            StructureRepairOperation(
                operation_id=_operation_id("demote_to_paragraph", [section.section_id]),
                type="demote_to_paragraph",
                source_ids=[section.section_id],
                output_id=title_paragraph_id,
                target_section_id=numbered_parent.section_id,
                reason="generic repeated structure title under numbered section",
            )
        )
    document.sections = repaired


def _promote_numbered_paragraphs(
    document: DocumentIR,
    operations: list[StructureRepairOperation],
) -> None:
    repaired: list[Section] = []
    for source_section in document.sections:
        active = Section(
            section_id=source_section.section_id,
            title=source_section.title,
            level=source_section.level,
            paragraphs=[],
        )
        repaired.append(active)
        for paragraph in source_section.paragraphs:
            heading = _strict_heading(paragraph, active)
            if heading is None:
                active.paragraphs.append(paragraph)
                continue
            title, number = heading
            generated_id = _stable_id("section", [paragraph.paragraph_id])
            active = Section(
                section_id=generated_id,
                title=title,
                level=len(number),
                paragraphs=[],
            )
            repaired.append(active)
            operations.append(
                StructureRepairOperation(
                    operation_id=_operation_id(
                        "promote_to_section",
                        [paragraph.paragraph_id],
                    ),
                    type="promote_to_section",
                    source_ids=[paragraph.paragraph_id],
                    output_id=generated_id,
                    reason="strict numbered heading with compatible local sequence",
                )
            )
    document.sections = repaired


def _page_boundary_ids(document: DocumentIR) -> tuple[set[str], set[str]]:
    by_page: dict[int, list[Paragraph]] = defaultdict(list)
    for section in document.sections:
        for paragraph in section.paragraphs:
            if paragraph.page_no is not None:
                by_page[paragraph.page_no].append(paragraph)
    first = {items[0].paragraph_id for items in by_page.values() if items}
    last = {items[-1].paragraph_id for items in by_page.values() if items}
    return first, last


def _remove_noise(
    document: DocumentIR,
    operations: list[StructureRepairOperation],
) -> None:
    first_ids, last_ids = _page_boundary_ids(document)
    occurrences: dict[str, list[Paragraph]] = defaultdict(list)
    for section in document.sections:
        for paragraph in section.paragraphs:
            normalized = re.sub(r"\s+", " ", paragraph.text).strip().casefold()
            occurrences[normalized].append(paragraph)
    repeated_boundary_ids: set[str] = set()
    repeated_metadata_ids: set[str] = set()
    repeated_punctuation_ids: set[str] = set()
    ordered_paragraphs = [
        paragraph
        for section in document.sections
        for paragraph in section.paragraphs
    ]
    ordered_index = {
        paragraph.paragraph_id: index
        for index, paragraph in enumerate(ordered_paragraphs)
    }
    for normalized, paragraphs in occurrences.items():
        pages = {paragraph.page_no for paragraph in paragraphs if paragraph.page_no}
        if (
            normalized
            and len(pages) >= 2
            and all(
                paragraph.paragraph_id in first_ids
                or paragraph.paragraph_id in last_ids
                for paragraph in paragraphs
            )
        ):
            repeated_boundary_ids.update(p.paragraph_id for p in paragraphs)
        if (
            len(paragraphs) >= 2
            and len(normalized) <= 100
            and _DOCUMENT_CODE_RE.fullmatch(paragraphs[0].text)
        ):
            repeated_metadata_ids.update(p.paragraph_id for p in paragraphs)
        if len(paragraphs) >= 2 and _PUNCTUATION_ONLY_RE.fullmatch(
            paragraphs[0].text
        ):
            for paragraph in paragraphs:
                index = ordered_index[paragraph.paragraph_id]
                neighbors = ordered_paragraphs[
                    max(0, index - 1) : min(len(ordered_paragraphs), index + 2)
                ]
                if any(
                    neighbor.paragraph_id != paragraph.paragraph_id
                    and not _PUNCTUATION_ONLY_RE.fullmatch(neighbor.text)
                    for neighbor in neighbors
                ):
                    repeated_punctuation_ids.add(paragraph.paragraph_id)

    for section in document.sections:
        retained: list[Paragraph] = []
        for paragraph in section.paragraphs:
            is_page_number = bool(
                _EXPLICIT_PAGE_NUMBER_RE.fullmatch(paragraph.text)
                or (
                    _PURE_NUMBER_RE.fullmatch(paragraph.text)
                    and (
                        paragraph.paragraph_id in first_ids
                        or paragraph.paragraph_id in last_ids
                    )
                )
            )
            is_placeholder = bool(_IMAGE_PLACEHOLDER_RE.fullmatch(paragraph.text))
            if (
                not is_page_number
                and not is_placeholder
                and paragraph.paragraph_id not in repeated_boundary_ids
                and paragraph.paragraph_id not in repeated_metadata_ids
                and paragraph.paragraph_id not in repeated_punctuation_ids
            ):
                retained.append(paragraph)
                continue
            operations.append(
                StructureRepairOperation(
                    operation_id=_operation_id("remove_noise", [paragraph.paragraph_id]),
                    type="remove_noise",
                    source_ids=[paragraph.paragraph_id],
                    reason=(
                        "printed page number"
                        if is_page_number
                        else (
                            "intentionally omitted image placeholder"
                            if is_placeholder
                            else (
                                "stable repeated document metadata"
                                if paragraph.paragraph_id in repeated_metadata_ids
                                else (
                                    "repeated isolated punctuation fragment"
                                    if paragraph.paragraph_id
                                    in repeated_punctuation_ids
                                    else "stable repeated page boundary text"
                                )
                            )
                        )
                    ),
                )
            )
        section.paragraphs = retained


@dataclass(frozen=True)
class _PageBoundaryItem:
    paragraph: Paragraph
    sentence_index: int | None
    text: str


def _has_table_sentence_rows(paragraph: Paragraph) -> bool:
    rows = [sentence.text for sentence in paragraph.sentences if sentence.text.strip()]
    return bool(rows) and all(
        row.lstrip().startswith("|") and row.rstrip().endswith("|")
        for row in rows
    )


def _adjudicate_page_boundary_noise(
    document: DocumentIR,
    provider: object | None,
    model: str,
    operations: list[StructureRepairOperation],
    decisions: list[StructureRepairDecision],
    rejected: list[RejectedStructureCandidate],
    *,
    review_changes: bool,
) -> None:
    if provider is None:
        return
    ordered_paragraphs = [
        paragraph
        for section in document.sections
        for paragraph in section.paragraphs
    ]
    ordered: list[_PageBoundaryItem] = []
    for paragraph in ordered_paragraphs:
        if paragraph.sentences and (
            _is_table(paragraph) or _has_table_sentence_rows(paragraph)
        ):
            ordered.extend(
                _PageBoundaryItem(paragraph, sentence_index, sentence.text)
                for sentence_index, sentence in enumerate(paragraph.sentences)
                if sentence.text.strip() and not _is_separator(sentence.text)
            )
        else:
            ordered.append(_PageBoundaryItem(paragraph, None, paragraph.text))
    by_page: dict[int, list[_PageBoundaryItem]] = defaultdict(list)
    for item in ordered:
        if item.paragraph.page_no is not None:
            by_page[item.paragraph.page_no].append(item)
    removed_paragraph_ids: set[str] = set()
    removed_sentence_indexes: dict[str, set[int]] = defaultdict(set)
    row_removal_judgments: dict[str, list[tuple[float, str]]] = defaultdict(list)

    def page_edges(
        page_items: list[_PageBoundaryItem],
    ) -> tuple[list[_PageBoundaryItem], list[_PageBoundaryItem]]:
        start: list[_PageBoundaryItem] = []
        end: list[_PageBoundaryItem] = []
        item_count = len(page_items)
        for index, item in enumerate(page_items):
            start_distance = index
            end_distance = item_count - index - 1
            if start_distance <= end_distance and start_distance < 6:
                start.append(item)
            elif end_distance < 6:
                end.append(item)
        end.reverse()
        return start, end

    pages = sorted(by_page)
    edges = {page_no: page_edges(by_page[page_no]) for page_no in pages}
    batches: list[
        tuple[str, list[tuple[str, _PageBoundaryItem]]]
    ] = []
    if pages:
        first_page = pages[0]
        batches.append(
            (
                f"document-start:{first_page}",
                [("document_start", item) for item in edges[first_page][0]],
            )
        )
        for previous_page, next_page in zip(pages, pages[1:], strict=False):
            batches.append(
                (
                    f"pages:{previous_page}:{next_page}",
                    [
                        ("previous_page_end", item)
                        for item in edges[previous_page][1]
                    ]
                    + [
                        ("next_page_start", item)
                        for item in edges[next_page][0]
                    ],
                )
            )
        last_page = pages[-1]
        batches.append(
            (
                f"document-end:{last_page}",
                [("document_end", item) for item in edges[last_page][1]],
            )
        )

    for boundary_source, targets in batches:
        if not targets:
            continue
        boundary_id = _stable_id(
            "page-boundary",
            [ALGORITHM_VERSION, document.doc_id, boundary_source],
        )
        request_items = [
            {
                "id": f"L{index:02d}",
                "position": position,
                "text": item.text,
            }
            for index, (position, item) in enumerate(targets, start=1)
        ]
        initial, failure_code = adjudicate_page_noise_batch(
            boundary_id,
            request_items,
            provider,
            model,
        )
        if initial is None:
            rejected.append(
                RejectedStructureCandidate(
                    candidate_id=boundary_id,
                    code=f"initial_{failure_code}",
                    reason="page-boundary batch initial judgment was invalid",
                )
            )
            continue
        for (_, item), label in zip(targets, initial, strict=True):
            decisions.append(
                StructureRepairDecision(
                    candidate_id=f"{boundary_id}:initial:{label.item_id}",
                    action=label.action,
                    source_ids=[item.paragraph.paragraph_id],
                    target_section_id="",
                    confidence=label.confidence,
                    reason=label.reason,
                )
            )
        if not any(
            label.action == "remove_as_page_noise" for label in initial
        ):
            continue
        if review_changes:
            review, failure_code = review_page_noise_batch(
                boundary_id,
                request_items,
                initial,
                provider,
                model,
            )
            if review is None:
                rejected.append(
                    RejectedStructureCandidate(
                        candidate_id=boundary_id,
                        code=f"review_{failure_code}",
                        reason="page-boundary batch review was invalid",
                    )
                )
                continue
            for (_, item), label in zip(targets, review, strict=True):
                decisions.append(
                    StructureRepairDecision(
                        candidate_id=f"{boundary_id}:review:{label.item_id}",
                        action=label.action,
                        source_ids=[item.paragraph.paragraph_id],
                        target_section_id="",
                        confidence=label.confidence,
                        reason=label.reason,
                    )
                )
            for index, (initial_label, review_label) in enumerate(
                zip(initial, review, strict=True)
            ):
                if initial_label.action == review_label.action:
                    continue
                rejected.append(
                    RejectedStructureCandidate(
                        candidate_id=(
                            f"{boundary_id}:{request_items[index]['id']}"
                        ),
                        code="page_noise_review_disagreement",
                        reason=(
                            "initial and review labels disagree; the fixed item is kept"
                        ),
                    )
                )
        else:
            review = initial

        removable: set[str] = set()
        sides = dict.fromkeys(position for position, _ in targets)
        for position in sides:
            edge_is_open = True
            for index, (target_position, _item) in enumerate(targets):
                if target_position != position:
                    continue
                agreed = (
                    initial[index].action == "remove_as_page_noise"
                    and review[index].action == "remove_as_page_noise"
                )
                if edge_is_open and agreed:
                    removable.add(request_items[index]["id"])
                else:
                    edge_is_open = False
                    if agreed:
                        rejected.append(
                            RejectedStructureCandidate(
                                candidate_id=(
                                    f"{boundary_id}:{request_items[index]['id']}"
                                ),
                                code="non_contiguous_page_noise",
                                reason=(
                                    "a retained outer item prevents deleting this "
                                    "more-internal candidate"
                                ),
                            )
                        )

        for index, (_position, item) in enumerate(targets):
            item_id = request_items[index]["id"]
            if item_id not in removable:
                continue
            confidence = min(
                initial[index].confidence,
                review[index].confidence,
            )
            reason = initial[index].reason
            if review_changes:
                reason = f"initial: {reason}; review: {review[index].reason}"
            if item.sentence_index is None:
                removed_paragraph_ids.add(item.paragraph.paragraph_id)
                operations.append(
                    StructureRepairOperation(
                        operation_id=_operation_id(
                            "remove_noise",
                            [item.paragraph.paragraph_id],
                        ),
                        type="remove_noise",
                        source_ids=[item.paragraph.paragraph_id],
                        reason=reason,
                        actor="llm",
                        confidence=confidence,
                    )
                )
            else:
                removed_sentence_indexes[item.paragraph.paragraph_id].add(
                    item.sentence_index
                )
                following_index = item.sentence_index + 1
                if (
                    following_index < len(item.paragraph.sentences)
                    and _is_separator(
                        item.paragraph.sentences[following_index].text
                    )
                ):
                    removed_sentence_indexes[item.paragraph.paragraph_id].add(
                        following_index
                    )
                row_removal_judgments[item.paragraph.paragraph_id].append(
                    (confidence, reason)
                )

    for paragraph_id, sentence_indexes in removed_sentence_indexes.items():
        judgments = row_removal_judgments[paragraph_id]
        operations.append(
            StructureRepairOperation(
                operation_id=_operation_id(
                    "remove_noise_rows",
                    [
                        paragraph_id,
                        *(str(index) for index in sorted(sentence_indexes)),
                    ],
                ),
                type="remove_noise_rows",
                source_ids=[paragraph_id],
                source_sentence_indexes=sorted(sentence_indexes),
                reason="; ".join(dict.fromkeys(reason for _, reason in judgments)),
                actor="llm",
                confidence=min(confidence for confidence, _ in judgments),
            )
        )

    if not removed_paragraph_ids and not removed_sentence_indexes:
        return
    for section in document.sections:
        for paragraph in section.paragraphs:
            sentence_indexes = removed_sentence_indexes.get(
                paragraph.paragraph_id
            )
            if not sentence_indexes:
                continue
            paragraph.sentences = [
                sentence
                for sentence_index, sentence in enumerate(paragraph.sentences)
                if sentence_index not in sentence_indexes
            ]
            paragraph.text = "\n".join(
                sentence.text for sentence in paragraph.sentences
            )
            if not paragraph.sentences:
                removed_paragraph_ids.add(paragraph.paragraph_id)
        section.paragraphs = [
            paragraph
            for paragraph in section.paragraphs
            if paragraph.paragraph_id not in removed_paragraph_ids
        ]


def _is_table(paragraph: Paragraph) -> bool:
    lines = [line for line in paragraph.text.splitlines() if line.strip()]
    return bool(lines) and all(_TABLE_ROW_RE.match(line) for line in lines)


def _is_heading_like(text: str) -> bool:
    value = text.strip()
    return bool(
        _NUMBERED_TITLE_RE.match(value)
        or _GENERIC_STRUCTURE_RE.match(value)
        or (len(value) <= 20 and not _TERMINAL_RE.search(value))
    )


def _has_heading_like_boundary(
    previous: Paragraph,
    following: Paragraph,
) -> bool:
    following_value = following.text.strip()
    following_is_explicit_heading = bool(
        _NUMBERED_TITLE_RE.match(following_value)
        or _GENERIC_STRUCTURE_RE.match(following_value)
    )
    return _is_heading_like(previous.text) or following_is_explicit_heading


def _split_join_list_marker(text: str) -> tuple[str, str]:
    match = _JOIN_LIST_MARKER_RE.match(text)
    if match is None:
        return "", text
    return match.group("marker"), match.group("body")


def _join_boundary_text(previous: str, following: str) -> tuple[str, str]:
    removed_marker, following_body = _split_join_list_marker(following.lstrip())
    left = previous.rstrip()
    right = following_body.lstrip()
    separator = ""
    if (
        left
        and right
        and left[-1].isascii()
        and right[0].isascii()
        and left[-1].isalnum()
        and right[0].isalnum()
    ):
        separator = " "
    return left + separator + right, removed_marker


def _certain_continuation(previous: Paragraph, following: Paragraph) -> bool:
    left = previous.text.rstrip()
    right = following.text.lstrip()
    if (
        not left
        or not right
        or _is_table(previous)
        or _is_table(following)
        or _has_heading_like_boundary(previous, following)
    ):
        return False
    if _IMAGE_PLACEHOLDER_RE.fullmatch(previous.text) or _IMAGE_PLACEHOLDER_RE.fullmatch(following.text):
        return False
    if _TERMINAL_RE.search(left):
        return False
    if previous.page_no is None or following.page_no is None:
        return False
    if following.page_no - previous.page_no not in {0, 1}:
        return False
    has_explicit_continuity = left.endswith(
        _CONTINUATION_SUFFIXES
    ) or right.startswith(_CONTINUATION_PREFIXES)
    if not has_explicit_continuity:
        return False
    return True


def _table_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _is_separator(row: str) -> bool:
    cells = _table_cells(row)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _adjudicate_unnumbered_sections(
    document: DocumentIR,
    provider: object | None,
    model: str,
    operations: list[StructureRepairOperation],
    decisions: list[StructureRepairDecision],
    rejected: list[RejectedStructureCandidate],
    *,
    review_changes: bool,
) -> None:
    if provider is None:
        return
    for index in range(1, len(document.sections)):
        previous = document.sections[index - 1]
        section = document.sections[index]
        if (
            _section_number(previous.title) is None
            or _section_number(section.title) is not None
            or _GENERIC_STRUCTURE_RE.match(section.title)
            or section.level > previous.level
        ):
            continue
        judgment, rejection_code = adjudicate_section_parent(
            section,
            previous,
            provider,
            model,
        )
        candidate_id = f"section:{section.section_id}"
        if judgment is None:
            rejected.append(
                RejectedStructureCandidate(
                    candidate_id=candidate_id,
                    code=rejection_code,
                    reason="LLM judgment was unavailable or failed validation",
                )
            )
            continue
        if review_changes and judgment.action == "move_to_section":
            review, review_code = adjudicate_section_parent(
                section,
                previous,
                provider,
                model,
            )
            if review is None or review.action != judgment.action:
                rejected.append(
                    RejectedStructureCandidate(
                        candidate_id=candidate_id,
                        code=(
                            f"review_{review_code}"
                            if review is None
                            else "section_parent_review_disagreement"
                        ),
                        reason="section move was not confirmed by review",
                    )
                )
                continue
            judgment = replace(
                judgment,
                confidence=min(judgment.confidence, review.confidence),
                reason=f"initial: {judgment.reason}; review: {review.reason}",
            )
        decisions.append(
            StructureRepairDecision(
                candidate_id=candidate_id,
                action=judgment.action,
                source_ids=[section.section_id],
                target_section_id=previous.section_id,
                confidence=judgment.confidence,
                reason=judgment.reason,
            )
        )
        if judgment.action == "keep":
            continue
        section.level = previous.level + 1
        operations.append(
            StructureRepairOperation(
                operation_id=_operation_id(
                    "move_to_section",
                    [section.section_id, previous.section_id],
                ),
                type="move_to_section",
                source_ids=[section.section_id],
                output_id=section.section_id,
                target_section_id=previous.section_id,
                reason=judgment.reason,
                actor="llm",
                confidence=judgment.confidence,
            )
        )


def _is_ambiguous_fragment_candidate(
    previous: Paragraph,
    following: Paragraph,
) -> bool:
    if (
        previous.page_no is None
        or following.page_no is None
        or following.page_no - previous.page_no not in {0, 1}
        or _is_table(previous)
        or _is_table(following)
        or _TERMINAL_RE.search(previous.text.rstrip())
        or _has_heading_like_boundary(previous, following)
    ):
        return False
    return 0 < len(previous.text) + len(following.text) <= 1600


def _adjudicate_paragraph_fragments(
    document: DocumentIR,
    provider: object | None,
    model: str,
    operations: list[StructureRepairOperation],
    decisions: list[StructureRepairDecision],
    rejected: list[RejectedStructureCandidate],
    *,
    review_changes: bool,
) -> None:
    if provider is None:
        return
    section_paths: dict[str, tuple[str, ...]] = {}
    stack: list[tuple[int, str]] = []
    for current_section in document.sections:
        while stack and stack[-1][0] >= current_section.level:
            stack.pop()
        stack.append((current_section.level, current_section.title))
        section_paths[current_section.section_id] = tuple(
            title for _level, title in stack
        )
    for section in document.sections:
        repaired = list(section.paragraphs)
        end_pages = {
            paragraph.paragraph_id: paragraph.page_no
            for paragraph in repaired
        }
        index = 0
        while index + 1 < len(repaired):
            previous = repaired[index]
            following = repaired[index + 1]
            previous_boundary_view = replace(
                previous,
                page_no=end_pages.get(previous.paragraph_id, previous.page_no),
            )
            if not (
                _certain_continuation(previous_boundary_view, following)
                or _is_ambiguous_fragment_candidate(
                    previous_boundary_view,
                    following,
                )
            ):
                index += 1
                continue
            context = repaired[
                max(0, index - 6) : min(len(repaired), index + 8)
            ]
            judgment, rejection_code = adjudicate_paragraph_merge(
                section,
                previous_boundary_view,
                following,
                context,
                provider,
                model,
                rule_evidence=(
                    ("explicit_syntactic_continuity",)
                    if _certain_continuation(previous_boundary_view, following)
                    else ()
                ),
                document_title=document.title,
                section_path=section_paths.get(section.section_id, (section.title,)),
            )
            candidate_id = (
                f"paragraphs:{previous.paragraph_id}:{following.paragraph_id}"
            )
            if judgment is None:
                rejected.append(
                    RejectedStructureCandidate(
                        candidate_id=candidate_id,
                        code=rejection_code,
                        reason="LLM judgment was unavailable or failed validation",
                    )
                )
                index += 1
                continue
            if review_changes and judgment.action == "merge_paragraphs":
                review, review_code = adjudicate_paragraph_merge(
                    section,
                    previous_boundary_view,
                    following,
                    context,
                    provider,
                    model,
                    rule_evidence=(
                        ("explicit_syntactic_continuity",)
                        if _certain_continuation(previous_boundary_view, following)
                        else ()
                    ),
                    document_title=document.title,
                    section_path=section_paths.get(section.section_id, (section.title,)),
                )
                if review is None or review.action != judgment.action:
                    rejected.append(
                        RejectedStructureCandidate(
                            candidate_id=candidate_id,
                            code=(
                                f"review_{review_code}"
                                if review is None
                                else "paragraph_merge_review_disagreement"
                            ),
                            reason="paragraph merge was not confirmed by review",
                        )
                    )
                    index += 1
                    continue
                judgment = replace(
                    judgment,
                    confidence=min(judgment.confidence, review.confidence),
                    reason=f"initial: {judgment.reason}; review: {review.reason}",
                )
            decisions.append(
                StructureRepairDecision(
                    candidate_id=candidate_id,
                    action=judgment.action,
                    source_ids=[
                        previous.paragraph_id,
                        following.paragraph_id,
                    ],
                    target_section_id=section.section_id,
                    confidence=judgment.confidence,
                    reason=judgment.reason,
                )
            )
            if judgment.action == "keep":
                index += 1
                continue
            output_id = _stable_id(
                "paragraph",
                [previous.paragraph_id, following.paragraph_id],
            )
            merged_text, removed_marker = _join_boundary_text(
                previous.text,
                following.text,
            )
            previous_sentences = list(previous.sentences) or [Sentence(previous.text)]
            following_sentences = list(following.sentences) or [Sentence(following.text)]
            merged_boundary, sentence_removed_marker = _join_boundary_text(
                previous_sentences[-1].text,
                following_sentences[0].text,
            )
            if sentence_removed_marker != removed_marker:
                raise ValueError("paragraph and sentence join markers disagree")
            merged_sentences = [
                *previous_sentences[:-1],
                Sentence(merged_boundary),
                *following_sentences[1:],
            ]
            merged = Paragraph(
                paragraph_id=output_id,
                text=merged_text,
                sentences=merged_sentences,
                page_no=previous.page_no,
            )
            end_pages[output_id] = end_pages.get(
                following.paragraph_id,
                following.page_no,
            )
            repaired[index : index + 2] = [merged]
            operations.append(
                StructureRepairOperation(
                    operation_id=_operation_id(
                        "merge_paragraphs",
                        [previous.paragraph_id, following.paragraph_id],
                    ),
                    type="merge_paragraphs",
                    source_ids=[previous.paragraph_id, following.paragraph_id],
                    output_id=output_id,
                    target_section_id=section.section_id,
                    reason=judgment.reason,
                    actor="llm",
                    confidence=judgment.confidence,
                    removed_text=removed_marker,
                )
            )
            index = max(0, index - 1)
        section.paragraphs = repaired


def _rebuild_plain_text(document: DocumentIR) -> None:
    document.plain_text = "\n".join(
        paragraph.text
        for section in document.sections
        for paragraph in section.paragraphs
    )


def _characters(value: str) -> Counter[str]:
    return Counter(character for character in value if not character.isspace())


def _content_counter(document: DocumentIR) -> Counter[str]:
    content: Counter[str] = Counter()
    for section in document.sections:
        content.update(_characters(_normalize_title(section.title)))
        for paragraph in section.paragraphs:
            content.update(_characters(paragraph.text))
    return content


def _source_paragraphs(document: DocumentIR) -> dict[str, Paragraph]:
    return {
        paragraph.paragraph_id: paragraph
        for section in document.sections
        for paragraph in section.paragraphs
    }


def _allowed_removed_content(
    raw: DocumentIR,
    operations: list[StructureRepairOperation],
) -> Counter[str]:
    paragraphs = _source_paragraphs(raw)
    removed: Counter[str] = Counter()
    for operation in operations:
        if operation.type == "remove_noise":
            for source_id in operation.source_ids:
                paragraph = paragraphs.get(source_id)
                if paragraph is None:
                    raise ValueError("noise operation references an unknown paragraph")
                removed.update(_characters(paragraph.text))
        elif operation.type == "remove_noise_rows":
            if len(operation.source_ids) != 1:
                raise ValueError("row noise operation must reference one paragraph")
            paragraph = paragraphs.get(operation.source_ids[0])
            if paragraph is None:
                raise ValueError("row noise operation references an unknown paragraph")
            if len(operation.source_sentence_indexes) != len(
                set(operation.source_sentence_indexes)
            ):
                raise ValueError("row noise operation contains duplicate indexes")
            for sentence_index in operation.source_sentence_indexes:
                if not 0 <= sentence_index < len(paragraph.sentences):
                    raise ValueError("row noise operation references an unknown sentence")
                removed.update(_characters(paragraph.sentences[sentence_index].text))
        elif operation.type == "merge_paragraphs" and operation.removed_text:
            if len(operation.source_ids) != 2:
                raise ValueError("paragraph merge must reference exactly two paragraphs")
            following = paragraphs.get(operation.source_ids[1])
            if following is None:
                raise ValueError("paragraph merge references an unknown paragraph")
            marker, _ = _split_join_list_marker(following.text.lstrip())
            if marker != operation.removed_text:
                raise ValueError("paragraph merge removed marker does not match source")
            removed.update(_characters(marker))
    return removed


def _has_unique_ids(document: DocumentIR) -> bool:
    section_ids = [section.section_id for section in document.sections]
    paragraph_ids = [
        paragraph.paragraph_id
        for section in document.sections
        for paragraph in section.paragraphs
    ]
    return (
        len(section_ids) == len(set(section_ids))
        and len(paragraph_ids) == len(set(paragraph_ids))
    )


def _content_is_conserved(
    raw: DocumentIR,
    repaired: DocumentIR,
    operations: list[StructureRepairOperation],
) -> bool:
    try:
        expected = _content_counter(raw)
        expected.subtract(_allowed_removed_content(raw, operations))
    except ValueError:
        return False
    expected = Counter({key: count for key, count in expected.items() if count})
    return (
        not any(count < 0 for count in expected.values())
        and expected == _content_counter(repaired)
        and _has_unique_ids(repaired)
    )


def repair_document(
    document: DocumentIR,
    *,
    provider: object | None = None,
    model: str = "",
    review_changes: bool = False,
) -> StructureRepairResult:
    """Return a conservative normalized copy and a replayable audit trace."""
    raw_hash = _document_hash(document)
    repaired = deepcopy(document)
    operations: list[StructureRepairOperation] = []
    decisions: list[StructureRepairDecision] = []
    rejected: list[RejectedStructureCandidate] = []
    warnings: list[str] = []

    _normalize_titles(repaired, operations)
    _demote_generic_sections(repaired, operations)
    _promote_numbered_paragraphs(repaired, operations)
    _remove_noise(repaired, operations)
    _adjudicate_page_boundary_noise(
        repaired,
        provider,
        model,
        operations,
        decisions,
        rejected,
        review_changes=review_changes,
    )
    _adjudicate_paragraph_fragments(
        repaired,
        provider,
        model,
        operations,
        decisions,
        rejected,
        review_changes=review_changes,
    )
    _adjudicate_unnumbered_sections(
        repaired,
        provider,
        model,
        operations,
        decisions,
        rejected,
        review_changes=review_changes,
    )
    _rebuild_plain_text(repaired)

    if not _content_is_conserved(document, repaired, operations):
        fallback = deepcopy(document)
        fallback_hash = _document_hash(fallback)
        trace = StructureRepairTrace(
            schema_version=SCHEMA_VERSION,
            algorithm_version=ALGORITHM_VERSION,
            doc_id=document.doc_id,
            raw_hash=raw_hash,
            normalized_hash=fallback_hash,
            status="fallback",
            operations=operations,
            decisions=decisions,
            rejected=rejected,
            warnings=["content_conservation_failed"],
        )
        return StructureRepairResult(
            document=fallback,
            trace=trace,
            status="fallback",
            warnings=["content_conservation_failed"],
        )

    status = "repaired" if operations else "unchanged"
    trace = StructureRepairTrace(
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        doc_id=document.doc_id,
        raw_hash=raw_hash,
        normalized_hash=_document_hash(repaired),
        status=status,
        operations=operations,
        decisions=decisions,
        rejected=rejected,
        warnings=warnings,
    )
    return StructureRepairResult(
        document=repaired,
        trace=trace,
        status=status,
        warnings=warnings,
    )
