"""Bounded, page-aware context for cross-page table adjudication."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.normalization.table_trace import SourceRowRef
from app.core.types import DocumentIR


@dataclass(frozen=True)
class BoundaryContextItem:
    item_id: str
    section_id: str
    paragraph_id: str
    sentence_index: int | None
    page_no: int | None
    kind: Literal["table_row", "paragraph"]
    text: str


@dataclass(frozen=True)
class TableBoundaryContext:
    boundary_id: str
    side: Literal["baseline", "target"]
    previous_page_no: int | None
    next_page_no: int | None
    inferred: bool
    items: tuple[BoundaryContextItem, ...]


def _logical_items(document: DocumentIR) -> list[BoundaryContextItem]:
    items: list[BoundaryContextItem] = []
    for section in document.sections:
        for paragraph in section.paragraphs:
            table_sentences = [
                (index, sentence.text.strip())
                for index, sentence in enumerate(paragraph.sentences)
                if "|" in sentence.text and sentence.text.strip()
            ]
            if table_sentences:
                for sentence_index, text in table_sentences:
                    items.append(
                        BoundaryContextItem(
                            item_id=(
                                f"{section.section_id}:{paragraph.paragraph_id}:"
                                f"{sentence_index}"
                            ),
                            section_id=section.section_id,
                            paragraph_id=paragraph.paragraph_id,
                            sentence_index=sentence_index,
                            page_no=paragraph.page_no,
                            kind="table_row",
                            text=text,
                        )
                    )
            elif paragraph.text.strip():
                items.append(
                    BoundaryContextItem(
                        item_id=f"{section.section_id}:{paragraph.paragraph_id}:paragraph",
                        section_id=section.section_id,
                        paragraph_id=paragraph.paragraph_id,
                        sentence_index=None,
                        page_no=paragraph.page_no,
                        kind="paragraph",
                        text=paragraph.text.strip(),
                    )
                )
    return items


def _find_source(items: list[BoundaryContextItem], source: SourceRowRef) -> int | None:
    for index, item in enumerate(items):
        if (
            item.section_id == source.section_id
            and item.paragraph_id == source.paragraph_id
            and item.sentence_index == source.sentence_index
        ):
            return index
    return None


def locate_table_boundary_context(
    document: DocumentIR,
    side: Literal["baseline", "target"],
    previous_row: SourceRowRef,
    continuation_row: SourceRowRef,
    *,
    max_items_per_page: int = 6,
) -> TableBoundaryContext | None:
    """Return a bounded physical-page boundary, with a legacy inferred fallback."""
    items = _logical_items(document)
    previous_index = _find_source(items, previous_row)
    continuation_index = _find_source(items, continuation_row)
    if (
        previous_index is None
        or continuation_index is None
        or previous_index >= continuation_index
    ):
        return None

    previous_page = items[previous_index].page_no
    next_page = items[continuation_index].page_no
    inferred = previous_page is None or next_page is None
    if not inferred:
        if next_page != previous_page + 1:
            return None
        previous_items = [
            item
            for item in items[: previous_index + 1]
            if item.page_no == previous_page
        ][-max_items_per_page:]
        next_items = [
            item
            for item in items[previous_index + 1 :]
            if item.page_no == next_page
        ][:max_items_per_page]
        selected = [*previous_items, *next_items]
    else:
        start = max(0, previous_index - max_items_per_page + 1)
        stop = min(len(items), continuation_index + max_items_per_page)
        selected = items[start:stop]

    if not any(item.item_id == items[previous_index].item_id for item in selected):
        return None
    if not any(item.item_id == items[continuation_index].item_id for item in selected):
        return None

    boundary_id = (
        f"{side}:{previous_row.section_id}:{previous_row.paragraph_id}:"
        f"{previous_row.sentence_index}->{continuation_row.paragraph_id}:"
        f"{continuation_row.sentence_index}"
    )
    return TableBoundaryContext(
        boundary_id=boundary_id,
        side=side,
        previous_page_no=previous_page,
        next_page_no=next_page,
        inferred=inferred,
        items=tuple(selected),
    )
