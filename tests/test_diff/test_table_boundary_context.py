from __future__ import annotations

from app.core.normalization.table_trace import SourceRowRef
from app.core.types import DocumentIR, Paragraph, Section, Sentence


def _document(page_numbers: list[int | None]) -> DocumentIR:
    texts = [
        "设计校核表",
        "| 1 | 跨页前半行 |",
        "设计校核表",
        "|   | 跨页后半行 |",
        "| 2 | 下一完整行 |",
    ]
    paragraphs = [
        Paragraph(
            paragraph_id=f"p{index}",
            text=text,
            sentences=[Sentence(text=text)],
            page_no=page_numbers[index],
        )
        for index, text in enumerate(texts)
    ]
    return DocumentIR(
        "doc",
        "Title",
        "hash",
        [Section("section", "正文", 1, paragraphs)],
    )


def test_boundary_context_uses_adjacent_physical_pages_and_keeps_page_header():
    from app.core.normalization.table_boundary_context import locate_table_boundary_context

    context = locate_table_boundary_context(
        _document([1, 1, 2, 2, 2]),
        "baseline",
        SourceRowRef("section", "p1", 0),
        SourceRowRef("section", "p3", 0),
    )

    assert context is not None
    assert context.previous_page_no == 1
    assert context.next_page_no == 2
    assert any(item.text == "设计校核表" and item.paragraph_id == "p2" for item in context.items)
    assert len([item for item in context.items if item.page_no == 1]) <= 6
    assert len([item for item in context.items if item.page_no == 2]) <= 6


def test_boundary_context_rejects_non_adjacent_known_pages():
    from app.core.normalization.table_boundary_context import locate_table_boundary_context

    context = locate_table_boundary_context(
        _document([1, 1, 3, 3, 3]),
        "baseline",
        SourceRowRef("section", "p1", 0),
        SourceRowRef("section", "p3", 0),
    )

    assert context is None


def test_boundary_context_falls_back_to_inferred_window_for_legacy_ir():
    from app.core.normalization.table_boundary_context import locate_table_boundary_context

    context = locate_table_boundary_context(
        _document([None] * 5),
        "target",
        SourceRowRef("section", "p1", 0),
        SourceRowRef("section", "p3", 0),
    )

    assert context is not None
    assert context.previous_page_no is None
    assert context.next_page_no is None
    assert context.inferred is True
    assert {item.paragraph_id for item in context.items} >= {"p1", "p2", "p3"}
