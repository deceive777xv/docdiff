"""Tests for shared Markdown-to-DocumentIR conversion."""
from __future__ import annotations


def test_headingless_content_creates_default_section():
    from app.core.parser.markdown_ir import parse_markdown

    ir = parse_markdown(
        "Some text without any heading.\nMore text here.",
        "test_doc",
        "abc123",
    )
    assert len(ir.sections) == 1
    assert ir.sections[0].title == "正文"
    assert ir.sections[0].level == 1
    assert "Some text" in ir.sections[0].paragraphs[0].text


def test_single_heading_with_body():
    from app.core.parser.markdown_ir import parse_markdown

    ir = parse_markdown(
        "# Introduction\n\nThis is the intro text.\n\nMore intro text.",
        "test_doc",
        "abc123",
    )
    assert len(ir.sections) == 1
    assert ir.sections[0].title == "Introduction"
    assert ir.sections[0].level == 1
    assert len(ir.sections[0].paragraphs) == 2


def test_multi_level_headings():
    from app.core.parser.markdown_ir import parse_markdown

    markdown = (
        "# Chapter 1\n\nIntro paragraph.\n\n"
        "## Section 1.1\n\nSub content here.\n\n"
        "### Subsection 1.1.1\n\nDeep content.\n\n"
        "## Section 1.2\n\nAnother section."
    )
    ir = parse_markdown(markdown, "test_doc", "abc123")
    assert [(section.title, section.level) for section in ir.sections] == [
        ("Chapter 1", 1),
        ("Section 1.1", 2),
        ("Subsection 1.1.1", 3),
        ("Section 1.2", 2),
    ]


def test_table_rows_share_one_paragraph_and_keep_unknown_page_number():
    from app.core.parser.markdown_ir import parse_markdown

    ir = parse_markdown(
        "# Data\n\n| Name | Value |\n| --- | --- |\n| A | 1 |",
        "test_doc",
        "abc123",
    )
    paragraph = ir.sections[0].paragraphs[0]
    assert len(paragraph.sentences) == 3
    assert paragraph.page_no is None
