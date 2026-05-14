"""Tests for markitdown_adapter."""
from __future__ import annotations

import pytest


def test_is_available_returns_bool():
    from app.core.parser.markitdown_adapter import is_available
    assert isinstance(is_available(), bool)


def test_headingless_content_creates_default_section():
    """Content with no heading auto-inserts a '正文' section."""
    from app.core.parser.markitdown_adapter import _parse_markdown

    md = "Some text without any heading.\nMore text here."
    ir = _parse_markdown(md, "test_doc", "abc123")

    assert len(ir.sections) == 1
    assert ir.sections[0].title == "正文"
    assert ir.sections[0].level == 1
    assert len(ir.sections[0].paragraphs) >= 1
    assert "Some text" in ir.sections[0].paragraphs[0].text


def test_single_heading_with_body():
    from app.core.parser.markitdown_adapter import _parse_markdown

    md = "# Introduction\n\nThis is the intro text.\n\nMore intro text."
    ir = _parse_markdown(md, "test_doc", "abc123")

    assert len(ir.sections) == 1
    assert ir.sections[0].title == "Introduction"
    assert ir.sections[0].level == 1
    assert len(ir.sections[0].paragraphs) == 2


def test_multi_level_headings():
    from app.core.parser.markitdown_adapter import _parse_markdown

    md = (
        "# Chapter 1\n\nIntro paragraph.\n\n"
        "## Section 1.1\n\nSub content here.\n\n"
        "### Subsection 1.1.1\n\nDeep content.\n\n"
        "## Section 1.2\n\nAnother section."
    )
    ir = _parse_markdown(md, "test_doc", "abc123")

    assert len(ir.sections) == 4
    assert ir.sections[0].title == "Chapter 1"
    assert ir.sections[0].level == 1
    assert ir.sections[1].title == "Section 1.1"
    assert ir.sections[1].level == 2
    assert ir.sections[2].title == "Subsection 1.1.1"
    assert ir.sections[2].level == 3
    assert ir.sections[3].title == "Section 1.2"
    assert ir.sections[3].level == 2


def test_extract_with_no_llm_client(tmp_path):
    """extract() with llm_client=None must not raise."""
    test_file = tmp_path / "test.html"
    test_file.write_text("<h1>Hello</h1><p>World</p>", encoding="utf-8")

    from app.core.parser.markitdown_adapter import extract
    ir = extract(str(test_file), llm_client=None, llm_model="")

    assert ir.title == "test"
    assert len(ir.sections) >= 1


def test_extract_populates_doc_id_and_file_hash(tmp_path):
    test_file = tmp_path / "sample.html"
    test_file.write_text("<h1>Title</h1><p>Content here.</p>", encoding="utf-8")

    from app.core.parser.markitdown_adapter import extract
    ir = extract(str(test_file))

    assert ir.doc_id != ""
    assert ir.file_hash != ""


@pytest.mark.xfail(
    reason="Pending Task 4: Paragraph.page_no not yet removed from types.py",
    strict=True,
)
def test_paragraph_has_no_page_no(tmp_path):
    """After migration Paragraph must not have a page_no attribute."""
    test_file = tmp_path / "test.html"
    test_file.write_text("<h1>Section</h1><p>Text here.</p>", encoding="utf-8")

    from app.core.parser.markitdown_adapter import extract
    ir = extract(str(test_file))

    para = ir.sections[0].paragraphs[0]
    assert not hasattr(para, "page_no"), "Paragraph.page_no must not exist after migration"
