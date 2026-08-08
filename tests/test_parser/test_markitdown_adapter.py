"""Tests for the MarkItDown compatibility adapter."""
from __future__ import annotations


def test_is_available_returns_bool():
    from app.core.parser.markitdown_adapter import is_available

    assert isinstance(is_available(), bool)


def test_extract_with_no_llm_client(tmp_path):
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
    assert ir.doc_id
    assert ir.file_hash


def test_non_pdf_paragraph_has_unknown_page_number(tmp_path):
    test_file = tmp_path / "test.html"
    test_file.write_text("<h1>Section</h1><p>Text here.</p>", encoding="utf-8")

    from app.core.parser.markitdown_adapter import extract

    ir = extract(str(test_file))
    assert ir.sections[0].paragraphs[0].page_no is None
