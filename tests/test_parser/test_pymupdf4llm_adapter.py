"""Tests for the page-aware PyMuPDF4LLM adapter."""
from __future__ import annotations

import sys


def test_extract_assigns_physical_page_numbers_without_printed_page_labels(
    tmp_path,
    monkeypatch,
):
    import pymupdf

    pdf_path = tmp_path / "two-pages.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "First physical page.")
    document.new_page().insert_text((72, 72), "Second physical page.")
    document.save(pdf_path)
    document.close()

    class ForbiddenPdfPlumber:
        def __getattr__(self, name):
            raise AssertionError("pdfplumber must not be used by the PDF parser")

    monkeypatch.setitem(sys.modules, "pdfplumber", ForbiddenPdfPlumber())

    from app.core.parser.pymupdf4llm_adapter import extract

    ir = extract(str(pdf_path))
    paragraphs = [para for section in ir.sections for para in section.paragraphs]

    assert [para.page_no for para in paragraphs] == [1, 2]
    assert "First physical page" in paragraphs[0].text
    assert "Second physical page" in paragraphs[1].text


def test_page_chunks_preserve_section_state_across_page_boundaries(monkeypatch, tmp_path):
    pdf_path = tmp_path / "section.pdf"
    pdf_path.write_bytes(b"fake-pdf")

    chunks = [
        {
            "metadata": {"page_number": 1},
            "text": "# Shared section\n\nParagraph on page one.",
        },
        {
            "metadata": {"page_number": 2},
            "text": "Paragraph on page two.",
        },
    ]

    import pymupdf4llm

    monkeypatch.setattr(pymupdf4llm, "to_markdown", lambda *_args, **_kwargs: chunks)

    from app.core.parser.pymupdf4llm_adapter import extract

    ir = extract(str(pdf_path))

    assert len(ir.sections) == 1
    assert ir.sections[0].title == "Shared section"
    assert [para.page_no for para in ir.sections[0].paragraphs] == [1, 2]
    assert [para.text for para in ir.sections[0].paragraphs] == [
        "Paragraph on page one.",
        "Paragraph on page two.",
    ]
