"""Tests for app/core/parser/ir_builder.py"""
from __future__ import annotations
import uuid
import pytest

from app.core.types import DocumentIR, Section, Paragraph, Sentence
from app.core.parser.ir_builder import build_chunks


def _make_paragraph(text: str) -> Paragraph:
    return Paragraph(
        paragraph_id=str(uuid.uuid4()),
        text=text,
        sentences=[Sentence(text=text)],
    )


def _make_ir(sections_data: list[tuple[str, list[tuple[str, list[str]]]]]) -> DocumentIR:
    """Helper: build a DocumentIR from (section_title, [(para_text, [sent_text, ...]), ...])."""
    sections = []
    for sec_title, paras in sections_data:
        sec = Section(
            section_id=str(uuid.uuid4()),
            title=sec_title,
            level=1,
        )
        for para_text, sent_texts in paras:
            sentences = [Sentence(text=s) for s in sent_texts]
            sec.paragraphs.append(Paragraph(
                paragraph_id=str(uuid.uuid4()),
                text=para_text,
                sentences=sentences,
            ))
        sections.append(sec)
    return DocumentIR(
        doc_id=str(uuid.uuid4()),
        title="Test Doc",
        file_hash="abc123",
        sections=sections,
    )


def test_build_chunks_basic():
    """2 sections × 2 paragraphs → 4 chunks with correct version_id and section_path."""
    ir = _make_ir([
        ("Section A", [
            ("Para A1 text.", ["Para A1 text."]),
            ("Para A2 text.", ["Para A2 text."]),
        ]),
        ("Section B", [
            ("Para B1 text.", ["Para B1 text."]),
            ("Para B2 text.", ["Para B2 text."]),
        ]),
    ])

    chunks = build_chunks(ir, "v1")

    assert len(chunks) == 4
    for chunk in chunks:
        assert chunk.version_id == "v1"
    # Section paths
    assert chunks[0].section_path == "Section A"
    assert chunks[1].section_path == "Section A"
    assert chunks[2].section_path == "Section B"
    assert chunks[3].section_path == "Section B"


def test_build_chunks_keeps_paragraph_below_default_limit_intact():
    """Paragraphs below the 2000-character default remain one authoritative chunk."""
    long_sent_1 = "A" * 200 + "."
    long_sent_2 = "B" * 200 + "."
    long_sent_3 = "C" * 200 + "."
    long_text = long_sent_1 + " " + long_sent_2 + " " + long_sent_3
    assert len(long_text) > 500

    ir = _make_ir([
        ("Main", [
            (long_text, [long_sent_1, long_sent_2, long_sent_3]),
        ]),
    ])

    chunks = build_chunks(ir, "v2")

    assert len(chunks) == 1
    assert chunks[0].text == long_text


def test_chunk_no_is_sequential():
    """chunk_no values must be 0, 1, 2, ... in order."""
    ir = _make_ir([
        ("S1", [
            ("Para 1.", ["Para 1."]),
            ("Para 2.", ["Para 2."]),
            ("Para 3.", ["Para 3."]),
        ]),
    ])

    chunks = build_chunks(ir, "v3")

    assert len(chunks) == 3
    for expected, chunk in enumerate(chunks):
        assert chunk.chunk_no == expected


def test_chunks_inherit_paragraph_page_number():
    para = Paragraph(
        paragraph_id="p1",
        text="page-aware content",
        sentences=[Sentence(text="page-aware content")],
        page_no=3,
    )
    ir = DocumentIR(
        doc_id="doc",
        title="Title",
        file_hash="hash",
        sections=[Section("section", "正文", 1, [para])],
    )

    chunks = build_chunks(ir, "v1")

    assert chunks[0].page_no == 3


def test_chunks_use_hierarchical_section_path():
    ir = DocumentIR(
        doc_id="doc",
        title="Title",
        file_hash="hash",
        sections=[
            Section("s1", "1 总则", 1, [_make_paragraph("总则正文")]),
            Section("s2", "1.1 功能", 2, [_make_paragraph("功能正文")]),
            Section("s3", "附表", 3, [_make_paragraph("附表正文")]),
            Section("s4", "2 其他", 1, [_make_paragraph("其他正文")]),
        ],
    )

    chunks = build_chunks(ir, "v1")

    assert [chunk.section_path for chunk in chunks] == [
        "1 总则",
        "1 总则 > 1.1 功能",
        "1 总则 > 1.1 功能 > 附表",
        "2 其他",
    ]
