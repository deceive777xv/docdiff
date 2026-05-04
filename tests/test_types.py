"""Tests for app/core/types.py shared data classes."""

import pytest
from app.core.types import (
    DocumentIR,
    Section,
    Paragraph,
    Sentence,
    DiffItem,
    ComparePolicy,
    RetrievalScope,
    ParseQualityReport,
)


def test_document_ir_construction():
    """DocumentIR can be constructed with sections and paragraphs."""
    sentence = Sentence(text="Hello world.")
    paragraph = Paragraph(
        paragraph_id="p1",
        page_no=1,
        text="Hello world.",
        sentences=[sentence],
    )
    section = Section(
        section_id="s1",
        title="Introduction",
        level=1,
        paragraphs=[paragraph],
    )
    doc = DocumentIR(
        doc_id="doc1",
        title="Test Document",
        file_hash="abc123",
        sections=[section],
        plain_text="Hello world.",
    )

    assert doc.doc_id == "doc1"
    assert doc.title == "Test Document"
    assert doc.file_hash == "abc123"
    assert len(doc.sections) == 1
    assert doc.sections[0].section_id == "s1"
    assert doc.sections[0].title == "Introduction"
    assert doc.sections[0].level == 1
    assert len(doc.sections[0].paragraphs) == 1
    assert doc.sections[0].paragraphs[0].paragraph_id == "p1"
    assert doc.sections[0].paragraphs[0].page_no == 1
    assert doc.sections[0].paragraphs[0].text == "Hello world."
    assert len(doc.sections[0].paragraphs[0].sentences) == 1
    assert doc.sections[0].paragraphs[0].sentences[0].text == "Hello world."
    assert doc.plain_text == "Hello world."


def test_diff_item_stores_all_fields():
    """DiffItem stores all fields correctly."""
    diff = DiffItem(
        diff_id="d1",
        section_path="1.2.3",
        diff_type="实质修改",
        risk_level="high",
        baseline_text="Original clause text.",
        target_text="Revised clause text.",
        similarity_score=0.42,
        explanation="Material change in obligation scope.",
        baseline_page=5,
        target_page=6,
    )

    assert diff.diff_id == "d1"
    assert diff.section_path == "1.2.3"
    assert diff.diff_type == "实质修改"
    assert diff.risk_level == "high"
    assert diff.baseline_text == "Original clause text."
    assert diff.target_text == "Revised clause text."
    assert diff.similarity_score == pytest.approx(0.42)
    assert diff.explanation == "Material change in obligation scope."
    assert diff.baseline_page == 5
    assert diff.target_page == 6


def test_compare_policy_defaults():
    """ComparePolicy defaults are threshold=0.75, use_llm_classify=True, rule_strengthen=True."""
    policy = ComparePolicy()

    assert policy.similarity_threshold == pytest.approx(0.75)
    assert policy.use_llm_classify is True
    assert policy.rule_strengthen is True


def test_retrieval_scope_enum_values():
    """RetrievalScope enum values are correct strings."""
    assert RetrievalScope.CURRENT_DOC.value == "current_doc"
    assert RetrievalScope.BASELINE.value == "baseline"
    assert RetrievalScope.TARGET.value == "target"
    assert RetrievalScope.STANDARD_LIB.value == "standard_lib"
    assert RetrievalScope.ALL.value == "all"


def test_parse_quality_report_with_ocr():
    """ParseQualityReport with needs_ocr=True and non-empty ocr_pages."""
    report = ParseQualityReport(
        quality_score=0.55,
        needs_ocr=True,
        ocr_pages=[2, 3, 7],
        warnings=["Low resolution on page 2"],
    )

    assert report.quality_score == pytest.approx(0.55)
    assert report.needs_ocr is True
    assert report.ocr_pages == [2, 3, 7]
    assert len(report.warnings) == 1
    assert report.warnings[0] == "Low resolution on page 2"
