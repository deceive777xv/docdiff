"""Tests for app/core/parser/pymupdf_extractor.py."""
import pytest
import fitz
from pathlib import Path

from app.core.types import DocumentIR


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "第一章 总则", fontsize=18)
    page.insert_text((72, 120), "本协议自双方签署之日起生效。", fontsize=12)
    page.insert_text((72, 150), "甲方应在签署后30日内完成付款。", fontsize=12)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "第二章 违约责任", fontsize=18)
    page2.insert_text((72, 120), "违约方需支付合同金额的20%作为违约金。", fontsize=12)
    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_extract_returns_document_ir(sample_pdf):
    """extract() returns a DocumentIR instance."""
    from app.core.parser.pymupdf_extractor import extract
    ir, report = extract(str(sample_pdf))
    assert isinstance(ir, DocumentIR)


def test_sections_extracted(sample_pdf):
    """extract() produces at least 1 section."""
    from app.core.parser.pymupdf_extractor import extract
    ir, report = extract(str(sample_pdf))
    assert len(ir.sections) >= 1


def test_paragraphs_have_page_no(sample_pdf):
    """All paragraphs have page_no >= 1."""
    from app.core.parser.pymupdf_extractor import extract
    ir, report = extract(str(sample_pdf))
    for section in ir.sections:
        for para in section.paragraphs:
            assert para.page_no >= 1


def test_quality_report(sample_pdf):
    """ParseQualityReport for a normal PDF: needs_ocr=False, quality_score > 0.5."""
    from app.core.parser.pymupdf_extractor import extract
    ir, report = extract(str(sample_pdf))
    assert report.needs_ocr is False
    assert report.quality_score > 0.5


def test_file_hash_is_sha256(sample_pdf):
    """file_hash is a 64-character hex string (SHA-256)."""
    from app.core.parser.pymupdf_extractor import extract
    ir, report = extract(str(sample_pdf))
    assert len(ir.file_hash) == 64
    assert all(c in "0123456789abcdef" for c in ir.file_hash)
