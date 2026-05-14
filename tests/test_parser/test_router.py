"""Tests for router.parse_document."""
from __future__ import annotations

import inspect
import dataclasses

import pytest


def test_supported_extensions_set():
    from app.core.parser.router import SUPPORTED_EXTENSIONS
    for ext in (".pdf", ".docx", ".pptx", ".xlsx", ".html", ".csv", ".epub"):
        assert ext in SUPPORTED_EXTENSIONS


def test_unsupported_extension_raises_value_error(tmp_path):
    from app.core.parser.router import parse_document
    bad = tmp_path / "file.xyz"
    bad.write_text("content")
    with pytest.raises(ValueError, match="Unsupported format"):
        parse_document(str(bad))


def test_parse_document_returns_ir_and_report(tmp_path):
    from app.core.parser.router import parse_document
    from app.core.types import DocumentIR, ParseQualityReport

    f = tmp_path / "doc.html"
    f.write_text("<h1>Title</h1><p>Some content here.</p>", encoding="utf-8")

    ir, report = parse_document(str(f))
    assert isinstance(ir, DocumentIR)
    assert isinstance(report, ParseQualityReport)


def test_parse_document_none_llm_client_does_not_raise(tmp_path):
    from app.core.parser.router import parse_document

    f = tmp_path / "doc.html"
    f.write_text("<p>Hello world</p>", encoding="utf-8")

    ir, report = parse_document(str(f), llm_client=None, llm_model="")
    assert ir is not None
    assert report is not None


def test_parse_document_has_no_mode_parameter():
    from app.core.parser import router
    sig = inspect.signature(router.parse_document)
    assert "mode" not in sig.parameters


@pytest.mark.xfail(reason="ocr_pages will be removed from ParseQualityReport in Task 4")
def test_quality_report_has_no_ocr_pages_field():
    from app.core.types import ParseQualityReport
    fields = {f.name for f in dataclasses.fields(ParseQualityReport)}
    assert "ocr_pages" not in fields
