"""Tests for router.parse_document."""
from __future__ import annotations

import inspect
import dataclasses

import pytest


def test_supported_extension_sets_are_disjoint_and_complete():
    from app.core.parser.router import (
        ANYDOC_EXTENSIONS,
        MARKITDOWN_EXTENSIONS,
        PDF_EXTENSIONS,
        SUPPORTED_EXTENSIONS,
    )

    assert PDF_EXTENSIONS.isdisjoint(ANYDOC_EXTENSIONS)
    assert PDF_EXTENSIONS.isdisjoint(MARKITDOWN_EXTENSIONS)
    assert ANYDOC_EXTENSIONS.isdisjoint(MARKITDOWN_EXTENSIONS)
    assert SUPPORTED_EXTENSIONS == (
        PDF_EXTENSIONS | ANYDOC_EXTENSIONS | MARKITDOWN_EXTENSIONS
    )
    assert PDF_EXTENSIONS == {".pdf"}
    assert ANYDOC_EXTENSIONS == {
        ".doc", ".docx", ".docm",
        ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
        ".xls", ".xlsx", ".xlsm", ".xlsb",
        ".odt", ".ods", ".odp", ".rtf", ".epub", ".csv",
    }
    assert MARKITDOWN_EXTENSIONS == {
        ".html", ".htm", ".json", ".xml", ".txt", ".md", ".markdown",
    }


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


def test_parse_document_accepts_markdown_file(tmp_path):
    from app.core.parser.router import parse_document

    f = tmp_path / "sample.md"
    f.write_text(
        "# 标题\n\n| 序号 | 名称 |\n| :--: | ---- |\n| 1 | 示例 |",
        encoding="utf-8",
    )

    ir, report = parse_document(str(f), llm_client=None, llm_model="")

    assert ir.title == "sample"
    assert ir.sections[0].title == "标题"
    assert "| 序号 | 名称 |" in ir.plain_text
    assert report is not None


def test_parse_document_has_no_mode_parameter():
    from app.core.parser import router
    sig = inspect.signature(router.parse_document)
    assert "mode" not in sig.parameters


def test_pdf_dispatches_only_to_pymupdf(tmp_path, monkeypatch):
    from app.core.parser import router
    from app.core.types import DocumentIR, Paragraph, Section, Sentence

    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-fixture")
    ir = DocumentIR(
        "doc", "Title", "hash",
        [Section("section", "正文", 1, [Paragraph("p", "content", [Sentence("content")])])],
    )
    calls: list[str] = []
    monkeypatch.setattr(
        router.pymupdf4llm_adapter,
        "extract",
        lambda path: calls.append(path) or ir,
    )
    monkeypatch.setattr(
        router.anydoc_adapter,
        "extract",
        lambda *_args: pytest.fail("AnyDoc must not be called"),
    )
    monkeypatch.setattr(
        router.markitdown_adapter,
        "extract",
        lambda *_args, **_kwargs: pytest.fail("MarkItDown must not be called"),
    )

    router.parse_document(str(source))
    assert calls == [str(source)]


@pytest.mark.parametrize("extension", [".docx", ".ppt", ".xlsb", ".odt", ".rtf", ".csv"])
def test_anydoc_extensions_dispatch_only_to_anydoc(extension, tmp_path, monkeypatch):
    from app.core.parser import router
    from app.core.types import DocumentIR, Paragraph, Section, Sentence

    source = tmp_path / f"sample{extension}"
    source.write_bytes(b"fixture")
    ir = DocumentIR(
        "doc", "Title", "hash",
        [Section("section", "正文", 1, [Paragraph("p", "content", [Sentence("content")])])],
    )
    calls: list[str] = []
    monkeypatch.setattr(router.anydoc_adapter, "extract", lambda path: calls.append(path) or ir)
    monkeypatch.setattr(
        router.markitdown_adapter,
        "extract",
        lambda *_args, **_kwargs: pytest.fail("MarkItDown must not be called"),
    )
    monkeypatch.setattr(
        router.pymupdf4llm_adapter,
        "extract",
        lambda *_args, **_kwargs: pytest.fail("PDF parser must not be called"),
    )

    router.parse_document(str(source))
    assert calls == [str(source)]


def test_anydoc_failure_does_not_fall_back_to_markitdown(tmp_path, monkeypatch):
    from app.core.parser import router

    source = tmp_path / "broken.docx"
    source.write_bytes(b"broken")
    monkeypatch.setattr(
        router.anydoc_adapter,
        "extract",
        lambda _path: (_ for _ in ()).throw(RuntimeError("anydoc failed")),
    )
    monkeypatch.setattr(
        router.markitdown_adapter,
        "extract",
        lambda *_args, **_kwargs: pytest.fail("MarkItDown fallback is forbidden"),
    )

    with pytest.raises(RuntimeError, match="anydoc failed"):
        router.parse_document(str(source))


@pytest.mark.parametrize("extension", [".html", ".json", ".xml", ".txt", ".md"])
def test_compatibility_extensions_dispatch_only_to_markitdown(
    extension, tmp_path, monkeypatch
):
    from app.core.parser import router
    from app.core.types import DocumentIR, Paragraph, Section, Sentence

    source = tmp_path / f"sample{extension}"
    source.write_text("content", encoding="utf-8")
    ir = DocumentIR(
        "doc", "Title", "hash",
        [Section("section", "正文", 1, [Paragraph("p", "content", [Sentence("content")])])],
    )
    calls: list[tuple[str, object, str]] = []
    monkeypatch.setattr(
        router.markitdown_adapter,
        "extract",
        lambda path, client, model: calls.append((path, client, model)) or ir,
    )
    monkeypatch.setattr(
        router.anydoc_adapter,
        "extract",
        lambda *_args: pytest.fail("AnyDoc must not be called"),
    )

    client = object()
    router.parse_document(str(source), llm_client=client, llm_model="model")
    assert calls == [(str(source), client, "model")]


def test_quality_report_has_no_ocr_pages_field():
    from app.core.types import ParseQualityReport
    fields = {f.name for f in dataclasses.fields(ParseQualityReport)}
    assert "ocr_pages" not in fields
