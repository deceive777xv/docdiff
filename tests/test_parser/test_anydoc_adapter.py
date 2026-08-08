"""Tests for the AnyDoc adapter."""
from __future__ import annotations

import sys
import builtins
from types import ModuleType

import pytest


def _fake_anydoc(markdown_or_error) -> ModuleType:
    module = ModuleType("anydoc")

    def to_markdown(_path: str) -> str:
        if isinstance(markdown_or_error, BaseException):
            raise markdown_or_error
        return markdown_or_error

    module.to_markdown = to_markdown  # type: ignore[attr-defined]
    return module


def test_is_available_returns_bool():
    from app.core.parser.anydoc_adapter import is_available

    assert isinstance(is_available(), bool)


def test_extract_reports_missing_firecrawl_anydoc_dependency(tmp_path, monkeypatch):
    source = tmp_path / "sample.docx"
    source.write_bytes(b"fixture")
    real_import = builtins.__import__

    def import_without_anydoc(name, *args, **kwargs):
        if name == "anydoc":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_anydoc)

    from app.core.parser.anydoc_adapter import extract

    with pytest.raises(RuntimeError, match="firecrawl-anydoc is not installed"):
        extract(str(source))


def test_extract_converts_markdown_cleans_dispimg_and_preserves_nan(tmp_path, monkeypatch):
    source = tmp_path / "sample.docx"
    source.write_bytes(b"fixture")
    monkeypatch.setitem(
        sys.modules,
        "anydoc",
        _fake_anydoc(
            "# Title\n\n| Image | Value |\n| --- | --- |\n"
            '| =DISPIMG("ID_1") | NaN |'
        ),
    )

    from app.core.parser.anydoc_adapter import extract

    ir = extract(str(source))
    assert ir.title == "sample"
    assert ir.file_hash
    assert "DISPIMG" not in ir.plain_text
    assert "NaN" in ir.plain_text
    assert ir.sections[0].paragraphs[0].page_no is None


def test_extract_propagates_anydoc_conversion_error(tmp_path, monkeypatch):
    source = tmp_path / "broken.docx"
    source.write_bytes(b"broken")

    class ConversionFailure(Exception):
        pass

    failure = ConversionFailure("cannot convert")
    monkeypatch.setitem(sys.modules, "anydoc", _fake_anydoc(failure))

    from app.core.parser.anydoc_adapter import extract

    with pytest.raises(ConversionFailure, match="cannot convert"):
        extract(str(source))


def test_real_anydoc_csv_smoke(tmp_path):
    pytest.importorskip("anydoc")
    source = tmp_path / "sample.csv"
    source.write_text("name,value\nalpha,1\n", encoding="utf-8")

    from app.core.parser.anydoc_adapter import extract
    from app.core.structure_repair.storage import prepare_import_ir

    ir = extract(str(source))
    assert "alpha" in ir.plain_text
    assert "1" in ir.plain_text
    artifacts = prepare_import_ir(tmp_path / "csv-data", ir)
    assert artifacts.normalized_path.exists()


def test_real_anydoc_docx_and_xlsx_enter_import_pipeline(tmp_path):
    pytest.importorskip("anydoc")
    docx = pytest.importorskip("docx")
    openpyxl = pytest.importorskip("openpyxl")

    docx_path = tmp_path / "sample.docx"
    document = docx.Document()
    document.add_heading("Contract", level=1)
    document.add_paragraph("Alpha obligation")
    document.save(docx_path)

    xlsx_path = tmp_path / "sample.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    sheet.append(["name", "value"])
    sheet.append(["alpha", 1])
    workbook.save(xlsx_path)

    from app.core.parser.anydoc_adapter import extract
    from app.core.structure_repair.storage import prepare_import_ir

    for source, expected in ((docx_path, "Alpha obligation"), (xlsx_path, "alpha")):
        ir = extract(str(source))
        assert expected in ir.plain_text
        artifacts = prepare_import_ir(tmp_path / f"{source.suffix[1:]}-data", ir)
        assert artifacts.raw_path.exists()
        assert artifacts.normalized_path.exists()
