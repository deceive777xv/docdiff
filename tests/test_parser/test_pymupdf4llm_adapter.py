"""Tests for PDF Markdown repair in pymupdf4llm_adapter."""
from __future__ import annotations


def test_pdfplumber_table_repair_replaces_flattened_table_text():
    from app.core.parser.pymupdf4llm_adapter import (
        _merge_pdf_tables_into_markdown,
        _parse_markdown,
    )

    md = (
        "|编号|标题|\n"
        "|---|---|\n"
        "|57|怎么画好男人的屁股|\n\n"
        "58 如何画拿武器的刀剑客姿势漫画设计 "
        "59 色彩与层次技术-魅力人物着色方法 "
        "60 东方Touhou项目CG插图教程指南\n"
    )
    tables = [
        [
            ["58", "如何画拿武器的刀剑客姿势漫画设计"],
            ["59", "色彩与层次技术-魅力人物着色方法"],
            ["60", "东方Touhou项目CG插图教程指南"],
        ]
    ]

    repaired = _merge_pdf_tables_into_markdown(md, tables)
    ir = _parse_markdown(repaired, "books", "hash")
    paragraphs = ir.sections[0].paragraphs

    assert "58 如何画拿武器" not in repaired
    assert "|列1|列2|" in repaired
    assert "|58|如何画拿武器的刀剑客姿势漫画设计|" in repaired
    assert any(
        "|59|色彩与层次技术-魅力人物着色方法|" in sent.text
        for para in paragraphs
        for sent in para.sentences
    )


def test_pdfplumber_table_repair_skips_existing_markdown_tables():
    from app.core.parser.pymupdf4llm_adapter import _merge_pdf_tables_into_markdown

    md = (
        "|编号|标题|\n"
        "|---|---|\n"
        "|58|如何画拿武器的刀剑客姿势漫画设计|\n"
        "|59|色彩与层次技术-魅力人物着色方法|\n"
    )
    tables = [
        [
            ["58", "如何画拿武器的刀剑客姿势漫画设计"],
            ["59", "色彩与层次技术-魅力人物着色方法"],
        ]
    ]

    repaired = _merge_pdf_tables_into_markdown(md, tables)

    assert repaired == md
