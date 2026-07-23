from __future__ import annotations

import json

from app.core.types import DocumentIR, Paragraph, Section, Sentence


def test_document_ir_codec_round_trip_preserves_page_numbers(tmp_path):
    from app.core.document_ir_codec import (
        document_ir_from_dict,
        document_ir_to_dict,
        load_document_ir,
    )

    original = DocumentIR(
        doc_id="doc-1",
        title="Title",
        file_hash="hash",
        sections=[
            Section(
                section_id="section-1",
                title="正文",
                level=1,
                paragraphs=[
                    Paragraph(
                        paragraph_id="paragraph-1",
                        text="content",
                        sentences=[Sentence(text="content")],
                        page_no=7,
                    )
                ],
            )
        ],
        plain_text="content",
    )

    payload = document_ir_to_dict(original)
    restored = document_ir_from_dict(payload)
    path = tmp_path / "ir.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert restored == original
    assert load_document_ir(path) == original
    assert payload["sections"][0]["paragraphs"][0]["page_no"] == 7


def test_document_ir_codec_loads_legacy_json_without_page_number():
    from app.core.document_ir_codec import document_ir_from_dict

    restored = document_ir_from_dict(
        {
            "doc_id": "legacy",
            "title": "Legacy",
            "file_hash": "hash",
            "sections": [
                {
                    "section_id": "section",
                    "title": "正文",
                    "level": 1,
                    "paragraphs": [
                        {
                            "paragraph_id": "paragraph",
                            "text": "old content",
                            "sentences": [{"text": "old content"}],
                        }
                    ],
                }
            ],
            "plain_text": "old content",
        }
    )

    assert restored.sections[0].paragraphs[0].page_no is None


def test_compare_service_loader_preserves_page_number(tmp_path, monkeypatch):
    from app.core.document_ir_codec import document_ir_to_dict
    from app.services import compare_service

    document = DocumentIR(
        "doc",
        "Title",
        "hash",
        [Section("section", "正文", 1, [Paragraph("p", "text", page_no=4)])],
    )
    path = tmp_path / "ir.json"
    path.write_text(
        json.dumps(document_ir_to_dict(document), ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        compare_service.document_repo,
        "get_version_by_id",
        lambda _conn, _version_id: {"parsed_json_path": str(path)},
    )

    loaded = compare_service._load_ir("version", object())

    assert loaded.sections[0].paragraphs[0].page_no == 4


def test_compare_page_dictionary_loader_preserves_page_number():
    from app.ui.pages.compare_page import _ir_from_dict

    loaded = _ir_from_dict(
        {
            "doc_id": "doc",
            "title": "Title",
            "file_hash": "hash",
            "sections": [
                {
                    "section_id": "section",
                    "title": "正文",
                    "level": 1,
                    "paragraphs": [
                        {
                            "paragraph_id": "p",
                            "text": "text",
                            "sentences": [],
                            "page_no": 9,
                        }
                    ],
                }
            ],
        }
    )

    assert loaded.sections[0].paragraphs[0].page_no == 9
