"""Canonical serialization helpers for :class:`DocumentIR`."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from app.core.types import DocumentIR, Paragraph, Section, Sentence


def document_ir_from_dict(data: Mapping[str, Any]) -> DocumentIR:
    """Build a DocumentIR from current or legacy JSON data."""
    sections: list[Section] = []
    for raw_section in data.get("sections", []):
        paragraphs: list[Paragraph] = []
        for raw_paragraph in raw_section.get("paragraphs", []):
            paragraphs.append(
                Paragraph(
                    paragraph_id=str(raw_paragraph.get("paragraph_id", "")),
                    text=str(raw_paragraph.get("text", "")),
                    sentences=[
                        Sentence(text=str(raw_sentence.get("text", "")))
                        for raw_sentence in raw_paragraph.get("sentences", [])
                        if raw_sentence.get("text", "") is not None
                    ],
                    page_no=_optional_positive_int(raw_paragraph.get("page_no")),
                )
            )
        sections.append(
            Section(
                section_id=str(raw_section.get("section_id", "")),
                title=str(raw_section.get("title", "")),
                level=int(raw_section.get("level", 1) or 1),
                paragraphs=paragraphs,
            )
        )
    return DocumentIR(
        doc_id=str(data.get("doc_id", "")),
        title=str(data.get("title", "")),
        file_hash=str(data.get("file_hash", "")),
        sections=sections,
        plain_text=str(data.get("plain_text", "")),
    )


def document_ir_to_dict(document: DocumentIR) -> dict[str, Any]:
    """Return the canonical JSON-compatible representation."""
    return asdict(document)


def load_document_ir(path: str | Path) -> DocumentIR:
    """Load a DocumentIR JSON file using the compatibility codec."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DocumentIR JSON root must be an object")
    return document_ir_from_dict(payload)


def _optional_positive_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        page_no = int(value)
    except (TypeError, ValueError):
        return None
    return page_no if page_no > 0 else None
