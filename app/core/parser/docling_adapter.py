"""Docling-based parser adapter. Falls back gracefully if docling is unavailable."""
from __future__ import annotations
import logging
import re
import uuid
from pathlib import Path

from app.core.types import DocumentIR, Paragraph, ParseQualityReport, Section, Sentence
from app.core.utils import file_hash as compute_file_hash

logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False
    logger.warning("docling not installed — Docling adapter will not be available")


def is_available() -> bool:
    return _DOCLING_AVAILABLE


def _split_sentences(text: str) -> list[Sentence]:
    parts = re.split(r'(?<=[。！？.!?])\s*', text.strip())
    return [Sentence(text=p.strip()) for p in parts if p.strip()]


def extract(file_path: str) -> tuple[DocumentIR, ParseQualityReport]:
    """
    Parse file with Docling. Raises RuntimeError if docling is not installed.
    """
    if not _DOCLING_AVAILABLE:
        raise RuntimeError(
            "docling is not installed. Install it with: pip install docling"
        )

    path = Path(file_path)
    converter = DocumentConverter()
    result = converter.convert(str(path))
    dl_doc = result.document

    doc_hash = compute_file_hash(path)

    doc_id = str(uuid.uuid4())
    title = dl_doc.name or path.stem

    sections: list[Section] = []
    current_section: Section | None = None
    default_section = Section(
        section_id=str(uuid.uuid4()),
        title="正文",
        level=1,
    )

    for item, _ in dl_doc.iterate_items():
        item_type = type(item).__name__

        if item_type in ("SectionHeaderItem", "TitleItem"):
            text = getattr(item, "text", "") or ""
            level = getattr(item, "level", 1) or 1
            level = min(max(int(level), 1), 3)
            current_section = Section(
                section_id=str(uuid.uuid4()),
                title=text.strip(),
                level=level,
            )
            sections.append(current_section)

        elif item_type in ("TextItem", "ParagraphItem"):
            text = getattr(item, "text", "") or ""
            text = text.strip()
            if not text:
                continue
            target = current_section if current_section else default_section
            if target is default_section and default_section not in sections:
                sections.insert(0, default_section)
            prov = getattr(item, "prov", [])
            page_no = prov[0].page_no if prov else 0
            p = Paragraph(
                paragraph_id=str(uuid.uuid4()),
                page_no=page_no,
                text=text,
                sentences=_split_sentences(text),
            )
            target.paragraphs.append(p)

    plain_text = "\n".join(
        p.text for sec in sections for p in sec.paragraphs
    )

    ir = DocumentIR(
        doc_id=doc_id,
        title=title,
        file_hash=doc_hash,
        sections=sections,
        plain_text=plain_text,
    )
    report = ParseQualityReport(
        quality_score=1.0,
        needs_ocr=False,
    )
    return ir, report
