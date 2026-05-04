"""Extract DocumentIR from PDF files using PyMuPDF (fitz)."""
from __future__ import annotations
import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF

from app.core.types import DocumentIR, Paragraph, ParseQualityReport, Section, Sentence
from app.core.utils import file_hash


_HEADING_PATTERN = re.compile(
    r'^(第[一二三四五六七八九十百\d]+[章节条款项]|[\d一二三四五六七八九十]+[、.]\s*\S)'
)


def _split_sentences(text: str) -> list[Sentence]:
    parts = re.split(r'(?<=[。！？.!?])\s*', text.strip())
    return [Sentence(text=p.strip()) for p in parts if p.strip()]


def _is_likely_heading(text: str, font_size: float, page_max_size: float) -> bool:
    """Heuristic: large font OR matches common heading patterns."""
    if font_size >= page_max_size * 0.85 and len(text) < 60:
        return True
    return bool(_HEADING_PATTERN.match(text.strip()))


def _extract_page_blocks(page: fitz.Page) -> list[tuple[str, float]]:
    """
    Return (block_text, max_font_size) for each non-empty text block.
    Single-pass via rawdict gives both text and font metrics together.
    Falls back to plain blocks mode if rawdict returns no text.
    """
    items: list[tuple[str, float]] = []
    try:
        raw = page.get_text("rawdict")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            spans_in_block = [s for ln in block.get("lines", []) for s in ln.get("spans", [])]
            if not spans_in_block:
                continue
            text = "".join(s.get("text", "") for s in spans_in_block).strip()
            if not text:
                continue
            max_size = max((s.get("size", 12.0) for s in spans_in_block), default=12.0)
            items.append((text, max_size))
    except Exception:
        pass

    if not items:
        # Fallback for PDFs where rawdict yields no text (e.g., annotation-based inserts)
        for block in page.get_text("blocks"):
            if block[6] == 0 and block[4].strip():
                items.append((block[4].strip(), 12.0))
    return items


def extract(file_path: str) -> tuple[DocumentIR, ParseQualityReport]:
    """
    Parse a PDF into DocumentIR using PyMuPDF.
    Returns (DocumentIR, ParseQualityReport).
    """
    path = Path(file_path)
    doc = fitz.open(str(path))
    doc_hash = file_hash(path)
    doc_id = str(uuid.uuid4())
    total_pages = len(doc)

    sections: list[Section] = []
    current_section: Section | None = None
    total_chars = 0
    low_text_pages: list[int] = []
    warnings: list[str] = []

    default_section = Section(
        section_id=str(uuid.uuid4()),
        title="正文",
        level=1,
    )

    for page_num, page in enumerate(doc, start=1):
        block_items = _extract_page_blocks(page)
        page_char_count = sum(len(t) for t, _ in block_items)
        total_chars += page_char_count

        if page_char_count < 20:
            low_text_pages.append(page_num)

        page_max_size = max((sz for _, sz in block_items), default=12.0)

        for text, block_size in block_items:
            if _is_likely_heading(text, block_size, page_max_size):
                current_section = Section(
                    section_id=str(uuid.uuid4()),
                    title=text,
                    level=1,
                )
                sections.append(current_section)
            else:
                target = current_section if current_section else default_section
                if target is default_section and default_section not in sections:
                    sections.insert(0, default_section)
                p = Paragraph(
                    paragraph_id=str(uuid.uuid4()),
                    page_no=page_num,
                    text=text,
                    sentences=_split_sentences(text),
                )
                target.paragraphs.append(p)

    doc.close()

    plain_text = "\n".join(
        p.text
        for sec in sections
        for p in sec.paragraphs
    )

    needs_ocr = len(low_text_pages) > 0
    if needs_ocr:
        warnings.append(f"Low-text pages detected (possible scans): {low_text_pages}")

    quality_score = (
        1.0 if not low_text_pages
        else max(0.1, 1.0 - len(low_text_pages) / max(total_pages, 1))
    )

    title = sections[0].title if sections else path.stem

    ir = DocumentIR(
        doc_id=doc_id,
        title=title,
        file_hash=doc_hash,
        sections=sections,
        plain_text=plain_text,
    )
    report = ParseQualityReport(
        quality_score=round(quality_score, 2),
        needs_ocr=needs_ocr,
        ocr_pages=low_text_pages,
        warnings=warnings,
    )
    return ir, report
