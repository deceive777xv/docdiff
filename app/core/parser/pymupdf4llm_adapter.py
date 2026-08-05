"""Page-aware PDF parsing backed exclusively by PyMuPDF4LLM."""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.core.types import DocumentIR, Paragraph, Section, Sentence


SENTENCE_END_PATTERN = re.compile(
    r"(?:(?<!\d)[.!?](?!\d))\s+"
    r"|[。！？.!?](?=\s|$)"
)
TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$")


def is_available() -> bool:
    try:
        import pymupdf4llm  # noqa: F401

        return True
    except ImportError:
        return False


def extract(file_path: str) -> DocumentIR:
    if not is_available():
        raise RuntimeError("pymupdf4llm is not installed")

    import pymupdf4llm

    page_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    if not isinstance(page_chunks, list):
        raise TypeError("PyMuPDF4LLM page_chunks output must be a list")

    path = Path(file_path)
    return _parse_page_chunks(
        page_chunks,
        title=path.stem,
        doc_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _split_sentences(text: str, max_buffer_len: int = 500) -> list[str]:
    lines = text.split("\n")
    buffer: list[str] = []
    sentences: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        merged = " ".join(buffer).strip()
        if not merged:
            buffer.clear()
            return

        if len(merged) <= max_buffer_len:
            sentences.append(merged)
        else:
            parts = SENTENCE_END_PATTERN.split(merged)
            ends = SENTENCE_END_PATTERN.findall(merged)
            
            raw_sentences = []
            for i, part in enumerate(parts):
                if not part.strip():
                    continue
                end = ends[i] if i < len(ends) else ''
                raw_sentences.append(part + end)

            current_chunk = ""
            for s in raw_sentences:
                if not current_chunk:
                    current_chunk = s
                else:
                    candidate = current_chunk + s
                    if len(candidate) <= max_buffer_len:
                        current_chunk = candidate
                    else:
                        sentences.append(current_chunk.strip())
                        current_chunk = s
            if current_chunk:
                sentences.append(current_chunk.strip())

        buffer.clear()

    for line in lines:
        if TABLE_ROW_PATTERN.match(line):
            flush_buffer()
            if line.strip():
                sentences.append(line.strip())
        else:
            buffer.append(line)
    flush_buffer()
    return sentences


class _MarkdownPageParser:
    def __init__(self) -> None:
        self.sections: list[Section] = []
        self.current_section: Section | None = None
        self.para_buffer: list[str] = []
        self.table_buffer: list[str] = []
        self.page_no: int | None = None
        self.heading_re = re.compile(r"^(#{1,3})\s+(.+)")

    def parse_page(self, markdown: str, page_no: int | None) -> None:
        self.flush()
        self.page_no = page_no
        for line in markdown.splitlines():
            heading = self.heading_re.match(line)
            if heading:
                self.flush()
                self.current_section = Section(
                    section_id=str(uuid.uuid4()),
                    title=heading.group(2).strip(),
                    level=len(heading.group(1)),
                    paragraphs=[],
                )
                self.sections.append(self.current_section)
            elif not line.strip():
                self.flush()
            elif TABLE_ROW_PATTERN.match(line):
                self._flush_text()
                self.table_buffer.append(line)
            else:
                self._flush_table()
                self.para_buffer.append(line)
        self.flush()

    def flush(self) -> None:
        self._flush_text()
        self._flush_table()

    def _ensure_section(self) -> Section:
        if self.current_section is None:
            self.current_section = Section(
                section_id=str(uuid.uuid4()),
                title="正文",
                level=1,
                paragraphs=[],
            )
            self.sections.append(self.current_section)
        return self.current_section

    def _flush_text(self) -> None:
        if not self.para_buffer:
            return
        joined = "\n".join(self.para_buffer).strip()
        self.para_buffer.clear()
        if joined:
            self._append_paragraph(joined, _split_sentences(joined))

    def _flush_table(self) -> None:
        if not self.table_buffer:
            return
        rows = [row.strip() for row in self.table_buffer if row.strip()]
        self.table_buffer.clear()
        if rows:
            self._append_paragraph("\n".join(rows), rows)

    def _append_paragraph(self, text: str, sentence_texts: Iterable[str]) -> None:
        self._ensure_section().paragraphs.append(
            Paragraph(
                paragraph_id=str(uuid.uuid4()),
                text=text,
                sentences=[Sentence(text=value) for value in sentence_texts],
                page_no=self.page_no,
            )
        )


def _parse_page_chunks(
    chunks: list[Mapping[str, Any]],
    title: str,
    doc_hash: str,
) -> DocumentIR:
    parser = _MarkdownPageParser()
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        raw_page_no = metadata.get("page_number", index)
        try:
            page_no = int(raw_page_no)
        except (TypeError, ValueError):
            page_no = index
        parser.parse_page(str(chunk.get("text", "")), page_no)
    return _build_document(parser.sections, title, doc_hash)


def _parse_markdown(md_text: str, title: str, doc_hash: str) -> DocumentIR:
    """Compatibility helper for callers parsing a single Markdown string."""
    parser = _MarkdownPageParser()
    parser.parse_page(md_text, None)
    return _build_document(parser.sections, title, doc_hash)


def _build_document(
    sections: list[Section],
    title: str,
    doc_hash: str,
) -> DocumentIR:
    plain_text = "\n".join(
        paragraph.text
        for section in sections
        for paragraph in section.paragraphs
    )
    return DocumentIR(
        doc_id=str(uuid.uuid4()),
        title=title,
        file_hash=doc_hash,
        sections=sections,
        plain_text=plain_text,
    )
