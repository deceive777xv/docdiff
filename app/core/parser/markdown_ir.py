"""Convert parser-produced Markdown into the application's DocumentIR."""
from __future__ import annotations

import re
import uuid

from app.core.types import DocumentIR, Paragraph, Section, Sentence


SENTENCE_END_PATTERN = re.compile(
    r"(?:(?<!\d)[.!?](?!\d))\s+"
    r"|[。！？.!?](?=\s|$)"
)
TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$")


def split_sentences(text: str, max_buffer_len: int = 500) -> list[str]:
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
            raw_sentences: list[str] = []
            for index, part in enumerate(parts):
                if not part.strip():
                    continue
                end = ends[index] if index < len(ends) else ""
                raw_sentences.append(part + end)

            current_chunk = ""
            for sentence in raw_sentences:
                if not current_chunk:
                    current_chunk = sentence
                    continue
                candidate = current_chunk + sentence
                if len(candidate) <= max_buffer_len:
                    current_chunk = candidate
                else:
                    sentences.append(current_chunk.strip())
                    current_chunk = sentence
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


def parse_markdown(md_text: str, title: str, doc_hash: str) -> DocumentIR:
    sections: list[Section] = []
    current_section: Section | None = None
    paragraph_buffer: list[str] = []
    table_buffer: list[str] = []

    def ensure_section() -> None:
        nonlocal current_section
        if current_section is None:
            current_section = Section(
                section_id=str(uuid.uuid4()),
                title="正文",
                level=1,
                paragraphs=[],
            )
            sections.append(current_section)

    def flush_text() -> None:
        nonlocal current_section
        if not paragraph_buffer:
            return
        joined = "\n".join(paragraph_buffer).strip()
        if joined:
            ensure_section()
            assert current_section is not None
            current_section.paragraphs.append(Paragraph(
                paragraph_id=str(uuid.uuid4()),
                text=joined,
                sentences=[Sentence(text=text) for text in split_sentences(joined)],
            ))
        paragraph_buffer.clear()

    def flush_table() -> None:
        nonlocal current_section
        if not table_buffer:
            return
        rows = [row.strip() for row in table_buffer]
        table_text = "\n".join(rows)
        if table_text:
            ensure_section()
            assert current_section is not None
            current_section.paragraphs.append(Paragraph(
                paragraph_id=str(uuid.uuid4()),
                text=table_text,
                sentences=[Sentence(text=row) for row in rows if row],
            ))
        table_buffer.clear()

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)")
    for line in md_text.splitlines():
        heading_match = heading_pattern.match(line)
        if heading_match:
            flush_text()
            flush_table()
            current_section = Section(
                section_id=str(uuid.uuid4()),
                title=heading_match.group(2).strip(),
                level=len(heading_match.group(1)),
                paragraphs=[],
            )
            sections.append(current_section)
            continue

        if not line.strip():
            flush_text()
            flush_table()
        elif TABLE_ROW_PATTERN.match(line):
            flush_text()
            table_buffer.append(line)
        else:
            flush_table()
            paragraph_buffer.append(line)

    flush_text()
    flush_table()
    plain_text = "\n".join(
        paragraph.text for section in sections for paragraph in section.paragraphs
    )
    return DocumentIR(
        doc_id=str(uuid.uuid4()),
        title=title,
        file_hash=doc_hash,
        sections=sections,
        plain_text=plain_text,
    )
