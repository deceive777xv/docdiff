"""Utilities for building and post-processing DocumentIR."""
from __future__ import annotations
from app.core.types import DocumentIR, Chunk
import uuid


def build_chunks(ir: DocumentIR, version_id: str, max_chars: int = 2000) -> list[Chunk]:
    """
    Slice a DocumentIR into Chunks suitable for embedding and retrieval.
    Each paragraph becomes one chunk. If a paragraph exceeds max_chars,
    it is split into sentence-level chunks.
    """
    chunks: list[Chunk] = []
    chunk_no = 0

    section_stack: list[str] = []
    for section in ir.sections:
        level = max(1, int(section.level or 1))
        section_stack = section_stack[: level - 1]
        while len(section_stack) < level - 1:
            section_stack.append("")
        section_stack.append(section.title)
        section_path = " > ".join(part for part in section_stack if part)
        for para in section.paragraphs:
            if len(para.text) <= max_chars:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    version_id=version_id,
                    chunk_no=chunk_no,
                    section_path=section_path,
                    text=para.text,
                    page_no=para.page_no or 0,
                ))
                chunk_no += 1
            else:
                # Split into sentence-level chunks
                for sent in para.sentences:
                    if not sent.text:
                        continue
                    chunks.append(Chunk(
                        id=str(uuid.uuid4()),
                        version_id=version_id,
                        chunk_no=chunk_no,
                        section_path=section_path,
                        text=sent.text,
                        page_no=para.page_no or 0,
                    ))
                    chunk_no += 1

    return chunks
