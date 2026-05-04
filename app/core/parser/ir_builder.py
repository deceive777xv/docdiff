"""Utilities for building and post-processing DocumentIR."""
from __future__ import annotations
from app.core.types import DocumentIR, Chunk
import uuid


def build_chunks(ir: DocumentIR, version_id: str, max_chars: int = 500) -> list[Chunk]:
    """
    Slice a DocumentIR into Chunks suitable for embedding and retrieval.
    Each paragraph becomes one chunk. If a paragraph exceeds max_chars,
    it is split into sentence-level chunks.
    """
    chunks: list[Chunk] = []
    chunk_no = 0

    for section in ir.sections:
        section_path = section.title
        for para in section.paragraphs:
            if len(para.text) <= max_chars:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    version_id=version_id,
                    chunk_no=chunk_no,
                    section_path=section_path,
                    page_no=para.page_no,
                    text=para.text,
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
                        page_no=para.page_no,
                        text=sent.text,
                    ))
                    chunk_no += 1

    return chunks
