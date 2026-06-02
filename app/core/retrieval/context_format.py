"""Formatting helpers for retrieval context shown to QA models."""
from __future__ import annotations

from app.core.types import ChunkHit


def format_chunk_location(chunk) -> str:
    """Return a readable location, falling back to paragraph order when pages are absent."""
    parts: list[str] = []
    if chunk.section_path:
        parts.append(f"章节：{chunk.section_path}")
    if chunk.page_no and chunk.page_no > 0:
        parts.append(f"第{chunk.page_no}页")
    else:
        parts.append(f"段落：第{max(0, int(chunk.chunk_no)) + 1}段")
    return "，".join(parts)


def format_hits_context(hits: list[ChunkHit]) -> str:
    """Format retrieval hits for inclusion in QA prompts."""
    parts = []
    for i, hit in enumerate(hits, 1):
        chunk = hit.chunk
        location = format_chunk_location(chunk)
        ref = f"[{i}] "
        if location:
            ref += f"{location}，"
        ref += f"内容：{chunk.text}"
        parts.append(ref)
    return "\n\n".join(parts)
