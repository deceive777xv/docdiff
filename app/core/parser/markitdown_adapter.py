"""Markitdown-based document parser — converts any supported file to DocumentIR."""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from app.core.types import DocumentIR, Paragraph, Section


def is_available() -> bool:
    try:
        import markitdown  # noqa: F401
        return True
    except ImportError:
        return False


def extract(
    file_path: str,
    llm_client=None,
    llm_model: str = "",
) -> DocumentIR:
    if not is_available():
        raise RuntimeError("markitdown is not installed")

    from markitdown import MarkItDown

    md = MarkItDown(
        enable_plugins=bool(llm_client),
        llm_client=llm_client or None,
        llm_model=llm_model or None,
    )
    result = md.convert(file_path)
    title = Path(file_path).stem
    file_hash = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
    return _parse_markdown(result.markdown, title, file_hash)


def _parse_markdown(md_text: str, title: str, doc_hash: str) -> DocumentIR:
    sections: list[Section] = []
    current_section: Section | None = None
    para_buffer: list[str] = []

    def _flush() -> None:
        if current_section is not None and para_buffer:
            joined = " ".join(para_buffer).strip()
            if joined:
                current_section.paragraphs.append(
                    Paragraph(paragraph_id=str(uuid.uuid4()), page_no=0, text=joined)  # TODO Task 4: remove page_no=0
                )
        para_buffer.clear()

    heading_re = re.compile(r"^(#{1,3})\s+(.+)")

    for line in md_text.splitlines():
        m = heading_re.match(line)
        if m:
            _flush()
            level = len(m.group(1))
            current_section = Section(
                section_id=str(uuid.uuid4()),
                title=m.group(2).strip(),
                level=level,
                paragraphs=[],
            )
            sections.append(current_section)
        elif line.strip() == "":
            _flush()
        else:
            if current_section is None:
                current_section = Section(
                    section_id=str(uuid.uuid4()),
                    title="正文",
                    level=1,
                    paragraphs=[],
                )
                sections.append(current_section)
            para_buffer.append(line.strip())

    _flush()

    plain_text = "\n".join(
        para.text for sec in sections for para in sec.paragraphs
    )
    return DocumentIR(
        doc_id=str(uuid.uuid4()),
        title=title,
        file_hash=doc_hash,
        sections=sections,
        plain_text=plain_text,
    )
