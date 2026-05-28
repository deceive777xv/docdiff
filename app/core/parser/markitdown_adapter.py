"""Markitdown-based document parser — converts any supported file to DocumentIR."""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from app.core.types import DocumentIR, Paragraph, Section, Sentence


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

def _split_sentences(text: str) -> list[str]:
    """
    将段落文本切分为句子列表。
    - 普通文本：按 .!?。！？ 后跟空白切分（支持跨行句子）
    - Markdown 表格行：每行作为一个句子（整行保留，不再按标点切分）
    """
    TABLE_ROW_PATTERN = re.compile(r'^\s*\|.*\|\s*$')  # 匹配以 | 开头并结尾的行
    SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?。！？])\s+')

    lines = text.split('\n')
    buffer: list[str] = []
    sentences: list[str] = []

    for line in lines:
        if TABLE_ROW_PATTERN.match(line):
            # 先处理缓冲区中的普通文本
            if buffer:
                merged = ' '.join(buffer)
                for sent in SENTENCE_SPLIT_PATTERN.split(merged):
                    sent = sent.strip()
                    if sent:
                        sentences.append(sent)
                buffer.clear()
            # 表格行整体作为一个句子
            cleaned = line.strip()
            if cleaned:
                sentences.append(cleaned)
        else:
            buffer.append(line)

    # 处理最后剩余的普通文本
    if buffer:
        merged = ' '.join(buffer)
        for sent in SENTENCE_SPLIT_PATTERN.split(merged):
            sent = sent.strip()
            if sent:
                sentences.append(sent)

    return sentences

def _parse_markdown(md_text: str, title: str, doc_hash: str) -> DocumentIR:
    sections: list[Section] = []
    current_section: Section | None = None
    para_buffer: list[str] = []

    def _flush() -> None:
        if current_section is not None and para_buffer:
            joined = " ".join(para_buffer).strip()
            if joined:
                sentences = _split_sentences(joined)
                current_section.paragraphs.append(
                    Paragraph(
                        paragraph_id=str(uuid.uuid4()),
                        text=joined,
                        sentences=[Sentence(text=t) for t in sentences],
                    )
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
