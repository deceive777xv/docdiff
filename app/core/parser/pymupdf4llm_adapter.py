"""Markitdown-based document parser — converts any supported file to DocumentIR."""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from app.core.types import DocumentIR, Paragraph, Section, Sentence


def is_available() -> bool:
    try:
        import pymupdf4llm  # noqa: F401
        return True
    except ImportError:
        return False


def extract(
    file_path: str,
) -> DocumentIR:
    if not is_available():
        raise RuntimeError("pymupdf4llm is not installed")

    import pymupdf4llm  # noqa: F401

    result = pymupdf4llm.to_markdown(file_path)
    title = Path(file_path).stem
    file_hash = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
    return _parse_markdown(result, title, file_hash)

SENTENCE_END_PATTERN = re.compile(
    r'(?:(?<!\d)[.!?](?!\d))\s+'   # 英文句末标点：点前非数字，点后非数字（防止日期、小数等）
    r'|(?<=[。！？])\s*'            # 中文句末标点
)

# 匹配 Markdown 表格行（整行以 | 开头和结尾）
TABLE_ROW_PATTERN = re.compile(r'^\s*\|.*\|\s*$')

def _split_sentences(text: str) -> list[str]:
    """
    将段落文本切分为句子列表。
    - 普通文本：按增强的句末标点规则切分（避免切分数字编号）
    - Markdown 表格行：每行作为一个独立的句子
    """
    lines = text.split('\n')
    buffer: list[str] = []       # 存放非表格行的文本行
    sentences: list[str] = []

    for line in lines:
        if TABLE_ROW_PATTERN.match(line):
            # 1. 先清空缓存中的普通文本
            if buffer:
                merged = ' '.join(buffer)
                for sent in SENTENCE_END_PATTERN.split(merged):
                    sent = sent.strip()
                    if sent:
                        sentences.append(sent)
                buffer.clear()
            # 2. 表格行整体作为一个句子
            cleaned = line.strip()
            if cleaned:
                sentences.append(cleaned)
        else:
            buffer.append(line)

    # 3. 处理最后剩余的普通文本
    if buffer:
        merged = ' '.join(buffer)
        for sent in SENTENCE_END_PATTERN.split(merged):
            sent = sent.strip()
            if sent:
                sentences.append(sent)

    return sentences

def _parse_markdown(md_text: str, title: str, doc_hash: str) -> DocumentIR:
    sections: list[Section] = []
    current_section: Section | None = None
    para_buffer: list[str] = []      # 存放非表格的文本行
    table_buffer: list[str] = []     # 存放连续的表格行

    def _flush_text() -> None:
        """将 para_buffer 中的普通文本落盘为一个 Paragraph"""
        nonlocal current_section
        if para_buffer:
            # 用换行符保留原始段落中的换行，而不是用空格
            joined = "\n".join(para_buffer).strip()
            if joined:
                ensure_section()
                sentences = _split_sentences(joined)
                current_section.paragraphs.append(Paragraph(
                    paragraph_id=str(uuid.uuid4()),
                    text=joined,
                    sentences=[Sentence(text=t) for t in sentences],
                ))
            para_buffer.clear()

    def _flush_table() -> None:
        """将 table_buffer 中的连续表格行落盘为一个 Paragraph"""
        nonlocal current_section
        if table_buffer:
            # 保留原始空白：不 strip 单元格内部，只去除行首尾的空白
            cleaned_rows = [row.strip() for row in table_buffer]
            table_text = "\n".join(cleaned_rows)
            if table_text:
                ensure_section()
                # 表格的每一行作为一个 sentence
                sentences = [Sentence(text=row) for row in cleaned_rows if row]
                current_section.paragraphs.append(Paragraph(
                    paragraph_id=str(uuid.uuid4()),
                    text=table_text,
                    sentences=sentences,
                ))
            table_buffer.clear()

    def ensure_section():
        nonlocal current_section
        if current_section is None:
            current_section = Section(
                section_id=str(uuid.uuid4()),
                title="正文",
                level=1,
                paragraphs=[],
            )
            sections.append(current_section)

    heading_re = re.compile(r"^(#{1,3})\s+(.+)")

    for line in md_text.splitlines():
        if heading_re.match(line):
            _flush_text()
            _flush_table()
            level = len(heading_re.match(line).group(1))
            current_section = Section(
                section_id=str(uuid.uuid4()),
                title=heading_re.match(line).group(2).strip(),
                level=level,
                paragraphs=[],
            )
            sections.append(current_section)
            continue

        if line.strip() == "":
            # 空行：结束当前文本块和表格块
            _flush_text()
            _flush_table()
            continue

        # 判断当前行是否为表格行
        if TABLE_ROW_PATTERN.match(line):
            # 遇到表格行时，先把之前的普通文本落盘，然后加入表格缓冲区
            _flush_text()
            table_buffer.append(line)
        else:
            # 普通文本行：先把之前的表格落盘，然后加入文本缓冲区
            _flush_table()
            para_buffer.append(line)

    # 文件末尾，清空所有缓冲
    _flush_text()
    _flush_table()

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
