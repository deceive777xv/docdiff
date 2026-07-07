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
    result = _merge_pdf_tables_into_markdown(result, _extract_pdf_tables(file_path))
    title = Path(file_path).stem
    file_hash = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
    return _parse_markdown(result, title, file_hash)

SENTENCE_END_PATTERN = re.compile(
    r'(?:(?<!\d)[.!?](?!\d))\s+'   # 英文句末标点：点前非数字，点后非数字（防止日期、小数等）
    r'|(?<=[。！？])\s*'            # 中文句末标点
)

# 匹配 Markdown 表格行（整行以 | 开头和结尾）
TABLE_ROW_PATTERN = re.compile(r'^\s*\|.*\|\s*$')
TABLE_SEPARATOR_PATTERN = re.compile(r'^\s*\|[\s\-:|]+\|\s*$')


def _extract_pdf_tables(file_path: str) -> list[list[list[str]]]:
    try:
        import pdfplumber
    except ImportError:
        return []

    tables: list[list[list[str]]] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                for raw_table in page.extract_tables() or []:
                    table = _normalize_pdf_table(raw_table)
                    if _is_useful_pdf_table(table):
                        tables.append(table)
    except Exception:
        return []
    return tables


def _merge_pdf_tables_into_markdown(
    md_text: str,
    pdf_tables: list[list[list[object]]],
) -> str:
    if not pdf_tables:
        return md_text

    repaired = md_text
    table_signatures = _markdown_table_row_signatures(repaired)
    for raw_table in pdf_tables:
        table = _normalize_pdf_table(raw_table)
        if not _is_useful_pdf_table(table):
            continue

        row_signatures = _table_row_signatures(table)
        if _table_already_present(row_signatures, table_signatures):
            continue

        table_md = _pdf_table_to_markdown(table)
        repaired, replaced = _replace_flattened_table_text(
            repaired,
            row_signatures,
            table_md,
        )
        if not replaced:
            repaired = f"{repaired.rstrip()}\n\n{table_md}\n"
        table_signatures.update(row_signatures)

    return repaired


def _normalize_pdf_table(raw_table: list[list[object]]) -> list[list[str]]:
    rows: list[list[str]] = []
    max_cols = 0
    for raw_row in raw_table:
        cells = [_clean_pdf_cell(cell) for cell in raw_row]
        if any(cells):
            rows.append(cells)
            max_cols = max(max_cols, len(cells))

    if not rows or max_cols == 0:
        return []

    padded = [row + [""] * (max_cols - len(row)) for row in rows]
    non_empty_cols = [
        index
        for index in range(max_cols)
        if any(row[index] for row in padded)
    ]
    if not non_empty_cols:
        return []

    return [[row[index] for index in non_empty_cols] for row in padded]


def _clean_pdf_cell(cell: object) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _is_useful_pdf_table(table: list[list[str]]) -> bool:
    if len(table) < 2:
        return False
    return max((len(row) for row in table), default=0) >= 2


def _markdown_table_row_signatures(md_text: str) -> set[str]:
    signatures: set[str] = set()
    for line in md_text.splitlines():
        stripped = line.strip()
        if not TABLE_ROW_PATTERN.match(stripped):
            continue
        if TABLE_SEPARATOR_PATTERN.match(stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        signature = _row_signature(cells)
        if signature:
            signatures.add(signature)
    return signatures


def _table_row_signatures(table: list[list[str]]) -> list[str]:
    signatures: list[str] = []
    for row in table:
        signature = _row_signature(row)
        if signature:
            signatures.append(signature)
    return signatures


def _row_signature(row: list[str]) -> str:
    return _normalize_text(" ".join(cell for cell in row if cell))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _table_already_present(
    row_signatures: list[str],
    existing_signatures: set[str],
) -> bool:
    if not row_signatures:
        return True
    present = sum(1 for signature in row_signatures if signature in existing_signatures)
    return present / len(row_signatures) >= 0.7


def _pdf_table_to_markdown(table: list[list[str]]) -> str:
    col_count = max((len(row) for row in table), default=0)
    header = [f"列{index + 1}" for index in range(col_count)]
    lines = [
        _format_markdown_row(header),
        _format_markdown_row(["---"] * col_count),
    ]
    for row in table:
        padded = row + [""] * (col_count - len(row))
        lines.append(_format_markdown_row(padded))
    return "\n".join(lines)


def _format_markdown_row(cells: list[str]) -> str:
    return "|" + "|".join(_escape_markdown_cell(cell) for cell in cells) + "|"


def _escape_markdown_cell(cell: str) -> str:
    return cell.replace("|", r"\|")


def _replace_flattened_table_text(
    md_text: str,
    row_signatures: list[str],
    table_md: str,
) -> tuple[str, bool]:
    if not row_signatures:
        return md_text, False

    lines = md_text.splitlines()
    min_hits = min(3, len(row_signatures))
    first_signature = row_signatures[0]
    last_signature = row_signatures[-1]

    for index, line in enumerate(lines):
        normalized = _normalize_text(line)
        hit_count = _count_signature_hits(normalized, row_signatures)
        if (
            first_signature in normalized
            and last_signature in normalized
            and hit_count >= min_hits
        ):
            lines[index] = table_md
            return "\n".join(lines), True

    for start in range(len(lines)):
        if first_signature not in _normalize_text(lines[start]):
            continue
        for end in range(start, min(len(lines), start + len(row_signatures) + 3)):
            block = _normalize_text(" ".join(lines[start:end + 1]))
            hit_count = _count_signature_hits(block, row_signatures)
            if last_signature in block and hit_count >= min_hits:
                lines[start:end + 1] = [table_md]
                return "\n".join(lines), True

    return md_text, False


def _count_signature_hits(text: str, row_signatures: list[str]) -> int:
    return sum(1 for signature in row_signatures if signature in text)

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
