"""Small Markdown helpers for safe UI rendering and plain-text exports."""
from __future__ import annotations

import html
import re

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_ORDERED_LIST_RE = re.compile(r"^\s*\d+[.)]\s+(.+)$")
_UNORDERED_LIST_RE = re.compile(r"^\s*[-*+]\s+(.+)$")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_TABLE_SEPARATOR_CELL_RE = re.compile(r":?-{3,}:?")
_HTML_TAG_RE = re.compile(r"</?[^>\n]+>")


def _needs_inline_space(left: str, right: str) -> bool:
    return left.isascii() and right.isascii() and left.isalnum() and right.isalnum()


def _normalize_inline_markdown(text: str) -> str:
    parts = [part.strip() for part in normalize_markdown_breaks(text).split("\n") if part.strip()]
    if not parts:
        return ""

    merged = parts[0]
    for part in parts[1:]:
        if merged and _needs_inline_space(merged[-1], part[0]):
            merged += " "
        merged += part
    return merged


def normalize_markdown_breaks(text: str) -> str:
    """Normalize common stored break markers before rendering or stripping."""
    normalized = html.unescape(text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return _BR_RE.sub("\n", normalized)


def split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def is_markdown_table_separator(line: str) -> bool:
    if "|" not in line:
        return False
    cells = split_markdown_table_row(line)
    return bool(cells) and all(_TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells)


def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and is_markdown_table_separator(lines[index + 1])
    )


def render_inline_markdown(text: str) -> str:
    """Render a conservative, escaped inline Markdown subset."""
    rendered = html.escape(_normalize_inline_markdown(text), quote=False)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"__([^_\n]+)__", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", rendered)
    rendered = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"<em>\1</em>", rendered)
    rendered = re.sub(r"~~([^~\n]+)~~", r"<s>\1</s>", rendered)
    rendered = re.sub(
        r"!\[([^\]]*)\]\([^)]+\)",
        lambda m: html.escape(m.group(1), quote=False),
        rendered,
    )
    rendered = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", rendered)
    rendered = rendered.replace("`", "")
    rendered = re.sub(r"(?<!\w)[*_~]+|[*_~]+(?!\w)", "", rendered)
    return rendered


def _render_markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    header_cells = "".join(f"<th>{render_inline_markdown(cell)}</th>" for cell in rows[0])
    body_rows = []
    for row in rows[1:]:
        cells = "".join(f"<td>{render_inline_markdown(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")

    body_html = "".join(body_rows)
    return (
        '<div class="markdown-table-wrap"><table>'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table></div>"
    )


def render_markdown_fragment(markdown_text: str) -> str:
    """Convert stored Markdown text into safe, readable HTML."""
    lines = normalize_markdown_breaks(markdown_text).split("\n")
    blocks: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            code = html.escape("\n".join(code_lines), quote=False)
            blocks.append(f"<pre><code>{code}</code></pre>")
            continue

        if _is_markdown_table_start(lines, i):
            table_rows = [split_markdown_table_row(lines[i])]
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                if not is_markdown_table_separator(lines[i]):
                    table_rows.append(split_markdown_table_row(lines[i]))
                i += 1
            blocks.append(_render_markdown_table(table_rows))
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            level = min(6, len(heading.group(1)) + 3)
            blocks.append(f"<h{level}>{render_inline_markdown(heading.group(2).strip())}</h{level}>")
            i += 1
            continue

        unordered = _UNORDERED_LIST_RE.match(line)
        if unordered:
            items: list[str] = []
            while i < len(lines):
                match = _UNORDERED_LIST_RE.match(lines[i])
                if not match:
                    break
                items.append(f"<li>{render_inline_markdown(match.group(1).strip())}</li>")
                i += 1
            blocks.append(f"<ul>{''.join(items)}</ul>")
            continue

        ordered = _ORDERED_LIST_RE.match(line)
        if ordered:
            items = []
            while i < len(lines):
                match = _ORDERED_LIST_RE.match(lines[i])
                if not match:
                    break
                items.append(f"<li>{render_inline_markdown(match.group(1).strip())}</li>")
                i += 1
            blocks.append(f"<ol>{''.join(items)}</ol>")
            continue

        quote = _BLOCKQUOTE_RE.match(line)
        if quote:
            quote_lines: list[str] = []
            while i < len(lines):
                match = _BLOCKQUOTE_RE.match(lines[i])
                if not match:
                    break
                quote_lines.append(match.group(1).strip())
                i += 1
            quote_html = " ".join(render_inline_markdown(q) for q in quote_lines)
            blocks.append(f"<blockquote>{quote_html}</blockquote>")
            continue

        paragraph_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            current = lines[i].strip()
            starts_new_block = (
                _is_markdown_table_start(lines, i)
                or _HEADING_RE.match(current)
                or _UNORDERED_LIST_RE.match(lines[i])
                or _ORDERED_LIST_RE.match(lines[i])
                or _BLOCKQUOTE_RE.match(lines[i])
                or current.startswith("```")
            )
            if paragraph_lines and starts_new_block:
                break
            paragraph_lines.append(current)
            i += 1
        paragraph = " ".join(render_inline_markdown(p) for p in paragraph_lines)
        blocks.append(f"<p>{paragraph}</p>")

    return "".join(block for block in blocks if block)


def strip_markdown_formatting(markdown_text: str) -> str:
    """Remove Markdown and simple HTML markers while preserving readable text."""
    cleaned_lines: list[str] = []
    in_code_block = False

    text = html.unescape(markdown_text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not line:
            continue
        if not in_code_block and is_markdown_table_separator(line):
            continue
        if not in_code_block and line.startswith("|") and line.endswith("|"):
            line = " ".join(cell for cell in split_markdown_table_row(line) if cell)
        elif not in_code_block:
            line = re.sub(r"^#{1,6}\s+", "", line)
            line = re.sub(r"^>\s?", "", line)
            line = re.sub(r"^[-*+]\s+", "", line)
            line = re.sub(r"^\d+[.)]\s+", "", line)

        line = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = line.replace("`", "")
        line = re.sub(r"[*_~]+", "", line)
        line = _HTML_TAG_RE.sub(" ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)

    return re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()
