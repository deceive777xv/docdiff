"""Match paragraphs between aligned section pairs using embedding similarity."""
from __future__ import annotations
import html
import re
from dataclasses import dataclass

import numpy as np

from app.core.diff.structure_aligner import SectionPair
from app.core.model.base_provider import BaseProvider
from app.core.types import Paragraph, Sentence


@dataclass
class ParagraphPair:
    baseline_para: Paragraph | None
    target_para: Paragraph | None
    similarity: float   # cosine similarity, -1..1 (1 = identical)
    section_path: str = ""
    split_unit: bool = False
    baseline_match_text: str | None = None
    target_match_text: str | None = None


@dataclass
class _ParagraphUnit:
    para: Paragraph
    split_unit: bool
    match_text: str | None = None
    table_values: list[str] | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


_RULE_PATTERNS = [
    re.compile(r'\d+[\.,]\d*'),          # numbers
    re.compile(r'[不无未没]'),            # negations
    re.compile(r'(?:应|须|必须|不得|禁止)'),  # obligation words
]

_FINE_GRAINED_MAX_CHARS = 500
_TABLE_SEPARATOR_CELL_RE = re.compile(r":?-{2,}:?")
_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[^>\n]+>")
_LEXICAL_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:[\.,]\d+)?%?|[\u4e00-\u9fff]", re.IGNORECASE)
_NUMERIC_TOKEN_RE = re.compile(r"\d+(?:[\.,]\d+)?%?")
_EMPHASIZED_PART_RE = re.compile(
    r"^\s*(?:\*\*.+\*\*|__.+__|<(?:strong|b)>.*</(?:strong|b)>)\s*$",
    re.IGNORECASE,
)


def _rule_score_delta(text_a: str, text_b: str) -> float:
    """Return a small penalty (0..0.2) if key rule-patterns differ between texts."""
    score = 0.0
    for pat in _RULE_PATTERNS:
        hits_a = set(pat.findall(text_a))
        hits_b = set(pat.findall(text_b))
        if hits_a != hits_b:
            score += 0.067   # ~0.2 / 3 patterns
    return score


def _set_cosine(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / ((len(left) * len(right)) ** 0.5)


def _lexical_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(_LEXICAL_TOKEN_RE.findall(text_a.lower()))
    tokens_b = set(_LEXICAL_TOKEN_RE.findall(text_b.lower()))
    token_sim = _set_cosine(tokens_a, tokens_b)

    non_numeric_a = {token for token in tokens_a if not _NUMERIC_TOKEN_RE.fullmatch(token)}
    non_numeric_b = {token for token in tokens_b if not _NUMERIC_TOKEN_RE.fullmatch(token)}
    if len(non_numeric_a) >= 2 and len(non_numeric_b) >= 2:
        token_sim = max(token_sim, _set_cosine(non_numeric_a, non_numeric_b) * 0.95)
    return token_sim


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _strip_cell_markup(text: str) -> str:
    text = html.unescape(text or "")
    text = _HTML_TAG_RE.sub("", text)
    text = re.sub(r"[`*_~]+", "", text)
    return re.sub(r"\s+", "", text).strip()


def _raw_cell_parts(cell: str) -> list[str]:
    normalized = _HTML_BREAK_RE.sub("\n", html.unescape(cell or ""))
    return [
        part.strip()
        for part in normalized.splitlines()
        if _strip_cell_markup(part)
    ]


def _cell_parts(cell: str) -> list[str]:
    return [_strip_cell_markup(part) for part in _raw_cell_parts(cell)]


def _plain_cell(cell: str) -> str:
    return "".join(_cell_parts(cell)) or _strip_cell_markup(cell)


def _normalize_cell(cell: str) -> str:
    return _plain_cell(cell).strip().lower().rstrip("：:")


def _header_label_from_cell(cell: str) -> str:
    return _normalize_cell(cell)


def _cell_part_is_emphasized(raw_part: str) -> bool:
    return bool(_EMPHASIZED_PART_RE.fullmatch(raw_part.strip()))


def _cell_looks_like_header_value(cell: str) -> bool:
    raw_parts = _raw_cell_parts(cell)
    if len(raw_parts) < 2:
        return False

    first = raw_parts[0].strip()
    if first.endswith((":", "：")):
        return True

    first_is_emphasized = _cell_part_is_emphasized(first)
    rest_are_emphasized = [_cell_part_is_emphasized(part) for part in raw_parts[1:]]
    return first_is_emphasized and not all(rest_are_emphasized)


def _row_has_embedded_header_values(line: str) -> bool:
    if "|" not in line:
        return False
    cells = [cell for cell in _split_table_row(line) if _plain_cell(cell)]
    if not cells:
        return False
    embedded_count = sum(1 for cell in cells if _cell_looks_like_header_value(cell))
    if len(cells) == 1:
        return embedded_count == 1
    return embedded_count >= 2 and embedded_count >= len(cells) / 2


def _cell_value(cell: str, row_has_embedded_header_values: bool = False) -> str:
    parts = _cell_parts(cell)
    if row_has_embedded_header_values and _cell_looks_like_header_value(cell):
        return "".join(parts[1:])
    return "".join(parts)


def _table_header_labels(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if "|" not in line or _is_table_separator_row(line) or _is_empty_table_row(line):
            continue
        following = next(
            (
                later
                for later in lines[index + 1 :]
                if "|" in later and not _is_empty_table_row(later)
            ),
            "",
        )
        if _is_table_separator_row(following) and not _row_has_embedded_header_values(line):
            return [_header_label_from_cell(cell) for cell in _split_table_row(line)]
        return []
    return []


def _is_table_header_row(line: str, header_labels: list[str]) -> bool:
    if "|" not in line or not header_labels or _row_has_embedded_header_values(line):
        return False
    labels = [_header_label_from_cell(cell) for cell in _split_table_row(line)]
    return labels == header_labels


def _is_table_separator_row(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(_TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells)


def _is_empty_table_row(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(not _plain_cell(cell) for cell in cells)


def _table_row_values(line: str) -> list[str]:
    cells = _split_table_row(line)
    embedded_header_values = _row_has_embedded_header_values(line)
    return [_cell_value(cell, embedded_header_values) for cell in cells]


def _table_row_match_text(line: str) -> str:
    if "|" not in line:
        return _plain_cell(line)
    return " | ".join(_table_row_values(line)).strip()


def _join_table_values(values: list[str]) -> str:
    return " | ".join(values).strip()


def _classification_texts(
    baseline_unit: _ParagraphUnit,
    target_unit: _ParagraphUnit,
) -> tuple[str | None, str | None]:
    baseline_text = baseline_unit.match_text
    target_text = target_unit.match_text
    b_values = baseline_unit.table_values
    t_values = target_unit.table_values
    if (
        b_values is not None
        and t_values is not None
        and len(b_values) == len(t_values)
        and len(b_values) > 1
        and b_values[1:] == t_values[1:]
        and b_values[0] != t_values[0]
    ):
        shared_content = _join_table_values(b_values[1:])
        return shared_content, shared_content
    return baseline_text, target_text


def _looks_like_table(para: Paragraph) -> bool:
    row_count = sum(1 for sent in para.sentences if "|" in sent.text)
    return row_count >= 2


def _should_split_para(para: Paragraph) -> bool:
    sentences = [sent.text.strip() for sent in para.sentences if sent.text.strip()]
    return len(sentences) > 1 and (len(para.text) > _FINE_GRAINED_MAX_CHARS or _looks_like_table(para))


def _expand_paragraphs(paras: list[Paragraph]) -> list[_ParagraphUnit]:
    units: list[_ParagraphUnit] = []
    for para in paras:
        if not _should_split_para(para):
            units.append(_ParagraphUnit(para=para, split_unit=False, match_text=para.text))
            continue

        is_table = _looks_like_table(para)
        table_lines = [sent.text.strip() for sent in para.sentences if sent.text.strip()] if is_table else []
        header_labels = _table_header_labels(table_lines) if is_table else []
        for index, sent in enumerate(para.sentences):
            text = sent.text.strip()
            if not text:
                continue
            if is_table and (
                _is_table_separator_row(text)
                or _is_empty_table_row(text)
                or _is_table_header_row(text, header_labels)
            ):
                continue

            unit_para = Paragraph(
                paragraph_id=f"{para.paragraph_id}#u{index}",
                text=text,
                sentences=[Sentence(text=text)],
            )
            units.append(
                _ParagraphUnit(
                    para=unit_para,
                    split_unit=True,
                    match_text=_table_row_match_text(text),
                    table_values=_table_row_values(text) if is_table else None,
                )
            )
    return units


def match_paragraphs(
    pairs: list[SectionPair],
    embedder: BaseProvider,
    similarity_threshold: float = 0.75,
) -> list[ParagraphPair]:
    """
    For each SectionPair, match paragraphs by embedding similarity.
    Returns flat list of ParagraphPairs across all section pairs.
    """
    results: list[ParagraphPair] = []

    for sp in pairs:
        b_paras = sp.baseline_section.paragraphs if sp.baseline_section else []
        t_paras = sp.target_section.paragraphs if sp.target_section else []
        b_units = _expand_paragraphs(b_paras)
        t_units = _expand_paragraphs(t_paras)
        sec_path = (
            sp.baseline_section.title if sp.baseline_section else
            sp.target_section.title if sp.target_section else ""
        ) or ""

        if not b_units and not t_units:
            continue

        # Sections with no match in other doc → all paragraphs are added/removed
        if not b_units:
            for unit in t_units:
                results.append(ParagraphPair(
                    None,
                    unit.para,
                    0.0,
                    section_path=sec_path,
                    split_unit=unit.split_unit,
                    target_match_text=unit.match_text,
                ))
            continue
        if not t_units:
            for unit in b_units:
                results.append(ParagraphPair(
                    unit.para,
                    None,
                    0.0,
                    section_path=sec_path,
                    split_unit=unit.split_unit,
                    baseline_match_text=unit.match_text,
                ))
            continue

        # Embed all paragraphs in both sections in one batch
        all_texts = [
            unit.match_text if unit.match_text is not None else unit.para.text
            for unit in b_units
        ] + [
            unit.match_text if unit.match_text is not None else unit.para.text
            for unit in t_units
        ]
        all_embeds = embedder.embed(all_texts)
        b_embeds = all_embeds[: len(b_units)]
        t_embeds = all_embeds[len(b_units) :]

        candidates: list[tuple[float, int, int]] = []
        for i, b_unit in enumerate(b_units):
            for j, t_unit in enumerate(t_units):
                b_match_text = b_unit.match_text if b_unit.match_text is not None else b_unit.para.text
                t_match_text = t_unit.match_text if t_unit.match_text is not None else t_unit.para.text
                sim = max(
                    _cosine(b_embeds[i], t_embeds[j]),
                    _lexical_similarity(b_match_text, t_match_text),
                )
                sim -= _rule_score_delta(b_unit.para.text, t_unit.para.text)
                if sim >= similarity_threshold:
                    candidates.append((sim, i, j))

        b_matched: dict[int, tuple[int, float]] = {}
        t_used: set[int] = set()
        for sim, i, j in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
            if i in b_matched or j in t_used:
                continue
            b_matched[i] = (j, sim)
            t_used.add(j)

        for i, b_unit in enumerate(b_units):
            if i in b_matched:
                best_j, similarity = b_matched[i]
                target_unit = t_units[best_j]
                baseline_match_text, target_match_text = _classification_texts(b_unit, target_unit)
                results.append(ParagraphPair(
                    b_unit.para,
                    target_unit.para,
                    similarity,
                    section_path=sec_path,
                    split_unit=b_unit.split_unit or target_unit.split_unit,
                    baseline_match_text=baseline_match_text,
                    target_match_text=target_match_text,
                ))
            else:
                results.append(ParagraphPair(
                    b_unit.para,
                    None,
                    0.0,
                    section_path=sec_path,
                    split_unit=b_unit.split_unit,
                    baseline_match_text=b_unit.match_text,
                ))

        for j, t_unit in enumerate(t_units):
            if j not in t_used:
                results.append(ParagraphPair(
                    None,
                    t_unit.para,
                    0.0,
                    section_path=sec_path,
                    split_unit=t_unit.split_unit,
                    target_match_text=t_unit.match_text,
                ))

    return results
