"""Match paragraphs between aligned section pairs using embedding similarity."""
from __future__ import annotations
import html
import json
import logging
import re
from dataclasses import dataclass

import numpy as np

from app.core.diff.structure_aligner import SectionPair
from app.core.model.base_provider import BaseProvider
from app.core.types import Paragraph, Sentence

logger = logging.getLogger(__name__)


@dataclass
class ParagraphPair:
    baseline_para: Paragraph | None
    target_para: Paragraph | None
    similarity: float   # cosine similarity, -1..1 (1 = identical)
    section_path: str = ""
    split_unit: bool = False
    baseline_match_text: str | None = None
    target_match_text: str | None = None
    baseline_table_header: bool = False
    target_table_header: bool = False


@dataclass
class _ParagraphUnit:
    para: Paragraph
    split_unit: bool
    match_text: str | None = None
    table_values: list[str] | None = None
    table_header: bool = False


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


_RULE_PATTERNS = [
    re.compile(r'\d+[\.,]\d*'),          # numbers
    re.compile(r'[不无未没]'),            # negations
    re.compile(r'(?:应|须|必须|不得|禁止)'),  # obligation words
]

_TABLE_SEPARATOR_CELL_RE = re.compile(r":?-{2,}:?")
_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[^>\n]+>")
_LEXICAL_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:[\.,]\d+)?%?|[\u4e00-\u9fff]", re.IGNORECASE)
_NUMERIC_TOKEN_RE = re.compile(r"\d+(?:[\.,]\d+)?%?")
_EMPHASIZED_PART_RE = re.compile(
    r"^\s*(?:\*\*.+\*\*|__.+__|<(?:strong|b)>.*</(?:strong|b)>)\s*$",
    re.IGNORECASE,
)
_LLM_RERANK_TOP_K = 5
_LLM_RERANK_SCORE_GAP = 0.15
_LLM_RERANK_MIN_SCORE = 0.45
_LLM_RERANK_MIN_CONFIDENCE = 0.6

_MATCH_RERANK_PROMPT = """你是文档表格行匹配助手。请判断“基准行”应该和哪一个候选行视为同一条记录。

匹配原则：
- 优先看核心内容、名称、标题、主体是否一致。
- 行号、页码、展示序号、位置变化可以不同。
- 数量、金额、日期等如果属于标题/业务内容的一部分，不要忽略。
- 如果没有合适候选，返回 null。

基准行：
原文：{baseline_text}
匹配文本：{baseline_match}

候选行：
{candidates}

请只输出 JSON：
{{
  "matched_candidate": 1,
  "confidence": 0.0,
  "reason": "简短原因"
}}
"""


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
    values = _table_row_values(line)
    leading_count = _leading_numeric_cell_count(values)
    if 0 < leading_count < len(values):
        values = values[leading_count:]
    return " | ".join(values).strip()


def _join_table_values(values: list[str]) -> str:
    return " | ".join(values).strip()


def _is_numeric_cell(value: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+(?:[\.,]\d+)?%?", value.strip()))


def _is_item_number_cell(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)+", value.strip()))


def _looks_like_structural_table_header(line: str, following: str = "") -> bool:
    if "|" not in line or not _is_table_separator_row(following):
        return False
    if _row_has_embedded_header_values(line):
        return False

    values = [value for value in _table_row_values(line) if value]
    if len(values) < 2:
        return False
    if _leading_numeric_cell_count(values) > 0:
        return False
    if any(_is_item_number_cell(value) for value in values):
        return False
    return True


def _leading_numeric_cell_count(values: list[str]) -> int:
    count = 0
    for value in values:
        if not _is_numeric_cell(value):
            break
        count += 1
    return count


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
    ):
        leading_count = min(_leading_numeric_cell_count(b_values), _leading_numeric_cell_count(t_values))
        if (
            leading_count > 0
            and leading_count < len(b_values)
            and b_values[leading_count:] == t_values[leading_count:]
            and b_values[:leading_count] != t_values[:leading_count]
        ):
            shared_content = _join_table_values(b_values[leading_count:])
            return shared_content, shared_content
    return baseline_text, target_text


def _unit_match_text(unit: _ParagraphUnit) -> str:
    return unit.match_text if unit.match_text is not None else unit.para.text


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip().lower()


def _table_header_signature(unit: _ParagraphUnit) -> str:
    if not unit.table_header:
        return ""
    values = unit.table_values
    if values is None:
        values = _table_row_values(unit.para.text) if "|" in unit.para.text else [_unit_match_text(unit)]
    normalized = [_normalize_match_text(value) for value in values if _normalize_match_text(value)]
    return "\x1f".join(normalized)


def _table_header_signature_counts(units: list[_ParagraphUnit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        signature = _table_header_signature(unit)
        if signature:
            counts[signature] = counts.get(signature, 0) + 1
    return counts


def _table_header_values_look_like_schema_labels(unit: _ParagraphUnit) -> bool:
    values = [value for value in (unit.table_values or []) if value]
    if len(values) < 2:
        return False
    if _leading_numeric_cell_count(values) > 0:
        return False
    if any(_is_item_number_cell(value) for value in values):
        return False
    return True


def _should_suppress_unmatched_table_header(
    unit: _ParagraphUnit,
    same_header_counts: dict[str, int],
    other_header_counts: dict[str, int],
) -> bool:
    signature = _table_header_signature(unit)
    if not signature:
        return False
    return (
        _table_header_values_look_like_schema_labels(unit)
        or same_header_counts.get(signature, 0) > 1
        or other_header_counts.get(signature, 0) > 0
    )


def _should_llm_rerank_candidates(
    baseline_unit: _ParagraphUnit,
    target_units: list[_ParagraphUnit],
    candidates: list[tuple[float, int]],
) -> bool:
    if baseline_unit.table_values is None or len(candidates) < 2:
        return False

    top_score, top_j = candidates[0]
    second_score, _ = candidates[1]
    baseline_text = _normalize_match_text(_unit_match_text(baseline_unit))
    top_text = _normalize_match_text(_unit_match_text(target_units[top_j]))
    if baseline_text and baseline_text == top_text:
        return False
    return top_score - second_score <= _LLM_RERANK_SCORE_GAP


def _llm_rerank_candidate(
    baseline_unit: _ParagraphUnit,
    target_units: list[_ParagraphUnit],
    candidates: list[tuple[float, int]],
    provider: BaseProvider,
) -> int | None:
    candidate_lines = []
    for index, (score, target_index) in enumerate(candidates, start=1):
        target_unit = target_units[target_index]
        candidate_lines.append(
            (
                f"{index}. 相似度：{score:.3f}\n"
                f"   原文：{target_unit.para.text[:300]}\n"
                f"   匹配文本：{_unit_match_text(target_unit)[:300]}"
            )
        )

    prompt = _MATCH_RERANK_PROMPT.format(
        baseline_text=baseline_unit.para.text[:500],
        baseline_match=_unit_match_text(baseline_unit)[:500],
        candidates="\n".join(candidate_lines),
    )
    try:
        response = provider.chat([{"role": "user", "content": prompt}])
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        raw_index = (
            data.get("matched_candidate")
            if "matched_candidate" in data
            else data.get("candidate_index", data.get("index"))
        )
        if raw_index is None:
            return None
        confidence = float(data.get("confidence", 0.0))
        if confidence < _LLM_RERANK_MIN_CONFIDENCE:
            return None
        candidate_index = int(raw_index) - 1
        if not 0 <= candidate_index < len(candidates):
            return None
        return candidates[candidate_index][1]
    except Exception as exc:
        logger.warning("LLM match rerank failed, using embedding score: %s", exc)
        return None


def _looks_like_table(para: Paragraph) -> bool:
    row_count = sum(1 for sent in para.sentences if "|" in sent.text)
    return row_count >= 2


def _should_split_para(para: Paragraph) -> bool:
    sentences = [sent.text.strip() for sent in para.sentences if sent.text.strip()]
    return len(sentences) > 1


def _expand_paragraphs(paras: list[Paragraph]) -> list[_ParagraphUnit]:
    units: list[_ParagraphUnit] = []
    for para in paras:
        if not _should_split_para(para):
            units.append(_ParagraphUnit(para=para, split_unit=False, match_text=para.text))
            continue

        is_table = _looks_like_table(para)
        sentences = [sent.text.strip() for sent in para.sentences if sent.text.strip()]
        for index, text in enumerate(sentences):
            if is_table and (
                _is_table_separator_row(text)
                or _is_empty_table_row(text)
            ):
                continue

            unit_para = Paragraph(
                paragraph_id=f"{para.paragraph_id}#u{index}",
                text=text,
                sentences=[Sentence(text=text)],
            )
            following = sentences[index + 1] if index + 1 < len(sentences) else ""
            units.append(
                _ParagraphUnit(
                    para=unit_para,
                    split_unit=True,
                    match_text=_table_row_match_text(text) if is_table else text,
                    table_values=_table_row_values(text) if is_table else None,
                    table_header=_looks_like_structural_table_header(text, following) if is_table else False,
                )
            )
    return units


def match_paragraphs(
    pairs: list[SectionPair],
    embedder: BaseProvider,
    similarity_threshold: float = 0.75,
    *,
    rerank_provider: BaseProvider | None = None,
    use_llm_rerank: bool = True,
    suppress_unmatched_table_headers: bool = True,
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
        b_header_counts = _table_header_signature_counts(b_units)
        t_header_counts = _table_header_signature_counts(t_units)
        sec_path = (
            sp.baseline_section.title if sp.baseline_section else
            sp.target_section.title if sp.target_section else ""
        ) or ""

        if not b_units and not t_units:
            continue

        # Sections with no match in other doc → all paragraphs are added/removed
        if not b_units:
            for unit in t_units:
                if (
                    suppress_unmatched_table_headers
                    and _should_suppress_unmatched_table_header(unit, t_header_counts, b_header_counts)
                ):
                    continue
                results.append(ParagraphPair(
                    None,
                    unit.para,
                    0.0,
                    section_path=sec_path,
                    split_unit=unit.split_unit,
                    target_match_text=unit.match_text,
                    target_table_header=unit.table_header,
                ))
            continue
        if not t_units:
            for unit in b_units:
                if (
                    suppress_unmatched_table_headers
                    and _should_suppress_unmatched_table_header(unit, b_header_counts, t_header_counts)
                ):
                    continue
                results.append(ParagraphPair(
                    unit.para,
                    None,
                    0.0,
                    section_path=sec_path,
                    split_unit=unit.split_unit,
                    baseline_match_text=unit.match_text,
                    baseline_table_header=unit.table_header,
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

        candidates: list[tuple[float, float, int, int]] = []
        candidates_by_baseline: dict[int, list[tuple[float, int]]] = {}
        rerank_floor = min(similarity_threshold, _LLM_RERANK_MIN_SCORE)
        for i, b_unit in enumerate(b_units):
            for j, t_unit in enumerate(t_units):
                b_match_text = _unit_match_text(b_unit)
                t_match_text = _unit_match_text(t_unit)
                sim = max(
                    _cosine(b_embeds[i], t_embeds[j]),
                    _lexical_similarity(b_match_text, t_match_text),
                )
                sim -= _rule_score_delta(b_unit.para.text, t_unit.para.text)
                if sim >= rerank_floor:
                    candidates_by_baseline.setdefault(i, []).append((sim, j))
                if sim >= similarity_threshold:
                    candidates.append((sim, sim, i, j))

        if rerank_provider is not None and use_llm_rerank:
            for i, row_candidates in candidates_by_baseline.items():
                ranked = sorted(row_candidates, key=lambda item: (-item[0], item[1]))[:_LLM_RERANK_TOP_K]
                if not _should_llm_rerank_candidates(b_units[i], t_units, ranked):
                    continue
                selected_j = _llm_rerank_candidate(
                    b_units[i],
                    t_units,
                    ranked,
                    rerank_provider,
                )
                if selected_j is None:
                    continue
                selected_score = next(
                    score for score, candidate_j in ranked if candidate_j == selected_j
                )
                rank_score = max(ranked[0][0] + 0.001, similarity_threshold)
                candidates.append((rank_score, selected_score, i, selected_j))

        b_matched: dict[int, tuple[int, float]] = {}
        t_used: set[int] = set()
        for _, sim, i, j in sorted(candidates, key=lambda item: (-item[0], item[2], item[3])):
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
                    baseline_table_header=b_unit.table_header,
                    target_table_header=target_unit.table_header,
                ))
            else:
                if (
                    suppress_unmatched_table_headers
                    and _should_suppress_unmatched_table_header(b_unit, b_header_counts, t_header_counts)
                ):
                    continue
                results.append(ParagraphPair(
                    b_unit.para,
                    None,
                    0.0,
                    section_path=sec_path,
                    split_unit=b_unit.split_unit,
                    baseline_match_text=b_unit.match_text,
                    baseline_table_header=b_unit.table_header,
                ))

        for j, t_unit in enumerate(t_units):
            if j not in t_used:
                if (
                    suppress_unmatched_table_headers
                    and _should_suppress_unmatched_table_header(t_unit, t_header_counts, b_header_counts)
                ):
                    continue
                results.append(ParagraphPair(
                    None,
                    t_unit.para,
                    0.0,
                    section_path=sec_path,
                    split_unit=t_unit.split_unit,
                    target_match_text=t_unit.match_text,
                    target_table_header=t_unit.table_header,
                ))

    return results
