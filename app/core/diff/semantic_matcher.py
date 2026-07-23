"""Match paragraphs between aligned section pairs using embedding similarity."""
from __future__ import annotations
import html
import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

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
_LLM_CONFLICT_MAX_BASELINES = 6
_LLM_CONFLICT_MAX_TARGETS = 8

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

_CONFLICT_RERANK_PROMPT = """你是文档表格行匹配助手。下面是一组局部冲突候选：多条基准行和多条目标行内容相近，可能发生插入、删除或序号偏移。

匹配原则：
- 优先看核心内容、项目名称、主体、方向、条件、尺寸/数值要求是否一致。
- 行号、序号、页码、展示位置可以不同，不能因为序号相同就强行匹配。
- 如果某条基准行在目标文档没有对应行，target 返回 null。
- 每条目标行最多匹配一条基准行。

基准行：
{baseline_rows}

目标行：
{target_rows}

候选相似度：
{score_rows}

请只输出 JSON：
{{
  "matches": [
    {{"baseline": 1, "target": 2, "confidence": 0.95}},
    {{"baseline": 2, "target": null, "confidence": 0.90}}
  ],
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


def _rank_candidates_by_baseline(
    candidates_by_baseline: dict[int, list[tuple[float, int]]],
) -> dict[int, list[tuple[float, int]]]:
    return {
        i: sorted(row_candidates, key=lambda item: (-item[0], item[1]))[:_LLM_RERANK_TOP_K]
        for i, row_candidates in candidates_by_baseline.items()
    }


def _candidate_score_map(
    ranked_by_baseline: dict[int, list[tuple[float, int]]],
) -> dict[tuple[int, int], float]:
    return {
        (i, j): score
        for i, row_candidates in ranked_by_baseline.items()
        for score, j in row_candidates
    }


def _find_conflict_clusters(
    b_units: list[_ParagraphUnit],
    t_units: list[_ParagraphUnit],
    ranked_by_baseline: dict[int, list[tuple[float, int]]],
) -> list[tuple[list[int], list[int]]]:
    top_target_to_baselines: dict[int, list[int]] = {}
    for i, row_candidates in ranked_by_baseline.items():
        if not row_candidates or b_units[i].table_values is None:
            continue
        _, top_j = row_candidates[0]
        if t_units[top_j].table_values is None:
            continue
        top_target_to_baselines.setdefault(top_j, []).append(i)

    clusters: list[tuple[list[int], list[int]]] = []
    seen_baselines: set[int] = set()
    for _, initial_baselines in sorted(top_target_to_baselines.items()):
        if len(initial_baselines) < 2:
            continue

        baseline_set = set(initial_baselines)
        target_set: set[int] = set()
        for i in baseline_set:
            for _, j in ranked_by_baseline.get(i, []):
                if t_units[j].table_values is not None:
                    target_set.add(j)

        if len(baseline_set) < 2 or not target_set:
            continue
        if baseline_set & seen_baselines:
            continue
        baseline_indices = sorted(baseline_set)
        target_indices = sorted(target_set)
        if (
            len(baseline_indices) > _LLM_CONFLICT_MAX_BASELINES
            or len(target_indices) > _LLM_CONFLICT_MAX_TARGETS
        ):
            continue
        seen_baselines.update(baseline_indices)
        clusters.append((baseline_indices, target_indices))
    return clusters


def _format_conflict_rows(
    units: list[_ParagraphUnit],
    indices: list[int],
) -> str:
    lines: list[str] = []
    for local_index, unit_index in enumerate(indices, start=1):
        unit = units[unit_index]
        lines.append(
            (
                f"{local_index}. 原文：{unit.para.text[:300]}\n"
                f"   匹配文本：{_unit_match_text(unit)[:300]}"
            )
        )
    return "\n".join(lines)


def _format_conflict_scores(
    baseline_indices: list[int],
    target_indices: list[int],
    score_map: dict[tuple[int, int], float],
) -> str:
    target_lookup = {target_index: local for local, target_index in enumerate(target_indices, start=1)}
    rows: list[str] = []
    for baseline_local, baseline_index in enumerate(baseline_indices, start=1):
        parts = []
        for target_index in target_indices:
            score = score_map.get((baseline_index, target_index))
            if score is not None:
                parts.append(f"T{target_lookup[target_index]}={score:.3f}")
        rows.append(f"B{baseline_local}: " + (", ".join(parts) if parts else "无候选"))
    return "\n".join(rows)


def _parse_nullable_index(value: object, upper_bound: int) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "null", "none", "no_match", "unmatched", "无", "无匹配"}:
        return None
    index = int(float(normalized))
    if not 1 <= index <= upper_bound:
        return None
    return index


def _llm_rerank_conflict_cluster(
    b_units: list[_ParagraphUnit],
    t_units: list[_ParagraphUnit],
    baseline_indices: list[int],
    target_indices: list[int],
    score_map: dict[tuple[int, int], float],
    provider: BaseProvider,
) -> tuple[list[tuple[int, int, float, float]], set[int]]:
    prompt = _CONFLICT_RERANK_PROMPT.format(
        baseline_rows=_format_conflict_rows(b_units, baseline_indices),
        target_rows=_format_conflict_rows(t_units, target_indices),
        score_rows=_format_conflict_scores(baseline_indices, target_indices, score_map),
    )
    try:
        response = provider.chat([{"role": "user", "content": prompt}])
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return [], set()
        data = json.loads(match.group())
        raw_matches = data.get("matches", [])
        if not isinstance(raw_matches, list):
            return [], set()

        selected: list[tuple[int, int, float, float]] = []
        unmatched_baselines: set[int] = set()
        used_targets: set[int] = set()
        max_cluster_score = max(
            (
                score
                for (i, j), score in score_map.items()
                if i in baseline_indices and j in target_indices
            ),
            default=0.0,
        )
        for item in raw_matches:
            if not isinstance(item, dict):
                continue
            try:
                baseline_local = int(float(str(item.get("baseline")).strip()))
                if not 1 <= baseline_local <= len(baseline_indices):
                    continue
                target_local = _parse_nullable_index(item.get("target"), len(target_indices))
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if confidence < _LLM_RERANK_MIN_CONFIDENCE:
                continue

            baseline_index = baseline_indices[baseline_local - 1]
            if target_local is None:
                unmatched_baselines.add(baseline_index)
                continue
            target_index = target_indices[target_local - 1]
            if target_index in used_targets:
                continue
            selected_score = score_map.get((baseline_index, target_index), 0.0)
            rank_score = max(max_cluster_score + 0.002, selected_score, _LLM_RERANK_MIN_SCORE)
            selected.append((baseline_index, target_index, rank_score, selected_score))
            used_targets.add(target_index)
        return selected, unmatched_baselines
    except Exception as exc:
        logger.warning("LLM conflict rerank failed, using embedding score: %s", exc)
        return [], set()


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
            stripped_text = para.text.strip()
            is_table_row = (
                stripped_text.startswith("|")
                and stripped_text.endswith("|")
            )
            units.append(
                _ParagraphUnit(
                    para=para,
                    split_unit=False,
                    match_text=(
                        _table_row_match_text(para.text)
                        if is_table_row
                        else para.text
                    ),
                    table_values=(
                        _table_row_values(para.text)
                        if is_table_row
                        else None
                    ),
                )
            )
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
                page_no=para.page_no,
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


def _is_ordinary_unit(unit: _ParagraphUnit) -> bool:
    return unit.table_values is None


def _context_side_similarity(left: str, right: str) -> float | None:
    if not left and not right:
        return None
    if not left or not right:
        return None
    normalized_left = _normalize_match_text(left)
    normalized_right = _normalize_match_text(right)
    if normalized_left == normalized_right:
        return 1.0
    return min(0.65, max(
        _lexical_similarity(left, right),
        SequenceMatcher(None, normalized_left, normalized_right).ratio(),
    ))


def _ordinary_context_similarity(
    b_units: list[_ParagraphUnit],
    t_units: list[_ParagraphUnit],
    i: int,
    j: int,
) -> float:
    weighted_scores: list[tuple[float, float]] = []
    for offset, weight in ((-1, 2.0), (1, 2.0), (-2, 1.0), (2, 1.0)):
        b_index = i + offset
        t_index = j + offset
        b_text = (
            _unit_match_text(b_units[b_index])
            if 0 <= b_index < len(b_units) and _is_ordinary_unit(b_units[b_index])
            else ""
        )
        t_text = (
            _unit_match_text(t_units[t_index])
            if 0 <= t_index < len(t_units) and _is_ordinary_unit(t_units[t_index])
            else ""
        )
        score = _context_side_similarity(b_text, t_text)
        if score is not None:
            weighted_scores.append((score, weight))
    if not weighted_scores:
        return 1.0
    return sum(score * weight for score, weight in weighted_scores) / sum(
        weight for _, weight in weighted_scores
    )


def _relative_position_similarity(
    i: int,
    baseline_count: int,
    j: int,
    target_count: int,
) -> float:
    baseline_position = i / max(1, baseline_count - 1)
    target_position = j / max(1, target_count - 1)
    return max(0.0, 1.0 - abs(baseline_position - target_position))


def _ordinary_match_score(
    base_score: float,
    b_units: list[_ParagraphUnit],
    t_units: list[_ParagraphUnit],
    i: int,
    j: int,
) -> float:
    b_text = _normalize_match_text(_unit_match_text(b_units[i]))
    t_text = _normalize_match_text(_unit_match_text(t_units[j]))
    b_duplicates = sum(
        _normalize_match_text(_unit_match_text(unit)) == b_text
        for unit in b_units
        if _is_ordinary_unit(unit)
    )
    t_duplicates = sum(
        _normalize_match_text(_unit_match_text(unit)) == t_text
        for unit in t_units
        if _is_ordinary_unit(unit)
    )
    ambiguous = (
        min(len(b_text), len(t_text)) <= 24
        or b_duplicates > 1
        or t_duplicates > 1
    )
    if not ambiguous:
        return base_score

    context_score = _ordinary_context_similarity(b_units, t_units, i, j)
    position_score = _relative_position_similarity(i, len(b_units), j, len(t_units))
    role_score = 1.0
    return (
        0.55 * base_score
        + 0.30 * context_score
        + 0.10 * position_score
        + 0.05 * role_score
    )


def _select_monotonic_ordinary_matches(
    candidates: list[tuple[float, float, int, int]],
    b_units: list[_ParagraphUnit],
    t_units: list[_ParagraphUnit],
) -> dict[int, tuple[int, float]]:
    baseline_indices = [
        index for index, unit in enumerate(b_units) if _is_ordinary_unit(unit)
    ]
    target_indices = [
        index for index, unit in enumerate(t_units) if _is_ordinary_unit(unit)
    ]
    score_map: dict[tuple[int, int], tuple[float, float]] = {}
    for rank_score, similarity, i, j in candidates:
        if not (_is_ordinary_unit(b_units[i]) and _is_ordinary_unit(t_units[j])):
            continue
        previous = score_map.get((i, j))
        if previous is None or rank_score > previous[0]:
            score_map[(i, j)] = (rank_score, similarity)

    rows = len(baseline_indices)
    columns = len(target_indices)
    dp = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    choice = [[""] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        choice[row][0] = "up"
    for column in range(1, columns + 1):
        choice[0][column] = "left"

    for row in range(1, rows + 1):
        i = baseline_indices[row - 1]
        for column in range(1, columns + 1):
            j = target_indices[column - 1]
            best = dp[row - 1][column]
            direction = "up"
            if dp[row][column - 1] > best:
                best = dp[row][column - 1]
                direction = "left"
            candidate = score_map.get((i, j))
            if candidate is not None:
                match_total = dp[row - 1][column - 1] + candidate[0]
                if match_total >= best:
                    best = match_total
                    direction = "match"
            dp[row][column] = best
            choice[row][column] = direction

    selected: dict[int, tuple[int, float]] = {}
    row, column = rows, columns
    while row > 0 or column > 0:
        direction = choice[row][column]
        if direction == "match":
            i = baseline_indices[row - 1]
            j = target_indices[column - 1]
            selected[i] = (j, score_map[(i, j)][1])
            row -= 1
            column -= 1
        elif direction == "up":
            row -= 1
        else:
            column -= 1
    return selected


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
                if _is_ordinary_unit(b_unit) != _is_ordinary_unit(t_unit):
                    continue
                b_match_text = _unit_match_text(b_unit)
                t_match_text = _unit_match_text(t_unit)
                base_sim = max(
                    _cosine(b_embeds[i], t_embeds[j]),
                    _lexical_similarity(b_match_text, t_match_text),
                )
                base_sim -= _rule_score_delta(b_unit.para.text, t_unit.para.text)
                sim = (
                    _ordinary_match_score(
                        base_sim,
                        b_units,
                        t_units,
                        i,
                        j,
                    )
                    if _is_ordinary_unit(b_unit)
                    else base_sim
                )
                if sim >= rerank_floor:
                    candidates_by_baseline.setdefault(i, []).append((sim, j))
                if sim >= similarity_threshold:
                    candidates.append((sim, sim, i, j))

        llm_unmatched_baselines: set[int] = set()
        if rerank_provider is not None and use_llm_rerank:
            ranked_by_baseline = _rank_candidates_by_baseline(candidates_by_baseline)
            score_map = _candidate_score_map(ranked_by_baseline)
            for baseline_indices, target_indices in _find_conflict_clusters(
                b_units,
                t_units,
                ranked_by_baseline,
            ):
                selected_pairs, unmatched_baselines = _llm_rerank_conflict_cluster(
                    b_units,
                    t_units,
                    baseline_indices,
                    target_indices,
                    score_map,
                    rerank_provider,
                )
                llm_unmatched_baselines.update(unmatched_baselines)
                for i, j, rank_score, selected_score in selected_pairs:
                    candidates.append((rank_score, selected_score, i, j))

            for i, ranked in ranked_by_baseline.items():
                if i in llm_unmatched_baselines:
                    continue
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

        b_matched = _select_monotonic_ordinary_matches(
            candidates,
            b_units,
            t_units,
        )
        t_used: set[int] = {
            target_index for target_index, _ in b_matched.values()
        }
        for _, sim, i, j in sorted(candidates, key=lambda item: (-item[0], item[2], item[3])):
            if _is_ordinary_unit(b_units[i]):
                continue
            if i in b_matched or j in t_used:
                continue
            if i in llm_unmatched_baselines:
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
