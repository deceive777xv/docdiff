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
from app.core.types import Paragraph, Section, Sentence

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


@dataclass
class _ParagraphUnit:
    para: Paragraph
    split_unit: bool
    match_text: str | None = None
    table_values: list[str] | None = None
    section_path: str = ""
    section_level: int = 0
    document_title: str = ""
    source_paragraph_id: str = ""


@dataclass
class _SectionMatchScope:
    baseline_sections: list[Section]
    target_sections: list[Section]
    title: str


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
_EXACT_WINDOW_MIN_CHARS = 24
_EXACT_CROSS_PARAGRAPH_WINDOW_MIN_CHARS = 4
_EXACT_WINDOW_MAX_CHARS = 2000

_MATCH_RERANK_PROMPT = """你是文档表格行匹配助手。请判断“基准行”应该和哪一个候选行视为同一条记录。

匹配原则：
- 优先看核心内容、名称、标题、主体是否一致。
- 行号、页码、展示序号、位置变化可以不同。
- 数量、金额、日期等如果属于标题/业务内容的一部分，不要忽略。
- 如果没有合适候选，返回 null。

基准行：
文档/章节：{baseline_context}
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
    without_list_marker = re.sub(
        r"^\s*(?:[-*+•]\s+)+",
        "",
        text or "",
    )
    return re.sub(r"\s+", "", without_list_marker).strip().lower()


def _pair_text(pair: ParagraphPair, side: str) -> str:
    if side == "baseline":
        return pair.baseline_match_text or (
            pair.baseline_para.text if pair.baseline_para is not None else ""
        )
    return pair.target_match_text or (
        pair.target_para.text if pair.target_para is not None else ""
    )


def _reconcile_unique_exact_unmatched(
    results: list[ParagraphPair],
) -> list[ParagraphPair]:
    baseline_indexes = [
        index
        for index, pair in enumerate(results)
        if pair.baseline_para is not None
        and pair.target_para is None
        and not pair.split_unit
        and "|" not in pair.baseline_para.text
    ]
    target_indexes = [
        index
        for index, pair in enumerate(results)
        if pair.baseline_para is None
        and pair.target_para is not None
        and not pair.split_unit
        and "|" not in pair.target_para.text
    ]
    selected: dict[int, int] = {}
    target_used: set[int] = set()

    def select_unique(
        baseline_key,
        target_key,
        *,
        predicate,
        pair_filter=lambda _baseline, _target: True,
    ) -> None:
        baseline_groups: dict[object, list[int]] = {}
        target_groups: dict[object, list[int]] = {}
        for index in baseline_indexes:
            if index in selected:
                continue
            key = baseline_key(index)
            if predicate(key):
                baseline_groups.setdefault(key, []).append(index)
        for index in target_indexes:
            if index in target_used:
                continue
            key = target_key(index)
            if predicate(key):
                target_groups.setdefault(key, []).append(index)
        for key, source_indexes in baseline_groups.items():
            candidate_indexes = target_groups.get(key, [])
            if (
                len(source_indexes) == 1
                and len(candidate_indexes) == 1
                and pair_filter(source_indexes[0], candidate_indexes[0])
            ):
                selected[source_indexes[0]] = candidate_indexes[0]
                target_used.add(candidate_indexes[0])

    select_unique(
        lambda index: (
            results[index].section_path,
            _normalize_match_text(_pair_text(results[index], "baseline")),
        ),
        lambda index: (
            results[index].section_path,
            _normalize_match_text(_pair_text(results[index], "target")),
        ),
        predicate=lambda key: bool(key[1]) and len(key[1]) <= 24,
    )
    select_unique(
        lambda index: _normalize_match_text(
            _pair_text(results[index], "baseline")
        ),
        lambda index: _normalize_match_text(
            _pair_text(results[index], "target")
        ),
        predicate=lambda key: len(key) >= 24,
        pair_filter=lambda baseline_index, target_index: (
            results[baseline_index].section_path
            != results[target_index].section_path
        ),
    )

    reconciled: list[ParagraphPair] = []
    for index, pair in enumerate(results):
        if index in target_used:
            continue
        target_index = selected.get(index)
        if target_index is None:
            reconciled.append(pair)
            continue
        target_pair = results[target_index]
        reconciled.append(
            ParagraphPair(
                baseline_para=pair.baseline_para,
                target_para=target_pair.target_para,
                similarity=1.0,
                section_path=pair.section_path or target_pair.section_path,
                split_unit=False,
                baseline_match_text=pair.baseline_match_text,
                target_match_text=target_pair.target_match_text,
            )
        )
    return reconciled


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
                f"   文档/章节：{target_unit.document_title} / {target_unit.section_path}\n"
                f"   原文：{target_unit.para.text[:300]}\n"
                f"   匹配文本：{_unit_match_text(target_unit)[:300]}"
            )
        )

    prompt = _MATCH_RERANK_PROMPT.format(
        baseline_context=(
            f"{baseline_unit.document_title} / {baseline_unit.section_path}"
        ),
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
                f"{local_index}. 文档/章节：{unit.document_title} / {unit.section_path}\n"
                f"   原文：{unit.para.text[:300]}\n"
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


def _expand_paragraphs(
    paras: list[Paragraph],
    section_path: str = "",
    section_level: int = 0,
    document_title: str = "",
) -> list[_ParagraphUnit]:
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
                    section_path=section_path,
                    section_level=section_level,
                    document_title=document_title,
                    source_paragraph_id=para.paragraph_id,
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
            units.append(
                _ParagraphUnit(
                    para=unit_para,
                    split_unit=True,
                    match_text=_table_row_match_text(text) if is_table else text,
                    table_values=_table_row_values(text) if is_table else None,
                    section_path=section_path,
                    section_level=section_level,
                    document_title=document_title,
                    source_paragraph_id=para.paragraph_id,
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
    b_ordinary_units: list[_ParagraphUnit],
    t_ordinary_units: list[_ParagraphUnit],
    i: int,
    j: int,
) -> float:
    weighted_scores: list[tuple[float, float]] = []
    for offset, weight in ((-1, 2.0), (1, 2.0), (-2, 1.0), (2, 1.0)):
        b_index = i + offset
        t_index = j + offset
        b_text = (
            _unit_match_text(b_ordinary_units[b_index])
            if 0 <= b_index < len(b_ordinary_units)
            else ""
        )
        t_text = (
            _unit_match_text(t_ordinary_units[t_index])
            if 0 <= t_index < len(t_ordinary_units)
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
    b_ordinary_units: list[_ParagraphUnit],
    t_ordinary_units: list[_ParagraphUnit],
    i: int,
    j: int,
) -> float:
    b_text = _normalize_match_text(_unit_match_text(b_ordinary_units[i]))
    t_text = _normalize_match_text(_unit_match_text(t_ordinary_units[j]))
    b_duplicates = sum(
        _normalize_match_text(_unit_match_text(unit)) == b_text
        for unit in b_ordinary_units
    )
    t_duplicates = sum(
        _normalize_match_text(_unit_match_text(unit)) == t_text
        for unit in t_ordinary_units
    )
    ambiguous = (
        min(len(b_text), len(t_text)) <= 24
        or b_duplicates > 1
        or t_duplicates > 1
    )
    if not ambiguous:
        return base_score

    context_score = _ordinary_context_similarity(
        b_ordinary_units,
        t_ordinary_units,
        i,
        j,
    )
    position_score = _relative_position_similarity(
        i,
        len(b_ordinary_units),
        j,
        len(t_ordinary_units),
    )
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


def _section_match_scopes(pairs: list[SectionPair]) -> list[_SectionMatchScope]:
    scopes: list[_SectionMatchScope] = []
    scope_by_pair: dict[int, _SectionMatchScope] = {}

    for pair in pairs:
        if pair.baseline_section is None or pair.target_section is None:
            continue
        scope = _SectionMatchScope(
            baseline_sections=[pair.baseline_section],
            target_sections=[pair.target_section],
            title=pair.baseline_section.title or pair.target_section.title,
        )
        scopes.append(scope)
        scope_by_pair[id(pair)] = scope

    def attach_side(side: str) -> None:
        entries: list[tuple[int, Section, SectionPair]] = []
        for pair in pairs:
            section = (
                pair.baseline_section
                if side == "baseline"
                else pair.target_section
            )
            index = (
                pair.baseline_index
                if side == "baseline"
                else pair.target_index
            )
            if section is not None and index is not None:
                entries.append((index, section, pair))

        stack: list[tuple[int, _SectionMatchScope]] = []
        for _, section, pair in sorted(entries, key=lambda item: item[0]):
            while stack and stack[-1][0] >= section.level:
                stack.pop()

            scope = scope_by_pair.get(id(pair))
            if scope is None:
                scope = stack[-1][1] if stack else None
                if scope is None:
                    scope = _SectionMatchScope(
                        baseline_sections=[],
                        target_sections=[],
                        title=section.title,
                    )
                    scopes.append(scope)
                if side == "baseline":
                    scope.baseline_sections.append(section)
                else:
                    scope.target_sections.append(section)
                scope_by_pair[id(pair)] = scope

            stack.append((section.level, scope))

    attach_side("baseline")
    attach_side("target")

    for pair in pairs:
        if id(pair) in scope_by_pair:
            continue
        baseline_sections = (
            [pair.baseline_section] if pair.baseline_section is not None else []
        )
        target_sections = (
            [pair.target_section] if pair.target_section is not None else []
        )
        title = (
            pair.baseline_section.title
            if pair.baseline_section is not None
            else pair.target_section.title
            if pair.target_section is not None
            else ""
        )
        scope = _SectionMatchScope(baseline_sections, target_sections, title)
        scopes.append(scope)
        scope_by_pair[id(pair)] = scope

    return scopes


def _full_section_paths(
    pairs: list[SectionPair],
    side: str,
) -> dict[int, str]:
    paths: dict[int, str] = {}
    stack: list[tuple[int, str]] = []
    attribute = "baseline_section" if side == "baseline" else "target_section"
    index_attribute = "baseline_index" if side == "baseline" else "target_index"
    ordered_pairs = sorted(
        (
            pair
            for pair in pairs
            if getattr(pair, attribute) is not None
            and getattr(pair, index_attribute) is not None
        ),
        key=lambda pair: getattr(pair, index_attribute),
    )
    for pair in ordered_pairs:
        section = getattr(pair, attribute)
        if section is None or id(section) in paths:
            continue
        while stack and stack[-1][0] >= section.level:
            stack.pop()
        stack.append((section.level, section.title))
        paths[id(section)] = " / ".join(title for _level, title in stack)
    return paths


def _window_normalized_text(units: list[_ParagraphUnit]) -> str:
    return "".join(_normalize_match_text(_unit_match_text(unit)) for unit in units)


def _merge_window_units(units: list[_ParagraphUnit]) -> _ParagraphUnit:
    source_ids = [unit.para.paragraph_id for unit in units]
    text = "\n".join(unit.para.text for unit in units)
    match_text = "\n".join(_unit_match_text(unit) for unit in units)
    sentences = [
        sentence
        for unit in units
        for sentence in unit.para.sentences
    ]
    most_specific = max(units, key=lambda unit: unit.section_level)
    return _ParagraphUnit(
        para=Paragraph(
            paragraph_id="window:" + ":".join(source_ids),
            text=text,
            sentences=sentences,
            page_no=units[0].para.page_no,
        ),
        split_unit=True,
        match_text=match_text,
        section_path=most_specific.section_path,
        section_level=most_specific.section_level,
        document_title=most_specific.document_title,
        source_paragraph_id=(
            units[0].source_paragraph_id
            if len({unit.source_paragraph_id for unit in units}) == 1
            else ""
        ),
    )


def _specific_section_path(
    baseline_unit: _ParagraphUnit,
    target_unit: _ParagraphUnit,
    default: str,
) -> str:
    candidates = [
        unit
        for unit in (baseline_unit, target_unit)
        if unit.section_path
    ]
    if not candidates:
        return default
    return max(candidates, key=lambda unit: unit.section_level).section_path


def _exact_adjacent_window_matches(
    b_units: list[_ParagraphUnit],
    t_units: list[_ParagraphUnit],
) -> list[
    tuple[
        tuple[int, ...],
        tuple[int, ...],
        _ParagraphUnit,
        _ParagraphUnit,
    ]
]:
    baseline_single = [
        _normalize_match_text(_unit_match_text(unit))
        for unit in b_units
    ]
    target_single = [
        _normalize_match_text(_unit_match_text(unit))
        for unit in t_units
    ]
    baseline_positions: dict[str, list[int]] = {}
    target_positions: dict[str, list[int]] = {}
    for index, normalized in enumerate(baseline_single):
        if normalized:
            baseline_positions.setdefault(normalized, []).append(index)
    for index, normalized in enumerate(target_single):
        if normalized:
            target_positions.setdefault(normalized, []).append(index)
    anchored_baseline: set[int] = set()
    anchored_target: set[int] = set()
    for normalized, baseline_indexes in baseline_positions.items():
        target_indexes = target_positions.get(normalized, [])
        if len(baseline_indexes) == len(target_indexes) == 1:
            anchored_baseline.add(baseline_indexes[0])
            anchored_target.add(target_indexes[0])

    def windows(
        units: list[_ParagraphUnit],
        anchored: set[int],
    ) -> dict[str, list[tuple[tuple[int, ...], _ParagraphUnit]]]:
        by_text: dict[
            str,
            list[tuple[tuple[int, ...], _ParagraphUnit]],
        ] = {}
        for size in range(1, len(units) + 1):
            for start in range(len(units) - size + 1):
                indexes = tuple(range(start, start + size))
                if anchored.intersection(indexes):
                    continue
                window = units[start : start + size]
                end = start + size
                if (
                    start > 0
                    and units[start - 1].source_paragraph_id
                    == window[0].source_paragraph_id
                ):
                    continue
                if (
                    end < len(units)
                    and units[end].source_paragraph_id
                    == window[-1].source_paragraph_id
                ):
                    continue
                if len({unit.source_paragraph_id for unit in window}) > 2:
                    continue
                if len({unit.section_path for unit in window}) != 1:
                    continue
                if not all(_is_ordinary_unit(unit) for unit in window):
                    continue
                normalized = _window_normalized_text(window)
                if not (
                    _EXACT_CROSS_PARAGRAPH_WINDOW_MIN_CHARS
                    <= len(normalized)
                    <= _EXACT_WINDOW_MAX_CHARS
                ):
                    continue
                by_text.setdefault(normalized, []).append(
                    (
                        indexes,
                        window[0] if size == 1 else _merge_window_units(window),
                    )
                )
        return by_text

    baseline_windows = windows(b_units, anchored_baseline)
    target_windows = windows(t_units, anchored_target)
    candidates: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            _ParagraphUnit,
            _ParagraphUnit,
            str,
        ]
    ] = []
    for normalized in baseline_windows.keys() & target_windows.keys():
        for baseline_indexes, baseline_unit in baseline_windows[normalized]:
            for target_indexes, target_unit in target_windows[normalized]:
                if (
                    len(normalized) < _EXACT_WINDOW_MIN_CHARS
                    and baseline_unit.source_paragraph_id
                    and target_unit.source_paragraph_id
                ):
                    continue
                if (
                    not baseline_unit.source_paragraph_id
                    or not target_unit.source_paragraph_id
                ) and baseline_unit.section_level != target_unit.section_level:
                    continue
                baseline_segment = sum(
                    anchor < baseline_indexes[0] for anchor in anchored_baseline
                )
                target_segment = sum(
                    anchor < target_indexes[0] for anchor in anchored_target
                )
                if baseline_segment != target_segment:
                    continue
                if len(baseline_indexes) == len(target_indexes) == 1:
                    continue
                if (
                    len(baseline_indexes) == len(target_indexes)
                    and all(
                        baseline_single[baseline_index]
                        == target_single[target_index]
                        for baseline_index, target_index in zip(
                            baseline_indexes,
                            target_indexes,
                        )
                    )
                ):
                    continue
                candidates.append(
                    (
                        baseline_indexes,
                        target_indexes,
                        baseline_unit,
                        target_unit,
                        normalized,
                    )
                )

    signature_counts: dict[tuple[int, int, str], int] = {}
    for baseline_indexes, target_indexes, _, _, normalized in candidates:
        signature = (len(baseline_indexes), len(target_indexes), normalized)
        signature_counts[signature] = signature_counts.get(signature, 0) + 1

    selected = []
    used_baseline: set[int] = set()
    used_target: set[int] = set()
    last_baseline = -1
    last_target = -1
    for baseline_indexes, target_indexes, baseline_unit, target_unit, normalized in sorted(
        candidates,
        key=lambda item: (
            item[0][0],
            item[1][0],
            len(item[0]) + len(item[1]),
        ),
    ):
        signature = (len(baseline_indexes), len(target_indexes), normalized)
        if signature_counts[signature] != 1:
            continue
        if (
            baseline_indexes[0] <= last_baseline
            or target_indexes[0] <= last_target
            or used_baseline.intersection(baseline_indexes)
            or used_target.intersection(target_indexes)
        ):
            continue
        selected.append(
            (
                baseline_indexes,
                target_indexes,
                baseline_unit,
                target_unit,
            )
        )
        used_baseline.update(baseline_indexes)
        used_target.update(target_indexes)
        last_baseline = baseline_indexes[-1]
        last_target = target_indexes[-1]
    return selected


def match_paragraphs(
    pairs: list[SectionPair],
    embedder: BaseProvider,
    similarity_threshold: float = 0.75,
    *,
    rerank_provider: BaseProvider | None = None,
    use_llm_rerank: bool = True,
    baseline_document_title: str = "",
    target_document_title: str = "",
) -> list[ParagraphPair]:
    """
    For each SectionPair, match paragraphs by embedding similarity.
    Returns flat list of ParagraphPairs across all section pairs.
    """
    results: list[ParagraphPair] = []
    baseline_paths = _full_section_paths(pairs, "baseline")
    target_paths = _full_section_paths(pairs, "target")

    for scope in _section_match_scopes(pairs):
        b_units = [
            unit
            for section in scope.baseline_sections
            for unit in _expand_paragraphs(
                section.paragraphs,
                baseline_paths.get(id(section), section.title),
                section.level,
                baseline_document_title,
            )
        ]
        t_units = [
            unit
            for section in scope.target_sections
            for unit in _expand_paragraphs(
                section.paragraphs,
                target_paths.get(id(section), section.title),
                section.level,
                target_document_title,
            )
        ]
        b_ordinary_units = [unit for unit in b_units if _is_ordinary_unit(unit)]
        t_ordinary_units = [unit for unit in t_units if _is_ordinary_unit(unit)]
        b_ordinary_indexes = {
            raw_index: ordinary_index
            for ordinary_index, raw_index in enumerate(
                index
                for index, unit in enumerate(b_units)
                if _is_ordinary_unit(unit)
            )
        }
        t_ordinary_indexes = {
            raw_index: ordinary_index
            for ordinary_index, raw_index in enumerate(
                index
                for index, unit in enumerate(t_units)
                if _is_ordinary_unit(unit)
            )
        }
        sec_path = scope.title or ""

        if not b_units and not t_units:
            continue

        # Sections with no match in other doc → all paragraphs are added/removed
        if not b_units:
            for unit in t_units:
                results.append(ParagraphPair(
                    None,
                    unit.para,
                    0.0,
                    section_path=unit.section_path or sec_path,
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
                    section_path=unit.section_path or sec_path,
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
        window_matches = _exact_adjacent_window_matches(b_units, t_units)
        window_baseline_used = {
            index
            for baseline_indexes, _, _, _ in window_matches
            for index in baseline_indexes
        }
        window_target_used = {
            index
            for _, target_indexes, _, _ in window_matches
            for index in target_indexes
        }
        baseline_window_segments = {
            index: sum(
                baseline_indexes[-1] < index
                for baseline_indexes, _, _, _ in window_matches
            )
            for index in range(len(b_units))
            if index not in window_baseline_used
        }
        target_window_segments = {
            index: sum(
                target_indexes[-1] < index
                for _, target_indexes, _, _ in window_matches
            )
            for index in range(len(t_units))
            if index not in window_target_used
        }

        candidates: list[tuple[float, float, int, int]] = []
        candidates_by_baseline: dict[int, list[tuple[float, int]]] = {}
        rerank_floor = min(similarity_threshold, _LLM_RERANK_MIN_SCORE)
        for i, b_unit in enumerate(b_units):
            if i in window_baseline_used:
                continue
            for j, t_unit in enumerate(t_units):
                if j in window_target_used:
                    continue
                if baseline_window_segments[i] != target_window_segments[j]:
                    continue
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
                        b_ordinary_units,
                        t_ordinary_units,
                        b_ordinary_indexes[i],
                        t_ordinary_indexes[j],
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
        } | window_target_used
        for _, sim, i, j in sorted(candidates, key=lambda item: (-item[0], item[2], item[3])):
            if _is_ordinary_unit(b_units[i]):
                continue
            if i in b_matched or j in t_used:
                continue
            if i in llm_unmatched_baselines:
                continue
            b_matched[i] = (j, sim)
            t_used.add(j)

        for _, _, baseline_unit, target_unit in window_matches:
            baseline_match_text, target_match_text = _classification_texts(
                baseline_unit,
                target_unit,
            )
            results.append(ParagraphPair(
                baseline_unit.para,
                target_unit.para,
                1.0,
                section_path=_specific_section_path(
                    baseline_unit,
                    target_unit,
                    sec_path,
                ),
                split_unit=True,
                baseline_match_text=baseline_match_text,
                target_match_text=target_match_text,
            ))

        for i, b_unit in enumerate(b_units):
            if i in window_baseline_used:
                continue
            if i in b_matched:
                best_j, similarity = b_matched[i]
                target_unit = t_units[best_j]
                baseline_match_text, target_match_text = _classification_texts(b_unit, target_unit)
                results.append(ParagraphPair(
                    b_unit.para,
                    target_unit.para,
                    similarity,
                    section_path=_specific_section_path(
                        b_unit,
                        target_unit,
                        sec_path,
                    ),
                    split_unit=b_unit.split_unit or target_unit.split_unit,
                    baseline_match_text=baseline_match_text,
                    target_match_text=target_match_text,
                ))
            else:
                results.append(ParagraphPair(
                    b_unit.para,
                    None,
                    0.0,
                    section_path=b_unit.section_path or sec_path,
                    split_unit=b_unit.split_unit,
                    baseline_match_text=b_unit.match_text,
                ))

        for j, t_unit in enumerate(t_units):
            if j not in t_used:
                results.append(ParagraphPair(
                    None,
                    t_unit.para,
                    0.0,
                    section_path=t_unit.section_path or sec_path,
                    split_unit=t_unit.split_unit,
                    target_match_text=t_unit.match_text,
                ))

    return _reconcile_unique_exact_unmatched(results)
