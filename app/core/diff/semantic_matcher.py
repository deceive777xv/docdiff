"""Match paragraphs between aligned section pairs using embedding similarity."""
from __future__ import annotations
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


@dataclass
class _ParagraphUnit:
    para: Paragraph
    split_unit: bool
    match_key: str | None = None


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
_TABLE_SEPARATOR_CELL_RE = re.compile(r":?-{3,}:?")


def _rule_score_delta(text_a: str, text_b: str) -> float:
    """Return a small penalty (0..0.2) if key rule-patterns differ between texts."""
    score = 0.0
    for pat in _RULE_PATTERNS:
        hits_a = set(pat.findall(text_a))
        hits_b = set(pat.findall(text_b))
        if hits_a != hits_b:
            score += 0.067   # ~0.2 / 3 patterns
    return score


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_separator_row(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(_TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells)


def _table_row_key(line: str) -> str | None:
    if "|" not in line or _is_table_separator_row(line):
        return None
    cells = [cell for cell in _split_table_row(line) if cell]
    if not cells:
        return None
    return re.sub(r"\s+", "", cells[0])


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
            units.append(_ParagraphUnit(para=para, split_unit=False))
            continue

        is_table = _looks_like_table(para)
        for index, sent in enumerate(para.sentences):
            text = sent.text.strip()
            if not text:
                continue
            if is_table and _is_table_separator_row(text):
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
                    match_key=_table_row_key(text) if is_table else None,
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
                results.append(ParagraphPair(None, unit.para, 0.0, section_path=sec_path, split_unit=unit.split_unit))
            continue
        if not t_units:
            for unit in b_units:
                results.append(ParagraphPair(unit.para, None, 0.0, section_path=sec_path, split_unit=unit.split_unit))
            continue

        # Embed all paragraphs in both sections in one batch
        all_texts = [unit.para.text for unit in b_units] + [unit.para.text for unit in t_units]
        all_embeds = embedder.embed(all_texts)
        b_embeds = all_embeds[: len(b_units)]
        t_embeds = all_embeds[len(b_units) :]

        t_used: set[int] = set()
        for i, b_unit in enumerate(b_units):
            best_sim = -1.0
            best_j = None
            best_key_matched = False
            for j, t_unit in enumerate(t_units):
                if j in t_used:
                    continue
                sim = _cosine(b_embeds[i], t_embeds[j])
                # Apply rule penalty
                sim -= _rule_score_delta(b_unit.para.text, t_unit.para.text)
                key_matched = (
                    b_unit.match_key is not None
                    and b_unit.match_key == t_unit.match_key
                    and b_unit.split_unit
                    and t_unit.split_unit
                )
                if key_matched:
                    if not best_key_matched or sim > best_sim:
                        best_sim = sim
                        best_j = j
                        best_key_matched = True
                elif not best_key_matched and sim > best_sim:
                    best_sim = sim
                    best_j = j
                    best_key_matched = False

            if best_j is not None and (best_sim >= similarity_threshold or best_key_matched):
                t_used.add(best_j)
                target_unit = t_units[best_j]
                similarity = max(best_sim, 0.5) if best_key_matched else best_sim
                results.append(
                    ParagraphPair(
                        b_unit.para,
                        target_unit.para,
                        similarity,
                        section_path=sec_path,
                        split_unit=b_unit.split_unit or target_unit.split_unit,
                    )
                )
            else:
                results.append(ParagraphPair(b_unit.para, None, 0.0, section_path=sec_path, split_unit=b_unit.split_unit))

        for j, t_unit in enumerate(t_units):
            if j not in t_used:
                results.append(ParagraphPair(None, t_unit.para, 0.0, section_path=sec_path, split_unit=t_unit.split_unit))

    return results
