"""Match paragraphs between aligned section pairs using embedding similarity."""
from __future__ import annotations
import re
from dataclasses import dataclass

import numpy as np

from app.core.diff.structure_aligner import SectionPair
from app.core.model.base_provider import BaseProvider
from app.core.types import Paragraph


@dataclass
class ParagraphPair:
    baseline_para: Paragraph | None
    target_para: Paragraph | None
    similarity: float   # cosine similarity, -1..1 (1 = identical)
    section_path: str = ""


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


_RULE_PATTERNS = [
    re.compile(r'\d+[\.,]\d*'),          # numbers
    re.compile(r'[不无未没]'),            # negations
    re.compile(r'(?:应|须|必须|不得|禁止)'),  # obligation words
]


def _rule_score_delta(text_a: str, text_b: str) -> float:
    """Return a small penalty (0..0.2) if key rule-patterns differ between texts."""
    score = 0.0
    for pat in _RULE_PATTERNS:
        hits_a = set(pat.findall(text_a))
        hits_b = set(pat.findall(text_b))
        if hits_a != hits_b:
            score += 0.067   # ~0.2 / 3 patterns
    return score


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
        sec_path = (
            sp.baseline_section.title if sp.baseline_section else
            sp.target_section.title if sp.target_section else ""
        ) or ""

        if not b_paras and not t_paras:
            continue

        # Sections with no match in other doc → all paragraphs are added/removed
        if not b_paras:
            for p in t_paras:
                results.append(ParagraphPair(None, p, 0.0, section_path=sec_path))
            continue
        if not t_paras:
            for p in b_paras:
                results.append(ParagraphPair(p, None, 0.0, section_path=sec_path))
            continue

        # Embed all paragraphs in both sections in one batch
        all_texts = [p.text for p in b_paras] + [p.text for p in t_paras]
        all_embeds = embedder.embed(all_texts)
        b_embeds = all_embeds[: len(b_paras)]
        t_embeds = all_embeds[len(b_paras) :]

        t_used: set[int] = set()
        for i, b_para in enumerate(b_paras):
            best_sim = -1.0
            best_j = None
            for j, t_para in enumerate(t_paras):
                if j in t_used:
                    continue
                sim = _cosine(b_embeds[i], t_embeds[j])
                # Apply rule penalty
                sim -= _rule_score_delta(b_para.text, t_para.text)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j

            if best_j is not None and best_sim >= similarity_threshold:
                t_used.add(best_j)
                results.append(ParagraphPair(b_para, t_paras[best_j], best_sim, section_path=sec_path))
            else:
                results.append(ParagraphPair(b_para, None, 0.0, section_path=sec_path))

        for j, t_para in enumerate(t_paras):
            if j not in t_used:
                results.append(ParagraphPair(None, t_para, 0.0, section_path=sec_path))

    return results
