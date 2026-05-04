"""Align sections between two DocumentIRs by title similarity."""
from __future__ import annotations
from dataclasses import dataclass

from app.core.types import DocumentIR, Section


@dataclass
class SectionPair:
    baseline_section: Section | None
    target_section: Section | None
    title_similarity: float   # 0.0–1.0


def _title_similarity(a: str, b: str) -> float:
    """Simple character-level Jaccard similarity between two titles."""
    a, b = a.strip().lower(), b.strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def align_sections(
    baseline: DocumentIR,
    target: DocumentIR,
    min_similarity: float = 0.3,
) -> list[SectionPair]:
    """
    Match sections between baseline and target by title similarity.
    Sections below min_similarity threshold are marked as added/removed.
    Returns list of SectionPairs covering all sections from both docs.
    """
    paired: list[SectionPair] = []
    target_used = set()

    for b_sec in baseline.sections:
        best_sim = 0.0
        best_t = None
        for i, t_sec in enumerate(target.sections):
            if i in target_used:
                continue
            sim = _title_similarity(b_sec.title, t_sec.title)
            if sim > best_sim:
                best_sim = sim
                best_t = (i, t_sec)

        if best_t and best_sim >= min_similarity:
            target_used.add(best_t[0])
            paired.append(SectionPair(
                baseline_section=b_sec,
                target_section=best_t[1],
                title_similarity=best_sim,
            ))
        else:
            # Baseline section not found in target → deleted
            paired.append(SectionPair(
                baseline_section=b_sec,
                target_section=None,
                title_similarity=0.0,
            ))

    # Remaining target sections not matched → added
    for i, t_sec in enumerate(target.sections):
        if i not in target_used:
            paired.append(SectionPair(
                baseline_section=None,
                target_section=t_sec,
                title_similarity=0.0,
            ))

    return paired
