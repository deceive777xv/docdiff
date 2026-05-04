"""Tests for semantic_matcher.py"""
from __future__ import annotations
import uuid
from typing import List

import pytest

from app.core.diff.structure_aligner import SectionPair
from app.core.model.base_provider import BaseProvider
from app.core.types import DocumentIR, Section, Paragraph, Sentence


# ---------------------------------------------------------------------------
# Mock embedder
# ---------------------------------------------------------------------------

class MockEmbedder(BaseProvider):
    """Returns consistent vectors: identical texts get identical vectors,
    different texts get orthogonal vectors."""

    def __init__(self):
        self._registry: dict[str, list[float]] = {}
        self._dim = 4
        self._counter = 0

    def _get_or_create(self, text: str) -> list[float]:
        if text not in self._registry:
            vec = [0.0] * self._dim
            vec[self._counter % self._dim] = 1.0
            self._counter += 1
            self._registry[text] = vec
        return self._registry[text]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._get_or_create(t) for t in texts]

    def chat(self, messages: list[dict], **kwargs) -> str:
        return ""

    def health_check(self) -> bool:
        return True


def make_para(text: str, page_no: int = 1) -> Paragraph:
    return Paragraph(
        paragraph_id=str(uuid.uuid4()),
        page_no=page_no,
        text=text,
        sentences=[Sentence(text=text)],
    )


def make_section(title: str, paras: list[Paragraph]) -> Section:
    sec = Section(section_id=str(uuid.uuid4()), title=title, level=1)
    sec.paragraphs = paras
    return sec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_matched_pair_above_threshold():
    """Baseline and target with same text → embedder returns identical vectors → matched."""
    from app.core.diff.semantic_matcher import match_paragraphs

    same_text = "本合同自签署之日起生效。"
    b_para = make_para(same_text)
    t_para = make_para(same_text)

    b_sec = make_section("第一章", [b_para])
    t_sec = make_section("第一章", [t_para])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    embedder = MockEmbedder()
    pairs = match_paragraphs([sp], embedder, similarity_threshold=0.75)

    matched = [p for p in pairs if p.baseline_para is not None and p.target_para is not None]
    assert len(matched) == 1
    assert matched[0].similarity > 0.75


def test_unmatched_goes_to_deleted():
    """Baseline para with no similar target → target_para=None (deleted)."""
    from app.core.diff.semantic_matcher import match_paragraphs

    b_para = make_para("甲方应在30日内完成交付。")
    t_para = make_para("完全不同的内容ZZZZZZ")

    b_sec = make_section("第二章", [b_para])
    t_sec = make_section("第二章", [t_para])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    embedder = MockEmbedder()
    # Use high threshold so orthogonal vectors don't match
    pairs = match_paragraphs([sp], embedder, similarity_threshold=0.75)

    # b_para should be unmatched (target_para=None) since orthogonal vectors give 0.0 similarity
    deleted = [p for p in pairs if p.baseline_para is not None and p.target_para is None]
    added = [p for p in pairs if p.target_para is not None and p.baseline_para is None]
    assert len(deleted) >= 1 or len(added) >= 1


def test_section_with_only_target_paras():
    """SectionPair with baseline_section=None → all target paras become ParagraphPair(None, para, 0.0)."""
    from app.core.diff.semantic_matcher import match_paragraphs

    t_para1 = make_para("新增段落一")
    t_para2 = make_para("新增段落二")
    t_sec = make_section("附则", [t_para1, t_para2])

    sp = SectionPair(baseline_section=None, target_section=t_sec, title_similarity=0.0)

    embedder = MockEmbedder()
    pairs = match_paragraphs([sp], embedder)

    assert len(pairs) == 2
    for p in pairs:
        assert p.baseline_para is None
        assert p.target_para is not None
        assert p.similarity == 0.0
