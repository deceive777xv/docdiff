"""Tests for semantic_matcher.py"""
from __future__ import annotations
import re
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


class TokenOverlapEmbedder(BaseProvider):
    """Small deterministic vectorizer for tests that need similarity, not identity."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        token_sets = [
            set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower()))
            for text in texts
        ]
        vocab = sorted(set().union(*token_sets))
        if not vocab:
            return [[0.0] for _ in texts]
        return [[1.0 if token in tokens else 0.0 for token in vocab] for tokens in token_sets]

    def chat(self, messages: list[dict], **kwargs) -> str:
        return ""

    def health_check(self) -> bool:
        return True


def make_para(text: str) -> Paragraph:
    return Paragraph(
        paragraph_id=str(uuid.uuid4()),
        text=text,
        sentences=[Sentence(text=text)],
    )


def make_para_with_sentences(sentences: list[str]) -> Paragraph:
    return Paragraph(
        paragraph_id=str(uuid.uuid4()),
        text="\n".join(sentences),
        sentences=[Sentence(text=text) for text in sentences],
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


def test_large_table_paragraph_is_matched_by_table_rows():
    """Large table paragraphs should be compared row-by-row, not as one blob."""
    from app.core.diff.semantic_matcher import match_paragraphs

    unchanged_rows = [f"| 项目{i} | 内容{i} |" for i in range(40)]
    b_rows = [
        "| 项目 | 取值 |",
        "| --- | --- |",
        "| 付款周期 | 30天 |",
        *unchanged_rows,
    ]
    t_rows = [
        "| 项目 | 取值 |",
        "| --- | --- |",
        "| 付款周期 | 60天 |",
        *unchanged_rows,
    ]
    b_sec = make_section("付款表", [make_para_with_sentences(b_rows)])
    t_sec = make_section("付款表", [make_para_with_sentences(t_rows)])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    pairs = match_paragraphs([sp], TokenOverlapEmbedder(), similarity_threshold=0.5)

    changed = [
        pair for pair in pairs
        if pair.baseline_para is not None
        and pair.target_para is not None
        and "付款周期" in pair.baseline_para.text
    ]
    assert len(changed) == 1
    assert changed[0].baseline_para.text == "| 付款周期 | 30天 |"
    assert changed[0].target_para.text == "| 付款周期 | 60天 |"
    assert "\n" not in changed[0].baseline_para.text


def test_table_rows_with_ordinal_first_column_match_by_content_not_sequence():
    """Ordinal table columns should not force mismatched rows after reordering."""
    from app.core.diff.semantic_matcher import match_paragraphs

    b_rows = [
        "| 序号 | 区域 | 销售代表 | 一月(万元) |",
        "| --- | --- | --- | --- |",
        "| 4 | 华南 | 赵六 | 87 |",
        "| 5 | 华东 | 周明 | 134 |",
    ]
    t_rows = [
        "| 序号 | 区域 | 销售代表 | 一月(万元) |",
        "| --- | --- | --- | --- |",
        "| 4 | 华东 | 周明 | 134 |",
        "| 5 | 华南 | 赵六 | 87 |",
    ]
    b_sec = make_section("销售业绩一览", [make_para_with_sentences(b_rows)])
    t_sec = make_section("销售业绩一览", [make_para_with_sentences(t_rows)])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    pairs = match_paragraphs([sp], TokenOverlapEmbedder(), similarity_threshold=0.5)

    zhaoliu = [
        pair for pair in pairs
        if pair.baseline_para is not None
        and pair.target_para is not None
        and "赵六" in pair.baseline_para.text
    ]
    assert len(zhaoliu) == 1
    assert "赵六" in zhaoliu[0].target_para.text


def test_table_row_matching_prefers_vector_similarity_over_first_column_key():
    """Generic tables should not hard-match rows by a positional/index column."""
    from app.core.diff.semantic_matcher import match_paragraphs

    b_rows = [
        "| Index | Label | Status | Amount |",
        "| --- | --- | --- | --- |",
        "| 4 | Atlas | Ready | 200 |",
        "| 5 | Boreal | Hold | 900 |",
    ]
    t_rows = [
        "| Index | Label | Status | Amount |",
        "| --- | --- | --- | --- |",
        "| 4 | Boreal | Hold | 900 |",
        "| 5 | Atlas | Ready | 200 |",
    ]
    b_sec = make_section("inventory", [make_para_with_sentences(b_rows)])
    t_sec = make_section("inventory", [make_para_with_sentences(t_rows)])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    pairs = match_paragraphs([sp], TokenOverlapEmbedder(), similarity_threshold=0.5)

    matched = [
        (pair.baseline_para.text, pair.target_para.text)
        for pair in pairs
        if pair.baseline_para is not None and pair.target_para is not None
    ]
    assert ("| 4 | Atlas | Ready | 200 |", "| 5 | Atlas | Ready | 200 |") in matched
    assert ("| 5 | Boreal | Hold | 900 |", "| 4 | Boreal | Hold | 900 |") in matched


def test_pdf_header_value_cells_do_not_require_known_header_names():
    """Header/value cells should be normalized generically, without business labels."""
    from app.core.diff.semantic_matcher import match_paragraphs

    b_rows = [
        (
            "|**Code**<br>`A7`|**Subject**<br>Northwind Premium|"
            "**Region**<br>East|**State**<br>Ready|**Score**<br>`23`|"
        ),
        "|---|---|---|---|---|",
    ]
    t_rows = [
        "|**Code**|**Subject**|**Region**|**State**|**Score**|",
        "|---|---|---|---|---|",
        "|`B2`|Northwind Premium|East|Ready|`20`|",
    ]
    b_sec = make_section("generic table", [make_para_with_sentences(b_rows)])
    t_sec = make_section("generic table", [make_para_with_sentences(t_rows)])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    pairs = match_paragraphs([sp], TokenOverlapEmbedder(), similarity_threshold=0.5)

    matched = [
        pair for pair in pairs
        if pair.baseline_para is not None and pair.target_para is not None
    ]
    assert len(matched) == 1
    assert "Northwind" in matched[0].baseline_para.text
    assert "Northwind" in matched[0].target_para.text


def test_pdf_styled_ordinal_header_matches_rows_by_business_content():
    """PDF extraction may split and style the ordinal header as Markdown/HTML."""
    from app.core.diff.semantic_matcher import match_paragraphs

    b_rows = [
        "|**序**<br>**号**|**区**<br>**域**|**销售**<br>**代表**|**一月**|",
        "|---|---|---|---|",
        "|`4`|华<br>东|周明|`134`|",
        "|`5`|华<br>南|赵六|`87`|",
    ]
    t_rows = [
        "|**序**<br>**号**|**区**<br>**域**|**销售**<br>**代表**|**一月**|",
        "|---|---|---|---|",
        "|`4`|华<br>南|赵六|`87`|",
        "|`5`|华<br>东|周明|`134`|",
    ]
    b_sec = make_section("销售业绩一览", [make_para_with_sentences(b_rows)])
    t_sec = make_section("销售业绩一览", [make_para_with_sentences(t_rows)])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    pairs = match_paragraphs([sp], TokenOverlapEmbedder(), similarity_threshold=0.5)

    zhaoliu = [
        pair for pair in pairs
        if pair.baseline_para is not None
        and pair.target_para is not None
        and "赵六" in pair.baseline_para.text
    ]
    assert len(zhaoliu) == 1
    assert "赵六" in zhaoliu[0].target_para.text


def test_pdf_key_value_table_row_matches_plain_row_by_values():
    """Some PDF rows are extracted as cells containing header and value."""
    from app.core.diff.semantic_matcher import match_paragraphs

    b_rows = [
        "|**序号**<br>`3`|**姓名**<br>赵六|**部门**<br>销售|**出勤天数**<br>`23`|",
        "|---|---|---|---|",
    ]
    t_rows = [
        "|**序号**|**姓名**|**部门**|**出勤天数**|",
        "|---|---|---|---|",
        "|`4`|赵六|销售|`20`|",
    ]
    b_sec = make_section("员工考勤摘要", [make_para_with_sentences(b_rows)])
    t_sec = make_section("员工考勤摘要", [make_para_with_sentences(t_rows)])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    pairs = match_paragraphs([sp], TokenOverlapEmbedder(), similarity_threshold=0.5)

    zhaoliu = [
        pair for pair in pairs
        if pair.baseline_para is not None
        and pair.target_para is not None
        and "赵六" in pair.baseline_para.text
    ]
    assert len(zhaoliu) == 1
    assert "赵六" in zhaoliu[0].target_para.text


def test_empty_pdf_table_rows_are_ignored():
    """PDF extraction can emit empty pipe-only rows at page splits."""
    from app.core.diff.semantic_matcher import match_paragraphs

    b_rows = [
        "|**序号**|**姓名**|**部门**|",
        "|---|---|---|",
        "|`1`|李四|销售|",
        "||||",
    ]
    t_rows = [
        "|**序号**|**姓名**|**部门**|",
        "|---|---|---|",
        "|`1`|李四|销售|",
    ]
    b_sec = make_section("员工考勤摘要", [make_para_with_sentences(b_rows)])
    t_sec = make_section("员工考勤摘要", [make_para_with_sentences(t_rows)])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    pairs = match_paragraphs([sp], MockEmbedder(), similarity_threshold=0.75)

    assert not any(
        (pair.baseline_para is not None and pair.baseline_para.text == "||||")
        or (pair.target_para is not None and pair.target_para.text == "||||")
        for pair in pairs
    )
