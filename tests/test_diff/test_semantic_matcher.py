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


class LeadingNumberBiasedEmbedder(BaseProvider):
    """Embedder that overweights the first number, like a model biased by row position."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        first_numbers = [
            (re.search(r"\d+", text).group(0) if re.search(r"\d+", text) else "")
            for text in texts
        ]
        number_vocab = sorted(set(first_numbers))
        return [
            [
                1.0 if first_number == number else 0.0
                for number in number_vocab
            ]
            for first_number in first_numbers
        ]

    def chat(self, messages: list[dict], **kwargs) -> str:
        return ""

    def health_check(self) -> bool:
        return True


class PositionNumberBiasedEmbedder(BaseProvider):
    """Embedder that overweights a non-leading numeric position marker."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        markers = [
            (re.search(r"位置\d+", text).group(0) if re.search(r"位置\d+", text) else "")
            for text in texts
        ]
        vocab = sorted(set(markers))
        return [
            [1.0 if marker == value else 0.0 for value in vocab]
            for marker in markers
        ]

    def chat(self, messages: list[dict], **kwargs) -> str:
        return ""

    def health_check(self) -> bool:
        return True


class CandidateRerankProvider(BaseProvider):
    """Fake LLM provider that chooses a configured candidate from the prompt."""

    def __init__(self, candidate_index: int):
        self.candidate_index = candidate_index
        self.prompts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def chat(self, messages: list[dict], **kwargs) -> str:
        self.prompts.append(messages[-1]["content"])
        return (
            '{"matched_candidate": %d, "confidence": 0.96, '
            '"reason": "候选标题和数量完全一致"}'
        ) % self.candidate_index

    def health_check(self) -> bool:
        return True


class ConflictClusterRerankProvider(BaseProvider):
    """Fake LLM provider that returns global pairings for a local conflict cluster."""

    def __init__(self):
        self.prompts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def chat(self, messages: list[dict], **kwargs) -> str:
        self.prompts.append(messages[-1]["content"])
        return (
            '{"matches": ['
            '{"baseline": 1, "target": 2, "confidence": 0.98}, '
            '{"baseline": 2, "target": null, "confidence": 0.96}'
            '], "reason": "按项目内容和方向成组匹配"}'
        )

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


def test_short_paragraph_boundary_changes_match_sentence_units():
    """Same sentences should match even when parser paragraph boundaries differ."""
    from app.core.diff.semantic_matcher import match_paragraphs

    sentences = [
        "Alpha warranty obligation remains unchanged.",
        "Beta delivery schedule remains unchanged.",
        "Gamma invoice approval remains unchanged.",
    ]
    b_para = Paragraph(
        paragraph_id=str(uuid.uuid4()),
        text="\n".join(sentences),
        sentences=[Sentence(text=text) for text in sentences],
    )
    b_sec = make_section("Terms", [b_para])
    t_sec = make_section("Terms", [make_para(text) for text in sentences])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    pairs = match_paragraphs([sp], TokenOverlapEmbedder(), similarity_threshold=0.75)

    matched = [
        (pair.baseline_para.text, pair.target_para.text)
        for pair in pairs
        if pair.baseline_para is not None and pair.target_para is not None
    ]
    assert matched == [(text, text) for text in sentences]
    assert not any(
        pair.baseline_para is None or pair.target_para is None
        for pair in pairs
    )


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


def test_table_row_matching_ignores_leading_number_metadata_for_insertions():
    """Identical content with shifted row numbers should beat same-number near matches."""
    from app.core.diff.semantic_matcher import match_paragraphs

    b_rows = [
        "| A | B |",
        "| --- | --- |",
        "| 46 | 51本日本漫画绘画教程电子书 |",
    ]
    t_rows = [
        "| A | B |",
        "| --- | --- |",
        "| 45 | 51本日本漫画绘画教程电子书 |",
        "| 46 | 52本日本漫画绘画教程电子书 |",
    ]
    b_sec = make_section("正文", [make_para_with_sentences(b_rows)])
    t_sec = make_section("正文", [make_para_with_sentences(t_rows)])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    pairs = match_paragraphs([sp], LeadingNumberBiasedEmbedder(), similarity_threshold=0.75)

    matched = [
        pair for pair in pairs
        if pair.baseline_para is not None
        and "51本日本漫画" in pair.baseline_para.text
    ]
    added = [
        pair for pair in pairs
        if pair.baseline_para is None
        and pair.target_para is not None
    ]
    assert len(matched) == 1
    assert "| 45 | 51本日本漫画绘画教程电子书 |" == matched[0].target_para.text
    assert any("| 46 | 52本日本漫画绘画教程电子书 |" == pair.target_para.text for pair in added)


def test_llm_rerank_selects_better_candidate_when_numbers_are_ambiguous():
    """LLM rerank can choose content-equivalent rows over same-number near matches."""
    from app.core.diff.semantic_matcher import match_paragraphs

    b_rows = [
        "| 序号 | 位置 | 标题 |",
        "| --- | --- | --- |",
        "| 1 | 位置46 | 51本日本漫画绘画教程电子书 |",
    ]
    t_rows = [
        "| 序号 | 位置 | 标题 |",
        "| --- | --- | --- |",
        "| 2 | 位置45 | 51本日本漫画绘画教程电子书 |",
        "| 3 | 位置46 | 52本日本漫画绘画教程电子书 |",
    ]
    b_sec = make_section("正文", [make_para_with_sentences(b_rows)])
    t_sec = make_section("正文", [make_para_with_sentences(t_rows)])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)
    reranker = CandidateRerankProvider(candidate_index=2)

    pairs = match_paragraphs(
        [sp],
        PositionNumberBiasedEmbedder(),
        similarity_threshold=0.75,
        rerank_provider=reranker,
        baseline_document_title="旧版车门需求",
        target_document_title="新版车门需求",
    )

    matched = [
        pair for pair in pairs
        if pair.baseline_para is not None
        and "51本日本漫画" in pair.baseline_para.text
    ]
    added = [
        pair for pair in pairs
        if pair.baseline_para is None and pair.target_para is not None
    ]
    assert len(matched) == 1
    assert matched[0].target_para.text == "| 2 | 位置45 | 51本日本漫画绘画教程电子书 |"
    assert any("| 3 | 位置46 | 52本日本漫画绘画教程电子书 |" == pair.target_para.text for pair in added)
    assert len(reranker.prompts) == 1
    assert "旧版车门需求 / 正文" in reranker.prompts[0]
    assert "新版车门需求 / 正文" in reranker.prompts[0]


def test_llm_conflict_cluster_rerank_resolves_shifted_table_rows():
    """A local LLM rerank can resolve rows that compete for the same target after insertion."""
    from app.core.diff.semantic_matcher import match_paragraphs

    b_rows = [
        "| 序号 | 类型 | 项目 | 要求 |",
        "| --- | --- | --- | --- |",
        "| 2.98 | 动态间隙 | 升降器托架与转轮间隙(Y向) | ≥3mm |",
        "| 2.99 | 动态间隙 | 升降器托架与转轮间隙(Z向) | ≥3mm |",
    ]
    t_rows = [
        "| 序号 | 类型 | 项目 | 要求 |",
        "| --- | --- | --- | --- |",
        "| 2.98 | 动态间隙 | 升降器电机座板与玻璃间隙 | ≥8mm |",
        "| 2.99 | 动态间隙 | 升降器托架与转轮间隙(Y向) | ≥3mm |",
    ]
    b_sec = make_section("设计点检表", [make_para_with_sentences(b_rows)])
    t_sec = make_section("设计点检表", [make_para_with_sentences(t_rows)])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)
    reranker = ConflictClusterRerankProvider()

    pairs = match_paragraphs(
        [sp],
        TokenOverlapEmbedder(),
        similarity_threshold=0.5,
        rerank_provider=reranker,
    )

    matched = [
        (pair.baseline_para.text, pair.target_para.text)
        for pair in pairs
        if pair.baseline_para is not None and pair.target_para is not None
    ]
    deleted = [
        pair.baseline_para.text
        for pair in pairs
        if pair.baseline_para is not None and pair.target_para is None
    ]
    added = [
        pair.target_para.text
        for pair in pairs
        if pair.baseline_para is None and pair.target_para is not None
    ]
    assert (
        "| 2.98 | 动态间隙 | 升降器托架与转轮间隙(Y向) | ≥3mm |",
        "| 2.99 | 动态间隙 | 升降器托架与转轮间隙(Y向) | ≥3mm |",
    ) in matched
    assert "| 2.99 | 动态间隙 | 升降器托架与转轮间隙(Z向) | ≥3mm |" in deleted
    assert "| 2.98 | 动态间隙 | 升降器电机座板与玻璃间隙 | ≥8mm |" in added
    assert len(reranker.prompts) == 1


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


def test_table_content_row_before_separator_is_not_treated_as_header():
    """A page-continuation data row followed by a separator must still be matched."""
    from app.core.diff.semantic_matcher import _expand_paragraphs

    rows = [
        "|3.2||滑轮是否借用件说明|等级1|等级2|等级3|等级4|借用那个项目/新开|",
        "|---|---|---|---|---|---|---|---|",
        "|3.3||电机底座是否借用件说明|等级1|等级2|等级3|等级4|借用那个项目/新开|",
    ]

    units = _expand_paragraphs([make_para_with_sentences(rows)])

    assert any("3.2" in unit.para.text for unit in units)
    assert any("3.3" in unit.para.text for unit in units)
    assert not any("---" in unit.para.text for unit in units)


def test_single_new_header_like_table_row_is_reported_as_added():
    """A one-off new row parsed like a header must not be silently suppressed."""
    from app.core.diff.semantic_matcher import match_paragraphs

    b_rows = [
        "| 编号 | 名称 |",
        "| --- | --- |",
        "| 1 | Alpha |",
    ]
    t_rows = [
        "| 编号 | 名称 |",
        "| --- | --- |",
        "| 1 | Alpha |",
        "| 2 | Beta |",
        "| --- | --- |",
        "| 3 | Gamma |",
    ]
    b_sec = make_section("清单", [make_para_with_sentences(b_rows)])
    t_sec = make_section("清单", [make_para_with_sentences(t_rows)])
    sp = SectionPair(baseline_section=b_sec, target_section=t_sec, title_similarity=1.0)

    pairs = match_paragraphs([sp], TokenOverlapEmbedder(), similarity_threshold=0.75)

    added = [
        pair.target_para.text
        for pair in pairs
        if pair.baseline_para is None and pair.target_para is not None
    ]
    assert "| 2 | Beta |" in added


def test_repeated_short_paragraph_uses_neighbor_context():
    from app.core.diff.semantic_matcher import match_paragraphs

    first_heading = make_para("A）工作条件")
    surviving_heading = make_para("A）工作条件")
    baseline = make_section(
        "要求",
        [
            first_heading,
            make_para("仅旧版本保留的甲项说明"),
            surviving_heading,
            make_para("稳定保留的乙项说明"),
        ],
    )
    target_heading = make_para("A）工作条件")
    target = make_section(
        "要求",
        [target_heading, make_para("稳定保留的乙项说明")],
    )

    pairs = match_paragraphs(
        [SectionPair(baseline, target, 1.0)],
        TokenOverlapEmbedder(),
        similarity_threshold=0.75,
    )

    heading_match = next(
        pair
        for pair in pairs
        if pair.baseline_para is not None
        and pair.target_para is target_heading
    )
    assert heading_match.baseline_para is surviving_heading


def test_repeated_short_paragraph_context_skips_all_table_units():
    from app.core.diff.semantic_matcher import match_paragraphs

    first_heading = make_para("A）工作条件")
    surviving_heading = make_para("A）工作条件")
    baseline = make_section(
        "要求",
        [
            first_heading,
            make_para("仅旧版本保留的甲项说明"),
            surviving_heading,
            make_para("稳定保留的乙项说明"),
        ],
    )
    table_rows = [
        "| 文件名称 | 文件编号 | 页码 |",
        "| --- | --- | --- |",
        "| 示例文档 | DOC-001 | 第11页 |",
    ]
    target_heading = make_para("A）工作条件")
    target = make_section(
        "要求",
        [
            target_heading,
            make_para_with_sentences(table_rows),
            make_para("稳定保留的乙项说明"),
        ],
    )

    pairs = match_paragraphs(
        [SectionPair(baseline, target, 1.0)],
        TokenOverlapEmbedder(),
        similarity_threshold=0.75,
    )

    heading_match = next(
        pair
        for pair in pairs
        if pair.baseline_para is not None
        and pair.target_para is target_heading
    )
    assert heading_match.baseline_para is surviving_heading


def test_ordinary_paragraph_matches_are_monotonic():
    from app.core.diff.semantic_matcher import match_paragraphs

    baseline_paras = [
        make_para("Alpha unique ordinary paragraph"),
        make_para("Beta unique ordinary paragraph"),
    ]
    target_paras = [
        make_para("Beta unique ordinary paragraph"),
        make_para("Alpha unique ordinary paragraph"),
    ]
    pairs = match_paragraphs(
        [
            SectionPair(
                make_section("正文", baseline_paras),
                make_section("正文", target_paras),
                1.0,
            )
        ],
        MockEmbedder(),
        similarity_threshold=0.75,
    )

    baseline_index = {id(para): index for index, para in enumerate(baseline_paras)}
    target_index = {id(para): index for index, para in enumerate(target_paras)}
    matched_indices = [
        (baseline_index[id(pair.baseline_para)], target_index[id(pair.target_para)])
        for pair in pairs
        if pair.baseline_para is not None and pair.target_para is not None
    ]

    assert matched_indices == sorted(matched_indices)
    assert all(
        left[1] < right[1]
        for left, right in zip(matched_indices, matched_indices[1:])
    )


def test_unique_long_text_can_match_across_parent_child_section_split():
    from app.core.diff.semantic_matcher import match_paragraphs
    from app.core.diff.structure_aligner import align_sections

    shared = "车门电动打开过程中遇到障碍物时立即停止，并悬停在当前位置。"
    baseline = DocumentIR(
        "baseline",
        "Baseline",
        "hash-a",
        [
            make_section("1.1.18 中断处理功能", []),
            make_section("附表：悬停方式对应悬停动作", [make_para(shared)]),
        ],
    )
    target = DocumentIR(
        "target",
        "Target",
        "hash-b",
        [make_section("1.1.18 中断处理功能", [make_para(shared)])],
    )

    pairs = match_paragraphs(
        align_sections(baseline, target),
        TokenOverlapEmbedder(),
        similarity_threshold=0.75,
    )

    assert len(pairs) == 1
    assert pairs[0].baseline_para is not None
    assert pairs[0].target_para is not None
    assert pairs[0].similarity == 1.0


def test_unique_same_section_short_text_ignores_markdown_list_marker():
    from app.core.diff.semantic_matcher import match_paragraphs

    baseline = make_section(
        "1.1.4 开启关闭防夹功能",
        [make_para("A）工作条件"), make_para("旧版上下文。")],
    )
    target = make_section(
        "1.1.4 开启关闭防夹功能",
        [make_para("- A）工作条件"), make_para("新版上下文。")],
    )

    pairs = match_paragraphs(
        [SectionPair(baseline, target, 1.0)],
        MockEmbedder(),
        similarity_threshold=0.99,
    )

    headings = [
        pair
        for pair in pairs
        if pair.baseline_para is not None
        and pair.baseline_para.text == "A）工作条件"
    ]
    assert len(headings) == 1
    assert headings[0].target_para is not None
    assert headings[0].target_para.text == "- A）工作条件"


def test_sentence_windows_do_not_cross_paragraphs_or_section_paths():
    from app.core.diff.semantic_matcher import match_paragraphs
    from app.core.diff.structure_aligner import align_sections

    baseline_parent = make_section("1.1.28 附件功能", [])
    baseline_parent.level = 3
    baseline_child = make_section(
        "失效保护",
        [
            make_para(
                "当检测到系统存在故障时，域控记录故障码并通过CAN向诊断仪输出。"
            )
        ],
    )
    baseline_child.level = 4
    target_parent = make_section(
        "1.1.28 附件功能",
        [
            make_para("当检测到系统存在故障时，域控记录故障码"),
            make_para("并通过CAN向诊断仪输出。"),
        ],
    )
    target_parent.level = 3
    baseline = DocumentIR(
        "baseline",
        "Baseline",
        "hash-a",
        [baseline_parent, baseline_child],
    )
    target = DocumentIR(
        "target",
        "Target",
        "hash-b",
        [target_parent],
    )

    pairs = match_paragraphs(
        align_sections(baseline, target),
        TokenOverlapEmbedder(),
        similarity_threshold=0.99,
    )

    matched = [
        pair
        for pair in pairs
        if pair.baseline_para is not None and pair.target_para is not None
    ]
    assert matched == []
    assert any(pair.baseline_para is not None and pair.target_para is None for pair in pairs)
    assert any(pair.baseline_para is None and pair.target_para is not None for pair in pairs)


def test_exact_window_does_not_join_three_distinct_paragraphs():
    from app.core.diff.semantic_matcher import match_paragraphs

    baseline = make_section(
        "正文",
        [
            make_para(
                "系统检测到故障后记录故障码，并通过CAN向诊断仪输出，"
                "以便维修人员查询具体故障信息。"
            )
        ],
    )
    target = make_section(
        "正文",
        [
            make_para("系统检测到故障后记录故障码，"),
            make_para("并通过CAN向诊断仪输出，"),
            make_para("以便维修人员查询具体故障信息。"),
        ],
    )

    pairs = match_paragraphs(
        [SectionPair(baseline, target, 1.0)],
        TokenOverlapEmbedder(),
        similarity_threshold=0.99,
    )

    matched = [
        pair
        for pair in pairs
        if pair.baseline_para is not None and pair.target_para is not None
    ]
    assert matched == []
    assert len([pair for pair in pairs if pair.target_para is not None]) == 3


def test_exact_window_matches_different_sentence_boundaries_inside_one_paragraph():
    from app.core.diff.semantic_matcher import match_paragraphs

    parts = [
        "开启过程中按下外把手电容开关；",
        "域控检测到外把手电容开关信号（什么变化？",
        "电流？",
        "）开启过程中刷下NFC；域控检测到NFC信号。",
    ]
    baseline_text = "".join(parts)
    baseline = make_section("开启控制", [make_para(baseline_text)])
    target_para = Paragraph(
        paragraph_id=str(uuid.uuid4()),
        text=baseline_text,
        sentences=[Sentence(part) for part in parts],
        page_no=9,
    )
    target = make_section("开启控制", [target_para])

    pairs = match_paragraphs(
        [SectionPair(baseline, target, 1.0)],
        MockEmbedder(),
        similarity_threshold=0.99,
    )

    matched = [
        pair
        for pair in pairs
        if pair.baseline_para is not None and pair.target_para is not None
    ]
    assert len(matched) == 1
    assert matched[0].similarity == 1.0
    assert not any(
        pair.baseline_para is None or pair.target_para is None
        for pair in pairs
    )


def test_exact_window_allows_minor_title_changes_in_an_aligned_section():
    from app.core.diff.semantic_matcher import match_paragraphs

    parts = ["系统检测到异常后记录故障码，", "并立即通知诊断模块。"]
    baseline = make_section("3.1 故障处理", [make_para("".join(parts))])
    target_para = Paragraph(
        paragraph_id=str(uuid.uuid4()),
        text="".join(parts),
        sentences=[Sentence(part) for part in parts],
        page_no=3,
    )
    target = make_section("3.1 故障处置", [target_para])

    pairs = match_paragraphs(
        [SectionPair(baseline, target, 0.9)],
        MockEmbedder(),
        similarity_threshold=0.99,
    )

    matched = [
        pair
        for pair in pairs
        if pair.baseline_para is not None and pair.target_para is not None
    ]
    assert len(matched) == 1
    assert matched[0].similarity == 1.0


def test_exact_window_never_synthesizes_a_cross_paragraph_match():
    from app.core.diff.semantic_matcher import match_paragraphs

    ordinary = "稍后出现的唯一普通段落，用于验证匹配顺序不能跨过窗口锚点。"
    combined = "系统检测到故障后记录故障码，并通过CAN向诊断仪输出。"
    baseline = make_section(
        "正文",
        [make_para(ordinary), make_para(combined)],
    )
    target = make_section(
        "正文",
        [
            make_para("系统检测到故障后记录故障码，"),
            make_para("并通过CAN向诊断仪输出。"),
            make_para(ordinary),
        ],
    )

    pairs = match_paragraphs(
        [SectionPair(baseline, target, 1.0)],
        TokenOverlapEmbedder(),
        similarity_threshold=0.99,
    )

    assert not any(
        pair.target_para is not None and "\n" in pair.target_para.text
        for pair in pairs
    )


def test_target_only_child_uses_the_more_specific_section_path():
    from app.core.diff.semantic_matcher import match_paragraphs
    from app.core.diff.structure_aligner import align_sections

    text = "目标侧子章节中的稳定正文内容，应保留最具体的章节路径。"
    baseline_parent = make_section("1.1.28 附件功能", [make_para(text)])
    baseline_parent.level = 3
    target_parent = make_section("1.1.28 附件功能", [])
    target_parent.level = 3
    target_child = make_section("失效保护", [make_para(text)])
    target_child.level = 4
    baseline = DocumentIR("baseline", "Baseline", "hash-a", [baseline_parent])
    target = DocumentIR(
        "target",
        "Target",
        "hash-b",
        [target_parent, target_child],
    )

    pairs = match_paragraphs(
        align_sections(baseline, target),
        TokenOverlapEmbedder(),
        similarity_threshold=0.99,
    )

    matched = [
        pair
        for pair in pairs
        if pair.baseline_para is not None and pair.target_para is not None
    ]
    assert len(matched) == 1
    assert matched[0].section_path == "1.1.28 附件功能 / 失效保护"


def test_inserted_child_path_is_not_attached_to_a_later_parent():
    from app.core.diff.semantic_matcher import match_paragraphs
    from app.core.diff.structure_aligner import align_sections

    text = "插入子章节中的正文应归属于前面的父章节。"
    baseline_parent = make_section("1 功能", [make_para(text)])
    baseline_parent.level = 1
    baseline_later = make_section("2 限制", [])
    baseline_later.level = 1
    target_parent = make_section("1 功能", [])
    target_parent.level = 1
    target_child = make_section("1.1 失效保护", [make_para(text)])
    target_child.level = 2
    target_later = make_section("2 限制", [])
    target_later.level = 1

    pairs = match_paragraphs(
        align_sections(
            DocumentIR(
                "baseline",
                "Baseline",
                "hash-a",
                [baseline_parent, baseline_later],
            ),
            DocumentIR(
                "target",
                "Target",
                "hash-b",
                [target_parent, target_child, target_later],
            ),
        ),
        TokenOverlapEmbedder(),
        similarity_threshold=0.99,
    )

    matched = next(
        pair
        for pair in pairs
        if pair.baseline_para is not None and pair.target_para is not None
    )
    assert matched.section_path == "1 功能 / 1.1 失效保护"


def test_exact_adjacent_paragraphs_match_one_combined_target_paragraph():
    from app.core.diff.semantic_matcher import match_paragraphs
    from app.core.diff.structure_aligner import align_sections

    first = "系统需采集用户操作信息并上传至平台进行后台监控；"
    second = "主要用于方便功能问题排查和后台监控车辆问题。"
    baseline = DocumentIR(
        "baseline-window",
        "Baseline",
        "hash-a",
        [make_section("1 功能", [make_para(first), make_para(second)])],
    )
    target = DocumentIR(
        "target-window",
        "Target",
        "hash-b",
        [make_section("1 功能", [make_para(first + second)])],
    )

    pairs = match_paragraphs(
        align_sections(baseline, target),
        TokenOverlapEmbedder(),
        similarity_threshold=0.75,
    )

    assert len(pairs) == 1
    assert pairs[0].baseline_para is not None
    assert pairs[0].target_para is not None
    assert pairs[0].baseline_para.text == first + "\n" + second
    assert pairs[0].target_para.text == first + second
    assert pairs[0].split_unit is True


def test_one_combined_baseline_paragraph_matches_exact_adjacent_targets():
    from app.core.diff.semantic_matcher import match_paragraphs
    from app.core.diff.structure_aligner import align_sections

    first = "系统需采集用户操作信息并上传至平台进行后台监控；"
    second = "主要用于方便功能问题排查和后台监控车辆问题。"
    baseline = DocumentIR(
        "baseline-window",
        "Baseline",
        "hash-a",
        [make_section("1 功能", [make_para(first + second)])],
    )
    target = DocumentIR(
        "target-window",
        "Target",
        "hash-b",
        [make_section("1 功能", [make_para(first), make_para(second)])],
    )

    pairs = match_paragraphs(
        align_sections(baseline, target),
        TokenOverlapEmbedder(),
        similarity_threshold=0.75,
    )

    assert len(pairs) == 1
    assert pairs[0].baseline_para is not None
    assert pairs[0].target_para is not None
    assert pairs[0].baseline_para.text == first + second
    assert pairs[0].target_para.text == first + "\n" + second
    assert pairs[0].split_unit is True


def test_short_adjacent_headings_match_one_combined_heading():
    from app.core.diff.semantic_matcher import match_paragraphs

    baseline = make_section("目录", [make_para("功能"), make_para("说明")])
    target = make_section("目录", [make_para("功能说明")])

    pairs = match_paragraphs(
        [SectionPair(baseline, target, 1.0)],
        TokenOverlapEmbedder(),
        similarity_threshold=0.99,
    )

    assert len(pairs) == 1
    assert pairs[0].baseline_para is not None
    assert pairs[0].target_para is not None
    assert pairs[0].baseline_para.text == "功能\n说明"
    assert pairs[0].target_para.text == "功能说明"


def test_exact_window_does_not_take_partial_units_from_adjacent_paragraphs():
    from app.core.diff.semantic_matcher import match_paragraphs

    baseline = make_section(
        "正文",
        [
            make_para_with_sentences(["前置说明。", "第一段末尾内容很长并且需要保持完整。"]),
            make_para_with_sentences(["第二段开头内容同样很长。", "后置说明。"]),
        ],
    )
    target = make_section(
        "正文",
        [make_para("第一段末尾内容很长并且需要保持完整。第二段开头内容同样很长。")],
    )

    pairs = match_paragraphs(
        [SectionPair(baseline, target, 1.0)],
        TokenOverlapEmbedder(),
        similarity_threshold=0.99,
    )

    assert not any(
        pair.baseline_para is not None
        and pair.target_para is not None
        and pair.split_unit
        for pair in pairs
    )
