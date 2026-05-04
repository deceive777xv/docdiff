"""Tests for diff_classifier.py"""
from __future__ import annotations
import uuid

import pytest

from app.core.diff.semantic_matcher import ParagraphPair
from app.core.types import ComparePolicy, Paragraph, Sentence


def make_para(text: str, page_no: int = 1) -> Paragraph:
    return Paragraph(
        paragraph_id=str(uuid.uuid4()),
        page_no=page_no,
        text=text,
        sentences=[Sentence(text=text)],
    )


_NO_LLM_POLICY = ComparePolicy(use_llm_classify=False, rule_strengthen=False)


def test_classifies_addition():
    """ParagraphPair(None, target_para, 0.0) → DiffItem with diff_type='新增'."""
    from app.core.diff.diff_classifier import classify

    t_para = make_para("目标文档新增的段落内容。")
    pp = ParagraphPair(baseline_para=None, target_para=t_para, similarity=0.0)

    result = classify(
        para_pairs=[pp],
        policy=_NO_LLM_POLICY,
        provider=None,  # type: ignore[arg-type]
        task_id="t1",
        baseline_version_id="b1",
        target_version_id="v1",
    )

    assert len(result.items) == 1
    assert result.items[0].diff_type == "新增"
    assert result.items[0].baseline_text == ""
    assert result.items[0].target_text == t_para.text


def test_classifies_deletion():
    """ParagraphPair(baseline_para, None, 0.0) → DiffItem with diff_type='删减'."""
    from app.core.diff.diff_classifier import classify

    b_para = make_para("基准文档中被删除的段落内容。")
    pp = ParagraphPair(baseline_para=b_para, target_para=None, similarity=0.0)

    result = classify(
        para_pairs=[pp],
        policy=_NO_LLM_POLICY,
        provider=None,  # type: ignore[arg-type]
        task_id="t1",
        baseline_version_id="b1",
        target_version_id="v1",
    )

    assert len(result.items) == 1
    assert result.items[0].diff_type == "删减"
    assert result.items[0].target_text == ""
    assert result.items[0].baseline_text == b_para.text


def test_rule_classify_format_change():
    """Identical texts with only whitespace difference → diff_type='格式变化'."""
    from app.core.diff.diff_classifier import classify

    b_para = make_para("本合同自签署之日起生效。")
    t_para = make_para("本合同自签署之日起生效。  ")  # trailing spaces
    pp = ParagraphPair(baseline_para=b_para, target_para=t_para, similarity=0.99)

    result = classify(
        para_pairs=[pp],
        policy=_NO_LLM_POLICY,
        provider=None,  # type: ignore[arg-type]
        task_id="t1",
        baseline_version_id="b1",
        target_version_id="v1",
    )

    assert len(result.items) == 1
    assert result.items[0].diff_type == "格式变化"


def test_rule_classify_substantial():
    """One text has '30日', other has '60日' → '实质修改', risk_level='high'."""
    from app.core.diff.diff_classifier import classify

    b_para = make_para("甲方应在30日内完成交付。")
    t_para = make_para("甲方应在60日内完成交付。")
    pp = ParagraphPair(baseline_para=b_para, target_para=t_para, similarity=0.85)

    result = classify(
        para_pairs=[pp],
        policy=_NO_LLM_POLICY,
        provider=None,  # type: ignore[arg-type]
        task_id="t1",
        baseline_version_id="b1",
        target_version_id="v1",
    )

    assert len(result.items) == 1
    assert result.items[0].diff_type == "实质修改"
    assert result.items[0].risk_level == "high"


def test_rule_classify_rewrite_on_low_similarity():
    """similarity < 0.3 → '重写', risk 'high' regardless of text patterns."""
    from app.core.diff.diff_classifier import _rule_classify

    dtype, risk, _ = _rule_classify(
        "甲方负责提供全部原材料及质检。",
        "乙方承担所有运输费用并负责到货验收。",
        similarity=0.15,
    )
    assert dtype == "重写"
    assert risk == "high"


def test_rule_classify_substantial_above_rewrite_threshold():
    """similarity >= 0.3 still detects 实质修改 when numbers differ."""
    from app.core.diff.diff_classifier import _rule_classify

    dtype, risk, _ = _rule_classify(
        "合同金额为100万元，付款期限30日。",
        "合同金额为200万元，付款期限60日。",
        similarity=0.85,
    )
    assert dtype == "实质修改"
    assert risk == "high"


def test_rule_classify_minor_above_rewrite_threshold():
    """similarity >= 0.3 and no structural triggers → '微调'."""
    from app.core.diff.diff_classifier import _rule_classify

    dtype, _, _ = _rule_classify(
        "甲方应当按时提交材料。",
        "甲方应当尽快提交材料。",
        similarity=0.85,
    )
    assert dtype == "微调"


def test_classify_passes_similarity_to_rule_classifier():
    """classify() with low similarity → DiffItem.diff_type == '重写'."""
    from app.core.diff.diff_classifier import classify

    b_para = make_para("甲方负责原材料供应及质量控制，费用由甲方承担。")
    t_para = make_para("乙方承担所有物流运输及到货验收责任，费用另行结算。")
    pp = ParagraphPair(baseline_para=b_para, target_para=t_para, similarity=0.1)

    result = classify(
        para_pairs=[pp],
        policy=_NO_LLM_POLICY,
        provider=None,  # type: ignore[arg-type]
        task_id="t2",
        baseline_version_id="b2",
        target_version_id="v2",
    )

    assert result.items[0].diff_type == "重写"
    assert result.items[0].risk_level == "high"
