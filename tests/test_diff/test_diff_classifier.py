"""Tests for diff_classifier.py"""
from __future__ import annotations
import uuid

import pytest

from app.core.diff.semantic_matcher import ParagraphPair
from app.core.types import ComparePolicy, Paragraph, Sentence


def make_para(text: str) -> Paragraph:
    return Paragraph(
        paragraph_id=str(uuid.uuid4()),
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


def test_addition_with_critical_terms_is_high_risk():
    """Added obligations with numbers should be promoted above the default risk."""
    from app.core.diff.diff_classifier import classify

    t_para = make_para("新增条款：乙方必须在5日内支付违约金100万元。")
    pp = ParagraphPair(baseline_para=None, target_para=t_para, similarity=0.0)

    result = classify(
        para_pairs=[pp],
        policy=_NO_LLM_POLICY,
        provider=None,  # type: ignore[arg-type]
        task_id="t-add-risk",
        baseline_version_id="b1",
        target_version_id="v1",
    )

    assert result.items[0].diff_type == "新增"
    assert result.items[0].risk_level == "high"


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
    """Identical texts with only whitespace difference → diff_type='格式变化', risk='none'."""
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
    assert result.items[0].risk_level == "none"


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


def test_llm_none_risk_is_not_strengthened_by_low_similarity():
    """LLM semantic judgment can mark a low-similarity rewrite as no risk."""
    from app.core.diff.diff_classifier import classify

    provider = type(
        "Provider",
        (),
        {
            "chat": lambda self, messages: (
                '{"diff_type": "格式变化", "risk_level": "none", '
                '"explanation": "语义一致，仅表达顺序调整"}'
            )
        },
    )()
    b_para = make_para("甲方应在收到发票后三十日内完成付款。")
    t_para = make_para("收到发票后，甲方付款期限为三十日。")
    pp = ParagraphPair(baseline_para=b_para, target_para=t_para, similarity=0.12)

    result = classify(
        para_pairs=[pp],
        policy=ComparePolicy(use_llm_classify=True, rule_strengthen=True),
        provider=provider,  # type: ignore[arg-type]
        task_id="t3",
        baseline_version_id="b3",
        target_version_id="v3",
    )

    assert result.items[0].diff_type == "格式变化"
    assert result.items[0].risk_level == "none"


def test_llm_can_suppress_semantically_identical_matched_paragraph():
    """An explicit boolean should_report=false omits a matched pair."""
    from app.core.diff.diff_classifier import classify

    provider = type(
        "Provider",
        (),
        {
            "chat": lambda self, messages: (
                '{"should_report": false, "diff_type": "格式变化", '
                '"risk_level": "none", "explanation": "语义完全一致"}'
            )
        },
    )()
    b_para = make_para("甲方应在收到发票后三十日内完成付款。")
    t_para = make_para("收到发票后，甲方付款期限为三十日。")
    pp = ParagraphPair(baseline_para=b_para, target_para=t_para, similarity=0.12)

    result = classify(
        para_pairs=[pp],
        policy=ComparePolicy(use_llm_classify=True, rule_strengthen=True),
        provider=provider,  # type: ignore[arg-type]
        task_id="t-identical",
        baseline_version_id="b-identical",
        target_version_id="v-identical",
    )

    assert result.items == []


def test_llm_string_should_report_false_does_not_suppress_matched_paragraph():
    """Only the JSON boolean false may suppress a matched pair."""
    from app.core.diff.diff_classifier import classify

    provider = type(
        "Provider",
        (),
        {
            "chat": lambda self, messages: (
                '{"should_report": "false", "diff_type": "格式变化", '
                '"risk_level": "none", "explanation": "语义完全一致"}'
            )
        },
    )()
    pp = ParagraphPair(
        baseline_para=make_para("甲方负责付款。"),
        target_para=make_para("付款由甲方负责。"),
        similarity=0.5,
    )

    result = classify(
        para_pairs=[pp],
        policy=ComparePolicy(use_llm_classify=True, rule_strengthen=True),
        provider=provider,  # type: ignore[arg-type]
        task_id="t-malformed-report",
        baseline_version_id="b-malformed-report",
        target_version_id="v-malformed-report",
    )

    assert len(result.items) == 1


def test_llm_can_suppress_single_sided_table_header():
    """Unmatched table headers can be delegated to LLM before reporting."""
    from app.core.diff.diff_classifier import classify

    class Provider:
        def __init__(self):
            self.prompts: list[str] = []

        def chat(self, messages):
            self.prompts.append(messages[-1]["content"])
            return (
                '{"should_report": "false", "diff_type": "格式变化", '
                '"risk_level": "none", "explanation": "仅表格列名"}'
            )

    provider = Provider()
    t_para = make_para("| 序号 | 姓名 | 部门 |")
    pp = ParagraphPair(
        baseline_para=None,
        target_para=t_para,
        similarity=0.0,
        target_table_header=True,
    )

    result = classify(
        para_pairs=[pp],
        policy=ComparePolicy(use_llm_classify=True, rule_strengthen=True),
        provider=provider,  # type: ignore[arg-type]
        task_id="t-header",
        baseline_version_id="b",
        target_version_id="t",
    )

    assert result.items == []
    assert len(provider.prompts) == 1
    assert "表头" in provider.prompts[0]


def test_llm_none_risk_is_not_strengthened_by_numeric_metadata_changes():
    """LLM no-risk judgment should not be overridden by metadata-like numbers."""
    from app.core.diff.diff_classifier import classify

    provider = type(
        "Provider",
        (),
        {
            "chat": lambda self, messages: (
                '{"diff_type": "格式变化", "risk_level": "none", '
                '"explanation": "仅数字序号变化，内容实质未变"}'
            )
        },
    )()
    b_para = make_para("56 55 漫画技法终极指导1-6全六集PDF原书教程")
    t_para = make_para("57 56 漫画技法终极指导1-6全六集PDF原书教程")
    pp = ParagraphPair(baseline_para=b_para, target_para=t_para, similarity=0.998)

    result = classify(
        para_pairs=[pp],
        policy=ComparePolicy(use_llm_classify=True, rule_strengthen=True),
        provider=provider,  # type: ignore[arg-type]
        task_id="t3-numeric-metadata",
        baseline_version_id="b3",
        target_version_id="v3",
    )

    assert result.items[0].diff_type == "格式变化"
    assert result.items[0].risk_level == "none"


def test_llm_low_risk_is_not_strengthened_to_high_by_low_similarity():
    """LLM low-risk judgment should survive the similarity fallback rule."""
    from app.core.diff.diff_classifier import classify

    provider = type(
        "Provider",
        (),
        {
            "chat": lambda self, messages: (
                '{"diff_type": "微调", "risk_level": "low", '
                '"explanation": "核心义务未变化"}'
            )
        },
    )()
    b_para = make_para("乙方负责交付前的质量检查并提交证明。")
    t_para = make_para("交付前质量检查及证明文件由乙方负责。")
    pp = ParagraphPair(baseline_para=b_para, target_para=t_para, similarity=0.18)

    result = classify(
        para_pairs=[pp],
        policy=ComparePolicy(use_llm_classify=True, rule_strengthen=True),
        provider=provider,  # type: ignore[arg-type]
        task_id="t4",
        baseline_version_id="b4",
        target_version_id="v4",
    )

    assert result.items[0].risk_level == "low"


def test_llm_low_risk_is_strengthened_when_rules_detect_critical_change():
    """Rules still catch concrete numeric or obligation changes after LLM classification."""
    from app.core.diff.diff_classifier import classify

    provider = type(
        "Provider",
        (),
        {
            "chat": lambda self, messages: (
                '{"diff_type": "微调", "risk_level": "low", '
                '"explanation": "措辞轻微调整"}'
            )
        },
    )()
    b_para = make_para("甲方应在30日内完成付款。")
    t_para = make_para("甲方应在60日内完成付款。")
    pp = ParagraphPair(baseline_para=b_para, target_para=t_para, similarity=0.92)

    result = classify(
        para_pairs=[pp],
        policy=ComparePolicy(use_llm_classify=True, rule_strengthen=True),
        provider=provider,  # type: ignore[arg-type]
        task_id="t5",
        baseline_version_id="b5",
        target_version_id="v5",
    )

    assert result.items[0].risk_level == "high"


def test_llm_chinese_risk_label_is_normalized_to_none():
    """LLM may answer with Chinese risk labels; normalize them before persistence."""
    from app.core.diff.diff_classifier import _llm_classify

    provider = type(
        "Provider",
        (),
        {
            "chat": lambda self, messages: (
                '{"diff_type": "格式变化", "risk_level": "无风险", '
                '"explanation": "语义完全一致"}'
            )
        },
    )()

    _, _, risk, _ = _llm_classify("甲方付款。", "甲方应付款。", provider, similarity=0.4)  # type: ignore[arg-type]

    assert risk == "none"
