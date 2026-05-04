"""Classify paragraph pairs into structured DiffItems."""
from __future__ import annotations
import json
import logging
import re
import uuid

from app.core.diff.semantic_matcher import ParagraphPair
from app.core.model.base_provider import BaseProvider
from app.core.types import ComparePolicy, DiffItem, DiffResult

logger = logging.getLogger(__name__)

_CLASSIFY_PROMPT = """你是一个专业的文档差异分析助手。请分析以下两段文本之间的差异，并给出结构化判断。

原文：
{baseline}

修改后：
{target}

向量相似度（0~1，越低越不同）：{similarity:.2f}

请以JSON格式回答，只输出JSON，不要有任何其他内容：
{{
  "diff_type": "微调|实质修改|重写|格式变化",
  "risk_level": "high|medium|low",
  "explanation": "简短的差异说明（30字以内）"
}}

判断规则：
- 格式变化：仅排版、标点、序号变化，语义完全相同
- 微调：措辞调整，核心意思不变（相似度通常 > 0.8）
- 实质修改：金额、日期、责任主体、权利义务等核心内容变化（相似度通常 0.3~0.8）
- 重写：段落大幅改写，原结构基本不保留（相似度通常 < 0.4）
"""


def _rule_classify(baseline: str, target: str, similarity: float = 1.0) -> tuple[str, str, str]:
    """Quick rule-based classification as fallback or supplement."""
    if re.sub(r'\s+', '', baseline) == re.sub(r'\s+', '', target):
        return "格式变化", "low", "仅格式变化"

    if similarity < 0.3:
        return "重写", "high", "文本结构大幅调整"

    numbers_b = set(re.findall(r'\d+[\.,]?\d*', baseline))
    numbers_t = set(re.findall(r'\d+[\.,]?\d*', target))
    neg_b = set(re.findall(r'[不无未没]', baseline))
    neg_t = set(re.findall(r'[不无未没]', target))
    oblig_b = set(re.findall(r'(?:应|须|必须|不得|禁止)', baseline))
    oblig_t = set(re.findall(r'(?:应|须|必须|不得|禁止)', target))

    if numbers_b != numbers_t or neg_b != neg_t or oblig_b != oblig_t:
        return "实质修改", "high", "关键数值或义务条款发生变化"

    return "微调", "medium", "措辞有所调整"


def _llm_classify(
    baseline: str,
    target: str,
    provider: BaseProvider,
    similarity: float = 1.0,
) -> tuple[str, str, str]:
    prompt = _CLASSIFY_PROMPT.format(
        baseline=baseline[:500],
        target=target[:500],
        similarity=similarity,
    )
    try:
        response = provider.chat([{"role": "user", "content": prompt}])
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return (
                data.get("diff_type", "微调"),
                data.get("risk_level", "medium"),
                data.get("explanation", ""),
            )
    except Exception as e:
        logger.warning("LLM classification failed, using rules: %s", e)
    return _rule_classify(baseline, target, similarity)


def classify(
    para_pairs: list[ParagraphPair],
    policy: ComparePolicy,
    provider: BaseProvider | None,
    task_id: str,
    baseline_version_id: str,
    target_version_id: str,
) -> DiffResult:
    items: list[DiffItem] = []
    for pp in para_pairs:
        if pp.baseline_para is None and pp.target_para is not None:
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type="新增",
                risk_level="medium",
                baseline_text="",
                target_text=pp.target_para.text,
                similarity_score=0.0,
                explanation="目标文档新增段落",
                baseline_page=0,
                target_page=pp.target_para.page_no,
            ))
        elif pp.baseline_para is not None and pp.target_para is None:
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type="删减",
                risk_level="medium",
                baseline_text=pp.baseline_para.text,
                target_text="",
                similarity_score=0.0,
                explanation="基准文档段落被删除",
                baseline_page=pp.baseline_para.page_no,
                target_page=0,
            ))
        elif pp.baseline_para is not None and pp.target_para is not None:
            if policy.use_llm_classify and provider is not None:
                diff_type, risk_level, explanation = _llm_classify(
                    pp.baseline_para.text, pp.target_para.text, provider, pp.similarity
                )
            else:
                diff_type, risk_level, explanation = _rule_classify(
                    pp.baseline_para.text, pp.target_para.text, pp.similarity
                )
            if policy.rule_strengthen:
                _, rule_risk, _ = _rule_classify(
                    pp.baseline_para.text, pp.target_para.text, pp.similarity
                )
                if rule_risk == "high" and risk_level != "high":
                    risk_level = "high"
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type=diff_type,
                risk_level=risk_level,
                baseline_text=pp.baseline_para.text,
                target_text=pp.target_para.text,
                similarity_score=pp.similarity,
                explanation=explanation,
                baseline_page=pp.baseline_para.page_no,
                target_page=pp.target_para.page_no,
            ))
    return DiffResult(
        task_id=task_id,
        baseline_version_id=baseline_version_id,
        target_version_id=target_version_id,
        items=items,
    )
