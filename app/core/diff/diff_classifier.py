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
  "should_report": true,
  "diff_type": "微调|实质修改|重写|格式变化",
  "risk_level": "high|medium|low|none",
  "explanation": "简短的差异说明（30字以内）"
}}

判断规则：
- should_report=false：两段语义、业务事实和权利义务完全一致，仅有格式、顺序或等价表达变化
- should_report=true：存在任何需要用户关注的内容变化，或无法确定两段完全等价
- 无风险（none）：仅格式、顺序、表达方式变化，语义和义务完全一致
- 低风险（low）：措辞调整，核心意思不变，业务影响很小
- 中风险（medium）：表达或范围有变化，但未触及关键金额、日期、责任主体、权利义务
- 高风险（high）：金额、日期、责任主体、权利义务、否定词、禁止/必须等关键内容变化
- 相似度只能作为参考；如果你判断语义一致，应给 low 或 none，不要仅因相似度低给 high
- 无风险（none）对应 should_report=false
"""

_VALID_RISK_LEVELS = {"high", "medium", "low", "none"}
_RISK_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
_RISK_ALIASES = {
    "高风险": "high",
    "high": "high",
    "中风险": "medium",
    "medium": "medium",
    "低风险": "low",
    "low": "low",
    "无风险": "none",
    "none": "none",
    "no": "none",
    "no_risk": "none",
    "norisk": "none",
}

_NUMBER_RE = re.compile(r'\d+[\.,]?\d*')
_NEGATION_RE = re.compile(r'[不无未没]')
_OBLIGATION_RE = re.compile(r'(?:必须|不得|禁止|应|须)')
_CRITICAL_OBLIGATION_RE = re.compile(r'(?:必须|不得|禁止)')


def _normalize_risk_level(value: str | None) -> str:
    if value is None:
        return "medium"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _VALID_RISK_LEVELS:
        return normalized
    return _RISK_ALIASES.get(normalized, "medium")


def _normalize_bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1", "是", "需要", "报告"}:
        return True
    if normalized in {"false", "no", "0", "否", "不需要", "不报告"}:
        return False
    return default


def _rule_classify(baseline: str, target: str, similarity: float = 1.0) -> tuple[str, str, str]:
    """Quick rule-based classification as fallback or supplement."""
    if re.sub(r'\s+', '', baseline) == re.sub(r'\s+', '', target):
        return "格式变化", "none", "仅格式变化"

    if similarity < 0.3:
        return "重写", "high", "文本结构大幅调整"

    numbers_b = set(_NUMBER_RE.findall(baseline))
    numbers_t = set(_NUMBER_RE.findall(target))
    neg_b = set(_NEGATION_RE.findall(baseline))
    neg_t = set(_NEGATION_RE.findall(target))
    oblig_b = set(_OBLIGATION_RE.findall(baseline))
    oblig_t = set(_OBLIGATION_RE.findall(target))

    if numbers_b != numbers_t or neg_b != neg_t or oblig_b != oblig_t:
        return "实质修改", "high", "关键数值或义务条款发生变化"

    if similarity < 0.75:
        return "微调", "medium", "措辞有所调整"
    return "微调", "low", "措辞有所调整"


def _critical_rule_risk(baseline: str, target: str) -> str | None:
    """Return high only for concrete legal/business triggers, not low similarity."""
    numbers_b = set(_NUMBER_RE.findall(baseline))
    numbers_t = set(_NUMBER_RE.findall(target))
    neg_b = set(_NEGATION_RE.findall(baseline))
    neg_t = set(_NEGATION_RE.findall(target))
    oblig_b = set(_CRITICAL_OBLIGATION_RE.findall(baseline))
    oblig_t = set(_CRITICAL_OBLIGATION_RE.findall(target))

    if numbers_b != numbers_t or neg_b != neg_t or oblig_b != oblig_t:
        return "high"
    return None


def _single_sided_risk(text: str) -> str:
    """Risk for added/deleted text where only one side exists."""
    return "high" if _critical_rule_risk("", text) == "high" else "medium"


def _max_risk(current: str, candidate: str | None) -> str:
    if candidate is None:
        return current
    return candidate if _RISK_RANK[candidate] > _RISK_RANK.get(current, 2) else current


def _should_strengthen_risk(diff_type: str, risk_level: str) -> bool:
    """Only strengthen when the primary classifier already sees content impact."""
    return diff_type != "格式变化" and risk_level != "none"


def _llm_classify(
    baseline: str,
    target: str,
    provider: BaseProvider,
    similarity: float = 1.0,
) -> tuple[bool, str, str, str]:
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
                data.get("should_report", True) is not False,
                data.get("diff_type", "微调"),
                _normalize_risk_level(data.get("risk_level")),
                data.get("explanation", ""),
            )
    except Exception as e:
        logger.warning("LLM classification failed, using rules: %s", e)
    return True, *_rule_classify(baseline, target, similarity)


def _same_text_ignoring_whitespace(baseline: str, target: str) -> bool:
    return re.sub(r"\s+", "", baseline) == re.sub(r"\s+", "", target)


_SEMANTIC_CORE_RE = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]+", re.IGNORECASE)


def _semantic_core(text: str) -> str:
  return "".join(_SEMANTIC_CORE_RE.findall(text or ""))


def _semantically_equivalent(baseline: str, target: str, similarity: float) -> bool:
  if similarity < 0.95:
    return False
  return _semantic_core(baseline) == _semantic_core(target)


def _pair_compare_texts(pp: ParagraphPair) -> tuple[str, str]:
    baseline_text = pp.baseline_para.text if pp.baseline_para is not None else ""
    target_text = pp.target_para.text if pp.target_para is not None else ""
    baseline_compare = pp.baseline_match_text if pp.baseline_match_text is not None else baseline_text
    target_compare = pp.target_match_text if pp.target_match_text is not None else target_text
    return baseline_compare, target_compare


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
                risk_level=_single_sided_risk(pp.target_para.text),
                baseline_text="",
                target_text=pp.target_para.text,
                similarity_score=0.0,
                explanation="目标文档新增段落",
            ))
        elif pp.baseline_para is not None and pp.target_para is None:
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type="删减",
                risk_level=_single_sided_risk(pp.baseline_para.text),
                baseline_text=pp.baseline_para.text,
                target_text="",
                similarity_score=0.0,
                explanation="基准文档段落被删除",
            ))
        elif pp.baseline_para is not None and pp.target_para is not None:
            if pp.split_unit and _same_text_ignoring_whitespace(pp.baseline_para.text, pp.target_para.text):
                continue
            baseline_compare, target_compare = _pair_compare_texts(pp)
            if pp.split_unit and _same_text_ignoring_whitespace(baseline_compare, target_compare):
                items.append(DiffItem(
                    diff_id=str(uuid.uuid4()),
                    section_path=pp.section_path,
                    diff_type="格式变化",
                    risk_level="none",
                    baseline_text=pp.baseline_para.text,
                    target_text=pp.target_para.text,
                    similarity_score=pp.similarity,
                    explanation="仅顺序或序号变化",
                ))
                continue
            if _semantically_equivalent(baseline_compare, target_compare, pp.similarity):
                continue
            if policy.use_llm_classify and provider is not None:
                should_report, diff_type, risk_level, explanation = _llm_classify(
                    baseline_compare, target_compare, provider, pp.similarity
                )
                if not should_report:
                    continue
            else:
                diff_type, risk_level, explanation = _rule_classify(
                    baseline_compare, target_compare, pp.similarity
                )
            if policy.rule_strengthen and _should_strengthen_risk(diff_type, risk_level):
                rule_risk = _critical_rule_risk(
                    baseline_compare, target_compare
                )
                risk_level = _max_risk(risk_level, rule_risk)
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type=diff_type,
                risk_level=risk_level,
                baseline_text=pp.baseline_para.text,
                target_text=pp.target_para.text,
                similarity_score=pp.similarity,
                explanation=explanation,
            ))
    return DiffResult(
        task_id=task_id,
        baseline_version_id=baseline_version_id,
        target_version_id=target_version_id,
        items=items,
    )
