"""Strict local LLM adjudication for ambiguous structure candidates."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core.llm_call_budget import LLMCallBudget
from app.core.types import Paragraph, Section


_SECTION_RESPONSE_FIELDS = {
    "candidate_id",
    "action",
    "source_ids",
    "target_section_id",
    "confidence",
    "reason",
}
_PARAGRAPH_RESPONSE_FIELDS = {
    "candidate_id",
    "action",
    "confidence",
    "reason",
}
_PAGE_NOISE_RESPONSE_FIELDS = {"boundary_id", "labels"}
_PAGE_NOISE_LABEL_FIELDS = {"id", "action", "confidence", "reason"}
_ALLOWED_ACTIONS = {"keep", "move_to_section"}
_MIN_CONFIDENCE = 0.85
_PAGE_NOISE_MIN_CONFIDENCE = 0.90
_MAX_REASON_LENGTH = 200
_PAGE_NOISE_SYSTEM_MESSAGE = """你需要对接近物理页面边界的固定文本项进行分类。

任务定义：
- 对每个提供的项，判断它是应该从规范化文档副本中移除的印刷页眉或页脚元素。
- 页眉页脚是文档逻辑内容之外的位置依赖型印刷材料，通常在各页之间重复，仅凭边界位置不足以作为判断依据。
- 一次性评判完整的边界批次。每一项都是操作目标，必须分配且仅分配一个标签。禁止省略、添加、重排序、替换、合并或拆分项。
- 不得改写、摘要、校正、生成、合并或移动文档文本。

内容安全规则：
- 数字、标准或文档标识符如果出现在正文行、章节标题或表格单元格中，则不视为噪声。但如果仅出现在顶端或底端边缘并重复，则属于典型页眉页脚，应移除。
- 含有序号或关键列及需求说明、绩效目标、标准、参考文献或附件文件名的一行是重要的业务内容。即使包含数字、.docx、Markdown 表格符号、空单元格、HTML <br> 标签或手动换行，也应保留。
- 保留真实章节标题、条目标题、正文、注释、业务列名表头、业务表格数据及跨页延续内容。
- 以表格形式呈现的重复印刷元数据仍可能是页眉页脚噪声，但表格语法、短长度或空单元格本身不足以判定为噪声。
- 重复性、边界接近性（顶部/底部）、通用描述性文本（如“页码”、“文档ID”、“机密”、“公司名”）是页眉页脚噪声的强烈证据，无论是否包含数字或标识符。除非属于连续数据表的一部分，否则应移除。
- 当内容与页眉页脚证据严格平衡时，倾向保留。但若项明确为重复定位元数据（如页码、文档标题、版权声明或位于精确顶/底端的日期），则证据不含糊，必须移除。
- 数据行的空单元格处理：若表格行中有很多空单元格，但已填单元格包含特定名词、代码或唯一值（如“Product-A”、“John Doe”、“100kg”），则属于合法数据记录，空单元格为可选字段，其内容可能与前文相关，应保留。

输入约定：
- `boundary_id` 为不透明固定字符串，请完全复制。
- `items` 为完整固定列表，每项仅包含 id、position 和 text。
- `id` 为请求局部且不透明，请完全复制且保持顺序。
- `position` 仅作为边界证据，其值为 `previous_page_end`、`next_page_start`、`document_start` 或 `document_end`。

输出约定：
- 返回恰好一个 JSON 对象，包含 `boundary_id` 和 `labels`。
- `labels` 必须与输入项顺序一致，每项一个对象。
- 每个标签必须包含 `id`、`action`、`confidence` 和 `reason`。
- `action` 必须为 `remove_as_page_noise` 或 `keep`。
- `confidence` 必须为 0.0 到 1.0 的 JSON 数字，不能加引号或使用布尔、null、NaN、无穷。
- `reason` 必须为非空 JSON 字符串，最多 200 个字符，仅讨论对应固定项。
- 不得使用 Markdown 标记或 JSON 对象外的文本，也不得添加字段。

合法的输出格式示例:
{"boundary_id":"copy-exactly","labels":[{"id":"L01","action":"remove_as_page_noise","confidence":0.99,"reason":"是典型的页眉噪声"}]}
"""
_PAGE_NOISE_REVIEW_SYSTEM_MESSAGE = _PAGE_NOISE_SYSTEM_MESSAGE + """
审核任务：
- 这是对所提供的 `initial_labels` 中相同固定项目的最终安全审核。请根据原始文本和完整边界批次重新评估每个项目。
- 特别注意编号业务行、需求、目标、标准、参考文献以及附件文件名的误删。
- 返回完整的替换标签列表。不要仅仅批准初始结果。
- 仅当两轮独立审核都返回有效的、高置信度的 `remove_as_page_noise` 标签时，候选项才会被移除。如不确定，请返回 `keep`。
"""
_PARAGRAPH_SYSTEM_MESSAGE = """你是一个文档段落合并分类器。
任务边界：
- 只判断 `candidate.continuation` 是否为 `candidate.previous` 的续写，并且两者属于同一原始段落。
- `candidate.previous` 和 `candidate.continuation` 是固定的槽位。绝不使用 `nearby_context` 中的文本替换它们。
- `nearby_context` 和 `rule_evidence` 仅作为背景证据。
- 文本结构仅作为辅助判断，尤其是列表结构符号“- ”可能是解析时产生的噪声。
- 不要重写、总结、纠正、移除或生成文档文本。

精确返回一个包含四个字段的 JSON 对象：candidate_id、action、confidence、reason。

字段要求：
- candidate_id：JSON 字符串。精确复制提供的 candidate_id。
- action：JSON 字符串。必须是 `merge_paragraphs` 或 `keep`。
- confidence：JSON 数字，从 0.0 到 1.0（含）。不要返回带引号的数字、布尔值、null、NaN 或无穷大。
- reason：非空 JSON 字符串，最多 200 个字符。仅解释固定 previous 和 continuation 槽位之间的关系。

严格 JSON 规则：
- 只输出一个 JSON 对象，且对象之前或之后不能有其他内容。
- 不要省略、添加或重复字段。
- 字段名和枚举值区分大小写。
- 不得发明或修改 candidate_id。

合法的输出示例:
{"candidate_id":"copy-exactly","action":"merge_paragraphs","confidence":0.92,"reason":"续写完成了未完成的句子。"}
"""
@dataclass(frozen=True)
class SectionMoveJudgment:
    action: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class ParagraphMergeJudgment:
    action: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class PageNoiseLabel:
    item_id: str
    action: str
    confidence: float
    reason: str


def _reject_duplicate_members(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    data: dict[str, object] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"duplicate JSON member: {key}")
        data[key] = value
    return data


def _chat(provider: object, messages: list[dict[str, str]], model: str) -> str:
    chat = getattr(provider, "chat", None)
    if callable(chat):
        return str(chat(messages))

    completions = getattr(getattr(provider, "chat", None), "completions", None)
    create = getattr(completions, "create", None)
    if not callable(create):
        raise TypeError("structure repair provider does not expose a supported chat API")
    response = create(model=model, messages=messages, temperature=0)
    return str(response.choices[0].message.content or "")


def _parse_page_noise_batch_response(
    raw: str,
    boundary_id: str,
    items: list[dict[str, str]],
) -> tuple[list[PageNoiseLabel] | None, str]:
    try:
        data: Any = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_members,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(data, dict) or set(data) != _PAGE_NOISE_RESPONSE_FIELDS:
        return None, "invalid_fields"
    if data["boundary_id"] != boundary_id:
        return None, "boundary_id_mismatch"
    labels = data["labels"]
    if not isinstance(labels, list) or len(labels) != len(items):
        return None, "label_set_mismatch"
    expected_ids = [item["id"] for item in items]
    parsed: list[PageNoiseLabel] = []
    for expected_id, label in zip(expected_ids, labels, strict=True):
        if not isinstance(label, dict) or set(label) != _PAGE_NOISE_LABEL_FIELDS:
            return None, "invalid_label_fields"
        if label["id"] != expected_id:
            return None, "label_set_mismatch"
        if label["action"] not in {"keep", "remove_as_page_noise"}:
            return None, "invalid_action"
        confidence = label["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None, "invalid_confidence"
        if not 0.0 <= float(confidence) <= 1.0:
            return None, "invalid_confidence"
        reason = label["reason"]
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > _MAX_REASON_LENGTH
        ):
            return None, "invalid_reason"
        action = str(label["action"])
        if (
            action == "remove_as_page_noise"
            and float(confidence) < _PAGE_NOISE_MIN_CONFIDENCE
        ):
            action = "keep"
        parsed.append(
            PageNoiseLabel(
                item_id=expected_id,
                action=action,
                confidence=float(confidence),
                reason=reason,
            )
        )
    return parsed, ""


def _chat_and_parse_with_validation_retry(
    provider: object,
    messages: list[dict[str, str]],
    model: str,
    parse: Any,
    call_budget: LLMCallBudget | None = None,
) -> tuple[Any | None, str]:
    budget = call_budget or LLMCallBudget()
    for attempt in range(2):
        if not budget.consume():
            return None, "llm_call_budget_exhausted"
        try:
            raw = _chat(provider, messages, model)
        except Exception:
            return None, "llm_error"
        parsed, failure_code = parse(raw)
        if parsed is not None or failure_code == "low_confidence":
            return parsed, failure_code
        if attempt == 1 or budget.remaining <= 0:
            return None, failure_code
    return None, "invalid_response"


def adjudicate_page_noise_batch(
    boundary_id: str,
    items: list[dict[str, str]],
    provider: object,
    model: str = "",
    call_budget: LLMCallBudget | None = None,
) -> tuple[list[PageNoiseLabel] | None, str]:
    """Label one fixed page-boundary batch with strict, fail-closed validation."""
    payload: dict[str, object] = {
        "boundary_id": boundary_id,
        "items": [
            {
                "id": item["id"],
                "position": item["position"],
                "text": item["text"][:1200],
            }
            for item in items
        ],
    }

    messages = [
        {"role": "system", "content": _PAGE_NOISE_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    return _chat_and_parse_with_validation_retry(
        provider,
        messages,
        model,
        lambda raw: _parse_page_noise_batch_response(raw, boundary_id, items),
        call_budget,
    )


def review_page_noise_batch(
    boundary_id: str,
    items: list[dict[str, str]],
    initial_labels: list[PageNoiseLabel],
    provider: object,
    model: str = "",
    call_budget: LLMCallBudget | None = None,
) -> tuple[list[PageNoiseLabel] | None, str]:
    """Review one fixed batch within the caller's shared LLM call budget."""
    payload: dict[str, object] = {
        "boundary_id": boundary_id,
        "items": [
            {
                "id": item["id"],
                "position": item["position"],
                "text": item["text"][:1200],
            }
            for item in items
        ],
        "initial_labels": [
            {
                "id": label.item_id,
                "action": label.action,
                "confidence": label.confidence,
                "reason": label.reason,
            }
            for label in initial_labels
        ],
    }
    messages = [
        {
            "role": "system",
            "content": _PAGE_NOISE_REVIEW_SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    review_items = [
        {"id": item["id"], "position": item["position"], "text": item["text"]}
        for item in items
    ]
    return _chat_and_parse_with_validation_retry(
        provider,
        messages,
        model,
        lambda raw: _parse_page_noise_batch_response(
            raw,
            boundary_id,
            review_items,
        ),
        call_budget,
    )


def adjudicate_section_parent(
    section: Section,
    previous: Section,
    provider: object,
    model: str = "",
    call_budget: LLMCallBudget | None = None,
) -> tuple[SectionMoveJudgment | None, str]:
    """Return a validated move judgment and a stable rejection code."""
    candidate_id = f"section:{section.section_id}"
    payload = {
        "candidate_id": candidate_id,
        "allowed_actions": sorted(_ALLOWED_ACTIONS),
        "candidate": {
            "section_id": section.section_id,
            "title": section.title,
            "level": section.level,
            "paragraphs": [paragraph.text[:800] for paragraph in section.paragraphs[:3]],
        },
        "previous_section": {
            "section_id": previous.section_id,
            "title": previous.title,
            "level": previous.level,
            "paragraphs": [paragraph.text[:800] for paragraph in previous.paragraphs[-3:]],
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "判断待处理的无编号章节是否应归入前一个有编号章节之下。请返回一个包含以下字段的 JSON 对象：\n"
                "- candidate_id：待处理章节的标识符，例如 \"section:xxx\"（字符串）\n"
                "- action：必须为 \"keep\" 或 \"move_to_section\"\n"
                "- source_ids：一个包含且仅包含一个元素（即 candidate_id 的值）的 JSON 数组\n"
                "- target_section_id：前一个章节的 section_id（字符串，不带前缀）\n"
                "- confidence：0.0 到 1.0 之间的数值\n"
                "- reason：解释您判断依据的非空字符串\n"
                "仅使用提供的 ID。切勿重写或生成文档文本。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    def parse(raw: str) -> tuple[SectionMoveJudgment | None, str]:
        try:
            data: Any = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_members,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, "invalid_json"
        if not isinstance(data, dict) or set(data) != _SECTION_RESPONSE_FIELDS:
            return None, "invalid_response"
        if (
            data["candidate_id"] != candidate_id
            or data["action"] not in _ALLOWED_ACTIONS
            or data["source_ids"] != [section.section_id]
            or data["target_section_id"] != previous.section_id
            or not isinstance(data["reason"], str)
            or not data["reason"].strip()
        ):
            return None, "invalid_response"
        confidence = data["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None, "invalid_response"
        if not 0.0 <= float(confidence) <= 1.0:
            return None, "invalid_response"
        if float(confidence) < _MIN_CONFIDENCE:
            return None, "low_confidence"
        return (
            SectionMoveJudgment(
                action=str(data["action"]),
                confidence=float(confidence),
                reason=str(data["reason"]),
            ),
            "",
        )

    return _chat_and_parse_with_validation_retry(
        provider,
        messages,
        model,
        parse,
        call_budget,
    )


def adjudicate_paragraph_merge(
    section: Section,
    previous: Paragraph,
    following: Paragraph,
    context: list[Paragraph],
    provider: object,
    model: str = "",
    rule_evidence: tuple[str, ...] = (),
    document_title: str = "",
    section_path: tuple[str, ...] = (),
    call_budget: LLMCallBudget | None = None,
) -> tuple[ParagraphMergeJudgment | None, str]:
    """Return a validated merge judgment for two existing paragraph IDs."""
    candidate_id = f"paragraphs:{previous.paragraph_id}:{following.paragraph_id}"
    payload = {
        "candidate_id": candidate_id,
        "candidate": {
            "previous": previous.text[:800],
            "continuation": following.text[:800],
        },
        "paragraph_context": {
            "document_title": document_title,
            "section_path": list(section_path) or [section.title],
            "current_section_title": section.title,
            "section_level": section.level,
        },
    }
    if previous.page_no is not None or following.page_no is not None:
        payload["pages"] = [previous.page_no, following.page_no]
    previous_index = next(
        (
            index
            for index, paragraph in enumerate(context)
            if paragraph.paragraph_id == previous.paragraph_id
        ),
        None,
    )
    following_index = next(
        (
            index
            for index, paragraph in enumerate(context)
            if paragraph.paragraph_id == following.paragraph_id
        ),
        None,
    )
    nearby: dict[str, list[str]] = {}
    if previous_index is not None:
        before = [paragraph.text[:800] for paragraph in context[:previous_index]][-6:]
        if before:
            nearby["before"] = before
    if following_index is not None:
        after = [paragraph.text[:800] for paragraph in context[following_index + 1 :]][:6]
        if after:
            nearby["after"] = after
    if nearby:
        payload["nearby_context"] = nearby
    if rule_evidence:
        payload["rule_evidence"] = list(rule_evidence)
    messages = [
        {"role": "system", "content": _PARAGRAPH_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    def parse(raw: str) -> tuple[ParagraphMergeJudgment | None, str]:
        try:
            data: Any = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_members,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, "invalid_json"
        if not isinstance(data, dict) or set(data) != _PARAGRAPH_RESPONSE_FIELDS:
            return None, "invalid_fields"
        if data["candidate_id"] != candidate_id:
            return None, "candidate_id_mismatch"
        if data["action"] not in {"keep", "merge_paragraphs"}:
            return None, "invalid_action"
        confidence = data["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None, "invalid_confidence"
        if not 0.0 <= float(confidence) <= 1.0:
            return None, "invalid_confidence"
        reason = data["reason"]
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > _MAX_REASON_LENGTH
        ):
            return None, "invalid_reason"
        if float(confidence) < _MIN_CONFIDENCE:
            return None, "low_confidence"
        return (
            ParagraphMergeJudgment(
                action=str(data["action"]),
                confidence=float(confidence),
                reason=reason,
            ),
            "",
        )

    return _chat_and_parse_with_validation_retry(
        provider,
        messages,
        model,
        parse,
        call_budget,
    )
