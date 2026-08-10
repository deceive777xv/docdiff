"""Bounded LLM adjudication for one fixed table-continuation candidate."""

from __future__ import annotations

import json
from typing import Literal, cast

from app.core.normalization.table_trace import LLMJudgment, SourceRowRef
from app.core.normalization.table_boundary_context import (
    BoundaryContextItem,
    TableBoundaryContext,
)
from app.core.normalization.tables import ContinuationCandidate, TableRowMatrix
from app.core.model.base_provider import BaseProvider
from app.core.llm_call_budget import LLMCallBudget


_RESPONSE_FIELDS = {
    "candidate_id",
    "continuation_role",
    "row_action",
    "table_action",
    "confidence",
    "reason",
}
_CONTINUATION_ROLES = {
    "continuation_row",
    "new_business_row",
    "table_header",
    "page_header",
    "page_footer",
    "ordinary_text",
    "new_table",
}
_ROW_ACTIONS = {"merge", "keep"}
_TABLE_ACTIONS = {"merge_fragments", "keep"}
_MAX_PEER_ROWS = 2
_MAX_REASON_LENGTH = 200
_SYSTEM_MESSAGE = """你是一个表格重建分类器。
任务范围：
- 做出两个独立的判断：判断两个表格片段是否属于同一个逻辑表，以及`candidate.continuation`与`candidate.previous`按照列映射是否属于同一逻辑行。
- `candidate.next`、`nearby_context`和`peer_rows`只是背景证据，它们永远不能成为判断目标。
- 结构线索：表格片段可能以 Markdown 风格的行出现，用竖线（|）分隔。
- 不要重写单元格、发明文本、选择列映射或替换固定的片段或行。
- 注意按列区分单元格，仅按照提供的列映射进行判断，不主动替换为正确的列映射。
- 仅判断`candidate.continuation`与`candidate.previous`，不要判断`candidate.continuation`之后的项。

准确返回一个 JSON 对象，且必须包含以下六个字段：
candidate_id, continuation_role, row_action, table_action, confidence, reason.
字段约定：
- candidate_id：JSON 字符串，完全按提供的`candidate_id`复制。
- continuation_role：JSON 字符串，仅对`candidate.continuation`进行分类。
    必须是以下之一：continuation_row, new_business_row, table_header, page_header, page_footer, ordinary_text, new_table。
    - `continuation_row`：当`candidate.continuation`的内容是`candidate.previous`同一行“对应列”的剩余部分时使用（通常存在对应列内容不完整、缺少结尾标点，或有延续性列表内容等），列映射必须正确。
    - `new_business_row`：当`candidate.continuation`是同一表格中新的一整行，而不是上一行的延续时使用。
- row_action：JSON 字符串，必须是 `merge` 或 `keep`。当 `continuation_role` 为 `continuation_row` 时，`row_action=merge` 才有效。对于重复的表头或新的业务行，即使判断属于同一表，也必须使用 `keep`。
- table_action：JSON 字符串，必须是 `merge_fragments` 或 `keep`。独立于 `row_action` 做出决定。`merge_fragments` 表示右边的片段是左边逻辑表的继续；这并不意味着边界行应合并。`new_table`、`page_header`、`page_footer` 和 `ordinary_text` 必须使用 `keep`。`table_action=merge_fragments`可在 `continuation_role` 为 `continuation_row` 或 `new_business_row` 时使用。
- confidence：JSON 数字，范围从 0.0 到 1.0（包括 0.0 和 1.0）。不要返回带引号的数字、布尔值、null、NaN 或 infinity。和 `row_action`、`table_action`相关：
    - row_action==merge && table_action==merge_fragments: confifence >= 0.9;
    - row_action==keep && table_action==merge_fragments: 0.8 <= confifence < 0.9;
    - row_action==keep && table_action==keep: confifence < 0.8;
- reason：非空 JSON 字符串，最多 200 个字符。解释两个结论的原因。

注意：想清楚原因再输出结论。

严格的 JSON 规则：
- 输出一个裸 JSON 对象，前后不要有任何其它内容。
- 不要使用所提供 Markdown 对象外的文本。
- 不省略、添加或重复字段。
- 字段名和枚举值区分大小写。
- 绝不编造或修改 `candidate_id`。

合法的输出示例:
{"candidate_id":"copy-exactly","continuation_role":"continuation_row",\
"row_action":"merge","table_action":"merge_fragments","confidence":0.92,\
"reason":"是前一行表格的延续，是同一个表格。"}
"""


def _reject_duplicate_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    data: dict[str, object] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"duplicate JSON member: {key}")
        data[key] = value
    return data


def _project_cells(
    row: TableRowMatrix | None,
    mapping: dict[int, int],
) -> list[str] | None:
    if row is None:
        return None
    if not mapping:
        return list(row.normalized_cells)
    width = max(mapping.values(), default=-1) + 1
    cells = [""] * width
    for physical_index, logical_index in sorted(mapping.items()):
        if physical_index < len(row.normalized_cells):
            cells[logical_index] = row.normalized_cells[physical_index]
    return cells


def _matches_source(item: BoundaryContextItem, source: SourceRowRef) -> bool:
    return (
        item.section_id == source.section_id
        and item.paragraph_id == source.paragraph_id
        and item.sentence_index == source.sentence_index
    )


def _nearby_context(
    candidate: ContinuationCandidate,
    context: TableBoundaryContext | None,
) -> dict[str, list[str]]:
    if context is None:
        return {}
    previous_index = next(
        (
            index
            for index, item in enumerate(context.items)
            if _matches_source(item, candidate.previous_row.source)
        ),
        None,
    )
    continuation_index = next(
        (
            index
            for index, item in enumerate(context.items)
            if _matches_source(item, candidate.continuation_row.source)
        ),
        None,
    )
    if previous_index is None or continuation_index is None:
        return {}
    nearby: dict[str, list[str]] = {}
    before = [item.text for item in context.items[:previous_index]]
    after = [
        item.text
        for index, item in enumerate(context.items[previous_index + 1 :], previous_index + 1)
        if index != continuation_index
    ]
    if before:
        nearby["before"] = before
    if after:
        nearby["after"] = after
    return nearby


def _prompt_payload(
    candidate: ContinuationCandidate,
    context: TableBoundaryContext | None,
) -> dict[str, object]:
    candidate_rows: dict[str, object] = {
        "previous": _project_cells(
            candidate.previous_row,
            candidate.previous_mapping.logical_by_physical,
        ),
        "continuation": _project_cells(
            candidate.continuation_row,
            candidate.mapping.logical_by_physical,
        ),
    }
    next_cells = _project_cells(
        candidate.next_full_row,
        candidate.mapping.logical_by_physical,
    )
    if next_cells is not None:
        candidate_rows["next"] = next_cells
    payload: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "candidate": candidate_rows,
    }
    nearby = _nearby_context(candidate, context)
    if nearby:
        payload["nearby_context"] = nearby
    peer_rows = [
        list(row.normalized_cells)
        for row in candidate.cross_version_rows[:_MAX_PEER_ROWS]
    ]
    if peer_rows:
        payload["peer_rows"] = peer_rows
    return payload


def _parse_response(
    response: str,
    candidate: ContinuationCandidate,
    provider: BaseProvider,
) -> LLMJudgment | None:
    data = json.loads(response, object_pairs_hook=_reject_duplicate_members)
    if not isinstance(data, dict) or set(data) != _RESPONSE_FIELDS:
        return None
    if data["candidate_id"] != candidate.candidate_id:
        return None
    continuation_role = data["continuation_role"]
    if (
        not isinstance(continuation_role, str)
        or continuation_role not in _CONTINUATION_ROLES
    ):
        return None
    row_action = data["row_action"]
    if not isinstance(row_action, str) or row_action not in _ROW_ACTIONS:
        return None
    if row_action == "merge" and continuation_role != "continuation_row":
        return None
    table_action = data["table_action"]
    if not isinstance(table_action, str) or table_action not in _TABLE_ACTIONS:
        return None
    if table_action == "merge_fragments" and continuation_role not in {
        "continuation_row",
        "table_header",
        "new_business_row",
    }:
        return None
    confidence = data["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if not 0.0 <= float(confidence) <= 1.0:
        return None
    reason = data["reason"]
    if (
        not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > _MAX_REASON_LENGTH
    ):
        return None
    merge = table_action == "merge_fragments" or row_action == "merge"
    return LLMJudgment(
        model=str(getattr(provider, "chat_model", provider.__class__.__name__)),
        decision=cast(
            Literal["merge", "keep_separate"],
            "merge" if merge else "keep_separate",
        ),
        confidence=float(confidence),
        reason=reason,
        mapping_id=f"{candidate.candidate_id}:mapping:0",
        roles={"continuation": continuation_role},
        row_action=cast(Literal["merge", "keep"], row_action),
        table_action=cast(Literal["merge_fragments", "keep"], table_action),
    )


def _try_parse_response(
    response: str,
    candidate: ContinuationCandidate,
    provider: BaseProvider,
) -> LLMJudgment | None:
    try:
        return _parse_response(response, candidate, provider)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _chat_with_validation_retry(
    provider: BaseProvider,
    messages: list[dict[str, str]],
    candidate: ContinuationCandidate,
    call_budget: LLMCallBudget | None = None,
    *,
    max_attempts: int = 2,
) -> LLMJudgment | None:
    budget = call_budget or LLMCallBudget()
    for attempt in range(max_attempts):
        if not budget.consume():
            return None
        try:
            response = provider.chat(messages)
        except Exception:
            return None
        judgment = _try_parse_response(response, candidate, provider)
        if judgment is not None:
            return judgment
        if attempt == max_attempts - 1:
            return None
    return None


def adjudicate_continuation(
    candidate: ContinuationCandidate,
    provider: BaseProvider,
    context: TableBoundaryContext | None = None,
    call_budget: LLMCallBudget | None = None,
    *,
    max_attempts: int = 2,
) -> LLMJudgment | None:
    """Request one strict judgment, retrying one validation failure within budget."""
    user_message = {
        "role": "user",
        "content": json.dumps(
            _prompt_payload(candidate, context),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    messages = [
        {"role": "system", "content": _SYSTEM_MESSAGE},
        user_message,
    ]
    return _chat_with_validation_retry(
        provider,
        messages,
        candidate,
        call_budget,
        max_attempts=max_attempts,
    )


def review_continuation(
    candidate: ContinuationCandidate,
    initial: LLMJudgment,
    provider: BaseProvider,
    context: TableBoundaryContext | None = None,
    call_budget: LLMCallBudget | None = None,
) -> LLMJudgment | None:
    """Review one accepted change once; invalid review fails closed."""
    payload = _prompt_payload(candidate, context)
    payload["initial_judgment"] = {
        "continuation_role": initial.roles.get("continuation", "continuation_row"),
        "row_action": initial.row_action,
        "table_action": initial.table_action,
        "confidence": initial.confidence,
        "reason": initial.reason,
    }
    messages = [
        {
            "role": "system",
            "content": (
                _SYSTEM_MESSAGE
                + "\n请根据原始固定候选和 initial_judgment 执行最终复审。"
                "初判仅作为辅助证据，不是必须确认的答案。"
                "请返回完整、独立可校验的替代结论；有效复审将作为最终 LLM 结论。"
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
    return _chat_with_validation_retry(
        provider,
        messages,
        candidate,
        call_budget,
        max_attempts=1,
    )
