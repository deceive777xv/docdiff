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
_SYSTEM_MESSAGE = """You are a table reconstruction classifier.

Task boundary:
- Make two independent judgments: whether the two table fragments belong to one logical
  table, and whether candidate.continuation belongs to the same row as
  candidate.previous.
- candidate.previous and candidate.continuation are fixed candidate slots. Never replace
  either slot with a row from candidate.next, nearby_context, or peer_rows.
- candidate.next, nearby_context, and peer_rows are background evidence only. They can
  never become the target of action.
- Structural clues: Table fragments may appear as Markdown-style rows with pipe (|) delimiters.
- Do not rewrite cells, invent text, choose a column mapping, or replace the fixed
  fragments or rows.

Return exactly one JSON object with exactly these six fields:
candidate_id, continuation_role, row_action, table_action, confidence, reason.

Field contract:
- candidate_id: JSON string. Copy the supplied candidate_id exactly.
- continuation_role: JSON string classifying candidate.continuation only. It must be one
  of: continuation_row, new_business_row, table_header, page_header, page_footer,
  ordinary_text, new_table.
  - `continuation_row`: Use when the content in `candidate.continuation`
    is the remaining part of the same row as `candidate.previous`.
  - `new_business_row`: Use when `candidate.continuation` is a complete new row of the same table,
    not a continuation of the previous row.
- row_action: JSON string, exactly merge or keep. row_action=merge is valid only when
  continuation_role is continuation_row. A repeated header or a new business row must
  use keep even when the fragments belong to one table. Use new_business_row when the
  fixed row starts a complete new record in the continued table.
- table_action: JSON string, exactly merge_fragments or keep. Decide this independently
  from row_action. merge_fragments means the right fragment continues the left logical
  table; it does not mean the boundary rows should be combined. new_table, page_header,
  page_footer, and ordinary_text must use keep. row_action=merge is valid when continuation_role
  is continuation_row or new_business_row.
- confidence: JSON number from 0.0 through 1.0 inclusive. Do not return a quoted number,
  boolean, null, NaN, or infinity.
- reason: non-empty JSON string of at most 200 characters. Explain both relationships.

Strict JSON rules:
- Output one bare JSON object and nothing before or after it.
- Do not use Markdown fences or prose outside the object.
- Do not omit, add, or duplicate fields.
- Field names and enum values are case-sensitive.
- Never invent or modify candidate_id.

Valid output example:
{"candidate_id":"copy-exactly","continuation_role":"continuation_row",\
"row_action":"merge","table_action":"merge_fragments","confidence":0.92,\
"reason":"Same table and completes the previous row."}
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
) -> LLMJudgment | None:
    budget = call_budget or LLMCallBudget()
    for attempt in range(2):
        if not budget.consume():
            return None
        try:
            response = provider.chat(messages)
        except Exception:
            return None
        judgment = _try_parse_response(response, candidate, provider)
        if judgment is not None:
            return judgment
        if attempt == 1:
            return None
    return None


def adjudicate_continuation(
    candidate: ContinuationCandidate,
    provider: BaseProvider,
    context: TableBoundaryContext | None = None,
    call_budget: LLMCallBudget | None = None,
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
                + "\nReview the supplied initial_judgment against the original fixed "
                "candidate. Return a complete independent replacement judgment. A "
                "change will be applied only when both rounds agree."
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
    )
