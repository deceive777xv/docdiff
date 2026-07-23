"""Bounded LLM adjudication for ambiguous table continuation candidates."""

from __future__ import annotations

import json
from typing import Literal, cast

from app.core.diff.reconstruction_trace import LLMJudgment
from app.core.diff.table_boundary_context import TableBoundaryContext
from app.core.diff.table_reconstruction import ContinuationCandidate, TableRowMatrix
from app.core.model.base_provider import BaseProvider


_RESPONSE_FIELDS = {
    "boundary_id",
    "roles",
    "row_action",
    "table_action",
    "confidence",
    "reason",
}
_ROW_ACTIONS = {"merge", "keep"}
_TABLE_ACTIONS = {"merge_fragments", "keep"}
_ROLES = {
    "body_row",
    "continuation_row",
    "table_header",
    "page_header",
    "page_footer",
    "ordinary_text",
    "new_table",
}
_MAX_CROSS_VERSION_ROWS = 3
_SYSTEM_MESSAGE = (
    "Classify a bounded physical-page boundary and decide table reconstruction actions. "
    "Return only one JSON object with exactly these fields: boundary_id, roles, row_action, "
    "table_action, confidence, reason. roles may reference only supplied IDs. row_action "
    "must be merge or keep; table_action must be merge_fragments or keep. Never invent "
    "text, IDs, or column mappings. Markdown fences, prose, and extra fields are invalid."
)


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


def _logical_column_roles(candidate: ContinuationCandidate) -> list[dict[str, object]]:
    previous = _project_types(
        candidate.previous_row,
        candidate.previous_mapping.logical_by_physical,
    ) or []
    continuation = _project_types(
        candidate.continuation_row,
        candidate.mapping.logical_by_physical,
    )
    next_types = _project_types(
        candidate.next_full_row,
        candidate.mapping.logical_by_physical,
    )
    width = max(len(previous), len(continuation or []), len(next_types or []))
    return [
        {
            "logical_index": index,
            "previous_type": previous[index] if index < len(previous) else "empty",
            "continuation_type": (
                continuation[index] if continuation is not None and index < len(continuation) else "empty"
            ),
            "next_type": next_types[index] if next_types is not None and index < len(next_types) else "empty",
        }
        for index in range(width)
    ]


def _project_types(
    row: TableRowMatrix | None,
    mapping: dict[int, int],
) -> list[str] | None:
    if row is None:
        return None
    if not mapping:
        return list(row.value_types)
    width = max(mapping.values(), default=-1) + 1
    values = ["empty"] * width
    for physical_index, logical_index in sorted(mapping.items()):
        if physical_index < len(row.value_types):
            values[logical_index] = row.value_types[physical_index]
    return values


def _prompt_payload(
    candidate: ContinuationCandidate,
    context: TableBoundaryContext | None,
) -> dict[str, object]:
    previous_mapping = candidate.previous_mapping.logical_by_physical
    boundary_id = context.boundary_id if context is not None else candidate.candidate_id
    return {
        "boundary_id": boundary_id,
        "candidate_id": candidate.candidate_id,
        "side": candidate.side,
        "previous_cells": _project_cells(candidate.previous_row, previous_mapping),
        "continuation_cells": _project_cells(
            candidate.continuation_row,
            candidate.mapping.logical_by_physical,
        ),
        "next_cells": _project_cells(
            candidate.next_full_row,
            candidate.mapping.logical_by_physical,
        ),
        "logical_column_roles": _logical_column_roles(candidate),
        "physical_mapping": {
            "source_columns": list(candidate.mapping.source_columns),
            "previous_logical_by_physical": [
                [physical, logical]
                for physical, logical in sorted(previous_mapping.items())
            ],
            "continuation_logical_by_physical": [
                [physical, logical]
                for physical, logical in sorted(candidate.mapping.logical_by_physical.items())
            ],
        },
        "rule_evidence": list(candidate.evidence),
        "rule_conflicts": list(dict.fromkeys((*candidate.conflicts, *candidate.vetoes))),
        "cross_version_rows": [
            list(row.normalized_cells)
            for row in candidate.cross_version_rows[:_MAX_CROSS_VERSION_ROWS]
        ],
        "context_items": [
            {
                "id": item.item_id,
                "page_no": item.page_no,
                "kind": item.kind,
                "text": item.text,
            }
            for item in (context.items if context is not None else ())
        ],
    }


def _parse_response(
    response: str,
    candidate: ContinuationCandidate,
    provider: BaseProvider,
    context: TableBoundaryContext | None,
) -> LLMJudgment | None:
    data = json.loads(response, object_pairs_hook=_reject_duplicate_members)
    if not isinstance(data, dict) or set(data) != _RESPONSE_FIELDS:
        return None
    boundary_id = context.boundary_id if context is not None else candidate.candidate_id
    if data["boundary_id"] != boundary_id:
        return None
    roles = data["roles"]
    if not isinstance(roles, dict):
        return None
    allowed_role_ids = {"previous_row", "continuation_row"}
    if candidate.next_full_row is not None:
        allowed_role_ids.add("next_full_row")
    if context is not None:
        allowed_role_ids.update(item.item_id for item in context.items)
    if not {"previous_row", "continuation_row"}.issubset(roles):
        return None
    if any(
        not isinstance(item_id, str)
        or item_id not in allowed_role_ids
        or not isinstance(role, str)
        or role not in _ROLES
        for item_id, role in roles.items()
    ):
        return None
    row_action = data["row_action"]
    if not isinstance(row_action, str) or row_action not in _ROW_ACTIONS:
        return None
    table_action = data["table_action"]
    if not isinstance(table_action, str) or table_action not in _TABLE_ACTIONS:
        return None
    confidence = data["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if not 0.0 <= float(confidence) <= 1.0:
        return None
    reason = data["reason"]
    if not isinstance(reason, str) or not reason.strip():
        return None
    return LLMJudgment(
        model=str(getattr(provider, "chat_model", provider.__class__.__name__)),
        decision=cast(
            Literal["merge", "keep_separate"],
            (
                "merge"
                if row_action == "merge" and table_action == "merge_fragments"
                else "keep_separate"
            ),
        ),
        confidence=float(confidence),
        reason=reason,
        roles={str(item_id): str(role) for item_id, role in roles.items()},
        row_action=cast(Literal["merge", "keep"], row_action),
        table_action=cast(Literal["merge_fragments", "keep"], table_action),
    )


def adjudicate_continuation(
    candidate: ContinuationCandidate,
    provider: BaseProvider,
    context: TableBoundaryContext | None = None,
) -> LLMJudgment | None:
    """Ask once for a strict, candidate-bound judgment; invalid output is nonfatal."""
    messages = [
        {"role": "system", "content": _SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": json.dumps(
                _prompt_payload(candidate, context),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    try:
        response = provider.chat(messages)
        return _parse_response(response, candidate, provider, context)
    except Exception:
        return None
