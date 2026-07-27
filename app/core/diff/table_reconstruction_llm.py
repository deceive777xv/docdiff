"""Bounded LLM adjudication for ambiguous table continuation candidates."""

from __future__ import annotations

import json
from typing import Literal, cast

from app.core.diff.reconstruction_trace import LLMJudgment
from app.core.diff.table_boundary_context import TableBoundaryContext
from app.core.diff.table_reconstruction import ContinuationCandidate, TableRowMatrix
from app.core.model.base_provider import BaseProvider


_ATOMIC_RESPONSE_FIELDS = {
    "boundary_id",
    "candidate_id",
    "roles",
    "action",
    "mapping_id",
    "confidence",
    "reason",
}
_LEGACY_RESPONSE_FIELDS = {
    "boundary_id",
    "roles",
    "row_action",
    "table_action",
    "confidence",
    "reason",
}
_ROW_ACTIONS = {"merge", "keep"}
_TABLE_ACTIONS = {"merge_fragments", "keep"}
_ATOMIC_ACTIONS = {"merge_row", "keep"}
_ROLES = {
    "previous_row",
    "continuation_row",
    "body_row",
    "table_header",
    "page_header",
    "page_footer",
    "ordinary_text",
    "new_table",
}
_MAX_CROSS_VERSION_ROWS = 3
_SYSTEM_MESSAGE = (
    "You are a table reconstruction classifier. Given context items from two consecutive "
    "pages around a page boundary, classify each item's role and decide whether the table "
    "continues across the boundary.\n\n"
    "Output a single JSON object with exactly these fields:\n"
    "- boundary_id: must match the supplied boundary_id.\n"
    "- candidate_id: must match the supplied candidate_id.\n"
    "- roles: a dict mapping context item IDs (context_items[*].id) to one of:\n"
    "  previous_row: the single row immediately before the boundary that is the last"
    "row of the table on the previous page (corresponds to previous_cells).\n"
    "  continuation_row: a row on the next page that continues the same table "
    "(corresponds to continuation_cells).\n"
    "  body_row: a table body row on the previous page that is the same table "
    "but is NOT the immediate boundary row (i.e. rows above previous_row).\n"
    "  table_header: a table header row"
    "  page_header: page-level artifacts (document title, page number, separator lines, "
    "etc.) that should be ignored.\n"
    "  page_footer: page-level footer artifacts.\n"
    "  ordinary_text: non-table text context.\n"
    "  new_table: a row belonging to a different table that starts on the next page.\n"
    "  Only one item should be previous_row. Items above it on the same page should be "
    "body_row, not previous_row or continuation_row.\n"
    "- action: merge_row if the continuation row(s) should be merged into the previous "
    "table, or keep if they should stay separate (e.g. a new table starts on the next "
    "page).\n"
    "- mapping_id: must be one of the supplied mapping candidate IDs.\n"
    "- confidence: a number between 0.0 and 1.0.\n"
    "- reason: a non-empty string explaining your decision.\n\n"
    "Rules:\n"
    "- Never invent text, IDs.\n"
    "- No Markdown fences, no prose outside the JSON object, no extra fields."
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
    mapping_id = f"{candidate.candidate_id}:mapping:0"
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
        "mapping_candidates": [
            {
                "mapping_id": mapping_id,
                "logical_by_physical": [
                    [physical, logical]
                    for physical, logical in sorted(
                        candidate.mapping.logical_by_physical.items()
                    )
                ],
                "score": candidate.mapping.score,
            }
        ],
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
    if not isinstance(data, dict):
        return None
    response_fields = frozenset(data)
    if response_fields not in {
        frozenset(_ATOMIC_RESPONSE_FIELDS),
        frozenset(_LEGACY_RESPONSE_FIELDS),
    }:
        return None
    boundary_id = context.boundary_id if context is not None else candidate.candidate_id
    if data["boundary_id"] != boundary_id:
        return None
    roles = data["roles"]
    if not isinstance(roles, dict):
        return None
    allowed_role_ids = set()
    if context is not None:
        allowed_role_ids.update(item.item_id for item in context.items)
    if any(
        not isinstance(item_id, str)
        or item_id not in allowed_role_ids
        or not isinstance(role, str)
        or role not in _ROLES
        for item_id, role in roles.items()
    ):
        return None
    if response_fields == _ATOMIC_RESPONSE_FIELDS:
        if data["candidate_id"] != candidate.candidate_id:
            return None
        mapping_id = f"{candidate.candidate_id}:mapping:0"
        if data["mapping_id"] != mapping_id:
            return None
        action = data["action"]
        if not isinstance(action, str) or action not in _ATOMIC_ACTIONS:
            return None
        row_action = "merge" if action == "merge_row" else "keep"
        table_action = "merge_fragments" if action == "merge_row" else "keep"
    else:
        mapping_id = ""
        row_action = data["row_action"]
        table_action = data["table_action"]
    if not isinstance(row_action, str) or row_action not in _ROW_ACTIONS:
        return None
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
        mapping_id=mapping_id,
        roles={str(item_id): str(role) for item_id, role in roles.items()},
        row_action=cast(Literal["merge", "keep"], row_action),
        table_action=cast(Literal["merge_fragments", "keep"], table_action),
    )


def adjudicate_continuation(
    candidate: ContinuationCandidate,
    provider: BaseProvider,
    context: TableBoundaryContext | None = None,
) -> LLMJudgment | None:
    """Request a strict candidate-bound judgment, retrying invalid JSON once."""
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
        judgment = _parse_response(response, candidate, provider, context)
    except Exception:
        return None
    if judgment is not None:
        return judgment

    retry_messages = [
        {
            "role": "system",
            "content": (
                _SYSTEM_MESSAGE
                + " The previous response failed strict validation. Correct only the JSON "
                "shape and reuse the supplied IDs."
            ),
        },
        messages[1],
    ]
    try:
        response = provider.chat(retry_messages)
        return _parse_response(response, candidate, provider, context)
    except Exception:
        return None
