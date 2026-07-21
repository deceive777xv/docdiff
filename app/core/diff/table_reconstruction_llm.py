"""Bounded LLM adjudication for ambiguous table continuation candidates."""

from __future__ import annotations

import json
from typing import Literal, cast

from app.core.diff.reconstruction_trace import LLMJudgment
from app.core.diff.table_reconstruction import ContinuationCandidate, TableRowMatrix
from app.core.model.base_provider import BaseProvider


_RESPONSE_FIELDS = {"candidate_id", "decision", "confidence", "reason"}
_DECISIONS = {"merge", "keep_separate"}
_MAX_CROSS_VERSION_ROWS = 3
_SYSTEM_MESSAGE = (
    "Adjudicate whether one fragmented table row continues the previous logical row. "
    "Return only one JSON object with exactly these four fields: candidate_id, decision, "
    "confidence, and reason. decision must be merge or keep_separate; confidence must be "
    "between 0 and 1. Markdown fences, prose, and additional fields are invalid."
)


def _reject_duplicate_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    data: dict[str, object] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"duplicate JSON member: {key}")
        data[key] = value
    return data


def _previous_mapping(candidate: ContinuationCandidate) -> dict[int, int]:
    logical_width = max(candidate.mapping.logical_by_physical.values(), default=-1) + 1
    physical_width = len(candidate.previous_row.normalized_cells)
    if logical_width <= 0:
        return {index: index for index in range(physical_width)}
    if physical_width <= logical_width:
        return {index: index for index in range(physical_width)}
    occupied = [
        index
        for index, value in enumerate(candidate.previous_row.normalized_cells)
        if value
    ]
    selected = set(occupied[:logical_width])

    def candidate_score(columns: tuple[int, ...]) -> tuple[object, ...]:
        gaps = [right - left for left, right in zip(columns, columns[1:])]
        variation = (
            sum(abs(gap * len(gaps) - sum(gaps)) for gap in gaps)
            if gaps
            else 0
        )
        return variation, -(columns[-1] - columns[0]), columns

    while len(selected) < logical_width:
        selected.add(
            min(
                (index for index in range(physical_width) if index not in selected),
                key=lambda index: candidate_score(tuple(sorted((*selected, index)))),
            )
        )
    return {
        physical_index: logical_index
        for logical_index, physical_index in enumerate(sorted(selected))
    }


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
    previous = _project_types(candidate.previous_row, _previous_mapping(candidate)) or []
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


def _prompt_payload(candidate: ContinuationCandidate) -> dict[str, object]:
    previous_mapping = _previous_mapping(candidate)
    return {
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
    }


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
    decision = data["decision"]
    if not isinstance(decision, str) or decision not in _DECISIONS:
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
        decision=cast(Literal["merge", "keep_separate"], decision),
        confidence=float(confidence),
        reason=reason,
    )


def adjudicate_continuation(
    candidate: ContinuationCandidate,
    provider: BaseProvider,
) -> LLMJudgment | None:
    """Ask once for a strict, candidate-bound judgment; invalid output is nonfatal."""
    messages = [
        {"role": "system", "content": _SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": json.dumps(
                _prompt_payload(candidate),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    try:
        response = provider.chat(messages)
        return _parse_response(response, candidate, provider)
    except Exception:
        return None
