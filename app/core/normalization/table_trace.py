"""Versioned import-normalization trace data for cross-page tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Literal, cast

from app.core.types import DocumentIR


SCHEMA_VERSION = 3
ALGORITHM_VERSION = "cross-page-table-v5"
_LEGACY_ALGORITHM_VERSION = "cross-page-table-v1"

_SIDES = {"baseline", "target"}
_DECISION_ACTIONS = {"merge", "keep_separate"}
_RULE_CONFIDENCES = {"high", "medium", "low"}
_OPERATION_TYPES = {
    "project_columns",
    "drop_boundary_rows",
    "drop_boundary_paragraphs",
    "drop_repeated_table_header",
    "merge_rows",
    "merge_fragments",
}


@dataclass(frozen=True)
class DocumentTraceRef:
    doc_id: str
    file_hash: str


@dataclass(frozen=True)
class SourceRowRef:
    section_id: str
    paragraph_id: str
    sentence_index: int


@dataclass(frozen=True)
class LLMJudgment:
    model: str
    decision: Literal["merge", "keep_separate"]
    confidence: float
    reason: str
    mapping_id: str = ""
    roles: dict[str, str] = field(default_factory=dict)
    row_action: Literal["merge", "keep"] = "keep"
    table_action: Literal["merge_fragments", "keep"] = "keep"


@dataclass(frozen=True)
class ReconstructionDecision:
    candidate_id: str
    side: Literal["baseline", "target"]
    source_rows: list[SourceRowRef]
    column_mapping: dict[int, int]
    rule_confidence: Literal["high", "medium", "low"]
    rule_evidence: list[str]
    rule_conflicts: list[str]
    llm: LLMJudgment | None
    final_action: Literal["merge", "keep_separate"]
    boundary_id: str = ""
    previous_page_no: int | None = None
    next_page_no: int | None = None
    context_refs: list[str] = field(default_factory=list)
    generated_row_id: str = ""
    review: LLMJudgment | None = None


@dataclass(frozen=True)
class ReconstructionOperation:
    operation_id: str
    side: Literal["baseline", "target"]
    type: Literal[
        "project_columns",
        "drop_boundary_rows",
        "drop_boundary_paragraphs",
        "drop_repeated_table_header",
        "merge_rows",
        "merge_fragments",
    ]
    source_rows: list[SourceRowRef] = field(default_factory=list)
    source_paragraph_ids: list[str] = field(default_factory=list)
    column_mapping: dict[int, int] = field(default_factory=dict)
    decision_id: str = ""
    generated_row_id: str = ""
    generated_paragraph_id: str = ""


@dataclass(frozen=True)
class ReconstructionTrace:
    schema_version: int
    algorithm_version: str
    baseline: DocumentTraceRef
    target: DocumentTraceRef
    decisions: list[ReconstructionDecision]
    operations: list[ReconstructionOperation]


def trace_to_dict(trace: ReconstructionTrace) -> dict[str, object]:
    """Convert a trace to JSON-safe primitives without losing typed mappings."""
    return cast(dict[str, object], asdict(trace))


def _require_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _require_string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{label} must be a required string")
    return value


def _require_enum(value: object, label: str, choices: set[str]) -> str:
    value = _require_string(value, label)
    if value not in choices:
        raise ValueError(f"{label} has unsupported value {value!r}")
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _parse_source_row(value: object, label: str) -> SourceRowRef:
    data = _require_object(value, label)
    sentence_index = data.get("sentence_index")
    if isinstance(sentence_index, bool) or not isinstance(sentence_index, int) or sentence_index < 0:
        raise ValueError(f"{label}.sentence_index must be a non-negative integer")
    return SourceRowRef(
        section_id=_require_string(data.get("section_id"), f"{label}.section_id"),
        paragraph_id=_require_string(data.get("paragraph_id"), f"{label}.paragraph_id"),
        sentence_index=sentence_index,
    )


def _parse_source_rows(value: object, label: str) -> list[SourceRowRef]:
    return [_parse_source_row(item, f"{label}[{index}]") for index, item in enumerate(_require_list(value, label))]


def _parse_string_list(value: object, label: str) -> list[str]:
    return [_require_string(item, f"{label}[{index}]") for index, item in enumerate(_require_list(value, label))]


def _parse_column_mapping(value: object, label: str) -> dict[int, int]:
    data = _require_object(value, label)
    mapping: dict[int, int] = {}
    for raw_key, raw_value in data.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, (str, int)):
            raise ValueError(f"{label} keys must be integers")
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} keys must be integers") from exc
        if str(key) != str(raw_key):
            raise ValueError(f"{label} keys must be integers")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValueError(f"{label} values must be integers")
        mapping[key] = raw_value
    return mapping


def _parse_llm(value: object, label: str) -> LLMJudgment | None:
    if value is None:
        return None
    data = _require_object(value, label)
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{label}.confidence must be between 0 and 1")
    decision = cast(Literal["merge", "keep_separate"], _require_enum(data.get("decision"), f"{label}.decision", _DECISION_ACTIONS))
    raw_roles = data.get("roles", {})
    if not isinstance(raw_roles, dict) or any(
        not isinstance(key, str) or not isinstance(role, str)
        for key, role in raw_roles.items()
    ):
        raise ValueError(f"{label}.roles must map strings to strings")
    default_row_action = "merge" if decision == "merge" else "keep"
    default_table_action = "merge_fragments" if decision == "merge" else "keep"
    row_action = _require_enum(
        data.get("row_action", default_row_action),
        f"{label}.row_action",
        {"merge", "keep"},
    )
    table_action = _require_enum(
        data.get("table_action", default_table_action),
        f"{label}.table_action",
        {"merge_fragments", "keep"},
    )
    mapping_id = _require_string(
        data.get("mapping_id", ""),
        f"{label}.mapping_id",
        allow_empty=True,
    )
    return LLMJudgment(
        model=_require_string(data.get("model"), f"{label}.model"),
        decision=decision,
        confidence=float(confidence),
        reason=_require_string(data.get("reason"), f"{label}.reason"),
        mapping_id=mapping_id,
        roles=cast(dict[str, str], dict(raw_roles)),
        row_action=cast(Literal["merge", "keep"], row_action),
        table_action=cast(Literal["merge_fragments", "keep"], table_action),
    )


def _parse_optional_page_no(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer or null")
    return value


def _parse_decision(value: object, label: str) -> ReconstructionDecision:
    data = _require_object(value, label)
    return ReconstructionDecision(
        candidate_id=_require_string(data.get("candidate_id"), f"{label}.candidate_id"),
        side=cast(Literal["baseline", "target"], _require_enum(data.get("side"), f"{label}.side", _SIDES)),
        source_rows=_parse_source_rows(data.get("source_rows"), f"{label}.source_rows"),
        column_mapping=_parse_column_mapping(data.get("column_mapping"), f"{label}.column_mapping"),
        rule_confidence=cast(Literal["high", "medium", "low"], _require_enum(data.get("rule_confidence"), f"{label}.rule_confidence", _RULE_CONFIDENCES)),
        rule_evidence=_parse_string_list(data.get("rule_evidence"), f"{label}.rule_evidence"),
        rule_conflicts=_parse_string_list(data.get("rule_conflicts"), f"{label}.rule_conflicts"),
        llm=_parse_llm(data.get("llm"), f"{label}.llm"),
        final_action=cast(Literal["merge", "keep_separate"], _require_enum(data.get("final_action"), f"{label}.final_action", _DECISION_ACTIONS)),
        boundary_id=_require_string(
            data.get("boundary_id", ""),
            f"{label}.boundary_id",
            allow_empty=True,
        ),
        previous_page_no=_parse_optional_page_no(
            data.get("previous_page_no"),
            f"{label}.previous_page_no",
        ),
        next_page_no=_parse_optional_page_no(
            data.get("next_page_no"),
            f"{label}.next_page_no",
        ),
        context_refs=_parse_string_list(
            data.get("context_refs", []),
            f"{label}.context_refs",
        ),
        generated_row_id=_require_string(data.get("generated_row_id", ""), f"{label}.generated_row_id", allow_empty=True),
        review=_parse_llm(data.get("review"), f"{label}.review"),
    )


def _parse_operation(value: object, label: str) -> ReconstructionOperation:
    data = _require_object(value, label)
    return ReconstructionOperation(
        operation_id=_require_string(data.get("operation_id"), f"{label}.operation_id"),
        side=cast(Literal["baseline", "target"], _require_enum(data.get("side"), f"{label}.side", _SIDES)),
        type=cast(Literal["project_columns", "drop_boundary_rows", "drop_boundary_paragraphs", "drop_repeated_table_header", "merge_rows", "merge_fragments"], _require_enum(data.get("type"), f"{label}.type", _OPERATION_TYPES)),
        source_rows=_parse_source_rows(data.get("source_rows", []), f"{label}.source_rows"),
        source_paragraph_ids=_parse_string_list(data.get("source_paragraph_ids", []), f"{label}.source_paragraph_ids"),
        column_mapping=_parse_column_mapping(data.get("column_mapping", {}), f"{label}.column_mapping"),
        decision_id=_require_string(data.get("decision_id", ""), f"{label}.decision_id", allow_empty=True),
        generated_row_id=_require_string(data.get("generated_row_id", ""), f"{label}.generated_row_id", allow_empty=True),
        generated_paragraph_id=_require_string(data.get("generated_paragraph_id", ""), f"{label}.generated_paragraph_id", allow_empty=True),
    )


def trace_from_dict(data: dict[str, object]) -> ReconstructionTrace:
    """Build a validated trace from its JSON representation."""
    if not isinstance(data, dict):
        raise ValueError("Reconstruction trace must be an object")
    schema_version = data.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in {1, SCHEMA_VERSION}
    ):
        raise ValueError("Unsupported reconstruction schema version")
    expected_algorithm = (
        _LEGACY_ALGORITHM_VERSION if schema_version == 1 else ALGORITHM_VERSION
    )
    if data.get("algorithm_version") != expected_algorithm:
        raise ValueError("Unsupported reconstruction algorithm version")

    baseline = _require_object(data.get("baseline"), "baseline")
    target = _require_object(data.get("target"), "target")
    return ReconstructionTrace(
        schema_version=schema_version,
        algorithm_version=expected_algorithm,
        baseline=DocumentTraceRef(
            _require_string(baseline.get("doc_id"), "baseline.doc_id"),
            _require_string(baseline.get("file_hash"), "baseline.file_hash"),
        ),
        target=DocumentTraceRef(
            _require_string(target.get("doc_id"), "target.doc_id"),
            _require_string(target.get("file_hash"), "target.file_hash"),
        ),
        decisions=[_parse_decision(item, f"decisions[{index}]") for index, item in enumerate(_require_list(data.get("decisions"), "decisions"))],
        operations=[_parse_operation(item, f"operations[{index}]") for index, item in enumerate(_require_list(data.get("operations"), "operations"))],
    )


def validate_trace_documents(trace: ReconstructionTrace, baseline_ir: DocumentIR, target_ir: DocumentIR) -> None:
    """Ensure trace provenance matches both inputs before replaying operations."""
    for side, expected, document in (
        ("baseline", trace.baseline, baseline_ir),
        ("target", trace.target, target_ir),
    ):
        if expected.doc_id != document.doc_id:
            raise ValueError(f"{side} document id does not match reconstruction trace")
        if expected.file_hash != document.file_hash:
            raise ValueError(f"{side} file hash does not match reconstruction trace")


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _write_temp_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def write_json_atomic(path: Path, payload: object) -> None:
    """Durably replace one JSON artifact without exposing a partial file."""
    temp_path: Path | None = None
    try:
        temp_path = _write_temp_bytes(path, _json_bytes(payload))
        temp_path.replace(path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
