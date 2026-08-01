from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from app.core.normalization.table_trace import (
    ALGORITHM_VERSION,
    SCHEMA_VERSION,
    DocumentTraceRef,
    LLMJudgment,
    ReconstructionDecision,
    ReconstructionOperation,
    ReconstructionTrace,
    SourceRowRef,
    trace_from_dict,
    trace_to_dict,
    validate_trace_documents,
    write_json_atomic,
)
from app.core.diff.result_storage import persist_compare_result
from app.core.types import DiffItem, DocumentIR


def make_trace_with_every_operation() -> ReconstructionTrace:
    source_rows = [SourceRowRef("section-1", "paragraph-1", 0)]
    return ReconstructionTrace(
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        baseline=DocumentTraceRef("baseline-doc", "baseline-hash"),
        target=DocumentTraceRef("target-doc", "target-hash"),
        decisions=[
            ReconstructionDecision(
                candidate_id="candidate-1",
                side="baseline",
                source_rows=source_rows,
                column_mapping={1: 0, 3: 1},
                rule_confidence="high",
                rule_evidence=["same table header"],
                rule_conflicts=[],
                llm=LLMJudgment(
                    "test-model",
                    "merge",
                    0.9,
                    "continued table",
                    roles={
                        "previous_row": "body_row",
                        "continuation_row": "continuation_row",
                    },
                    row_action="merge",
                    table_action="merge_fragments",
                    mapping_id="candidate-1:mapping:0",
                ),
                final_action="merge",
                boundary_id="boundary-1",
                previous_page_no=4,
                next_page_no=5,
                context_refs=["item-1", "item-2"],
                generated_row_id="row-1",
                review=LLMJudgment(
                    "review-model",
                    "merge",
                    0.88,
                    "independent review confirmed the continuation",
                    row_action="merge",
                    table_action="merge_fragments",
                ),
            )
        ],
        operations=[
            ReconstructionOperation("operation-1", "baseline", "project_columns", source_rows, column_mapping={1: 0}),
            ReconstructionOperation("operation-2", "baseline", "drop_boundary_rows", source_rows),
            ReconstructionOperation("operation-3", "target", "drop_boundary_paragraphs", source_paragraph_ids=["paragraph-2"]),
            ReconstructionOperation(
                "operation-header",
                "baseline",
                "drop_repeated_table_header",
                [
                    SourceRowRef("section-1", "paragraph-1", 0),
                    SourceRowRef("section-1", "paragraph-2", 0),
                ],
                decision_id="candidate-1",
            ),
            ReconstructionOperation("operation-4", "target", "merge_rows", source_rows, decision_id="candidate-1", generated_row_id="row-1"),
            ReconstructionOperation("operation-5", "target", "merge_fragments", source_rows, generated_paragraph_id="paragraph-3"),
        ],
    )


def make_document_pair() -> tuple[DocumentIR, DocumentIR]:
    return (
        DocumentIR("baseline-doc", "Baseline", "baseline-hash"),
        DocumentIR("target-doc", "Target", "target-hash"),
    )


def make_diff_item() -> DiffItem:
    return DiffItem(
        diff_id="diff-1",
        section_path="Section 1",
        diff_type="微调",
        risk_level="low",
        baseline_text="baseline",
        target_text="target",
        similarity_score=0.8,
        explanation="example",
    )


def test_trace_round_trip_preserves_typed_mappings_and_llm_judgment(tmp_path):
    trace = make_trace_with_every_operation()

    path = tmp_path / "task.reconstruction.json"
    write_json_atomic(path, trace_to_dict(trace))
    restored = trace_from_dict(json.loads(path.read_text(encoding="utf-8")))

    assert restored == trace
    assert restored.decisions[0].column_mapping == {1: 0, 3: 1}
    assert restored.algorithm_version == "cross-page-table-v4"
    assert restored.decisions[0].previous_page_no == 4
    assert restored.decisions[0].llm.roles["continuation_row"] == "continuation_row"
    assert restored.decisions[0].llm.mapping_id == "candidate-1:mapping:0"
    assert restored.decisions[0].review.model == "review-model"


def test_trace_dict_round_trip_preserves_mapping_id():
    restored = trace_from_dict(trace_to_dict(make_trace_with_every_operation()))

    assert restored.decisions[0].llm.mapping_id == "candidate-1:mapping:0"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", SCHEMA_VERSION + 1),
        ("schema_version", True),
        ("algorithm_version", "unknown-version"),
    ],
)
def test_trace_rejects_unsupported_versions(field, value):
    payload = trace_to_dict(make_trace_with_every_operation())
    payload[field] = value

    with pytest.raises(ValueError, match="Unsupported reconstruction"):
        trace_from_dict(payload)


def test_trace_loader_accepts_legacy_v1_payload():
    payload = trace_to_dict(make_trace_with_every_operation())
    payload["schema_version"] = 1
    payload["algorithm_version"] = "cross-page-table-v1"
    for decision in payload["decisions"]:
        decision.pop("boundary_id", None)
        decision.pop("previous_page_no", None)
        decision.pop("next_page_no", None)
        decision.pop("context_refs", None)
        decision.pop("review", None)
        if decision["llm"] is not None:
            decision["llm"].pop("mapping_id", None)
            decision["llm"].pop("roles", None)
            decision["llm"].pop("row_action", None)
            decision["llm"].pop("table_action", None)

    restored = trace_from_dict(payload)

    assert restored.schema_version == 1
    assert restored.algorithm_version == "cross-page-table-v1"
    assert restored.decisions[0].boundary_id == ""
    assert restored.decisions[0].llm.roles == {}
    assert restored.decisions[0].llm.mapping_id == ""


def test_trace_rejects_document_identity_mismatch():
    trace = make_trace_with_every_operation()
    baseline_ir, target_ir = make_document_pair()
    target_ir.file_hash = "different-hash"

    with pytest.raises(ValueError, match="target file hash"):
        validate_trace_documents(trace, baseline_ir, target_ir)


def test_persist_compare_result_writes_existing_result_shape_without_sidecar(tmp_path):
    result_path = persist_compare_result(tmp_path, "task-1", [make_diff_item()])

    assert json.loads(result_path.read_text(encoding="utf-8")) == [asdict(make_diff_item())]
    assert not list((tmp_path / "exports").glob("*.reconstruction.json"))


def test_persist_compare_result_cleans_temp_file_when_replace_fails(tmp_path, monkeypatch):
    original_replace = Path.replace

    def failing_result_replace(path: Path, target: Path):
        if target.name == "task-1.json":
            raise OSError("result replacement failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", failing_result_replace)

    with pytest.raises(OSError, match="result replacement failed"):
        persist_compare_result(tmp_path, "task-1", [make_diff_item()])

    exports = tmp_path / "exports"
    assert not any(exports.glob("*.tmp"))
