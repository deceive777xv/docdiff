from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from app.core.diff.reconstruction_trace import (
    ALGORITHM_VERSION,
    SCHEMA_VERSION,
    DocumentTraceRef,
    LLMJudgment,
    ReconstructionDecision,
    ReconstructionOperation,
    ReconstructionTrace,
    SourceRowRef,
    load_reconstruction_trace,
    persist_compare_artifacts,
    trace_from_dict,
    trace_to_dict,
    validate_trace_documents,
    write_json_atomic,
)
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
                llm=LLMJudgment("test-model", "merge", 0.9, "continued table"),
                final_action="merge",
                generated_row_id="row-1",
            )
        ],
        operations=[
            ReconstructionOperation("operation-1", "baseline", "project_columns", source_rows, column_mapping={1: 0}),
            ReconstructionOperation("operation-2", "baseline", "drop_boundary_rows", source_rows),
            ReconstructionOperation("operation-3", "target", "drop_boundary_paragraphs", source_paragraph_ids=["paragraph-2"]),
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
    restored = load_reconstruction_trace(path)

    assert restored == trace
    assert restored.decisions[0].column_mapping == {1: 0, 3: 1}
    assert restored.algorithm_version == "cross-page-table-v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("algorithm_version", "unknown-version"),
    ],
)
def test_trace_rejects_unsupported_versions(field, value):
    payload = trace_to_dict(make_trace_with_every_operation())
    payload[field] = value

    with pytest.raises(ValueError, match="Unsupported reconstruction"):
        trace_from_dict(payload)


def test_trace_rejects_document_identity_mismatch():
    trace = make_trace_with_every_operation()
    baseline_ir, target_ir = make_document_pair()
    target_ir.file_hash = "different-hash"

    with pytest.raises(ValueError, match="target file hash"):
        validate_trace_documents(trace, baseline_ir, target_ir)


def test_persist_compare_artifacts_writes_existing_result_shape_and_sidecar(tmp_path):
    result_path, trace_path = persist_compare_artifacts(
        tmp_path,
        "task-1",
        [make_diff_item()],
        make_trace_with_every_operation(),
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == [asdict(make_diff_item())]
    assert trace_path.name == "task-1.reconstruction.json"
    assert load_reconstruction_trace(trace_path) == make_trace_with_every_operation()


def test_persist_compare_artifacts_stages_both_files_before_first_replace(tmp_path, monkeypatch):
    observed_temp_files: list[tuple[bool, bool]] = []
    original_replace = Path.replace

    def recording_replace(path: Path, target: Path):
        exports = tmp_path / "exports"
        observed_temp_files.append(
            (
                any(exports.glob("task-1.json.*.tmp")),
                any(exports.glob("task-1.reconstruction.json.*.tmp")),
            )
        )
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    persist_compare_artifacts(tmp_path, "task-1", [make_diff_item()], make_trace_with_every_operation())

    assert observed_temp_files[0] == (True, True)


def test_persist_compare_artifacts_raises_when_result_replace_fails(tmp_path, monkeypatch):
    original_replace = Path.replace

    def failing_result_replace(path: Path, target: Path):
        if target.name == "task-1.json":
            raise OSError("result replacement failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", failing_result_replace)

    with pytest.raises(OSError, match="result replacement failed"):
        persist_compare_artifacts(tmp_path, "task-1", [make_diff_item()], make_trace_with_every_operation())

    exports = tmp_path / "exports"
    assert not any(exports.glob("*.tmp"))
