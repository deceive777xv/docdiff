"""Tests for compare_graph LangGraph workflow."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def base_state():
    return {
        "data_dir": "/tmp",
        "baseline_version_id": "ver-1",
        "target_version_id": "ver-2",
        "provider": MagicMock(),
        "embedder": MagicMock(),
        "conn": MagicMock(),
    }


def test_create_task_node(base_state):
    """create_task inserts a compare_tasks record and returns task_id."""
    from app.agent.compare_graph import create_task

    with (
        patch("app.agent.compare_graph.compare_repo.create_compare_task", return_value="task-001"),
        patch("app.agent.compare_graph.compare_repo.update_task_status"),
    ):
        result = create_task(base_state)

    assert result["task_id"] == "task-001"
    assert result.get("error") is None


def test_create_task_node_reuses_existing_task(base_state):
    """Recovery runs reuse the provided task id and prepare it for rerun."""
    from app.agent.compare_graph import create_task

    base_state["task_id"] = "task-existing"
    with (
        patch("app.agent.compare_graph.compare_repo.get_task_by_id", return_value={"id": "task-existing"}),
        patch("app.agent.compare_graph.compare_repo.prepare_task_for_rerun") as prepare,
        patch("app.agent.compare_graph.compare_repo.create_compare_task") as create,
    ):
        result = create_task(base_state)

    assert result["task_id"] == "task-existing"
    prepare.assert_called_once_with(base_state["conn"], "task-existing")
    create.assert_not_called()
    assert result.get("error") is None


def test_graph_sets_error_on_missing_version(base_state):
    """Graph sets error when a version's IR file is missing."""
    from app.agent.compare_graph import compare_graph

    with (
        patch("app.agent.compare_graph.compare_repo.create_compare_task", return_value="task-err"),
        patch("app.agent.compare_graph.compare_repo.update_task_status"),
        patch("app.agent.compare_graph.document_repo.get_version_by_id", return_value=None),
    ):
        result = compare_graph.invoke(base_state)

    assert result.get("error") is not None
    assert result.get("status") == "failed"


def test_compare_graph_reconstructs_before_matching_and_persists_trace(base_state, tmp_path):
    """Full happy path: both IRs load, align, compare, classify, persist."""
    from app.agent.compare_graph import compare_graph
    from app.core.diff.reconstruction_trace import (
        ALGORITHM_VERSION,
        SCHEMA_VERSION,
        DocumentTraceRef,
        ReconstructionTrace,
    )
    from app.core.diff.table_reconstruction_pipeline import ReconstructionResult
    from app.core.types import DiffResult, DocumentIR

    mock_ir = DocumentIR(doc_id="d1", title="T", file_hash="h", sections=[], plain_text="")
    mock_result = DiffResult(task_id="task-001", baseline_version_id="ver-1", target_version_id="ver-2", items=[])
    initial_pairs = [object()]
    reconstructed_pairs = [object()]
    trace = ReconstructionTrace(
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        baseline=DocumentTraceRef("d1", "h"),
        target=DocumentTraceRef("d1", "h"),
        decisions=[],
        operations=[],
    )
    reconstruction = ReconstructionResult(mock_ir, mock_ir, reconstructed_pairs, trace)
    calls: list[str] = []

    def record_align(*args):
        calls.append("align")
        return initial_pairs

    def record_reconstruct(*args, **kwargs):
        calls.append("reconstruct")
        assert args == (mock_ir, mock_ir)
        assert kwargs["section_pairs"] is initial_pairs
        return reconstruction

    def record_match(pairs, *args, **kwargs):
        calls.append("match")
        assert pairs is reconstructed_pairs
        return []

    def record_persist(data_dir, task_id, items, persisted_trace):
        calls.append("persist")
        assert persisted_trace is trace
        exports_dir = Path(data_dir) / "exports"
        return (
            exports_dir / f"{task_id}.json",
            exports_dir / f"{task_id}.reconstruction.json",
        )

    base_state["data_dir"] = str(tmp_path)

    with (
        patch("app.agent.compare_graph.compare_repo.create_compare_task", return_value="task-001"),
        patch("app.agent.compare_graph.compare_repo.update_task_status"),
        patch("app.agent.compare_graph.compare_repo.insert_diff_items"),
        patch("app.agent.compare_graph._load_ir", return_value=mock_ir),
        patch("app.agent.compare_graph.align_sections", side_effect=record_align),
        patch("app.agent.compare_graph.normalize_pair", side_effect=record_reconstruct),
        patch("app.agent.compare_graph.match_paragraphs", side_effect=record_match),
        patch("app.agent.compare_graph.classify", return_value=mock_result),
        patch("app.agent.compare_graph.persist_compare_artifacts", side_effect=record_persist),
    ):
        result = compare_graph.invoke(base_state)

    assert calls == ["align", "reconstruct", "match", "persist"]
    assert result.get("error") is None
    assert result["task_id"] == "task-001"
    assert result["status"] == "completed"
    assert result["_reconstruction_trace"] is trace


def test_graph_marks_failed_when_sidecar_publish_fails(base_state, tmp_path):
    """A dual-artifact publish failure must never be recorded as completed."""
    from app.agent.compare_graph import compare_graph
    from app.core.diff.reconstruction_trace import (
        ALGORITHM_VERSION,
        SCHEMA_VERSION,
        DocumentTraceRef,
        ReconstructionTrace,
    )
    from app.core.diff.table_reconstruction_pipeline import ReconstructionResult
    from app.core.types import DiffResult, DocumentIR

    mock_ir = DocumentIR(doc_id="d1", title="T", file_hash="h", sections=[], plain_text="")
    trace = ReconstructionTrace(
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        baseline=DocumentTraceRef("d1", "h"),
        target=DocumentTraceRef("d1", "h"),
        decisions=[],
        operations=[],
    )
    reconstruction = ReconstructionResult(mock_ir, mock_ir, [], trace)
    mock_result = DiffResult("task-001", "ver-1", "ver-2", [])
    statuses: list[str] = []
    base_state["data_dir"] = str(tmp_path)

    with (
        patch("app.agent.compare_graph.compare_repo.create_compare_task", return_value="task-001"),
        patch(
            "app.agent.compare_graph.compare_repo.update_task_status",
            side_effect=lambda conn, task_id, status, *args: statuses.append(status),
        ),
        patch("app.agent.compare_graph.compare_repo.insert_diff_items"),
        patch("app.agent.compare_graph._load_ir", return_value=mock_ir),
        patch("app.agent.compare_graph.align_sections", return_value=[]),
        patch("app.agent.compare_graph.normalize_pair", return_value=reconstruction),
        patch("app.agent.compare_graph.match_paragraphs", return_value=[]),
        patch("app.agent.compare_graph.classify", return_value=mock_result),
        patch(
            "app.agent.compare_graph.persist_compare_artifacts",
            side_effect=OSError("sidecar write failed"),
        ),
    ):
        result = compare_graph.invoke(base_state)

    assert result["status"] == "failed"
    assert result["error"] == "sidecar write failed"
    assert statuses[-1] == "failed"
    assert "completed" not in statuses
