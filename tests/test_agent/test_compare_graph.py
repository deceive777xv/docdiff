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


def test_compare_graph_matches_imported_irs_and_persists_result_only(base_state, tmp_path):
    """Compare consumes import-normalized IRs without a reconstruction stage."""
    from app.agent.compare_graph import compare_graph
    from app.core.types import DiffResult, DocumentIR

    mock_ir = DocumentIR(doc_id="d1", title="T", file_hash="h", sections=[], plain_text="")
    mock_result = DiffResult(task_id="task-001", baseline_version_id="ver-1", target_version_id="ver-2", items=[])
    initial_plan = object()
    calls: list[str] = []

    def record_align(*args, **kwargs):
        calls.append("align")
        return initial_plan

    def record_match(plan, *args, **kwargs):
        calls.append("match")
        assert plan is initial_plan
        assert kwargs["baseline_document_title"] == "T"
        assert kwargs["target_document_title"] == "T"
        return []

    def record_persist(data_dir, task_id, items):
        calls.append("persist")
        return Path(data_dir) / "exports" / f"{task_id}.json"

    base_state["data_dir"] = str(tmp_path)

    with (
        patch("app.agent.compare_graph.compare_repo.create_compare_task", return_value="task-001"),
        patch("app.agent.compare_graph.compare_repo.update_task_status"),
        patch("app.agent.compare_graph.compare_repo.insert_diff_items"),
        patch("app.agent.compare_graph._load_ir", return_value=mock_ir),
        patch("app.agent.compare_graph.align_compare_scopes", side_effect=record_align),
        patch("app.agent.compare_graph.match_paragraphs", side_effect=record_match),
        patch("app.agent.compare_graph.classify", return_value=mock_result),
        patch("app.agent.compare_graph.persist_compare_result", side_effect=record_persist),
    ):
        result = compare_graph.invoke(base_state)

    assert calls == ["align", "match", "persist"]
    assert result.get("error") is None
    assert result["task_id"] == "task-001"
    assert result["status"] == "completed"
    assert "_reconstruction_trace" not in result


def test_graph_marks_failed_when_result_publish_fails(base_state, tmp_path):
    """A result publish failure must never be recorded as completed."""
    from app.agent.compare_graph import compare_graph
    from app.core.types import DiffResult, DocumentIR

    mock_ir = DocumentIR(doc_id="d1", title="T", file_hash="h", sections=[], plain_text="")
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
        patch("app.agent.compare_graph.align_compare_scopes", return_value=object()),
        patch("app.agent.compare_graph.match_paragraphs", return_value=[]),
        patch("app.agent.compare_graph.classify", return_value=mock_result),
        patch(
            "app.agent.compare_graph.persist_compare_result",
            side_effect=OSError("result write failed"),
        ),
    ):
        result = compare_graph.invoke(base_state)

    assert result["status"] == "failed"
    assert result["error"] == "result write failed"
    assert statuses[-1] == "failed"
    assert "completed" not in statuses
