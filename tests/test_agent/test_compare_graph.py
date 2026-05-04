"""Tests for compare_graph LangGraph workflow."""
from __future__ import annotations

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


def test_graph_happy_path(base_state, tmp_path):
    """Full happy path: both IRs load, align, compare, classify, persist."""
    from app.agent.compare_graph import compare_graph
    from app.core.types import DiffResult, DocumentIR

    mock_ir = DocumentIR(doc_id="d1", title="T", file_hash="h", sections=[], plain_text="")
    mock_result = DiffResult(task_id="task-001", baseline_version_id="ver-1", target_version_id="ver-2", items=[])
    base_state["data_dir"] = str(tmp_path)

    with (
        patch("app.agent.compare_graph.compare_repo.create_compare_task", return_value="task-001"),
        patch("app.agent.compare_graph.compare_repo.update_task_status"),
        patch("app.agent.compare_graph.compare_repo.insert_diff_items"),
        patch("app.agent.compare_graph._load_ir", return_value=mock_ir),
        patch("app.agent.compare_graph.align_sections", return_value=[]),
        patch("app.agent.compare_graph.match_paragraphs", return_value=[]),
        patch("app.agent.compare_graph.classify", return_value=mock_result),
    ):
        result = compare_graph.invoke(base_state)

    assert result.get("error") is None
    assert result["task_id"] == "task-001"
    assert result["status"] == "completed"
