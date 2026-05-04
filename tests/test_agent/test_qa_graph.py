"""Tests for qa_graph LangGraph workflow."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def base_state():
    mock_provider = MagicMock()
    mock_provider.chat.return_value = "答案是X。"
    return {
        "data_dir": "/tmp",
        "question": "付款周期是多少天？",
        "scope": "current_doc",
        "current_version_ids": ["ver-1"],
        "provider": mock_provider,
        "embedder": MagicMock(),
        "conn": MagicMock(),
    }


def test_resolve_scope_current_doc(base_state):
    from app.agent.qa_graph import resolve_scope
    result = resolve_scope(base_state)
    assert result["_version_ids"] == ["ver-1"]
    assert result.get("error") is None


def test_resolve_scope_empty_current_doc_returns_error(base_state):
    from app.agent.qa_graph import resolve_scope
    base_state["current_version_ids"] = []
    result = resolve_scope(base_state)
    assert result.get("error") is not None
    assert result.get("status") == "failed"


def test_graph_happy_path(base_state):
    from app.agent.qa_graph import qa_graph

    mock_hit = MagicMock()
    mock_hit.chunk.section_path = "第一章"
    mock_hit.chunk.page_no = 1
    mock_hit.chunk.text = "付款周期为30天。"

    with patch("app.agent.qa_graph.search", return_value=[mock_hit]):
        result = qa_graph.invoke(base_state)

    assert result.get("error") is None
    assert result["answer"] == "答案是X。"
    assert len(result["citations"]) == 1
    assert result["status"] == "completed"


def test_graph_no_hits_returns_default_message(base_state):
    from app.agent.qa_graph import qa_graph

    with patch("app.agent.qa_graph.search", return_value=[]):
        result = qa_graph.invoke(base_state)

    assert result.get("error") is None
    assert "未找到" in result["answer"]
    assert result["citations"] == []


# ---------------------------------------------------------------------------
# Helpers and tests for compare scope (Task 4)
# ---------------------------------------------------------------------------

def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE documents (
            id TEXT PRIMARY KEY, doc_name TEXT, doc_type TEXT,
            file_path TEXT, file_hash TEXT, source_type TEXT,
            business_category TEXT, created_at TEXT, updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE document_versions (
            id TEXT PRIMARY KEY, document_id TEXT, version_no INTEGER,
            version_label TEXT, status TEXT, parsed_json_path TEXT,
            summary TEXT, created_at TEXT
        )
    """)
    conn.commit()
    return conn


def test_resolve_scope_compare_returns_provided_version_ids():
    """scope='compare' with current_version_ids → _version_ids equals input list."""
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {
        "scope": "compare",
        "current_version_ids": ["baseline-v1", "target-v1"],
        "conn": conn,
    }
    result = resolve_scope(state)
    assert set(result["_version_ids"]) == {"baseline-v1", "target-v1"}
    conn.close()


def test_resolve_scope_compare_error_when_no_ids():
    """scope='compare' with empty current_version_ids → error."""
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {
        "scope": "compare",
        "current_version_ids": [],
        "conn": conn,
    }
    result = resolve_scope(state)
    assert "error" in result
    conn.close()


def test_resolve_scope_current_doc_unchanged():
    """Existing 'current_doc' scope still works after the compare branch is added."""
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {
        "scope": "current_doc",
        "current_version_ids": ["v-abc"],
        "conn": conn,
    }
    result = resolve_scope(state)
    assert result["_version_ids"] == ["v-abc"]
    conn.close()
