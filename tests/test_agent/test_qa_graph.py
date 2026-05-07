"""Tests for qa_graph LangGraph workflow."""
from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import MagicMock

import pytest

from langchain_core.messages import AIMessageChunk

from app.core.types import Chunk, ChunkHit


# ── Fixtures ──────────────────────────────────────────────────────────────────

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


async def _mock_astream(messages):
    yield AIMessageChunk(content="答案是X。")


def _make_config(conn=None, embedder=None, lc_model=None, thread_id="t1"):
    mock_lc = MagicMock()
    mock_lc.astream = _mock_astream
    return {
        "configurable": {
            "thread_id": thread_id,
            "conn": conn or _make_conn(),
            "embedder": embedder or MagicMock(),
            "lc_model": lc_model if lc_model is not None else mock_lc,
        }
    }


# ── resolve_scope direct tests ─────────────────────────────────────────────────

def test_resolve_scope_current_doc():
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {"scope": "current_doc", "current_version_ids": ["ver-1"]}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert result["_version_ids"] == ["ver-1"]
    assert result.get("error") is None
    conn.close()


def test_resolve_scope_empty_current_doc_returns_error():
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {"scope": "current_doc", "current_version_ids": []}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert result.get("error") is not None
    assert result.get("status") == "failed"
    conn.close()


def test_resolve_scope_compare_returns_provided_version_ids():
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {"scope": "compare", "current_version_ids": ["baseline-v1", "target-v1"]}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert set(result["_version_ids"]) == {"baseline-v1", "target-v1"}
    conn.close()


def test_resolve_scope_compare_error_when_no_ids():
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {"scope": "compare", "current_version_ids": []}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert "error" in result
    conn.close()


def test_resolve_scope_current_doc_unchanged():
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {"scope": "current_doc", "current_version_ids": ["v-abc"]}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert result["_version_ids"] == ["v-abc"]
    conn.close()


# ── Full graph integration tests ───────────────────────────────────────────────

def test_graph_happy_path():
    from unittest.mock import patch
    from app.agent.qa_graph import qa_graph

    mock_hit = ChunkHit(
        chunk=Chunk(id="c1", version_id="v1", chunk_no=0,
                    section_path="第一章", page_no=1, text="付款周期为30天。"),
        score=0.9,
    )

    conn = _make_conn()
    config = _make_config(conn=conn, thread_id="happy-path-1")
    state_input = {
        "data_dir": "/tmp",
        "question": "付款周期是多少天？",
        "scope": "current_doc",
        "current_version_ids": ["ver-1"],
    }

    with patch("app.agent.qa_graph.search", return_value=[mock_hit]):
        result = asyncio.run(qa_graph.ainvoke(state_input, config))

    assert result.get("error") is None
    assert result["answer"] == "答案是X。"
    assert len(result["citations"]) == 1
    assert result["status"] == "completed"
    conn.close()


def test_graph_no_hits_returns_default_message():
    from unittest.mock import patch
    from app.agent.qa_graph import qa_graph

    conn = _make_conn()
    config = {"configurable": {
        "thread_id": "no-hits-1",
        "conn": conn,
        "embedder": MagicMock(),
        "lc_model": None,
    }}
    state_input = {
        "data_dir": "/tmp",
        "question": "付款周期是多少天？",
        "scope": "current_doc",
        "current_version_ids": ["ver-1"],
    }

    with patch("app.agent.qa_graph.search", return_value=[]):
        result = asyncio.run(qa_graph.ainvoke(state_input, config))

    assert result.get("error") is None
    assert "未找到" in result["answer"]
    assert result["citations"] == []
    conn.close()
