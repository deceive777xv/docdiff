"""Tests for async generate_answer and streaming graph behavior."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage


def _make_lc_model(response_text: str = "流式回答"):
    async def _astream(messages):
        yield AIMessageChunk(content=response_text)

    model = MagicMock()
    model.astream = _astream
    return model


def _make_config(lc_model=None, conn=None):
    return {
        "configurable": {
            "thread_id": "t1",
            "conn": conn or MagicMock(),
            "embedder": MagicMock(),
            "lc_model": lc_model,
        }
    }


# ── generate_answer unit tests ─────────────────────────────────────────────────

def test_generate_answer_streams_tokens_from_lc_model():
    from app.agent.qa_graph import generate_answer

    mock_hit = MagicMock()
    mock_hit.chunk.section_path = "第一章"
    mock_hit.chunk.page_no = 1
    mock_hit.chunk.text = "付款周期为三十天。"

    state = {
        "_hits": [mock_hit],
        "messages": [HumanMessage(content="付款周期是多少天？")],
    }
    config = _make_config(lc_model=_make_lc_model("三十天"))

    result = asyncio.run(generate_answer(state, config))
    assert result["answer"] == "三十天"
    assert result["status"] == "answered"


def test_generate_answer_appends_ai_message_to_messages():
    from app.agent.qa_graph import generate_answer

    mock_hit = MagicMock()
    mock_hit.chunk.text = "内容"
    mock_hit.chunk.section_path = ""
    mock_hit.chunk.page_no = 0

    state = {
        "_hits": [mock_hit],
        "messages": [HumanMessage(content="问题")],
    }
    config = _make_config(lc_model=_make_lc_model("答案"))

    result = asyncio.run(generate_answer(state, config))
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "答案"


def test_generate_answer_no_lc_model_returns_prompt():
    from app.agent.qa_graph import generate_answer

    mock_hit = MagicMock()
    mock_hit.chunk.text = "内容"
    mock_hit.chunk.section_path = ""
    mock_hit.chunk.page_no = 0

    state = {"_hits": [mock_hit], "messages": []}
    config = _make_config(lc_model=None)

    result = asyncio.run(generate_answer(state, config))
    assert "配置" in result["answer"]
    assert result["status"] == "answered"


def test_generate_answer_no_hits_returns_not_found():
    from app.agent.qa_graph import generate_answer

    state = {"_hits": [], "messages": [HumanMessage(content="问题")]}
    config = _make_config(lc_model=None)

    result = asyncio.run(generate_answer(state, config))
    assert "未找到" in result["answer"]
    assert result["status"] == "answered"


def test_generate_answer_uses_compare_context_without_chunk_hits():
    """Compare-result context should be enough to answer difference questions."""
    from app.agent.qa_graph import generate_answer

    sent_messages = []

    async def capture_astream(messages):
        sent_messages.extend(messages)
        yield AIMessageChunk(content="两者差异包括付款周期变化。")

    model = MagicMock()
    model.astream = capture_astream
    state = {
        "_hits": [],
        "_compare_context": "对比结果：差异总数：1\n[1] 章节：付款条款，类型：实质修改，目标：付款周期60天。",
        "messages": [HumanMessage(content="两者有什么差异？")],
    }

    result = asyncio.run(generate_answer(state, _make_config(lc_model=model)))

    assert result["answer"] == "两者差异包括付款周期变化。"
    assert "付款周期60天" in sent_messages[0].content


def test_generate_answer_truncates_history_to_6_messages():
    """Only last 6 messages are sent to model, not the full history."""
    from app.agent.qa_graph import generate_answer

    sent_messages = []

    async def capture_astream(messages):
        sent_messages.extend(messages)
        yield AIMessageChunk(content="ok")

    model = MagicMock()
    model.astream = capture_astream

    mock_hit = MagicMock()
    mock_hit.chunk.text = "内容"
    mock_hit.chunk.section_path = ""
    mock_hit.chunk.page_no = 0

    history = [HumanMessage(content=f"问题{i}") for i in range(10)]
    state = {"_hits": [mock_hit], "messages": history}
    config = _make_config(lc_model=model)

    asyncio.run(generate_answer(state, config))
    # messages_to_send = [system_msg] + last 6 history messages
    # so total = 1 system + 6 history = 7
    assert len(sent_messages) == 7


def test_generate_answer_applies_context_budget_before_streaming():
    """Very large compare/retrieval context should be clipped before calling the model."""
    from app.agent.qa_graph import generate_answer

    sent_messages = []

    async def capture_astream(messages):
        sent_messages.extend(messages)
        yield AIMessageChunk(content="ok")

    model = MagicMock()
    model.astream = capture_astream

    mock_hit = MagicMock()
    mock_hit.chunk.text = "检索片段" * 600
    mock_hit.chunk.section_path = "超长章节"
    mock_hit.chunk.page_no = 0

    state = {
        "_hits": [mock_hit],
        "_compare_context": "对比结果：" + ("差异内容" * 600),
        "messages": [HumanMessage(content="总结差异")],
    }
    config = _make_config(lc_model=model)
    config["configurable"]["qa_context_char_budget"] = 500

    result = asyncio.run(generate_answer(state, config))

    assert result["answer"] == "ok"
    system_content = sent_messages[0].content
    assert len(system_content) < 1200
    assert "已截断" in system_content
    assert "检索片段" in system_content


def test_generate_answer_trims_history_content_by_budget():
    """Long persisted history should not crowd out the latest question."""
    from app.agent.qa_graph import generate_answer

    sent_messages = []

    async def capture_astream(messages):
        sent_messages.extend(messages)
        yield AIMessageChunk(content="ok")

    model = MagicMock()
    model.astream = capture_astream

    mock_hit = MagicMock()
    mock_hit.chunk.text = "资料"
    mock_hit.chunk.section_path = ""
    mock_hit.chunk.page_no = 0

    history = [HumanMessage(content=f"旧问题{i}" + "旧" * 500) for i in range(5)]
    history.append(HumanMessage(content="当前问题：" + "新" * 500))
    state = {"_hits": [mock_hit], "messages": history}
    config = _make_config(lc_model=model)
    config["configurable"]["qa_history_char_budget"] = 180
    config["configurable"]["qa_history_message_char_limit"] = 80

    asyncio.run(generate_answer(state, config))

    sent_history = sent_messages[1:]
    assert sent_history[-1].content.startswith("当前问题")
    assert len(sent_history[-1].content) <= 90
    assert sum(len(message.content) for message in sent_history) <= 220


# ── resolve_scope with new config signature ────────────────────────────────────

def test_resolve_scope_reads_conn_from_config():
    import sqlite3
    from app.agent.qa_graph import resolve_scope

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE documents (id TEXT PRIMARY KEY, doc_name TEXT,
        doc_type TEXT, file_path TEXT, file_hash TEXT, source_type TEXT,
        business_category TEXT, created_at TEXT, updated_at TEXT)""")
    conn.execute("""CREATE TABLE document_versions (id TEXT PRIMARY KEY,
        document_id TEXT, version_no INTEGER, version_label TEXT, status TEXT,
        parsed_json_path TEXT, summary TEXT, created_at TEXT)""")
    conn.commit()

    state = {"scope": "current_doc", "current_version_ids": ["v1"]}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert result["_version_ids"] == ["v1"]
    conn.close()
