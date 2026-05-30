"""Tests for the local SQLite LangGraph checkpointer."""
from __future__ import annotations

import sqlite3

from app.core.types import Chunk, ChunkHit


def test_sqlite_checkpointer_persists_checkpoint_between_instances():
    """A checkpoint saved by one saver instance can be loaded by another."""
    from app.agent.sqlite_checkpointer import SQLiteCheckpointSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    serde = JsonPlusSerializer(allowed_msgpack_modules=[Chunk, ChunkHit])
    saver = SQLiteCheckpointSaver(serde=serde)
    config = {"configurable": {"thread_id": "qa-session-1", "checkpoint_ns": "", "conn": conn}}
    checkpoint = {
        "v": 1,
        "id": "00000000000000000000000000000001",
        "ts": "2026-05-30T00:00:00+00:00",
        "channel_values": {"messages": ["hello"], "status": "answered"},
        "channel_versions": {"messages": "1", "status": "1"},
        "versions_seen": {},
        "pending_sends": [],
    }

    saved_config = saver.put(
        config,
        checkpoint,
        {"source": "unit-test"},
        {"messages": "1", "status": "1"},
    )
    saver.put_writes(saved_config, [("messages", ["assistant reply"])], "task-1")

    reloaded = SQLiteCheckpointSaver(serde=serde)
    restored = reloaded.get_tuple(
        {"configurable": {"thread_id": "qa-session-1", "checkpoint_ns": "", "conn": conn}}
    )

    assert restored is not None
    assert restored.checkpoint["channel_values"]["messages"] == ["hello"]
    assert restored.metadata["source"] == "unit-test"
    assert restored.pending_writes == [("task-1", "messages", ["assistant reply"])]
    conn.close()
