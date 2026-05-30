"""Tests for persisted QA sessions and messages."""
from __future__ import annotations

import sqlite3

import pytest

from app.db.schema import DDL


@pytest.fixture()
def conn():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(DDL)
    db.commit()
    yield db
    db.close()


def test_qa_session_lifecycle(conn):
    """QA sessions keep metadata, messages, and can be deleted as a unit."""
    from app.db import qa_repo

    session_id = qa_repo.create_session(
        conn,
        title="两份合同差异",
        scope="compare",
        current_version_ids=["baseline-v1", "target-v1"],
        compare_task_id="task-1",
    )
    qa_repo.add_message(conn, session_id, "user", "两者有什么差异？")
    qa_repo.add_message(conn, session_id, "assistant", "主要差异是付款周期。")

    sessions = qa_repo.list_sessions(conn)
    messages = qa_repo.list_messages(conn, session_id)

    assert sessions[0]["id"] == session_id
    assert sessions[0]["title"] == "两份合同差异"
    assert sessions[0]["scope"] == "compare"
    assert qa_repo.decode_version_ids(sessions[0]["current_version_ids_json"]) == [
        "baseline-v1",
        "target-v1",
    ]
    assert [row["role"] for row in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "主要差异是付款周期。"

    qa_repo.delete_session(conn, session_id)

    assert qa_repo.get_session(conn, session_id) is None
    assert qa_repo.list_messages(conn, session_id) == []


def test_qa_repo_updates_session_context_and_title_from_question(conn):
    """Existing sessions can be retargeted when the user changes QA scope."""
    from app.db import qa_repo

    session_id = qa_repo.create_session(conn, title="", scope="all")

    qa_repo.update_session(
        conn,
        session_id,
        title="这份制度讲了什么？",
        scope="current_doc",
        current_version_ids=["doc-v1"],
        compare_task_id=None,
    )

    row = qa_repo.get_session(conn, session_id)
    assert row["title"] == "这份制度讲了什么？"
    assert row["scope"] == "current_doc"
    assert qa_repo.decode_version_ids(row["current_version_ids_json"]) == ["doc-v1"]
