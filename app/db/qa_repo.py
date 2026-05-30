"""Persistence helpers for QA sessions and chat messages."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Iterable


def _now() -> str:
    return datetime.now().isoformat()


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS qa_sessions (
            id                         TEXT PRIMARY KEY,
            title                      TEXT NOT NULL,
            scope                      TEXT NOT NULL,
            current_version_ids_json   TEXT NOT NULL DEFAULT '[]',
            compare_task_id            TEXT,
            created_at                 TEXT NOT NULL,
            updated_at                 TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS qa_messages (
            id              TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL REFERENCES qa_sessions(id) ON DELETE CASCADE,
            role            TEXT NOT NULL CHECK(role IN ('user','assistant')),
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL
        );
        """
    )


def encode_version_ids(version_ids: Iterable[str] | None) -> str:
    return json.dumps(list(version_ids or []), ensure_ascii=False)


def decode_version_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def title_from_question(question: str, limit: int = 28) -> str:
    title = " ".join((question or "").split())
    return title[:limit] if title else "新会话"


def create_session(
    conn: sqlite3.Connection,
    *,
    title: str,
    scope: str,
    current_version_ids: Iterable[str] | None = None,
    compare_task_id: str | None = None,
    session_id: str | None = None,
) -> str:
    _ensure_tables(conn)
    sid = session_id or str(uuid.uuid4())
    now = _now()
    conn.execute(
        """INSERT INTO qa_sessions
           (id, title, scope, current_version_ids_json, compare_task_id, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            sid,
            title or "新会话",
            scope,
            encode_version_ids(current_version_ids),
            compare_task_id,
            now,
            now,
        ),
    )
    conn.commit()
    return sid


def get_session(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    _ensure_tables(conn)
    return conn.execute("SELECT * FROM qa_sessions WHERE id = ?", (session_id,)).fetchone()


def list_sessions(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    _ensure_tables(conn)
    return conn.execute(
        "SELECT * FROM qa_sessions ORDER BY updated_at DESC, created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()


def update_session(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    title: str | None = None,
    scope: str | None = None,
    current_version_ids: Iterable[str] | None = None,
    compare_task_id: str | None = None,
) -> None:
    _ensure_tables(conn)
    existing = get_session(conn, session_id)
    if existing is None:
        raise ValueError(f"QA session not found: {session_id}")
    conn.execute(
        """UPDATE qa_sessions
           SET title = ?, scope = ?, current_version_ids_json = ?,
               compare_task_id = ?, updated_at = ?
           WHERE id = ?""",
        (
            title if title is not None else existing["title"],
            scope if scope is not None else existing["scope"],
            (
                encode_version_ids(current_version_ids)
                if current_version_ids is not None
                else existing["current_version_ids_json"]
            ),
            compare_task_id,
            _now(),
            session_id,
        ),
    )
    conn.commit()


def add_message(conn: sqlite3.Connection, session_id: str, role: str, content: str) -> str:
    _ensure_tables(conn)
    if role not in {"user", "assistant"}:
        raise ValueError(f"Unsupported QA message role: {role}")
    message_id = str(uuid.uuid4())
    now = _now()
    conn.execute(
        """INSERT INTO qa_messages (id, session_id, role, content, created_at)
           VALUES (?,?,?,?,?)""",
        (message_id, session_id, role, content, now),
    )
    conn.execute(
        "UPDATE qa_sessions SET updated_at = ? WHERE id = ?",
        (now, session_id),
    )
    conn.commit()
    return message_id


def list_messages(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    _ensure_tables(conn)
    return conn.execute(
        "SELECT * FROM qa_messages WHERE session_id = ? ORDER BY rowid",
        (session_id,),
    ).fetchall()


def delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    _ensure_tables(conn)
    conn.execute("DELETE FROM qa_messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM qa_sessions WHERE id = ?", (session_id,))
    for table in ("qa_checkpoint_writes", "qa_checkpoint_blobs", "qa_checkpoints"):
        conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (session_id,))
    conn.commit()
