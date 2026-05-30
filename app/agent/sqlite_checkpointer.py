"""SQLite-backed LangGraph checkpointer for local QA memory."""
from __future__ import annotations

import random
import sqlite3
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import datetime
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
    get_checkpoint_metadata,
)


_DDL = """
CREATE TABLE IF NOT EXISTS qa_checkpoints (
    thread_id               TEXT NOT NULL,
    checkpoint_ns           TEXT NOT NULL DEFAULT '',
    checkpoint_id           TEXT NOT NULL,
    checkpoint_type         TEXT NOT NULL,
    checkpoint_blob         BLOB NOT NULL,
    metadata_type           TEXT NOT NULL,
    metadata_blob           BLOB NOT NULL,
    parent_checkpoint_id    TEXT,
    created_at              TEXT NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS qa_checkpoint_writes (
    thread_id        TEXT NOT NULL,
    checkpoint_ns    TEXT NOT NULL DEFAULT '',
    checkpoint_id    TEXT NOT NULL,
    task_id          TEXT NOT NULL,
    idx              INTEGER NOT NULL,
    channel          TEXT NOT NULL,
    value_type       TEXT NOT NULL,
    value_blob       BLOB NOT NULL,
    task_path        TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE TABLE IF NOT EXISTS qa_checkpoint_blobs (
    thread_id        TEXT NOT NULL,
    checkpoint_ns    TEXT NOT NULL DEFAULT '',
    channel          TEXT NOT NULL,
    version          TEXT NOT NULL,
    value_type       TEXT NOT NULL,
    value_blob       BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
"""


class SQLiteCheckpointSaver(BaseCheckpointSaver[str]):
    """Persist LangGraph checkpoints into the app's local SQLite database."""

    def _conn(self, config: RunnableConfig) -> sqlite3.Connection:
        try:
            conn = config["configurable"]["conn"]
        except KeyError as exc:
            raise ValueError("SQLiteCheckpointSaver requires configurable.conn") from exc
        self._ensure_schema(conn)
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_DDL)

    def _dump(self, value: Any) -> tuple[str, bytes]:
        typ, blob = self.serde.dumps_typed(value)
        return typ, blob

    def _load(self, typ: str, blob: bytes) -> Any:
        return self.serde.loads_typed((typ, blob))

    def _load_blobs(
        self,
        conn: sqlite3.Connection,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        channel_values: dict[str, Any] = {}
        for channel, version in versions.items():
            row = conn.execute(
                """SELECT value_type, value_blob FROM qa_checkpoint_blobs
                   WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version = ?""",
                (thread_id, checkpoint_ns, channel, str(version)),
            ).fetchone()
            if row is not None and row["value_type"] != "empty":
                channel_values[channel] = self._load(row["value_type"], row["value_blob"])
        return channel_values

    def _pending_writes(
        self,
        conn: sqlite3.Connection,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[PendingWrite]:
        rows = conn.execute(
            """SELECT task_id, channel, value_type, value_blob
               FROM qa_checkpoint_writes
               WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
               ORDER BY task_id, idx""",
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()
        return [
            (row["task_id"], row["channel"], self._load(row["value_type"], row["value_blob"]))
            for row in rows
        ]

    def _tuple_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        config: RunnableConfig | None = None,
    ) -> CheckpointTuple:
        thread_id = row["thread_id"]
        checkpoint_ns = row["checkpoint_ns"]
        checkpoint_id = row["checkpoint_id"]
        checkpoint = self._load(row["checkpoint_type"], row["checkpoint_blob"])
        return CheckpointTuple(
            config=config or {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(
                    conn,
                    thread_id,
                    checkpoint_ns,
                    checkpoint["channel_versions"],
                ),
            },
            metadata=self._load(row["metadata_type"], row["metadata_blob"]),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row["parent_checkpoint_id"],
                    }
                }
                if row["parent_checkpoint_id"]
                else None
            ),
            pending_writes=self._pending_writes(conn, thread_id, checkpoint_ns, checkpoint_id),
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        conn = self._conn(config)
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            row = conn.execute(
                """SELECT * FROM qa_checkpoints
                   WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?""",
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT * FROM qa_checkpoints
                   WHERE thread_id = ? AND checkpoint_ns = ?
                   ORDER BY checkpoint_id DESC
                   LIMIT 1""",
                (thread_id, checkpoint_ns),
            ).fetchone()
        if row is None:
            return None
        tuple_config = {
            "configurable": {
                "thread_id": row["thread_id"],
                "checkpoint_ns": row["checkpoint_ns"],
                "checkpoint_id": row["checkpoint_id"],
                "conn": conn,
            }
        }
        return self._tuple_from_row(conn, row, config=tuple_config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            raise ValueError("SQLiteCheckpointSaver.list requires configurable.conn")
        conn = self._conn(config)
        params: list[Any] = []
        where: list[str] = []
        if thread_id := config["configurable"].get("thread_id"):
            where.append("thread_id = ?")
            params.append(thread_id)
        if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
            where.append("checkpoint_ns = ?")
            params.append(checkpoint_ns)
        if checkpoint_id := get_checkpoint_id(config):
            where.append("checkpoint_id = ?")
            params.append(checkpoint_id)
        if before and (before_checkpoint_id := get_checkpoint_id(before)):
            where.append("checkpoint_id < ?")
            params.append(before_checkpoint_id)
        sql = "SELECT * FROM qa_checkpoints"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY checkpoint_id DESC"
        yielded = 0
        for row in conn.execute(sql, params).fetchall():
            item = self._tuple_from_row(conn, row)
            if filter and not all(item.metadata.get(key) == value for key, value in filter.items()):
                continue
            yield item
            yielded += 1
            if limit is not None and yielded >= limit:
                break

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        conn = self._conn(config)
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_copy = checkpoint.copy()
        values: dict[str, Any] = checkpoint_copy.pop("channel_values")
        for channel, version in new_versions.items():
            value_type, value_blob = (
                self._dump(values[channel]) if channel in values else ("empty", b"")
            )
            conn.execute(
                """INSERT OR REPLACE INTO qa_checkpoint_blobs
                   (thread_id, checkpoint_ns, channel, version, value_type, value_blob)
                   VALUES (?,?,?,?,?,?)""",
                (thread_id, checkpoint_ns, channel, str(version), value_type, value_blob),
            )
        checkpoint_type, checkpoint_blob = self._dump(checkpoint_copy)
        metadata_type, metadata_blob = self._dump(get_checkpoint_metadata(config, metadata))
        conn.execute(
            """INSERT OR REPLACE INTO qa_checkpoints
               (thread_id, checkpoint_ns, checkpoint_id, checkpoint_type, checkpoint_blob,
                metadata_type, metadata_blob, parent_checkpoint_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                thread_id,
                checkpoint_ns,
                checkpoint["id"],
                checkpoint_type,
                checkpoint_blob,
                metadata_type,
                metadata_blob,
                config["configurable"].get("checkpoint_id"),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
                "conn": conn,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        conn = self._conn(config)
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            value_type, value_blob = self._dump(value)
            sql = "INSERT OR REPLACE" if write_idx < 0 else "INSERT OR IGNORE"
            conn.execute(
                f"""{sql} INTO qa_checkpoint_writes
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                     channel, value_type, value_blob, task_path)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    write_idx,
                    channel,
                    value_type,
                    value_blob,
                    task_path,
                ),
            )
        conn.commit()

    def delete_thread(self, thread_id: str) -> None:
        raise ValueError("delete_thread requires a DB connection; use delete_thread_from_conn")

    def delete_thread_from_conn(self, conn: sqlite3.Connection, thread_id: str) -> None:
        self._ensure_schema(conn)
        for table in ("qa_checkpoint_writes", "qa_checkpoint_blobs", "qa_checkpoints"):
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        conn.commit()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
