"""CRUD for chunks table."""
from __future__ import annotations
import sqlite3
import uuid

from app.core.types import Chunk


def insert_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    """Bulk-insert chunks. Uses executemany for efficiency."""
    rows = [
        (c.id, c.version_id, c.chunk_no, c.section_path, c.page_no, c.text, c.faiss_index_id)
        for c in chunks
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO chunks
           (id, version_id, chunk_no, section_path, page_no, text, faiss_index_id)
           VALUES (?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()


def get_chunks_by_version(conn: sqlite3.Connection, version_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chunks WHERE version_id = ? ORDER BY chunk_no",
        (version_id,),
    ).fetchall()


def update_faiss_ids(conn: sqlite3.Connection, chunk_id_to_faiss: dict[str, int]) -> None:
    """Update faiss_index_id for multiple chunks at once."""
    rows = [(fid, cid) for cid, fid in chunk_id_to_faiss.items()]
    conn.executemany(
        "UPDATE chunks SET faiss_index_id = ? WHERE id = ?", rows
    )
    conn.commit()


def get_chunk_by_faiss_id(
    conn: sqlite3.Connection, version_id: str, faiss_index_id: int
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM chunks WHERE version_id = ? AND faiss_index_id = ?",
        (version_id, faiss_index_id),
    ).fetchone()
