"""TypedDict state definitions for LangGraph workflows."""
from __future__ import annotations

from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class IngestState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    file_path: str
    data_dir: str
    source_type: str           # "standard" | "uploaded"
    document_id: Optional[str] # set when adding new version to existing doc
    embedder: Any
    conn: Any                  # sqlite3.Connection, opened and closed by caller

    # ── Node-internal intermediate values ───────────────────────────────────
    _file_hash: str
    _ir: Any                   # DocumentIR
    _chunks: list

    # ── Node outputs ────────────────────────────────────────────────────────
    doc_id: str
    version_id: str

    # ── Status ──────────────────────────────────────────────────────────────
    error: Optional[str]
    status: str


class CompareState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    data_dir: str
    baseline_version_id: str
    target_version_id: str
    provider: Any
    embedder: Any
    conn: Any

    # ── Node-internal ────────────────────────────────────────────────────────
    _baseline_ir: Any          # DocumentIR
    _target_ir: Any            # DocumentIR
    _section_pairs: list
    _para_pairs: list

    # ── Node outputs ────────────────────────────────────────────────────────
    task_id: str
    result: Any                # DiffResult

    # ── Status ──────────────────────────────────────────────────────────────
    error: Optional[str]
    status: str


class QAState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    data_dir: str
    question: str
    scope: str                 # "current_doc" | "standard_lib" | "all"
    current_version_ids: list  # version IDs in scope for "current_doc"

    # ── Session memory (accumulated via add_messages reducer) ────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Node-internal ────────────────────────────────────────────────────────
    _version_ids: list
    _hits: list                # list[ChunkHit]

    # ── Node outputs ────────────────────────────────────────────────────────
    answer: str
    citations: list

    # ── Status ──────────────────────────────────────────────────────────────
    error: Optional[str]
    status: str
