"""LangGraph StateGraph for the document ingest workflow."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

from langgraph.graph import END, StateGraph

from app.agent.states import IngestState
from app.core.parser.ir_builder import build_chunks
from app.core.parser.router import parse_document
from app.core.utils import file_hash as compute_file_hash
from app.db import chunk_repo, document_repo

logger = logging.getLogger(__name__)


def _route(state: IngestState) -> str:
    return "end" if state.get("error") else "continue"


def file_check(state: IngestState) -> dict:
    """Verify file exists, compute hash, detect duplicates."""
    try:
        path = Path(state["file_path"])
        if not path.exists():
            return {"error": f"文件不存在：{path}", "status": "failed"}

        file_hash = compute_file_hash(path)
        conn = state["conn"]

        if not state.get("document_id"):
            existing = document_repo.get_document_by_hash(conn, file_hash)
            if existing:
                return {
                    "error": (
                        f"文档已存在（hash {file_hash[:8]}...）。"
                        "如需新增版本，请选择文档后点击[新增版本]。"
                    ),
                    "status": "failed",
                }

        return {"_file_hash": file_hash, "status": "file_checked"}
    except Exception as e:
        logger.exception("file_check failed")
        return {"error": str(e), "status": "failed"}


def parse_doc(state: IngestState) -> dict:
    """Parse the file into a DocumentIR."""
    try:
        ir, quality = parse_document(state["file_path"])
        if quality.needs_ocr:
            logger.warning("Document needs OCR (Phase 2 feature): %s", quality.ocr_pages)
        return {"_ir": ir, "status": "parsed"}
    except Exception as e:
        logger.exception("parse_doc failed")
        return {"error": str(e), "status": "failed"}


def save_document(state: IngestState) -> dict:
    """Copy file, persist IR JSON, insert DB rows, insert chunks."""
    try:
        conn = state["conn"]
        data_dir = state["data_dir"]
        ir = state["_ir"]
        path = Path(state["file_path"])
        file_hash = state["_file_hash"]

        docs_dir = Path(data_dir) / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        dest = docs_dir / f"{file_hash}{path.suffix}"
        if not dest.exists():
            shutil.copy2(str(path), str(dest))

        parsed_dir = Path(data_dir) / "parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        ir_path = parsed_dir / f"{ir.doc_id}.json"
        ir_path.write_text(
            json.dumps(asdict(ir), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        document_id = state.get("document_id")
        if document_id:
            versions = document_repo.list_versions(conn, document_id)
            version_no = (max(v["version_no"] for v in versions) + 1) if versions else 1
            version_id = document_repo.insert_version(
                conn,
                document_id=document_id,
                version_no=version_no,
                parsed_json_path=str(ir_path),
                summary=ir.title,
            )
            doc_id = document_id
        else:
            doc_id = document_repo.insert_document(
                conn,
                doc_name=path.stem,
                doc_type=path.suffix.lstrip(".").lower(),
                file_path=str(dest),
                file_hash=file_hash,
                source_type=state.get("source_type", "standard"),
                business_category="",
            )
            version_id = document_repo.insert_version(
                conn,
                document_id=doc_id,
                version_no=1,
                parsed_json_path=str(ir_path),
                summary=ir.title,
            )

        chunks = build_chunks(ir, version_id)
        chunk_repo.insert_chunks(conn, chunks)

        return {"doc_id": doc_id, "version_id": version_id, "_chunks": chunks, "status": "saved"}
    except Exception as e:
        logger.exception("save_document failed")
        return {"error": str(e), "status": "failed"}


def build_embeddings(state: IngestState) -> dict:
    """Build FAISS index for the ingested version (skipped if no embedder)."""
    try:
        embedder = state.get("embedder")
        chunks = state.get("_chunks", [])
        if embedder and chunks:
            from app.core.retrieval.indexer import build_index
            build_index(state["data_dir"], state["conn"], state["version_id"], chunks, embedder)
        return {"status": "completed"}
    except Exception as e:
        logger.exception("build_embeddings failed")
        return {"error": str(e), "status": "failed"}


def _build_ingest_graph():
    graph = StateGraph(IngestState)
    graph.add_node("file_check", file_check)
    graph.add_node("parse_doc", parse_doc)
    graph.add_node("save_document", save_document)
    graph.add_node("build_embeddings", build_embeddings)

    graph.set_entry_point("file_check")
    graph.add_conditional_edges("file_check",    _route, {"continue": "parse_doc",       "end": END})
    graph.add_conditional_edges("parse_doc",     _route, {"continue": "save_document",   "end": END})
    graph.add_conditional_edges("save_document", _route, {"continue": "build_embeddings", "end": END})
    graph.add_edge("build_embeddings", END)
    return graph.compile()


ingest_graph = _build_ingest_graph()
