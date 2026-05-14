"""LangGraph StateGraph for the document comparison workflow."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from langgraph.graph import END, StateGraph

from app.agent.states import CompareState
from app.core.diff.diff_classifier import classify
from app.core.diff.semantic_matcher import match_paragraphs
from app.core.diff.structure_aligner import align_sections
from app.core.types import ComparePolicy, DocumentIR, Paragraph, Section, Sentence
from app.db import compare_repo, document_repo

logger = logging.getLogger(__name__)


def _route(state: CompareState) -> str:
    return "end" if state.get("error") else "continue"


def _load_ir(version_id: str, conn) -> DocumentIR:
    """Load DocumentIR from the parsed JSON path stored in DB."""
    row = document_repo.get_version_by_id(conn, version_id)
    if not row:
        raise ValueError(f"Version not found: {version_id}")
    ir_path = row["parsed_json_path"]
    if not ir_path or not Path(ir_path).exists():
        raise FileNotFoundError(f"Parsed IR not found: {ir_path}")
    data = json.loads(Path(ir_path).read_text(encoding="utf-8"))
    sections = []
    for sec in data.get("sections", []):
        paras = [
            Paragraph(
                paragraph_id=p["paragraph_id"],
                text=p["text"],
                sentences=[Sentence(text=s["text"]) for s in p.get("sentences", [])],
            )
            for p in sec.get("paragraphs", [])
        ]
        sections.append(Section(
            section_id=sec["section_id"],
            title=sec["title"],
            level=sec["level"],
            paragraphs=paras,
        ))
    return DocumentIR(
        doc_id=data["doc_id"],
        title=data["title"],
        file_hash=data["file_hash"],
        sections=sections,
        plain_text=data.get("plain_text", ""),
    )


def create_task(state: CompareState) -> dict:
    """Insert compare_tasks record and mark as running."""
    try:
        task_id = compare_repo.create_compare_task(
            state["conn"],
            baseline_version_id=state["baseline_version_id"],
            target_version_id=state["target_version_id"],
        )
        compare_repo.update_task_status(state["conn"], task_id, "running")
        return {"task_id": task_id, "status": "task_created"}
    except Exception as e:
        logger.exception("create_task failed")
        return {"error": str(e), "status": "failed"}


def ensure_parsed(state: CompareState) -> dict:
    """Load both DocumentIRs from DB-stored JSON paths."""
    try:
        baseline_ir = _load_ir(state["baseline_version_id"], state["conn"])
        target_ir = _load_ir(state["target_version_id"], state["conn"])
        return {"_baseline_ir": baseline_ir, "_target_ir": target_ir, "status": "irs_loaded"}
    except Exception as e:
        logger.exception("ensure_parsed failed")
        compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
        return {"error": str(e), "status": "failed"}


def do_align(state: CompareState) -> dict:
    """Align document sections using title similarity."""
    try:
        pairs = align_sections(state["_baseline_ir"], state["_target_ir"])
        return {"_section_pairs": pairs, "status": "aligned"}
    except Exception as e:
        logger.exception("do_align failed")
        compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
        return {"error": str(e), "status": "failed"}


def do_semantic_compare(state: CompareState) -> dict:
    """Match paragraphs by embedding cosine similarity."""
    try:
        policy = ComparePolicy()
        para_pairs = match_paragraphs(
            state["_section_pairs"], state["embedder"], policy.similarity_threshold
        )
        return {"_para_pairs": para_pairs, "status": "matched"}
    except Exception as e:
        logger.exception("do_semantic_compare failed")
        compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
        return {"error": str(e), "status": "failed"}


def do_classify(state: CompareState) -> dict:
    """Classify paragraph pairs with LLM and rule-based strengthening."""
    try:
        policy = ComparePolicy()
        result = classify(
            state["_para_pairs"],
            policy=policy,
            provider=state["provider"],
            task_id=state["task_id"],
            baseline_version_id=state["baseline_version_id"],
            target_version_id=state["target_version_id"],
        )
        return {"result": result, "status": "classified"}
    except Exception as e:
        logger.exception("do_classify failed")
        compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
        return {"error": str(e), "status": "failed"}


def persist_result(state: CompareState) -> dict:
    """Write diff_items to DB and save JSON export."""
    try:
        result = state["result"]
        conn = state["conn"]
        task_id = state["task_id"]

        compare_repo.insert_diff_items(conn, task_id, result.items)

        exports_dir = Path(state["data_dir"]) / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        result_path = exports_dir / f"{task_id}.json"
        result_path.write_text(
            json.dumps([asdict(i) for i in result.items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        compare_repo.update_task_status(conn, task_id, "completed", str(result_path))
        logger.info("Compare task %s completed: %d items", task_id, len(result.items))
        return {"status": "completed"}
    except Exception as e:
        logger.exception("persist_result failed")
        compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
        return {"error": str(e), "status": "failed"}


def _build_compare_graph():
    graph = StateGraph(CompareState)
    nodes = [
        ("create_task",         create_task),
        ("ensure_parsed",       ensure_parsed),
        ("do_align",            do_align),
        ("do_semantic_compare", do_semantic_compare),
        ("do_classify",         do_classify),
        ("persist_result",      persist_result),
    ]
    for name, fn in nodes:
        graph.add_node(name, fn)

    graph.set_entry_point("create_task")
    sequence = [n for n, _ in nodes]
    for i, src in enumerate(sequence[:-1]):
        dst = sequence[i + 1]
        graph.add_conditional_edges(src, _route, {"continue": dst, "end": END})
    graph.add_edge("persist_result", END)
    return graph.compile()


compare_graph = _build_compare_graph()
