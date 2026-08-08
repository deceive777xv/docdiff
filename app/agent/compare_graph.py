"""LangGraph StateGraph for the document comparison workflow."""
from __future__ import annotations

import logging
from pathlib import Path

from langgraph.graph import END, StateGraph

from app.agent.states import CompareState
from app.core.diff.diff_classifier import classify
from app.core.diff.result_storage import persist_compare_result
from app.core.diff.section_scope_aligner import align_compare_scopes
from app.core.diff.semantic_matcher import match_paragraphs
from app.core.document_ir_codec import load_document_ir
from app.core.types import ComparePolicy, DocumentIR
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
    return load_document_ir(ir_path)


def create_task(state: CompareState) -> dict:
    """Insert compare_tasks record and mark as running."""
    try:
        task_id = state.get("task_id")
        if task_id:
            task = compare_repo.get_task_by_id(state["conn"], task_id)
            if task is None:
                raise ValueError(f"Compare task not found: {task_id}")
            compare_repo.prepare_task_for_rerun(state["conn"], task_id)
        else:
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
    """Build compare-only logical section scopes."""
    try:
        policy = ComparePolicy()
        plan = align_compare_scopes(
            state["_baseline_ir"],
            state["_target_ir"],
            state["embedder"],
            similarity_threshold=policy.similarity_threshold,
        )
        return {"_section_alignment_plan": plan, "status": "aligned"}
    except Exception as e:
        logger.exception("do_align failed")
        compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
        return {"error": str(e), "status": "failed"}


def do_semantic_compare(state: CompareState) -> dict:
    """Match paragraphs by embedding cosine similarity."""
    try:
        policy = ComparePolicy()
        para_pairs = match_paragraphs(
            state["_section_alignment_plan"],
            state["embedder"],
            policy.similarity_threshold,
            rerank_provider=state["provider"] if policy.use_llm_match else None,
            use_llm_rerank=policy.use_llm_match,
            baseline_document_title=state["_baseline_ir"].title,
            target_document_title=state["_target_ir"].title,
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
    """Write diff items and atomically publish the compare result."""
    try:
        result = state["result"]
        conn = state["conn"]
        task_id = state["task_id"]

        compare_repo.insert_diff_items(conn, task_id, result.items)

        result_path = persist_compare_result(
            state["data_dir"],
            task_id,
            result.items,
        )
        compare_repo.update_task_status(conn, task_id, "completed", str(result_path))
        logger.debug(
            "Compare task %s result published: %s",
            task_id,
            result_path,
        )
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
