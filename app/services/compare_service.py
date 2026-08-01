"""Compare service — run semantic diff between two document versions."""
from __future__ import annotations
import logging
import sqlite3
from pathlib import Path

from app.core.document_ir_codec import load_document_ir
from app.core.diff.diff_classifier import classify
from app.core.diff.result_storage import persist_compare_result
from app.core.diff.semantic_matcher import match_paragraphs
from app.core.diff.structure_aligner import align_sections
from app.core.model.base_provider import BaseProvider
from app.core.types import ComparePolicy, DiffResult, DocumentIR
from app.db import compare_repo, document_repo

logger = logging.getLogger(__name__)


def _load_ir(version_id: str, conn: sqlite3.Connection) -> DocumentIR:
    """Load DocumentIR from the parsed JSON path stored in DB."""
    version_row = document_repo.get_version_by_id(conn, version_id)
    if not version_row:
        raise ValueError(f"Version not found: {version_id}")
    ir_path = version_row["parsed_json_path"]
    if not ir_path or not Path(ir_path).exists():
        raise FileNotFoundError(f"Parsed IR not found at {ir_path}")
    return load_document_ir(ir_path)


def run_compare(
    conn: sqlite3.Connection,
    data_dir: str,
    baseline_version_id: str,
    target_version_id: str,
    embedder: BaseProvider,
    provider: BaseProvider,
    policy: ComparePolicy | None = None,
    task_id: str | None = None,
) -> DiffResult:
    """
    Full compare pipeline:
    1. Load DocumentIRs
    2. Align sections
    3. Match paragraphs by embedding
    4. Classify diffs with LLM
    5. Persist results
    Returns DiffResult.
    """
    if policy is None:
        policy = ComparePolicy()

    if task_id:
        compare_repo.prepare_task_for_rerun(conn, task_id)
    else:
        task_id = compare_repo.create_compare_task(
            conn,
            baseline_version_id=baseline_version_id,
            target_version_id=target_version_id,
        )
        compare_repo.update_task_status(conn, task_id, "running")

    try:
        baseline_ir = _load_ir(baseline_version_id, conn)
        target_ir = _load_ir(target_version_id, conn)

        section_pairs = align_sections(baseline_ir, target_ir)
        para_pairs = match_paragraphs(
            section_pairs,
            embedder,
            policy.similarity_threshold,
            rerank_provider=provider if policy.use_llm_match else None,
            use_llm_rerank=policy.use_llm_match,
            baseline_document_title=baseline_ir.title,
            target_document_title=target_ir.title,
        )
        result = classify(
            para_pairs,
            policy=policy,
            provider=provider,
            task_id=task_id,
            baseline_version_id=baseline_version_id,
            target_version_id=target_version_id,
        )

        # Persist diff items
        compare_repo.insert_diff_items(conn, task_id, result.items)

        result_path = persist_compare_result(
            data_dir,
            task_id,
            result.items,
        )

        compare_repo.update_task_status(conn, task_id, "completed", str(result_path))
        logger.debug(
            "Compare task %s result published: %s",
            task_id,
            result_path,
        )
        logger.info("Compare task %s completed: %d diff items", task_id, len(result.items))
        return result

    except Exception as e:
        compare_repo.update_task_status(conn, task_id, "failed")
        logger.error("Compare task %s failed: %s", task_id, e)
        raise
