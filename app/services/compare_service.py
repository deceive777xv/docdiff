"""Compare service — run semantic diff between two document versions."""
from __future__ import annotations
import json
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path

from app.core.diff.diff_classifier import classify
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
    data = json.loads(Path(ir_path).read_text(encoding="utf-8"))
    from app.core.types import Section, Paragraph, Sentence
    sections = []
    for sec in data.get("sections", []):
        paras = []
        for p in sec.get("paragraphs", []):
            sents = [Sentence(text=s["text"]) for s in p.get("sentences", [])]
            paras.append(Paragraph(
                paragraph_id=p["paragraph_id"],
                page_no=p["page_no"],
                text=p["text"],
                sentences=sents,
            ))
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


def run_compare(
    conn: sqlite3.Connection,
    data_dir: str,
    baseline_version_id: str,
    target_version_id: str,
    embedder: BaseProvider,
    provider: BaseProvider,
    policy: ComparePolicy | None = None,
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
        para_pairs = match_paragraphs(section_pairs, embedder, policy.similarity_threshold)
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

        # Save result JSON
        exports_dir = Path(data_dir) / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        result_path = exports_dir / f"{task_id}.json"
        result_path.write_text(
            json.dumps([asdict(item) for item in result.items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        compare_repo.update_task_status(conn, task_id, "completed", str(result_path))
        logger.info("Compare task %s completed: %d diff items", task_id, len(result.items))
        return result

    except Exception as e:
        compare_repo.update_task_status(conn, task_id, "failed")
        logger.error("Compare task %s failed: %s", task_id, e)
        raise
