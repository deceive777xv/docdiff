"""Ingest service — import a document file into the local library."""
from __future__ import annotations
import json
import logging
import shutil
import sqlite3
from dataclasses import asdict
from pathlib import Path

from app.core.model.base_provider import BaseProvider
from app.core.parser.ir_builder import build_chunks
from app.core.parser.router import parse_document
from app.core.types import DocumentIR
from app.core.utils import file_hash as compute_file_hash
from app.db import chunk_repo, document_repo

logger = logging.getLogger(__name__)


def ingest_document(
    conn: sqlite3.Connection,
    data_dir: str,
    file_path: str,
    source_type: str = "standard",
    business_category: str = "",
    embedder: BaseProvider | None = None,
) -> tuple[str, str]:
    """
    Import a document file.

    1. Hash check — skip if already ingested
    2. Copy file to data_dir/docs/
    3. Parse → DocumentIR
    4. Save IR JSON to data_dir/parsed/
    5. Insert document + version rows
    6. Insert chunks
    7. Build FAISS index (if embedder provided)

    Returns (document_id, version_id).
    Raises FileExistsError if the file hash already exists.
    """
    path = Path(file_path)
    file_hash = compute_file_hash(path)

    existing = document_repo.get_document_by_hash(conn, file_hash)
    if existing:
        raise FileExistsError(
            f"Document already ingested (hash {file_hash[:8]}…). "
            "Use ingest_new_version() to add a new version."
        )

    # Copy file to docs dir
    docs_dir = Path(data_dir) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / f"{file_hash}{path.suffix}"
    if not dest.exists():
        shutil.copy2(str(path), str(dest))

    # Parse
    ir, quality = parse_document(str(path))
    if quality.needs_ocr:
        logger.warning("Low-quality document, OCR may be needed: %s", path)

    # Save IR JSON
    parsed_dir = Path(data_dir) / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    ir_path = parsed_dir / f"{ir.doc_id}.json"
    ir_path.write_text(
        json.dumps(asdict(ir), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # DB insert
    doc_id = document_repo.insert_document(
        conn,
        doc_name=path.stem,
        doc_type=path.suffix.lstrip(".").lower(),
        file_path=str(dest),
        file_hash=file_hash,
        source_type=source_type,
        business_category=business_category,
    )
    version_id = document_repo.insert_version(
        conn,
        document_id=doc_id,
        version_no=1,
        parsed_json_path=str(ir_path),
        summary=ir.title,
    )

    # Chunks
    chunks = build_chunks(ir, version_id)
    chunk_repo.insert_chunks(conn, chunks)

    # FAISS index
    if embedder and chunks:
        from app.core.retrieval.indexer import build_index
        build_index(data_dir, conn, version_id, chunks, embedder)

    logger.info("Ingested document %s (version %s)", doc_id, version_id)
    return doc_id, version_id


def ingest_new_version(
    conn: sqlite3.Connection,
    data_dir: str,
    file_path: str,
    document_id: str,
    version_label: str = "",
    embedder: BaseProvider | None = None,
) -> str:
    """Add a new version to an existing document. Returns version_id."""
    path = Path(file_path)

    # Get existing versions to determine next version_no
    versions = document_repo.list_versions(conn, document_id)
    version_no = (max(v["version_no"] for v in versions) + 1) if versions else 1

    ir, quality = parse_document(str(path))
    if quality.needs_ocr:
        logger.warning("Low-quality new version, OCR may be needed: %s", path)

    parsed_dir = Path(data_dir) / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    ir_path = parsed_dir / f"{ir.doc_id}.json"
    ir_path.write_text(
        json.dumps(asdict(ir), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    version_id = document_repo.insert_version(
        conn,
        document_id=document_id,
        version_no=version_no,
        version_label=version_label,
        parsed_json_path=str(ir_path),
        summary=ir.title,
    )

    chunks = build_chunks(ir, version_id)
    chunk_repo.insert_chunks(conn, chunks)

    if embedder and chunks:
        from app.core.retrieval.indexer import build_index
        build_index(data_dir, conn, version_id, chunks, embedder)

    return version_id
