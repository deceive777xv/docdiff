"""Tests for ingest_graph LangGraph workflow."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_file_check_missing_file():
    """file_check sets error when file does not exist."""
    from app.agent.ingest_graph import file_check

    result = file_check({
        "file_path": "/nonexistent/file.pdf",
        "data_dir": "/tmp",
        "source_type": "standard",
        "conn": MagicMock(),
    })
    assert result.get("error") is not None
    assert result.get("status") == "failed"


def test_file_check_duplicate(tmp_path):
    """file_check sets error when doc hash already exists in DB."""
    from app.agent.ingest_graph import file_check

    doc = tmp_path / "test.pdf"
    doc.write_bytes(b"%PDF-1.4 test")

    with patch(
        "app.agent.ingest_graph.document_repo.get_document_by_hash",
        return_value={"id": "existing-id"},
    ):
        result = file_check({
            "file_path": str(doc),
            "data_dir": str(tmp_path),
            "source_type": "standard",
            "document_id": None,
            "conn": MagicMock(),
        })
    assert result.get("error") is not None
    assert result.get("status") == "failed"


def test_graph_propagates_error_to_end():
    """Graph reaches END immediately when file_check sets error."""
    from app.agent.ingest_graph import ingest_graph

    result = ingest_graph.invoke({
        "file_path": "/nonexistent/file.pdf",
        "data_dir": "/tmp",
        "source_type": "standard",
        "conn": MagicMock(),
    })
    assert result.get("error") is not None
    assert result.get("status") == "failed"
    assert not result.get("doc_id")


def test_graph_happy_path(tmp_path):
    """Full happy path: file exists, no duplicate, parse succeeds."""
    from app.agent.ingest_graph import ingest_graph
    from app.core.types import DocumentIR, ParseQualityReport

    doc = tmp_path / "test.pdf"
    doc.write_bytes(b"%PDF-1.4 test")

    mock_ir = DocumentIR(
        doc_id="doc-uuid",
        title="Test",
        file_hash="abc123",
        sections=[],
        plain_text="",
    )
    # ParseQualityReport requires quality_score (float) and needs_ocr (bool);
    # ocr_pages and warnings have defaults via field(default_factory=list).
    # Using MagicMock() in the patch return value so no explicit instantiation needed.

    with (
        patch("app.agent.ingest_graph.document_repo.get_document_by_hash", return_value=None),
        patch("app.agent.ingest_graph.parse_document", return_value=(mock_ir, MagicMock())),
        patch("app.agent.ingest_graph.document_repo.insert_document", return_value="doc-123"),
        patch("app.agent.ingest_graph.document_repo.insert_version", return_value="ver-456"),
        patch("app.agent.ingest_graph.chunk_repo.insert_chunks"),
        patch("app.agent.ingest_graph.build_chunks", return_value=[]),
    ):
        result = ingest_graph.invoke({
            "file_path": str(doc),
            "data_dir": str(tmp_path),
            "source_type": "standard",
            "document_id": None,
            "embedder": None,
            "conn": MagicMock(),
        })

    assert result.get("error") is None
    assert result["doc_id"] == "doc-123"
    assert result["version_id"] == "ver-456"
    assert result["status"] == "completed"
