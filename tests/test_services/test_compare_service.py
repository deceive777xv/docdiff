"""Tests for app/services/compare_service.py"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def two_docx_versions(tmp_path):
    from docx import Document

    doc1 = Document()
    doc1.add_heading("第一章 总则", level=1)
    doc1.add_paragraph("付款周期为30天。")
    p1 = tmp_path / "v1.docx"
    doc1.save(str(p1))

    doc2 = Document()
    doc2.add_heading("第一章 总则", level=1)
    doc2.add_paragraph("付款周期调整为60天。")
    p2 = tmp_path / "v2.docx"
    doc2.save(str(p2))

    return p1, p2


@pytest.fixture
def db_conn(tmp_path):
    from app.db.schema import init_db

    conn = init_db(str(tmp_path))
    yield conn
    conn.close()


def _make_mock_embedder():
    mock_embedder = MagicMock()
    mock_embedder.embed.side_effect = lambda texts: [
        [float(hash(t) % 1000) / 1000.0] * 8 for t in texts
    ]
    return mock_embedder


def _make_mock_provider():
    mock_provider = MagicMock()
    mock_provider.chat.return_value = (
        '{"diff_type": "实质修改", "risk_level": "high", "explanation": "金额变化"}'
    )
    return mock_provider


def test_run_compare_returns_diff_result(tmp_path, two_docx_versions, db_conn):
    """run_compare returns a DiffResult with at least one item."""
    from app.services.ingest_service import ingest_document
    from app.services.compare_service import run_compare
    from app.core.types import DiffResult

    p1, p2 = two_docx_versions
    mock_embedder = _make_mock_embedder()
    mock_provider = _make_mock_provider()

    _, v1_id = ingest_document(db_conn, str(tmp_path), str(p1), embedder=None)
    _, v2_id = ingest_document(db_conn, str(tmp_path), str(p2), embedder=None)

    result = run_compare(
        conn=db_conn,
        data_dir=str(tmp_path),
        baseline_version_id=v1_id,
        target_version_id=v2_id,
        embedder=mock_embedder,
        provider=mock_provider,
    )

    assert isinstance(result, DiffResult)
    assert len(result.items) >= 1


def test_compare_task_status_completed(tmp_path, two_docx_versions, db_conn):
    """After run_compare, the task record in DB has status 'completed'."""
    from app.services.ingest_service import ingest_document
    from app.services.compare_service import run_compare
    from app.db import compare_repo

    p1, p2 = two_docx_versions
    mock_embedder = _make_mock_embedder()
    mock_provider = _make_mock_provider()

    _, v1_id = ingest_document(db_conn, str(tmp_path), str(p1), embedder=None)
    _, v2_id = ingest_document(db_conn, str(tmp_path), str(p2), embedder=None)

    result = run_compare(
        conn=db_conn,
        data_dir=str(tmp_path),
        baseline_version_id=v1_id,
        target_version_id=v2_id,
        embedder=mock_embedder,
        provider=mock_provider,
    )

    task = compare_repo.get_task_by_id(db_conn, result.task_id)
    assert task is not None
    assert task["status"] == "completed"


def test_compare_detects_change(tmp_path, two_docx_versions, db_conn):
    """run_compare detects at least one diff item between the two versions."""
    from app.services.ingest_service import ingest_document
    from app.services.compare_service import run_compare

    p1, p2 = two_docx_versions
    mock_embedder = _make_mock_embedder()
    mock_provider = _make_mock_provider()

    _, v1_id = ingest_document(db_conn, str(tmp_path), str(p1), embedder=None)
    _, v2_id = ingest_document(db_conn, str(tmp_path), str(p2), embedder=None)

    result = run_compare(
        conn=db_conn,
        data_dir=str(tmp_path),
        baseline_version_id=v1_id,
        target_version_id=v2_id,
        embedder=mock_embedder,
        provider=mock_provider,
    )

    assert len(result.items) >= 1
