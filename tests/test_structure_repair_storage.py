from __future__ import annotations

import json

from app.core.normalization import NormalizationDepth
from app.core.types import DocumentIR, Paragraph, Section, Sentence


def _raw_document() -> DocumentIR:
    return DocumentIR(
        doc_id="raw-doc",
        title="Raw",
        file_hash="raw-hash",
        sections=[
            Section(
                "s1",
                "**1.1 测试**",
                2,
                [
                    Paragraph(
                        "p1",
                        "原始正文。",
                        [Sentence("原始正文。")],
                        1,
                    )
                ],
            )
        ],
        plain_text="原始正文。",
    )


def test_prepare_import_ir_persists_raw_normalized_and_trace(tmp_path):
    from app.core.structure_repair.storage import prepare_import_ir

    artifacts = prepare_import_ir(
        tmp_path,
        _raw_document(),
        depth=NormalizationDepth.STANDARD,
    )

    assert artifacts.raw_path == tmp_path / "parsed" / "raw" / "raw-doc.json"
    assert artifacts.normalized_path == tmp_path / "parsed" / "raw-doc.json"
    assert artifacts.trace_path == (
        tmp_path / "parsed" / "traces" / "raw-doc.structure.json"
    )
    assert artifacts.normalization_trace_path == (
        tmp_path / "parsed" / "traces" / "raw-doc.normalization.json"
    )
    assert artifacts.boundary_profile_path == (
        tmp_path / "parsed" / "profiles" / "raw-doc.boundary.json"
    )
    raw = json.loads(artifacts.raw_path.read_text(encoding="utf-8"))
    normalized = json.loads(artifacts.normalized_path.read_text(encoding="utf-8"))
    trace = json.loads(artifacts.trace_path.read_text(encoding="utf-8"))
    normalization_trace = json.loads(
        artifacts.normalization_trace_path.read_text(encoding="utf-8")
    )
    boundary_profile = json.loads(
        artifacts.boundary_profile_path.read_text(encoding="utf-8")
    )
    assert raw["sections"][0]["title"] == "**1.1 测试**"
    assert normalized["sections"][0]["title"] == "1.1 测试"
    assert trace["status"] == "repaired"
    assert trace["raw_hash"] != trace["normalized_hash"]
    assert normalization_trace["scope"] == "document"
    assert normalization_trace["structure_trace"] == trace
    assert normalization_trace["table_trace"]["baseline"]["doc_id"] == "raw-doc"
    assert boundary_profile["doc_id"] == "raw-doc"
    assert "deferred_table_candidates" not in boundary_profile


def test_prepare_import_ir_persists_skipped_low_depth_artifacts(tmp_path):
    from app.core.structure_repair.storage import prepare_import_ir

    artifacts = prepare_import_ir(
        tmp_path,
        _raw_document(),
        depth=NormalizationDepth.OFF,
    )

    raw = json.loads(artifacts.raw_path.read_text(encoding="utf-8"))
    normalized = json.loads(artifacts.normalized_path.read_text(encoding="utf-8"))
    trace = json.loads(
        artifacts.normalization_trace_path.read_text(encoding="utf-8")
    )

    assert normalized == raw
    assert trace["status"] == "skipped"
    assert trace["normalization_depth"] == "off"


def test_prepare_import_ir_persists_diagnostic_output_when_conservation_disabled(
    tmp_path,
    monkeypatch,
):
    from app.core.structure_repair import pipeline
    from app.core.structure_repair.storage import prepare_import_ir

    def corrupt(document, _operations):
        document.sections[0].paragraphs.pop()

    monkeypatch.setattr(pipeline, "_remove_noise", corrupt)
    monkeypatch.setenv(pipeline.CONTENT_CONSERVATION_DISABLE_ENV, "1")
    raw_document = _raw_document()
    raw_document.sections[0].title = "1.1 测试"

    artifacts = prepare_import_ir(
        tmp_path,
        raw_document,
        depth=NormalizationDepth.STANDARD,
    )

    normalized = json.loads(artifacts.normalized_path.read_text(encoding="utf-8"))
    structure_trace = json.loads(artifacts.trace_path.read_text(encoding="utf-8"))
    normalization_trace = json.loads(
        artifacts.normalization_trace_path.read_text(encoding="utf-8")
    )

    assert normalized["sections"][0]["paragraphs"] == []
    assert structure_trace["operations"] == []
    assert structure_trace["status"] == "repaired"
    assert normalization_trace["status"] == "repaired"
    assert structure_trace["warnings"] == [
        "content_conservation_disabled",
        "content_conservation_failed:remove_noise",
    ]
    assert normalization_trace["warnings"] == structure_trace["warnings"]


def test_prepare_import_ir_falls_back_to_raw_when_repair_raises(
    tmp_path,
    monkeypatch,
):
    from app.core.structure_repair import storage

    def fail(*_args, **_kwargs):
        raise RuntimeError("repair failed")

    monkeypatch.setattr(storage, "normalize_document", fail)

    artifacts = storage.prepare_import_ir(tmp_path, _raw_document())

    raw = json.loads(artifacts.raw_path.read_text(encoding="utf-8"))
    normalized = json.loads(artifacts.normalized_path.read_text(encoding="utf-8"))
    trace = json.loads(artifacts.trace_path.read_text(encoding="utf-8"))
    normalization_trace = json.loads(
        artifacts.normalization_trace_path.read_text(encoding="utf-8")
    )
    assert normalized == raw
    assert artifacts.document == _raw_document()
    assert trace["status"] == "fallback"
    assert trace["warnings"] == ["RuntimeError: repair failed"]
    assert normalization_trace["status"] == "fallback"
