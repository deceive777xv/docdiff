from __future__ import annotations

from copy import deepcopy
import json

from app.core.model.base_provider import BaseProvider
from app.core.normalization import normalize_document, normalize_pair
from app.core.types import DocumentIR, Paragraph, Section, Sentence


class _EchoTableMergeProvider(BaseProvider):
    chat_model = "echo-table-merge"

    def chat(self, messages: list[dict], **kwargs) -> str:
        payload = json.loads(messages[-1]["content"])
        return json.dumps(
            {
                "candidate_id": payload["candidate_id"],
                "continuation_role": "continuation_row",
                "action": "merge",
                "confidence": 0.92,
                "reason": "the bounded row continues the preceding business cell",
            }
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("retrieval is outside normalization")

    def health_check(self) -> bool:
        return True


def _paragraph(paragraph_id: str, lines: list[str], page_no: int) -> Paragraph:
    return Paragraph(
        paragraph_id,
        "\n".join(lines),
        [Sentence(line) for line in lines],
        page_no=page_no,
    )


def _document_with_sparse_cross_page_table() -> DocumentIR:
    section = Section(
        "function",
        "3.1 系统功能",
        1,
        [
            _paragraph(
                "function-page-7",
                [
                    "||序号|功能|要求||备注||||",
                    "|---|---|---|---|---|---|---|---|---|",
                    "||20|开关门舒适性|关门反力：300 ± 30 N。||根据分析结果微调。||||",
                    "||21|防夹要求|关门过程：①电撑杆高位、中位、低位（峰值）≤100 N；开门过程：③电撑杆高位、中位、低位（峰值）≤120 N；||/||||",
                ],
                7,
            ),
            _paragraph(
                "function-page-8",
                [
                    "|||||文件名称|文件名称|文件名称|文件名称|文件名称|",
                    "|---|---|---|---|---|---|---|---|---|",
                    "|||||文件编号||版本：001|第8页||",
                    "||||电撑杆高位、中位、低位（有效值）≤120 N。<br>2、防夹响应时间：≤0.5 s；||||||",
                    "||22|障碍物检测|障碍物前停止距离：100-150 mm。||/||||",
                ],
                8,
            ),
        ],
    )
    return DocumentIR(
        "function-doc",
        "Function",
        "function-hash",
        [section],
        "\n".join(paragraph.text for paragraph in section.paragraphs),
    )


def test_normalize_document_reconstructs_cross_page_tables_without_a_peer_document():
    raw = _document_with_sparse_cross_page_table()
    original = deepcopy(raw)

    result = normalize_document(raw, provider=_EchoTableMergeProvider())

    merged_row = next(
        sentence.text
        for section in result.document.sections
        for paragraph in section.paragraphs
        for sentence in paragraph.sentences
        if "21" in sentence.text and "防夹要求" in sentence.text
    )
    assert "（有效值）≤120 N。" in merged_row
    assert raw == original
    assert any(
        decision.final_action == "merge"
        for decision in result.table_trace.decisions
    )
    assert any(
        operation.type == "merge_rows"
        for operation in result.table_trace.operations
    )


def test_normalize_pair_rechecks_only_document_candidates_deferred_at_import():
    raw = _document_with_sparse_cross_page_table()
    imported = normalize_document(raw, provider=None)
    target = DocumentIR("target-empty", "", "target-empty-hash", [])

    assert imported.trace.deferred_table_candidates

    result = normalize_pair(
        imported.document,
        target,
        provider=_EchoTableMergeProvider(),
        baseline_deferred=imported.trace.deferred_table_candidates,
        target_deferred=[],
    )

    assert any(
        decision.final_action == "merge" for decision in result.trace.decisions
    )
    assert any(
        operation.type == "merge_rows" for operation in result.trace.operations
    )


def test_normalize_pair_skips_reconstruction_when_import_has_no_deferred_candidates():
    baseline = _document_with_sparse_cross_page_table()
    target = DocumentIR("target-empty", "", "target-empty-hash", [])
    provider = _EchoTableMergeProvider()

    result = normalize_pair(
        baseline,
        target,
        provider=provider,
        baseline_deferred=[],
        target_deferred=[],
    )

    assert result.trace.decisions == []
    assert result.trace.operations == []
    assert result.baseline_ir == baseline
    assert result.baseline_ir is not baseline


def test_normalize_document_does_not_apply_table_changes_after_structure_fallback(
    monkeypatch,
):
    from app.core.normalization import pipeline
    from app.core.structure_repair.models import (
        StructureRepairResult,
        StructureRepairTrace,
    )

    raw = _document_with_sparse_cross_page_table()
    fallback_trace = StructureRepairTrace(
        schema_version=1,
        algorithm_version="test",
        doc_id=raw.doc_id,
        raw_hash="raw",
        normalized_hash="raw",
        status="fallback",
        warnings=["content_conservation_failed"],
    )
    monkeypatch.setattr(
        pipeline,
        "repair_document",
        lambda *args, **kwargs: StructureRepairResult(
            document=deepcopy(raw),
            trace=fallback_trace,
            status="fallback",
            warnings=["content_conservation_failed"],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "reconstruct_document_tables",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("table reconstruction must not run after fallback")
        ),
    )

    result = pipeline.normalize_document(raw, provider=_EchoTableMergeProvider())

    assert result.status == "fallback"
    assert result.document == raw
    assert result.table_trace.decisions == []
    assert result.table_trace.operations == []


def test_import_artifact_round_trips_deferred_table_candidates(tmp_path):
    from app.core.normalization import load_deferred_table_candidates
    from app.core.structure_repair.storage import prepare_import_ir

    artifacts = prepare_import_ir(
        tmp_path,
        _document_with_sparse_cross_page_table(),
        provider=None,
    )

    loaded = load_deferred_table_candidates(tmp_path, artifacts.document)

    assert loaded == artifacts.normalization_trace.deferred_table_candidates
    assert loaded
