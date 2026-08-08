from __future__ import annotations

from copy import deepcopy
import json

import pytest

from app.core.model.base_provider import BaseProvider
from app.core.normalization import (
    NormalizationDepth,
    normalize_document,
)
from app.core.types import DocumentIR, Paragraph, Section, Sentence


class _EchoTableMergeProvider(BaseProvider):
    chat_model = "echo-table-merge"

    def chat(self, messages: list[dict], **kwargs) -> str:
        payload = json.loads(messages[-1]["content"])
        continuation = payload["candidate"]["continuation"]
        is_row_continuation = not continuation[0]
        return json.dumps(
            {
                "candidate_id": payload["candidate_id"],
                "continuation_role": (
                    "continuation_row" if is_row_continuation else "table_header"
                ),
                "row_action": "merge" if is_row_continuation else "keep",
                "table_action": "merge_fragments",
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

    result = normalize_document(
        raw,
        provider=_EchoTableMergeProvider(),
        depth=NormalizationDepth.STANDARD,
    )

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


def test_low_depth_skips_all_normalization_and_provider_calls():
    class FailIfCalledProvider:
        def chat(self, *_args, **_kwargs):
            raise AssertionError("low depth must not call the normalization provider")

    raw = _document_with_sparse_cross_page_table()

    result = normalize_document(
        raw,
        provider=FailIfCalledProvider(),
        depth=NormalizationDepth.OFF,
    )

    assert result.document == raw
    assert result.document is not raw
    assert result.status == "skipped"
    assert result.trace.normalization_depth == "off"
    assert result.trace.structure_trace.operations == []
    assert result.trace.table_trace.operations == []


@pytest.mark.parametrize(
    ("depth", "expects_review"),
    [
        (NormalizationDepth.STANDARD, False),
        (NormalizationDepth.REVIEW, True),
    ],
)
def test_depth_controls_page_noise_review(depth, expects_review):
    class RecordingProvider:
        chat_model = "fake-model"

        def __init__(self):
            self.payloads: list[dict] = []

        def chat(self, messages, **_kwargs):
            payload = json.loads(messages[-1]["content"])
            self.payloads.append(payload)
            if "items" not in payload:
                raise AssertionError("fixture must only produce page-noise candidates")
            return json.dumps(
                {
                    "boundary_id": payload["boundary_id"],
                    "labels": [
                        {
                            "id": item["id"],
                            "action": (
                                "remove_as_page_noise"
                                if item["text"] == "重复页眉"
                                else "keep"
                            ),
                            "confidence": 0.99,
                            "reason": "fixed test judgment",
                        }
                        for item in payload["items"]
                    ],
                },
                ensure_ascii=False,
            )

    raw = DocumentIR(
        "depth-doc",
        "Depth",
        "depth-hash",
        [
            Section(
                "s1",
                "1 范围",
                1,
                [
                    Paragraph("noise", "重复页眉", [Sentence("重复页眉")], 1),
                    Paragraph("body-1", "第一页正文。", [Sentence("第一页正文。")], 1),
                    Paragraph("body-2", "第二页正文。", [Sentence("第二页正文。")], 2),
                ],
            )
        ],
        "重复页眉\n第一页正文。\n第二页正文。",
    )
    provider = RecordingProvider()

    result = normalize_document(raw, provider=provider, depth=depth)

    assert "重复页眉" not in result.document.plain_text
    assert any("initial_labels" in payload for payload in provider.payloads) is expects_review
    assert result.trace.normalization_depth == depth.value


def test_normalize_document_preserves_page_number_when_merging_table_fragments():
    raw = _document_with_sparse_cross_page_table()

    result = normalize_document(
        raw,
        provider=_EchoTableMergeProvider(),
        depth=NormalizationDepth.STANDARD,
    )

    paragraphs = [
        paragraph
        for section in result.document.sections
        for paragraph in section.paragraphs
    ]
    assert len(paragraphs) == 1
    assert paragraphs[0].page_no == 7


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

    result = pipeline.normalize_document(
        raw,
        provider=_EchoTableMergeProvider(),
        depth=NormalizationDepth.STANDARD,
    )

    assert result.status == "fallback"
    assert result.document == raw
    assert result.table_trace.decisions == []
    assert result.table_trace.operations == []


def test_normalize_document_preserves_structure_repairs_when_table_stage_raises(
    monkeypatch,
):
    from app.core.normalization import pipeline
    from app.core.structure_repair.models import (
        StructureRepairResult,
        StructureRepairTrace,
    )

    raw = _document_with_sparse_cross_page_table()
    repaired = deepcopy(raw)
    repaired.sections[0].title = "3.1 已规范化的系统功能"
    repaired_trace = StructureRepairTrace(
        schema_version=1,
        algorithm_version="test",
        doc_id=raw.doc_id,
        raw_hash="raw",
        normalized_hash="repaired",
        status="repaired",
    )
    monkeypatch.setattr(
        pipeline,
        "repair_document",
        lambda *args, **kwargs: StructureRepairResult(
            document=repaired,
            trace=repaired_trace,
            status="repaired",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "reconstruct_document_tables",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("key column conflict at logical column 0")
        ),
    )

    result = pipeline.normalize_document(
        raw,
        provider=_EchoTableMergeProvider(),
        depth=NormalizationDepth.STANDARD,
    )

    assert result.document.sections[0].title == "3.1 已规范化的系统功能"
    assert result.structure_trace is repaired_trace
    assert result.status == "fallback"
    assert result.table_trace.operations == []
    assert result.warnings == [
        "table_reconstruction_failed: ValueError: key column conflict at logical column 0"
    ]
