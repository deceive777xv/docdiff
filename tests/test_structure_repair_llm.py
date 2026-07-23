from __future__ import annotations

import json

from app.core.types import DocumentIR, Paragraph, Section, Sentence


class _FakeProvider:
    chat_model = "fake-model"

    def __init__(self, response: object):
        self.response = response
        self.messages: list[dict] | None = None

    def chat(self, messages: list[dict], **_kwargs) -> str:
        self.messages = messages
        if isinstance(self.response, BaseException):
            raise self.response
        return str(self.response)


def _document() -> DocumentIR:
    return DocumentIR(
        doc_id="doc-llm",
        title="测试",
        file_hash="hash-llm",
        sections=[
            Section(
                "s1",
                "1.1.18 悬停功能",
                3,
                [Paragraph("p1", "正文。", [Sentence("正文。")], 1)],
            ),
            Section(
                "s2",
                "附表：悬停方式对应悬停动作",
                1,
                [Paragraph("p2", "附表正文。", [Sentence("附表正文。")], 1)],
            ),
        ],
        plain_text="正文。\n附表正文。",
    )


def test_llm_can_move_unnumbered_section_under_existing_parent():
    from app.core.structure_repair import repair_document

    provider = _FakeProvider(
        json.dumps(
            {
                "candidate_id": "section:s2",
                "action": "move_to_section",
                "source_ids": ["s2"],
                "target_section_id": "s1",
                "confidence": 0.93,
                "reason": "附表属于前一编号章节",
            },
            ensure_ascii=False,
        )
    )

    result = repair_document(_document(), provider=provider)

    assert result.document.sections[1].level == 4
    assert result.trace.operations[-1].actor == "llm"
    assert result.trace.operations[-1].type == "move_to_section"
    assert result.trace.decisions[-1].action == "move_to_section"
    assert result.trace.decisions[-1].confidence == 0.93
    assert provider.messages is not None
    prompt = provider.messages[-1]["content"]
    assert "s1" in prompt and "s2" in prompt
    assert "附表正文。" in prompt


def test_invalid_or_low_confidence_llm_response_keeps_original_structure():
    from app.core.structure_repair import repair_document

    provider = _FakeProvider(
        json.dumps(
            {
                "candidate_id": "section:s2",
                "action": "move_to_section",
                "source_ids": ["unknown"],
                "target_section_id": "s1",
                "confidence": 0.50,
                "reason": "不确定",
            },
            ensure_ascii=False,
        )
    )

    result = repair_document(_document(), provider=provider)

    assert result.document.sections[1].level == 1
    assert result.trace.operations == []
    assert result.trace.rejected


def test_llm_exception_is_nonfatal():
    from app.core.structure_repair import repair_document

    result = repair_document(_document(), provider=_FakeProvider(RuntimeError("offline")))

    assert result.document.sections[1].level == 1
    assert result.status == "unchanged"
    assert result.trace.rejected[0].code == "llm_error"


def test_valid_keep_judgment_is_traced_without_operation():
    from app.core.structure_repair import repair_document

    provider = _FakeProvider(
        json.dumps(
            {
                "candidate_id": "section:s2",
                "action": "keep",
                "source_ids": ["s2"],
                "target_section_id": "s1",
                "confidence": 0.91,
                "reason": "该标题是独立章节",
            },
            ensure_ascii=False,
        )
    )

    result = repair_document(_document(), provider=provider)

    assert result.trace.operations == []
    assert len(result.trace.decisions) == 1
    assert result.trace.decisions[0].action == "keep"
    assert result.trace.decisions[0].reason == "该标题是独立章节"


def test_llm_can_merge_ambiguous_adjacent_page_fragments():
    from app.core.structure_repair import repair_document

    document = DocumentIR(
        doc_id="doc-fragment",
        title="测试",
        file_hash="hash-fragment",
        sections=[
            Section(
                "s1",
                "1.1 功能",
                2,
                [
                    Paragraph(
                        "p1",
                        "车窗开启过程",
                        [Sentence("车窗开启过程")],
                        1,
                    ),
                    Paragraph(
                        "p2",
                        "遇到障碍物时立即下降。",
                        [Sentence("遇到障碍物时立即下降。")],
                        2,
                    ),
                ],
            )
        ],
        plain_text="车窗开启过程\n遇到障碍物时立即下降。",
    )
    provider = _FakeProvider(
        json.dumps(
            {
                "candidate_id": "paragraphs:p1:p2",
                "action": "merge_paragraphs",
                "source_ids": ["p1", "p2"],
                "target_section_id": "s1",
                "confidence": 0.92,
                "reason": "后一页延续前一页未完成语句",
            },
            ensure_ascii=False,
        )
    )

    result = repair_document(document, provider=provider)

    paragraphs = result.document.sections[0].paragraphs
    assert [paragraph.text for paragraph in paragraphs] == [
        "车窗开启过程遇到障碍物时立即下降。"
    ]
    assert result.trace.operations[-1].type == "merge_paragraphs"
    assert result.trace.operations[-1].actor == "llm"
    assert result.trace.operations[-1].source_ids == ["p1", "p2"]
