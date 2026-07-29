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
                        "控制系统在车窗持续开启的整个运行过程中监测",
                        [Sentence("控制系统在车窗持续开启的整个运行过程中监测")],
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
        plain_text="控制系统在车窗持续开启的整个运行过程中监测\n遇到障碍物时立即下降。",
    )
    provider = _FakeProvider(
        json.dumps(
            {
                "candidate_id": "paragraphs:p1:p2",
                "action": "merge_paragraphs",
                "confidence": 0.92,
                "reason": "后一页延续前一页未完成语句",
            },
            ensure_ascii=False,
        )
    )

    result = repair_document(document, provider=provider)

    paragraphs = result.document.sections[0].paragraphs
    assert [paragraph.text for paragraph in paragraphs] == [
        "控制系统在车窗持续开启的整个运行过程中监测遇到障碍物时立即下降。"
    ]
    assert result.trace.operations[-1].type == "merge_paragraphs"
    assert result.trace.operations[-1].actor == "llm"
    assert result.trace.operations[-1].source_ids == ["p1", "p2"]


def test_small_heading_is_not_sent_to_paragraph_merge_llm():
    from app.core.structure_repair import repair_document

    class AlwaysMergeProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict], **_kwargs) -> str:
            self.calls += 1
            payload = json.loads(messages[-1]["content"])
            return json.dumps(
                {
                    "candidate_id": payload["candidate_id"],
                    "action": "merge_paragraphs",
                    "confidence": 0.99,
                    "reason": "always merge for regression coverage",
                }
            )

    document = DocumentIR(
        "small-heading",
        "Small heading",
        "small-heading-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [
                    Paragraph("heading", "C）功能约束", [Sentence("C）功能约束")], 1),
                    Paragraph(
                        "body",
                        "执行机构在整个运行过程中应当满足规定的负载要求。",
                        [Sentence("执行机构在整个运行过程中应当满足规定的负载要求。")],
                        1,
                    ),
                ],
            )
        ],
    )
    provider = AlwaysMergeProvider()

    result = repair_document(document, provider=provider)

    assert [
        paragraph.text for paragraph in result.document.sections[0].paragraphs
    ] == [
        "C）功能约束",
        "执行机构在整个运行过程中应当满足规定的负载要求。",
    ]
    assert provider.calls == 0
    assert all(
        operation.type != "merge_paragraphs"
        for operation in result.trace.operations
    )


def test_unmarked_small_heading_is_not_sent_to_paragraph_merge_llm():
    from app.core.structure_repair import repair_document

    class FailIfCalledProvider:
        chat_model = "fake-model"

        def chat(self, _messages: list[dict], **_kwargs) -> str:
            raise AssertionError("heading boundary must be rejected before LLM")

    document = DocumentIR(
        "unmarked-small-heading",
        "Unmarked small heading",
        "unmarked-small-heading-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [
                    Paragraph("heading", "环境要求", [Sentence("环境要求")], 1),
                    Paragraph(
                        "body",
                        "设备在规定的温度和湿度范围内应保持连续稳定运行。",
                        [Sentence("设备在规定的温度和湿度范围内应保持连续稳定运行。")],
                        1,
                    ),
                ],
            )
        ],
    )

    result = repair_document(document, provider=FailIfCalledProvider())

    assert [
        paragraph.text for paragraph in result.document.sections[0].paragraphs
    ] == ["环境要求", "设备在规定的温度和湿度范围内应保持连续稳定运行。"]
    assert all(
        operation.type != "merge_paragraphs"
        for operation in result.trace.operations
    )


def test_small_heading_in_continuation_slot_is_not_sent_to_paragraph_merge_llm():
    from app.core.structure_repair import repair_document

    class FailIfCalledProvider:
        chat_model = "fake-model"

        def chat(self, _messages: list[dict], **_kwargs) -> str:
            raise AssertionError("heading boundary must be rejected before LLM")

    document = DocumentIR(
        "continuation-small-heading",
        "Continuation small heading",
        "continuation-small-heading-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [
                    Paragraph(
                        "body",
                        "控制系统在检测到运行环境发生异常变化时应当",
                        [Sentence("控制系统在检测到运行环境发生异常变化时应当")],
                        1,
                    ),
                    Paragraph("heading", "（D）处置要求", [Sentence("（D）处置要求")], 2),
                ],
            )
        ],
    )

    result = repair_document(document, provider=FailIfCalledProvider())

    assert [
        paragraph.text for paragraph in result.document.sections[0].paragraphs
    ] == ["控制系统在检测到运行环境发生异常变化时应当", "（D）处置要求"]
    assert all(
        operation.type != "merge_paragraphs"
        for operation in result.trace.operations
    )


def test_import_normalization_adjudicates_every_eligible_paragraph_candidate():
    from app.core.structure_repair import repair_document

    class KeepAllProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.candidate_ids: list[str] = []

        def chat(self, messages: list[dict], **_kwargs) -> str:
            payload = json.loads(messages[-1]["content"])
            candidate_id = payload["candidate_id"]
            self.candidate_ids.append(candidate_id)
            return json.dumps(
                {
                    "candidate_id": candidate_id,
                    "action": "keep",
                    "confidence": 0.91,
                    "reason": "independent paragraphs",
                }
            )

    paragraphs = []
    for pair_index in range(25):
        page_no = pair_index + 1
        paragraphs.extend(
            [
                Paragraph(
                    f"p{pair_index * 2}",
                    f"Fragment {pair_index:02d} remains open",
                    [Sentence(f"Fragment {pair_index:02d} remains open")],
                    page_no,
                ),
                Paragraph(
                    f"p{pair_index * 2 + 1}",
                    f"Continuation paragraph content number {pair_index:02d}.",
                    [Sentence(f"Continuation paragraph content number {pair_index:02d}.")],
                    page_no,
                ),
            ]
        )
    document = DocumentIR(
        doc_id="doc-all-candidates",
        title="All candidates",
        file_hash="hash-all-candidates",
        sections=[Section("s1", "1 Scope", 1, paragraphs)],
        plain_text="\n".join(paragraph.text for paragraph in paragraphs),
    )
    provider = KeepAllProvider()

    result = repair_document(document, provider=provider)

    assert len(provider.candidate_ids) == 49
    assert len(result.trace.decisions) == 49
    assert provider.candidate_ids[-1] == "paragraphs:p48:p49"
    assert result.trace.decisions[-1].candidate_id == "paragraphs:p48:p49"


def test_llm_paragraph_merge_rechecks_the_new_fragment_across_three_pages():
    from app.core.structure_repair import repair_document

    class MergeProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.payloads: list[dict] = []

        def chat(self, messages: list[dict], **_kwargs) -> str:
            payload = json.loads(messages[-1]["content"])
            self.payloads.append(payload)
            return json.dumps(
                {
                    "candidate_id": payload["candidate_id"],
                    "action": "merge_paragraphs",
                    "confidence": 0.94,
                    "reason": "the fixed continuation completes the open sentence",
                }
            )

    document = DocumentIR(
        "three-page-fragment",
        "Three pages",
        "three-page-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [
                    Paragraph(
                        "p1",
                        "当控制系统持续监测运行区域内的异常状态并检测到",
                        [Sentence("当控制系统持续监测运行区域内的异常状态并检测到")],
                        1,
                    ),
                    Paragraph(
                        "p2",
                        "可能影响设备安全运行的障碍物或其他风险因素并",
                        [Sentence("可能影响设备安全运行的障碍物或其他风险因素并")],
                        2,
                    ),
                    Paragraph("p3", "立即执行回退。", [Sentence("立即执行回退。")], 3),
                ],
            )
        ],
        "当控制系统持续监测运行区域内的异常状态并检测到\n"
        "可能影响设备安全运行的障碍物或其他风险因素并\n"
        "立即执行回退。",
    )
    provider = MergeProvider()

    result = repair_document(document, provider=provider)

    assert [
        paragraph.text for paragraph in result.document.sections[0].paragraphs
    ] == [
        "当控制系统持续监测运行区域内的异常状态并检测到"
        "可能影响设备安全运行的障碍物或其他风险因素并立即执行回退。"
    ]
    assert len(provider.payloads) == 2
    assert all(set(payload) <= {
        "candidate_id",
        "candidate",
        "pages",
        "nearby_context",
        "rule_evidence",
    } for payload in provider.payloads)
    assert all("kind" not in payload for payload in provider.payloads)
    assert len(
        [
            operation
            for operation in result.trace.operations
            if operation.type == "merge_paragraphs" and operation.actor == "llm"
        ]
    ) == 2


def test_paragraph_llm_retries_extra_fields_and_enforces_detailed_contract():
    from app.core.structure_repair import repair_document

    class RetryProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.messages: list[list[dict]] = []

        def chat(self, messages: list[dict], **_kwargs) -> str:
            self.messages.append(messages)
            payload = json.loads(messages[-1]["content"])
            response = {
                "candidate_id": payload["candidate_id"],
                "action": "merge_paragraphs",
                "confidence": 0.93,
                "reason": "the continuation completes the fixed previous slot",
            }
            if len(self.messages) == 1:
                response["source_ids"] = ["p1", "p2"]
            return json.dumps(response)

    document = DocumentIR(
        "strict-paragraph",
        "Strict",
        "strict-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [
                    Paragraph(
                        "p1",
                        "控制系统在持续监测运行区域内的异常状态时检测到",
                        [Sentence("控制系统在持续监测运行区域内的异常状态时检测到")],
                        1,
                    ),
                    Paragraph("p2", "障碍物后回退。", [Sentence("障碍物后回退。")], 2),
                ],
            )
        ],
    )
    provider = RetryProvider()

    result = repair_document(document, provider=provider)

    assert len(provider.messages) == 2
    retry_prompt = provider.messages[1][0]["content"]
    assert "exactly these four fields" in retry_prompt
    assert "invalid_fields" in retry_prompt
    assert "fixed slots" in retry_prompt
    assert result.document.sections[0].paragraphs[0].text == (
        "控制系统在持续监测运行区域内的异常状态时检测到障碍物后回退。"
    )


def test_llm_preserves_simple_repeated_header_table_merge_capability():
    from app.core.structure_repair import repair_document

    first = "\n".join(
        ["| 序号 | 项目 |", "| --- | --- |", "| 1 | A |"]
    )
    second = "\n".join(
        ["| 序号 | 项目 |", "| --- | --- |", "| 2 | B |"]
    )
    provider = _FakeProvider(
        json.dumps(
            {
                "candidate_id": "tables:t1:t2",
                "action": "merge_fragments",
                "confidence": 0.95,
                "reason": "repeated header followed by a new row of the same table",
            }
        )
    )
    document = DocumentIR(
        "simple-table",
        "Simple table",
        "simple-table-hash",
        [
            Section(
                "s1",
                "1 Table",
                1,
                [
                    Paragraph("t1", first, [Sentence(row) for row in first.splitlines()], 1),
                    Paragraph("t2", second, [Sentence(row) for row in second.splitlines()], 2),
                ],
            )
        ],
    )

    result = repair_document(document, provider=provider)

    assert len(result.document.sections[0].paragraphs) == 1
    assert result.document.sections[0].paragraphs[0].text.endswith("| 2 | B |")
    operation = result.trace.operations[-1]
    assert operation.type == "merge_table_fragments"
    assert operation.actor == "llm"
    assert provider.messages is not None
    payload = json.loads(provider.messages[-1]["content"])
    assert set(payload) == {"candidate_id", "candidate", "pages"}
    assert "kind" not in payload
