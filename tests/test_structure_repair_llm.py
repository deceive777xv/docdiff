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


def _page_noise_response(payload: dict, action_for_text) -> str:
    return json.dumps(
        {
            "boundary_id": payload["boundary_id"],
            "labels": [
                {
                    "id": item["id"],
                    "action": action_for_text(item["text"]),
                    "confidence": 0.98,
                    "reason": "fixed boundary item classification",
                }
                for item in payload["items"]
            ],
        }
    )


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
    assert any(rejected.code.endswith("llm_error") for rejected in result.trace.rejected)


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
            payload = json.loads(messages[-1]["content"])
            if "items" in payload:
                return _page_noise_response(payload, lambda _text: "keep")
            self.calls += 1
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


def test_complex_page_header_is_removed_before_paragraph_merge_adjudication():
    from app.core.structure_repair import repair_document

    class BoundaryAwareProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.merge_payloads: list[dict] = []

        def chat(self, messages: list[dict], **_kwargs) -> str:
            payload = json.loads(messages[-1]["content"])
            if "items" in payload:
                return _page_noise_response(
                    payload,
                    lambda text: (
                        "remove_as_page_noise"
                        if text.startswith("文件名称：")
                        else "keep"
                    ),
                )
            self.merge_payloads.append(payload)
            return json.dumps(
                {
                    "candidate_id": payload["candidate_id"],
                    "action": "keep",
                    "confidence": 0.97,
                    "reason": "independent paragraphs",
                }
            )

    prefix = (
        "文件名称：EREA 项目蝴蝶门开闭系统设计说明书 "
        "文件编号：EREA_STS_A19-10006 版本：001"
    )
    document = DocumentIR(
        "complex-page-header",
        "Complex page header",
        "complex-page-header-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [
                    Paragraph(
                        "h10",
                        f"{prefix} 第10页",
                        [Sentence(f"{prefix} 第10页")],
                        10,
                    ),
                    Paragraph("b10", "第一页真实正文。", [Sentence("第一页真实正文。")], 10),
                    Paragraph(
                        "h11",
                        f"{prefix} 第11页",
                        [Sentence(f"{prefix} 第11页")],
                        11,
                    ),
                    Paragraph("b11", "第二页真实正文。", [Sentence("第二页真实正文。")], 11),
                ],
            )
        ],
    )
    provider = BoundaryAwareProvider()

    result = repair_document(document, provider=provider)

    assert [
        paragraph.text for paragraph in result.document.sections[0].paragraphs
    ] == ["第一页真实正文。", "第二页真实正文。"]
    assert all(
        prefix not in json.dumps(payload, ensure_ascii=False)
        for payload in provider.merge_payloads
    )
    assert [
        operation.source_ids
        for operation in result.trace.operations
        if operation.type == "remove_noise"
    ] == [["h10"], ["h11"]]


def test_table_page_header_sentences_are_removed_before_table_merge_adjudication():
    from app.core.structure_repair import repair_document

    class RowBoundaryProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.table_merge_payloads: list[dict] = []

        def chat(self, messages: list[dict], **_kwargs) -> str:
            payload = json.loads(messages[-1]["content"])
            if "items" in payload:
                return _page_noise_response(
                    payload,
                    lambda text: (
                        "remove_as_page_noise"
                        if text.startswith("| 文件名称：")
                        else "keep"
                    ),
                )
            self.table_merge_payloads.append(payload)
            return json.dumps(
                {
                    "candidate_id": payload["candidate_id"],
                    "action": "keep",
                    "confidence": 0.98,
                    "reason": "independent table content",
                }
            )

    header = (
        "| 文件名称：EREA 项目蝴蝶门开闭系统设计说明书 "
        "| 文件编号：EREA_STS_A19-10006 | 版本：001 | 第11页 |"
    )
    separator = "| --- | --- | --- | --- |"
    body = "| 33 | 异响 | 运行过程中无异常噪声 |"
    table = Paragraph(
        "mixed-table",
        "\n".join([header, separator, body]),
        [Sentence(header), Sentence(separator), Sentence(body)],
        11,
    )
    document = DocumentIR(
        "table-page-header",
        "Table page header",
        "table-page-header-hash",
        [Section("s1", "1 Scope", 1, [table])],
    )
    provider = RowBoundaryProvider()

    result = repair_document(document, provider=provider)

    repaired = result.document.sections[0].paragraphs
    assert len(repaired) == 1
    assert repaired[0].text == body
    assert [sentence.text for sentence in repaired[0].sentences] == [body]
    assert document.sections[0].paragraphs[0].text == "\n".join(
        [header, separator, body]
    )
    assert all(
        header not in json.dumps(payload, ensure_ascii=False)
        for payload in provider.table_merge_payloads
    )


def test_independent_table_page_header_is_removed_when_all_sentences_are_noise():
    from app.core.structure_repair import repair_document

    class HeaderProvider:
        chat_model = "fake-model"

        def chat(self, messages: list[dict], **_kwargs) -> str:
            payload = json.loads(messages[-1]["content"])
            if "items" not in payload:
                raise AssertionError("empty boundary table must not reach merge LLM")
            return _page_noise_response(
                payload,
                lambda _text: "remove_as_page_noise",
            )

    header = "| 文档名称：示例说明书 | 版本：002 | 第3页 |"
    separator = "| --- | --- | --- |"
    document = DocumentIR(
        "independent-table-header",
        "Independent table header",
        "independent-table-header-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [
                    Paragraph(
                        "header-table",
                        "\n".join([header, separator]),
                        [Sentence(header), Sentence(separator)],
                        3,
                    )
                ],
            )
        ],
    )

    result = repair_document(document, provider=HeaderProvider())

    assert result.document.sections[0].paragraphs == []
    row_operation = next(
        operation
        for operation in result.trace.operations
        if operation.type == "remove_noise_rows"
    )
    assert row_operation.source_ids == ["header-table"]
    assert row_operation.source_sentence_indexes == [0, 1]


def test_page_boundary_batch_review_keeps_business_rows_and_caps_each_boundary_at_two_calls():
    from app.core.structure_repair import repair_document

    class ReviewProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.legacy_calls = 0
            self.batch_calls: list[str] = []
            self.batch_payloads: list[dict] = []

        def chat(self, messages: list[dict], **_kwargs) -> str:
            payload = json.loads(messages[-1]["content"])
            if "items" not in payload:
                self.legacy_calls += 1
                return json.dumps(
                    {
                        "candidate_id": payload["candidate_id"],
                        "action": "remove_as_page_noise",
                        "confidence": 0.99,
                        "reason": "misread numeric business content as page metadata",
                    }
                )
            self.batch_calls.append(payload["boundary_id"])
            self.batch_payloads.append(payload)
            action = "keep" if "initial_labels" in payload else "remove_as_page_noise"
            return json.dumps(
                {
                    "boundary_id": payload["boundary_id"],
                    "labels": [
                        {
                            "id": item["id"],
                            "action": action,
                            "confidence": 0.99,
                            "reason": "review recognizes a numbered business table row",
                        }
                        for item in payload["items"]
                    ],
                }
            )

    rows = [
        "||18|尺寸工程|电动蝴蝶门系统是一个虚拟总成，不单独<br>设置尺寸性能目标，相关要求如下：||/||||",
        "||34|EMC|电动吸合门锁EMC<br>要求.docx<br>电动撑杆EMC要求.<br>doc||/||||",
        "|8|环境适应性：\n高温|85℃。|Q/BYDQ-A1906.4262—2018《乘用车侧开式侧门开闭\n系统技术条件》。|",
        "|9|环境适应性：\n低温|-40℃。|Q/BYDQ-A1906.4262—2018《乘用车侧开式侧门开闭\n系统技术条件》。|",
        "|10|环境适应性：\n低温、淋雨电|门锁总成耐低温工作性\n能、耐温度变化性、低温|/|",
        "||21|防夹要求|1、防夹力：关门过程：①电撑杆高位、中位、低位（峰值）≤100 N；||/||||",
    ]
    table = Paragraph(
        "business-table",
        "\n".join(rows),
        [Sentence(row) for row in rows],
        11,
    )
    document = DocumentIR(
        "business-table-page-noise",
        "Business table page noise",
        "business-table-page-noise-hash",
        [Section("s1", "1 Scope", 1, [table])],
    )
    provider = ReviewProvider()

    result = repair_document(document, provider=provider)

    repaired_rows = [
        sentence.text
        for paragraph in result.document.sections[0].paragraphs
        for sentence in paragraph.sentences
    ]
    assert repaired_rows == rows
    assert provider.legacy_calls == 0
    assert provider.batch_calls
    initial_item_texts = [
        item["text"]
        for payload in provider.batch_payloads
        if "initial_labels" not in payload
        for item in payload["items"]
    ]
    assert sorted(initial_item_texts) == sorted(rows)
    assert all(
        set(payload) in (
            {"boundary_id", "items"},
            {"boundary_id", "items", "initial_labels"},
        )
        for payload in provider.batch_payloads
    )
    assert all(
        set(item) == {"id", "position", "text"}
        for payload in provider.batch_payloads
        for item in payload["items"]
    )
    assert all(
        [item["id"] for item in payload["items"]]
        == [f"L{index:02d}" for index in range(1, len(payload["items"]) + 1)]
        for payload in provider.batch_payloads
    )
    assert all(
        provider.batch_calls.count(boundary_id) <= 2
        for boundary_id in set(provider.batch_calls)
    )
    assert any(
        candidate.code == "page_noise_review_disagreement"
        for candidate in result.trace.rejected
    )


def test_page_boundary_batch_cannot_delete_past_a_retained_outer_item():
    from app.core.structure_repair import repair_document

    class NonContiguousProvider:
        chat_model = "fake-model"

        def chat(self, messages: list[dict], **_kwargs) -> str:
            payload = json.loads(messages[-1]["content"])
            if "items" in payload:
                return _page_noise_response(
                    payload,
                    lambda text: (
                        "remove_as_page_noise"
                        if text == "内侧误判候选"
                        else "keep"
                    ),
                )
            return json.dumps(
                {
                    "candidate_id": payload["candidate_id"],
                    "action": "keep",
                    "confidence": 0.99,
                    "reason": "independent paragraphs",
                }
            )

    texts = ["外侧真实正文。", "内侧误判候选", "中间正文。", "页尾正文。"]
    document = DocumentIR(
        "non-contiguous-page-noise",
        "Non-contiguous page noise",
        "non-contiguous-page-noise-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [
                    Paragraph(f"p{index}", text, [Sentence(text)], 1)
                    for index, text in enumerate(texts)
                ],
            )
        ],
    )

    result = repair_document(document, provider=NonContiguousProvider())

    assert [
        paragraph.text for paragraph in result.document.sections[0].paragraphs
    ] == texts
    assert any(
        candidate.code == "non_contiguous_page_noise"
        for candidate in result.trace.rejected
    )


def test_page_boundary_batch_skips_review_when_initial_labels_all_keep():
    from app.core.structure_repair import repair_document

    class KeepProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.batch_calls: list[str] = []

        def chat(self, messages: list[dict], **_kwargs) -> str:
            payload = json.loads(messages[-1]["content"])
            if "items" in payload:
                self.batch_calls.append(payload["boundary_id"])
                return _page_noise_response(payload, lambda _text: "keep")
            return json.dumps(
                {
                    "candidate_id": payload["candidate_id"],
                    "action": "keep",
                    "confidence": 0.99,
                    "reason": "independent paragraphs",
                }
            )

    document = DocumentIR(
        "single-pass-page-noise",
        "Single pass page noise",
        "single-pass-page-noise-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [
                    Paragraph(
                        f"p{index}",
                        f"第{index + 1}条真实正文。",
                        [Sentence(f"第{index + 1}条真实正文。")],
                        1,
                    )
                    for index in range(6)
                ],
            )
        ],
    )
    provider = KeepProvider()

    repair_document(document, provider=provider)

    assert provider.batch_calls
    assert all(
        provider.batch_calls.count(boundary_id) == 1
        for boundary_id in set(provider.batch_calls)
    )


def test_page_boundary_batch_keeps_everything_when_review_ids_are_incomplete():
    from app.core.structure_repair import repair_document

    class IncompleteReviewProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict], **_kwargs) -> str:
            self.calls += 1
            payload = json.loads(messages[-1]["content"])
            if "initial_labels" not in payload:
                return _page_noise_response(
                    payload,
                    lambda _text: "remove_as_page_noise",
                )
            return json.dumps(
                {
                    "boundary_id": payload["boundary_id"],
                    "labels": [],
                }
            )

    body = "真实正文不得因不完整复审而删除。"
    document = DocumentIR(
        "incomplete-page-noise-review",
        "Incomplete page noise review",
        "incomplete-page-noise-review-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [Paragraph("body", body, [Sentence(body)], 1)],
            )
        ],
    )
    provider = IncompleteReviewProvider()

    result = repair_document(document, provider=provider)

    assert provider.calls == 2
    assert result.document.sections[0].paragraphs[0].text == body
    assert result.trace.rejected[0].code == "review_label_set_mismatch"


def test_page_noise_llm_does_not_retry_invalid_batch_fields():
    from app.core.structure_repair import repair_document

    class InvalidProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.messages: list[list[dict]] = []

        def chat(self, messages: list[dict], **_kwargs) -> str:
            self.messages.append(messages)
            payload = json.loads(messages[-1]["content"])
            response = {
                "boundary_id": payload["boundary_id"],
                "labels": [
                    {
                        "id": item["id"],
                        "action": "remove_as_page_noise",
                        "confidence": 0.99,
                        "reason": "incorrect deletion proposal",
                    }
                    for item in payload["items"]
                ],
                "selected_context": 1,
            }
            return json.dumps(response)

    document = DocumentIR(
        "strict-page-noise",
        "Strict page noise",
        "strict-page-noise-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [Paragraph("body", "真实正文。", [Sentence("真实正文。")], 1)],
            )
        ],
    )
    provider = InvalidProvider()

    result = repair_document(document, provider=provider)

    assert len(provider.messages) == 1
    system_prompt = provider.messages[0][0]["content"]
    assert "Every item has exactly id, position, and text" in system_prompt
    assert "labels must contain exactly one object per input item" in system_prompt
    assert "Never omit, add, reorder, replace, merge, or split items" in system_prompt
    assert [
        paragraph.text for paragraph in result.document.sections[0].paragraphs
    ] == ["真实正文。"]
    assert result.trace.rejected[0].code == "initial_invalid_fields"


def test_page_noise_llm_cannot_replace_a_fixed_batch_item():
    from app.core.structure_repair import repair_document

    class WrongCandidateProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict], **_kwargs) -> str:
            self.calls += 1
            payload = json.loads(messages[-1]["content"])
            return json.dumps(
                {
                    "boundary_id": payload["boundary_id"],
                    "labels": [
                        {
                            "id": "L99",
                            "action": "remove_as_page_noise",
                            "confidence": 0.99,
                            "reason": "selected a different batch item",
                        }
                        for _item in payload["items"]
                    ],
                }
            )

    document = DocumentIR(
        "wrong-page-noise-candidate",
        "Wrong page-noise candidate",
        "wrong-page-noise-candidate-hash",
        [
            Section(
                "s1",
                "1 Scope",
                1,
                [Paragraph("body", "真实正文。", [Sentence("真实正文。")], 1)],
            )
        ],
    )
    provider = WrongCandidateProvider()

    result = repair_document(document, provider=provider)

    assert provider.calls == 1
    assert [
        paragraph.text for paragraph in result.document.sections[0].paragraphs
    ] == ["真实正文。"]
    rejection = next(
        rejected
        for rejected in result.trace.rejected
        if rejected.candidate_id.startswith("page-boundary-")
    )
    assert rejection.code == "initial_label_set_mismatch"


def test_import_normalization_adjudicates_every_eligible_paragraph_candidate():
    from app.core.structure_repair import repair_document

    class KeepAllProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.candidate_ids: list[str] = []

        def chat(self, messages: list[dict], **_kwargs) -> str:
            payload = json.loads(messages[-1]["content"])
            if "items" in payload:
                return _page_noise_response(payload, lambda _text: "keep")
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

    paragraph_decisions = [
        decision
        for decision in result.trace.decisions
        if decision.candidate_id.startswith("paragraphs:")
    ]
    assert len(provider.candidate_ids) == 49
    assert len(paragraph_decisions) == 49
    assert provider.candidate_ids[-1] == "paragraphs:p48:p49"
    assert paragraph_decisions[-1].candidate_id == "paragraphs:p48:p49"


def test_llm_paragraph_merge_rechecks_the_new_fragment_across_three_pages():
    from app.core.structure_repair import repair_document

    class MergeProvider:
        chat_model = "fake-model"

        def __init__(self) -> None:
            self.payloads: list[dict] = []

        def chat(self, messages: list[dict], **_kwargs) -> str:
            payload = json.loads(messages[-1]["content"])
            if "items" in payload:
                return _page_noise_response(payload, lambda _text: "keep")
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
            payload = json.loads(messages[-1]["content"])
            if "items" in payload:
                return _page_noise_response(payload, lambda _text: "keep")
            self.messages.append(messages)
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
