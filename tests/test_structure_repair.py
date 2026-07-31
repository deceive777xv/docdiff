from __future__ import annotations

from copy import deepcopy

from app.core.types import DocumentIR, Paragraph, Section, Sentence


def _paragraph(paragraph_id: str, text: str, page_no: int | None = None) -> Paragraph:
    return Paragraph(
        paragraph_id=paragraph_id,
        text=text,
        sentences=[Sentence(text=text)],
        page_no=page_no,
    )


def _document(sections: list[Section]) -> DocumentIR:
    return DocumentIR(
        doc_id="doc-1",
        title="测试文档",
        file_hash="hash-1",
        sections=sections,
        plain_text="\n".join(
            paragraph.text
            for section in sections
            for paragraph in section.paragraphs
        ),
    )


def test_repair_normalizes_titles_and_promotes_strict_numbered_heading():
    from app.core.structure_repair import repair_document

    raw = _document(
        [
            Section(
                "s1",
                "**1.1.3 关闭功能**",
                3,
                [
                    _paragraph("p1", "关闭功能说明。", 1),
                    _paragraph("p2", "1.1.4 开启关闭防夹功能", 1),
                    _paragraph("p3", "开启功能说明。", 1),
                ],
            )
        ]
    )

    result = repair_document(raw)

    assert raw.sections[0].title == "**1.1.3 关闭功能**"
    assert [section.title for section in result.document.sections] == [
        "1.1.3 关闭功能",
        "1.1.4 开启关闭防夹功能",
    ]
    assert [p.text for p in result.document.sections[0].paragraphs] == ["关闭功能说明。"]
    assert [p.text for p in result.document.sections[1].paragraphs] == ["开启功能说明。"]
    assert result.document.sections[1].level == 3
    assert {operation.type for operation in result.trace.operations} == {
        "normalize_title",
        "promote_to_section",
    }


def test_repair_demotes_repeated_generic_sections_under_numbered_parent():
    from app.core.structure_repair import repair_document

    raw = _document(
        [
            Section("s1", "1.1.18 悬停功能", 3, [_paragraph("p1", "总述。")]),
            Section("s2", "A）触发条件", 1, [_paragraph("p2", "条件一。")]),
            Section("s3", "B）触发条件及执行动作", 1, [_paragraph("p3", "动作一。")]),
            Section("s4", "1.1.19 其他功能", 3, [_paragraph("p4", "其他。")]),
        ]
    )

    result = repair_document(raw)

    assert [section.title for section in result.document.sections] == [
        "1.1.18 悬停功能",
        "1.1.19 其他功能",
    ]
    assert [p.text for p in result.document.sections[0].paragraphs] == [
        "总述。",
        "A）触发条件",
        "条件一。",
        "B）触发条件及执行动作",
        "动作一。",
    ]
    assert all(
        operation.type == "demote_to_paragraph"
        for operation in result.trace.operations
    )


def test_repair_removes_only_stable_page_boundary_noise():
    from app.core.structure_repair import repair_document

    raw = _document(
        [
            Section(
                "s1",
                "1 范围",
                1,
                [
                    _paragraph("h1", "升降器设计校核表-20251213-v1", 1),
                    _paragraph("b1", "第一页正文。", 1),
                    _paragraph("f1", "第 1 页", 1),
                    _paragraph("h2", "升降器设计校核表-20251213-v1", 2),
                    _paragraph("b2", "第二页正文。", 2),
                    _paragraph("f2", "第 2 页", 2),
                ],
            )
        ]
    )

    result = repair_document(raw)

    assert [p.text for p in result.document.sections[0].paragraphs] == [
        "第一页正文。",
        "第二页正文。",
    ]
    removed_ids = {
        source_id
        for operation in result.trace.operations
        if operation.type == "remove_noise"
        for source_id in operation.source_ids
    }
    assert removed_ids == {"h1", "f1", "h2", "f2"}


def test_repair_keeps_pure_number_inside_page_body():
    from app.core.structure_repair import repair_document

    raw = _document(
        [
            Section(
                "s1",
                "1 范围",
                1,
                [
                    _paragraph("b1", "数值如下：", 1),
                    _paragraph("number", "510", 1),
                    _paragraph("b2", "单位为次。", 1),
                ],
            )
        ]
    )

    result = repair_document(raw)

    assert [p.text for p in result.document.sections[0].paragraphs] == [
        "数值如下：",
        "510",
        "单位为次。",
    ]


def test_repair_removes_strict_placeholders_and_repeated_document_codes_without_pages():
    from app.core.structure_repair import repair_document

    raw = _document(
        [
            Section(
                "s1",
                "1 范围",
                1,
                [
                    _paragraph("b1", "第一页正文。"),
                    _paragraph("m1", "FMA-272-A19-V01（20171013）"),
                    _paragraph(
                        "i1",
                        "==> picture [109 x 26] intentionally omitted <==",
                    ),
                    _paragraph("b2", "第二页正文。"),
                    _paragraph("m2", "FMA-272-A19-V01（20171013）"),
                ],
            )
        ]
    )

    result = repair_document(raw)

    assert [p.text for p in result.document.sections[0].paragraphs] == [
        "第一页正文。",
        "第二页正文。",
    ]
    assert {
        source_id
        for operation in result.trace.operations
        if operation.type == "remove_noise"
        for source_id in operation.source_ids
    } == {"m1", "i1", "m2"}


def test_repair_removes_repeated_isolated_punctuation_between_body_text():
    from app.core.structure_repair import repair_document

    raw = _document(
        [
            Section(
                "s1",
                "1 范围",
                1,
                [
                    _paragraph("b1", "第一项正文。"),
                    _paragraph("n1", "……"),
                    _paragraph("b2", "第二项正文。"),
                    _paragraph("n2", "……"),
                    _paragraph("b3", "第三项正文。"),
                ],
            )
        ]
    )

    result = repair_document(raw)

    assert [p.text for p in result.document.sections[0].paragraphs] == [
        "第一项正文。",
        "第二项正文。",
        "第三项正文。",
    ]


def test_repair_does_not_rule_merge_certain_sentence_fragments_without_llm():
    from app.core.structure_repair import repair_document

    raw = _document(
        [
            Section(
                "s1",
                "1 范围",
                1,
                [
                    _paragraph("p1", "当车窗上升过程中检测到", 1),
                    _paragraph("p2", "障碍物时，车窗立即下降。", 1),
                    _paragraph("p3", "短标题", 1),
                    _paragraph("p4", "这是独立正文。", 1),
                ],
            )
        ]
    )

    result = repair_document(raw)

    assert [p.text for p in result.document.sections[0].paragraphs] == [
        "当车窗上升过程中检测到",
        "障碍物时，车窗立即下降。",
        "短标题",
        "这是独立正文。",
    ]
    assert all(
        operation.type != "merge_paragraphs"
        for operation in result.trace.operations
    )


def test_repair_does_not_merge_fragments_without_physical_page_evidence():
    from app.core.structure_repair import repair_document

    raw = _document(
        [
            Section(
                "s1",
                "1 范围",
                1,
                [
                    _paragraph("p1", "开启过程中检测到"),
                    _paragraph("p2", "FMA-272-A19-V01（20171013）"),
                ],
            )
        ]
    )

    result = repair_document(raw)

    assert [p.text for p in result.document.sections[0].paragraphs] == [
        "开启过程中检测到",
        "FMA-272-A19-V01（20171013）",
    ]
    assert all(
        operation.type != "merge_paragraphs"
        for operation in result.trace.operations
    )


def test_repair_leaves_table_fragment_ownership_to_table_normalization():
    from app.core.structure_repair import repair_document

    first = "\n".join(
        [
            "| 序号 | 项目 |",
            "| --- | --- |",
            "| 1 | A |",
        ]
    )
    second = "\n".join(
        [
            "| 序号 | 项目 |",
            "| --- | --- |",
            "| 2 | B |",
        ]
    )
    raw = _document(
        [
            Section(
                "s1",
                "1 表格",
                1,
                [_paragraph("t1", first, 1), _paragraph("t2", second, 2)],
            )
        ]
    )

    result = repair_document(raw)

    assert [
        paragraph.text for paragraph in result.document.sections[0].paragraphs
    ] == [first, second]
    assert all(
        operation.type != "merge_table_fragments"
        for operation in result.trace.operations
    )


def test_repair_is_idempotent_and_preserves_raw_document():
    from app.core.structure_repair import repair_document
    from app.core.document_ir_codec import document_ir_to_dict

    raw = _document(
        [
            Section(
                "s1",
                "**1.1 测试**",
                2,
                [
                    _paragraph("p1", "第一段尚未结束", 1),
                    _paragraph("p2", "并在这里结束。", 1),
                ],
            )
        ]
    )
    snapshot = deepcopy(raw)

    first = repair_document(raw)
    second = repair_document(first.document)

    assert raw == snapshot
    assert document_ir_to_dict(first.document) == document_ir_to_dict(second.document)
    assert second.trace.operations == []
    assert second.status == "unchanged"


def test_unexplained_content_loss_rejects_only_the_unsafe_stage(monkeypatch):
    from app.core.structure_repair import pipeline
    from app.core.document_ir_codec import document_ir_to_dict
    from app.core.structure_repair.models import StructureRepairOperation

    raw = _document(
        [
            Section(
                "s1",
                "**1 范围**",
                1,
                [
                    _paragraph("p1", "必须保留的正文。", 1),
                    _paragraph("p2", "另一段正文。", 1),
                ],
            )
        ]
    )

    def corrupt(document, operations):
        document.sections[0].paragraphs.pop()
        operations.append(
            StructureRepairOperation(
                operation_id="bad-remove",
                type="remove_noise",
                source_ids=["p1"],
                reason="incorrectly classified as noise",
            )
        )

    monkeypatch.setattr(pipeline, "_remove_noise", corrupt)

    result = pipeline.repair_document(raw)

    assert result.status == "repaired"
    assert result.document.sections[0].title == "1 范围"
    assert document_ir_to_dict(result.document) != document_ir_to_dict(raw)
    assert [
        paragraph.paragraph_id
        for paragraph in result.document.sections[0].paragraphs
    ] == [
        "p1",
        "p2",
    ]
    assert [operation.type for operation in result.trace.operations] == [
        "normalize_title"
    ]
    assert result.trace.status == "repaired"
    assert result.trace.warnings == ["content_conservation_failed:remove_noise"]
    assert any(
        candidate.candidate_id == "bad-remove"
        and candidate.code == "content_conservation_failed"
        and "stage was discarded" in candidate.reason
        for candidate in result.trace.rejected
    )


def test_content_conservation_can_be_disabled_for_diagnostics(
    monkeypatch,
):
    from app.core.structure_repair import pipeline
    from app.core.structure_repair.models import StructureRepairOperation

    raw = _document(
        [
            Section(
                "s1",
                "**1 范围**",
                1,
                [
                    _paragraph("p1", "必须保留的正文。", 1),
                    _paragraph("p2", "另一段正文。", 1),
                ],
            )
        ]
    )

    def corrupt(document, operations):
        document.sections[0].paragraphs.pop()
        operations.append(
            StructureRepairOperation(
                operation_id="bad-remove",
                type="remove_noise",
                source_ids=["p1"],
                reason="incorrectly classified as noise",
            )
        )

    monkeypatch.setattr(pipeline, "_remove_noise", corrupt)
    monkeypatch.setenv(pipeline.CONTENT_CONSERVATION_DISABLE_ENV, "1")

    result = pipeline.repair_document(raw)

    assert [
        paragraph.paragraph_id
        for paragraph in result.document.sections[0].paragraphs
    ] == [
        "p1"
    ]
    assert [operation.type for operation in result.trace.operations] == [
        "normalize_title"
    ]
    assert any(
        candidate.candidate_id == "bad-remove"
        and candidate.code == "content_conservation_failed"
        and "diagnostic output was retained" in candidate.reason
        for candidate in result.trace.rejected
    )
    assert result.trace.warnings == [
        "content_conservation_disabled",
        "content_conservation_failed:remove_noise",
    ]
