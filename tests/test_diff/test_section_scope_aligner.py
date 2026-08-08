from __future__ import annotations

import re

from app.core.diff.section_scope_aligner import (
    align_compare_scopes,
    normalize_fake_title_evidence,
)
from app.core.diff.semantic_matcher import match_paragraphs
from app.core.model.base_provider import BaseProvider
from app.core.types import DocumentIR, Paragraph, Section, Sentence


class CharacterEmbedder(BaseProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        character_sets = [
            set(re.findall(r"[A-Za-z0-9]|[\u4e00-\u9fff]", text.lower()))
            for text in texts
        ]
        vocabulary = sorted(set().union(*character_sets))
        return [
            [1.0 if character in characters else 0.0 for character in vocabulary]
            for characters in character_sets
        ]

    def chat(self, messages, **kwargs) -> str:  # pragma: no cover
        raise NotImplementedError

    def health_check(self) -> bool:
        return True


class ConceptEmbedder(BaseProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "障碍物" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "安全距离" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    def chat(self, messages, **kwargs) -> str:  # pragma: no cover
        raise NotImplementedError

    def health_check(self) -> bool:
        return True


def paragraph(paragraph_id: str, text: str) -> Paragraph:
    return Paragraph(paragraph_id, text, [Sentence(text)])


def table_paragraph(paragraph_id: str, rows: list[str]) -> Paragraph:
    return Paragraph(
        paragraph_id,
        "\n".join(rows),
        [Sentence(row) for row in rows],
    )


def section(
    section_id: str,
    title: str,
    paragraphs: list[Paragraph],
    *,
    level: int = 1,
) -> Section:
    return Section(section_id, title, level, paragraphs)


def document(doc_id: str, sections: list[Section]) -> DocumentIR:
    return DocumentIR(doc_id, doc_id, doc_id, sections)


def group_for(plan, side: str, section_id: str):
    attribute = "baseline_sections" if side == "baseline" else "target_sections"
    return next(
        group
        for group in plan.groups
        if any(section.section_id == section_id for section in getattr(group, attribute))
    )


def test_title_identity_allows_reordered_sections():
    baseline = document(
        "baseline",
        [
            section("b-a", "A section", [paragraph("b-a-p", "Alpha body")]),
            section("b-b", "B section", [paragraph("b-b-p", "Beta body")]),
        ],
    )
    target = document(
        "target",
        [
            section("t-b", "B section", [paragraph("t-b-p", "Beta body")]),
            section("t-a", "A section", [paragraph("t-a-p", "Alpha body")]),
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    assert [section.section_id for section in group_for(plan, "baseline", "b-a").target_sections] == ["t-a"]
    assert [section.section_id for section in group_for(plan, "baseline", "b-b").target_sections] == ["t-b"]


def test_exact_body_anchors_align_fully_renamed_reordered_sections():
    shared_a = "唯一且足够长的第一章节正文内容用于证明章节身份。"
    shared_b = "另一段唯一且足够长的第二章节正文内容用于证明章节身份。"
    baseline = document(
        "baseline",
        [
            section("b-a", "Copper", [paragraph("b-a-p", shared_a)]),
            section("b-b", "Forest", [paragraph("b-b-p", shared_b)]),
        ],
    )
    target = document(
        "target",
        [
            section("t-b", "Ocean", [paragraph("t-b-p", shared_b)]),
            section("t-a", "Sky", [paragraph("t-a-p", shared_a)]),
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    assert [section.section_id for section in group_for(plan, "baseline", "b-a").target_sections] == ["t-a"]
    assert [section.section_id for section in group_for(plan, "baseline", "b-b").target_sections] == ["t-b"]
    assert all(group.evidence[0].kind == "body_exact" for group in plan.groups)


def test_semantic_body_evidence_aligns_renamed_section_with_modified_text():
    baseline = document(
        "baseline",
        [
            section(
                "b",
                "Copper",
                [
                    paragraph("b-1", "探测到障碍物后停止开门。"),
                    paragraph("b-2", "车门必须保持安全距离。"),
                ],
            )
        ],
    )
    target = document(
        "target",
        [
            section(
                "t",
                "Sky",
                [
                    paragraph("t-1", "发现障碍物时暂停开门动作。"),
                    paragraph("t-2", "车门应当维持安全距离。"),
                ],
            )
        ],
    )

    plan = align_compare_scopes(baseline, target, ConceptEmbedder())

    assert len(plan.groups) == 1
    assert plan.groups[0].evidence[0].kind == "body_semantic"


def test_fake_title_paragraph_evidence_opens_only_its_adjacent_boundary():
    baseline_main = section("b-main", "Main", [paragraph("b-main-p", "main")])
    fake = section(
        "b-fake",
        "1）安全距离要求。",
        [paragraph("b-fake-p", "需要跨边界参与匹配的正文。")],
    )
    baseline_next = section("b-next", "Next", [paragraph("b-next-p", "next")])
    target_main = section(
        "t-main",
        "Main",
        [paragraph("title-as-body", "1 安全距离要求")],
    )
    target_next = section("t-next", "Next", [paragraph("t-next-p", "next")])

    plan = align_compare_scopes(
        document("baseline", [baseline_main, fake, baseline_next]),
        document("target", [target_main, target_next]),
        CharacterEmbedder(),
    )

    group = group_for(plan, "baseline", "b-fake")
    assert [section.section_id for section in group.baseline_sections] == ["b-main", "b-fake"]
    assert group.baseline_crossable_boundaries == frozenset({("b-main", "b-fake")})
    evidence = next(item for item in group.evidence if item.kind == "fake_paragraph")
    assert evidence.content_ref is not None
    assert evidence.content_ref.paragraph_id == "title-as-body"


def test_fake_title_requires_uniqueness_inside_anchor_interval():
    baseline = document(
        "baseline",
        [
            section("b-main", "Main", []),
            section("b-fake", "Repeated title", [paragraph("body", "body")]),
            section("b-next", "Next", []),
        ],
    )
    target = document(
        "target",
        [
            section("t-main", "Main", [paragraph("first", "Repeated title")]),
            section("t-next", "Next", [paragraph("second", "Repeated title")]),
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    fake_group = group_for(plan, "baseline", "b-fake")
    assert [section.section_id for section in fake_group.baseline_sections] == ["b-fake"]
    assert not fake_group.baseline_crossable_boundaries
    assert not any(
        evidence.kind.startswith("fake_")
        for group in plan.groups
        for evidence in group.evidence
    )


def test_table_boundary_full_cell_can_prove_fake_title():
    baseline = document(
        "baseline",
        [
            section("b-main", "Main", []),
            section("b-fake", "Boundary item。", [paragraph("body", "body")]),
            section("b-next", "Next", []),
        ],
    )
    table = table_paragraph(
        "table",
        [
            "| Name | Value |",
            "| --- | --- |",
            "| Boundary item | kept |",
        ],
    )
    target = document(
        "target",
        [
            section("t-main", "Main", [table]),
            section("t-next", "Next", []),
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    group = group_for(plan, "baseline", "b-fake")
    evidence = next(
        evidence
        for evidence in group.evidence
        if evidence.kind == "fake_table_boundary_cell"
    )
    assert evidence.content_ref is not None
    assert evidence.content_ref.paragraph_id == "table"
    assert evidence.content_ref.sentence_index == 2
    assert group.baseline_crossable_boundaries == frozenset({("b-main", "b-fake")})


def test_table_middle_row_does_not_prove_fake_title():
    baseline = document(
        "baseline",
        [
            section("b-main", "Main", []),
            section("b-fake", "Middle item", [paragraph("body", "body")]),
            section("b-next", "Next", []),
        ],
    )
    table = table_paragraph(
        "table",
        [
            "| Name | Value |",
            "| --- | --- |",
            "| Middle item | ignored |",
            "| Last item | kept |",
        ],
    )
    target = document(
        "target",
        [
            section("t-main", "Main", [table]),
            section("t-next", "Next", []),
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    fake_group = group_for(plan, "baseline", "b-fake")
    assert [section.section_id for section in fake_group.baseline_sections] == ["b-fake"]
    assert not fake_group.baseline_crossable_boundaries


def test_fake_title_normalization_removes_punctuation_but_keeps_symbols():
    assert normalize_fake_title_evidence("Ａ）安全距离要求。") == "a安全距离要求"
    assert normalize_fake_title_evidence("阈值≥90%") != normalize_fake_title_evidence("阈值90%")


def test_fake_boundary_allows_exact_two_to_one_paragraph_window():
    first = "系统探测到障碍物以后立即停止当前开门动作。"
    second = "车门随后保持规定的安全距离并等待新的指令。"
    baseline = document(
        "baseline",
        [
            section("b-main", "Main", [paragraph("b-first", first)]),
            section(
                "b-fake",
                "安全距离处理。",
                [paragraph("b-second", second)],
            ),
            section("b-next", "Next", []),
        ],
    )
    target = document(
        "target",
        [
            section(
                "t-main",
                "Main",
                [
                    paragraph("title-as-body", "安全距离处理"),
                    paragraph("t-combined", first + second),
                ],
            ),
            section("t-next", "Next", []),
        ],
    )
    embedder = CharacterEmbedder()
    plan = align_compare_scopes(baseline, target, embedder)

    pairs = match_paragraphs(plan, embedder, similarity_threshold=0.99)

    assert len(pairs) == 1
    assert pairs[0].split_unit is True
    assert pairs[0].baseline_para is not None
    assert pairs[0].target_para is not None
    assert pairs[0].baseline_para.text == first + "\n" + second
    assert pairs[0].target_para.text == first + second


def test_fake_boundary_allows_bounded_many_to_one_paragraph_window():
    parts = [
        "系统探测到障碍物以后停止当前开门动作。",
        "车门保持规定的安全距离并等待新的指令。",
        "收到恢复条件以后继续执行剩余开门流程。",
    ]
    baseline = document(
        "baseline",
        [
            section("b-main", "Main", [paragraph("b-first", parts[0])]),
            section(
                "b-fake",
                "安全距离处理。",
                [
                    paragraph("b-second", parts[1]),
                    paragraph("b-third", parts[2]),
                ],
            ),
            section("b-next", "Next", []),
        ],
    )
    target = document(
        "target",
        [
            section(
                "t-main",
                "Main",
                [
                    paragraph("title-as-body", "安全距离处理"),
                    paragraph("t-combined", "".join(parts)),
                ],
            ),
            section("t-next", "Next", []),
        ],
    )
    embedder = CharacterEmbedder()
    plan = align_compare_scopes(baseline, target, embedder)

    pairs = match_paragraphs(plan, embedder, similarity_threshold=0.99)

    assert len(pairs) == 1
    assert pairs[0].split_unit is True
    assert pairs[0].baseline_para is not None
    assert pairs[0].baseline_para.text == "\n".join(parts)


def test_fake_scope_opening_does_not_force_unrelated_body_match():
    baseline = document(
        "baseline",
        [
            section("b-main", "Main", []),
            section(
                "b-fake",
                "安全距离处理。",
                [paragraph("b-body", "完全不同的旧正文。")],
            ),
            section("b-next", "Next", []),
        ],
    )
    target = document(
        "target",
        [
            section(
                "t-main",
                "Main",
                [
                    paragraph("title-as-body", "安全距离处理"),
                    paragraph("t-body", "没有关联的新正文。"),
                ],
            ),
            section("t-next", "Next", []),
        ],
    )
    embedder = CharacterEmbedder()
    plan = align_compare_scopes(baseline, target, embedder)

    pairs = match_paragraphs(plan, embedder, similarity_threshold=0.99)

    assert any(
        pair.baseline_para is not None
        and pair.baseline_para.text == "完全不同的旧正文。"
        and pair.target_para is None
        for pair in pairs
    )
    assert any(
        pair.target_para is not None
        and pair.target_para.text == "没有关联的新正文。"
        and pair.baseline_para is None
        for pair in pairs
    )
    assert not any(
        pair.baseline_para is None
        and pair.target_para is not None
        and pair.target_para.paragraph_id == "title-as-body"
        for pair in pairs
    )


def test_multisentence_paragraph_used_as_fake_title_evidence_is_silently_covered():
    evidence_text = "安全距离处理。继续执行。"
    baseline = document(
        "baseline",
        [
            section("b-main", "Main", []),
            section("b-fake", evidence_text, []),
            section("b-next", "Next", []),
        ],
    )
    target_evidence = Paragraph(
        "target-evidence",
        evidence_text,
        [Sentence("安全距离处理。"), Sentence("继续执行。")],
    )
    target = document(
        "target",
        [
            section("t-main", "Main", [target_evidence]),
            section("t-next", "Next", []),
        ],
    )
    embedder = CharacterEmbedder()
    plan = align_compare_scopes(baseline, target, embedder)

    pairs = match_paragraphs(plan, embedder, similarity_threshold=0.99)

    assert pairs == []


def test_unique_exact_title_anchor_allows_parser_level_drift():
    baseline = document(
        "baseline",
        [section("b", "Shared title", [paragraph("b-p", "old body")], level=1)],
    )
    target = document(
        "target",
        [section("t", "Shared title", [paragraph("t-p", "new body")], level=2)],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    assert [
        item.section_id
        for item in group_for(plan, "baseline", "b").target_sections
    ] == ["t"]


def test_weak_title_alignment_does_not_cross_document_order():
    baseline = document(
        "baseline",
        [section("b-a", "aaaaab", []), section("b-b", "xxxxxy", [])],
    )
    target = document(
        "target",
        [section("t-b", "xxxxxz", []), section("t-a", "aaaaac", [])],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    paired_positions = [
        (
            baseline.sections.index(group.baseline_sections[0]),
            target.sections.index(group.target_sections[0]),
        )
        for group in plan.groups
        if group.baseline_sections
        and group.target_sections
        and group.evidence
        and group.evidence[0].kind == "title"
    ]
    assert all(
        left[1] < right[1]
        for left, right in zip(paired_positions, paired_positions[1:])
    )
    assert len(paired_positions) == 1


def test_weak_title_alignment_requires_compatible_mapped_parent():
    baseline = document(
        "baseline",
        [
            section("b-parent-a", "Parent A", []),
            section("b-child", "Safety rules", [], level=2),
            section("b-parent-b", "Parent B", []),
        ],
    )
    target = document(
        "target",
        [
            section("t-parent-a", "Parent A", []),
            section("t-parent-b", "Parent B", []),
            section("t-child", "Safety rules changed", [], level=2),
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    assert group_for(plan, "baseline", "b-child") is not group_for(
        plan,
        "target",
        "t-child",
    )


def test_single_paragraph_semantic_identity_rejects_ambiguous_section_candidates():
    baseline = document(
        "baseline",
        [
            section("b-a", "甲方", [paragraph("b-a-p", "系统必须保持安全距离。")]),
            section("b-b", "乙方", [paragraph("b-b-p", "系统必须保持安全距离。")]),
        ],
    )
    target = document(
        "target",
        [
            section("t-a", "丙方", [paragraph("t-a-p", "系统应保持安全距离。")]),
            section("t-b", "丁方", [paragraph("t-b-p", "系统应保持安全距离。")]),
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    assert not any(
        group.baseline_sections
        and group.target_sections
        and any(item.kind == "body_semantic" for item in group.evidence)
        for group in plan.groups
    )


def test_semantic_identity_ignores_empty_paragraphs_and_malformed_embeddings():
    class MalformedEmbedder(CharacterEmbedder):
        def __init__(self, result):
            self.result = result

        def embed(self, texts: list[str]):
            return self.result

    empty_baseline = document(
        "baseline",
        [section("b", "Copper", [paragraph("b-empty", "   ")])],
    )
    empty_target = document(
        "target",
        [section("t", "Sky", [paragraph("t-empty", "\n")])],
    )
    empty_plan = align_compare_scopes(
        empty_baseline,
        empty_target,
        CharacterEmbedder(),
    )
    assert not any(
        group.baseline_sections and group.target_sections
        for group in empty_plan.groups
    )

    baseline = document(
        "baseline",
        [section("b", "Copper", [paragraph("b-body", "baseline only body")])],
    )
    target = document(
        "target",
        [section("t", "Sky", [paragraph("t-body", "target unrelated content")])],
    )
    for result in (
        [[float("nan")], [1.0]],
        [[1.0], [1.0, 2.0]],
        [["not-a-number"], ["not-a-number"]],
    ):
        plan = align_compare_scopes(
            baseline,
            target,
            MalformedEmbedder(result),
        )
        assert not any(
            group.baseline_sections and group.target_sections
            for group in plan.groups
        )


def test_exact_body_candidate_rejects_unique_anchor_pointing_to_another_section():
    anchor_a = "第一条足够长且全文唯一的正文锚点用于冲突检测并确认拒绝。"
    anchor_b = "第二条足够长且全文唯一的正文锚点用于冲突检测并确认拒绝。"
    baseline = document(
        "baseline",
        [
            section("b-taken", "Taken", []),
            section(
                "b-conflict",
                "Copper",
                [paragraph("b-a", anchor_a), paragraph("b-b", anchor_b)],
            ),
        ],
    )
    target = document(
        "target",
        [
            section("t-taken", "Taken", [paragraph("t-a", anchor_a)]),
            section("t-other", "Sky", [paragraph("t-b", anchor_b)]),
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    assert group_for(plan, "baseline", "b-conflict") is not group_for(
        plan,
        "target",
        "t-other",
    )


def test_fake_sections_cannot_reuse_the_same_content_reference():
    baseline = document(
        "baseline",
        [
            section("b-main", "Main", []),
            section("b-fake-1", "Repeated title", []),
            section("b-fake-2", "Repeated title", []),
            section("b-next", "Next", []),
        ],
    )
    target = document(
        "target",
        [
            section(
                "t-main",
                "Main",
                [paragraph("shared-evidence", "Repeated title")],
            ),
            section("t-next", "Next", []),
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    assert not any(
        evidence.kind.startswith("fake_")
        for group in plan.groups
        for evidence in group.evidence
    )
    assert all(not group.baseline_crossable_boundaries for group in plan.groups)


def test_fake_boundary_opens_only_between_fake_and_original_real_match():
    baseline = document(
        "baseline",
        [
            section("b-main", "Main", []),
            section("b-fake-1", "First fake", []),
            section("b-fake-2", "Second fake", []),
        ],
    )
    target = document(
        "target",
        [
            section(
                "t-main",
                "Main",
                [
                    paragraph("first-evidence", "First fake"),
                    paragraph("second-evidence", "Second fake"),
                ],
            ),
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())
    group = group_for(plan, "baseline", "b-main")

    assert group.baseline_crossable_boundaries == frozenset(
        {("b-main", "b-fake-1")}
    )


def test_escaped_table_pipe_does_not_create_a_fragment_cell_candidate():
    baseline = document(
        "baseline",
        [section("b-main", "Main", []), section("b-fake", "B", [])],
    )
    target = document(
        "target",
        [
            section(
                "t-main",
                "Main",
                [
                    table_paragraph(
                        "table",
                        ["| A\\|B | Value |", "| --- | --- |"],
                    )
                ],
            )
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    assert not any(
        evidence.kind.startswith("fake_")
        for group in plan.groups
        for evidence in group.evidence
    )


def test_single_anchor_fake_requires_a_mapped_compatible_parent_scope():
    baseline = document(
        "baseline",
        [
            section("b-main", "Main", []),
            section("b-unmapped-parent", "Unmapped parent", []),
            section("b-fake", "Evidence title", [], level=2),
        ],
    )
    target = document(
        "target",
        [
            section(
                "t-main",
                "Main",
                [paragraph("evidence", "Evidence title")],
            )
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())

    assert not any(
        evidence.kind.startswith("fake_")
        for group in plan.groups
        for evidence in group.evidence
    )


def test_single_anchor_fake_accepts_a_mapped_compatible_parent_scope():
    baseline = document(
        "baseline",
        [
            section("b-main", "Main", []),
            section("b-fake", "Evidence title", [], level=2),
        ],
    )
    target = document(
        "target",
        [
            section(
                "t-main",
                "Main",
                [paragraph("evidence", "Evidence title")],
            )
        ],
    )

    plan = align_compare_scopes(baseline, target, CharacterEmbedder())
    group = group_for(plan, "baseline", "b-fake")

    assert any(evidence.kind == "fake_paragraph" for evidence in group.evidence)
    assert group.baseline_crossable_boundaries == frozenset(
        {("b-main", "b-fake")}
    )


def test_semantic_margin_includes_runner_up_below_policy_threshold():
    class FixedScoreEmbedder(CharacterEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            vectors = {
                "baseline source": [1.0, 0.0],
                "target winner": [0.8, 0.6],
                "target runner": [0.7, 0.51 ** 0.5],
            }
            return [vectors[text] for text in texts]

    baseline = document(
        "baseline",
        [section("b", "Copper", [paragraph("b-p", "baseline source")])],
    )
    target = document(
        "target",
        [
            section("t-winner", "Sky", [paragraph("t-w", "target winner")]),
            section("t-runner", "Moon", [paragraph("t-r", "target runner")]),
        ],
    )

    plan = align_compare_scopes(
        baseline,
        target,
        FixedScoreEmbedder(),
        similarity_threshold=0.75,
    )

    assert not any(
        group.baseline_sections
        and group.target_sections
        and any(evidence.kind == "body_semantic" for evidence in group.evidence)
        for group in plan.groups
    )
