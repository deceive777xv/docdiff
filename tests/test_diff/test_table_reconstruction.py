from __future__ import annotations

from dataclasses import replace

import pytest

from app.core.diff.reconstruction_trace import SourceRowRef
from app.core.diff.table_reconstruction import (
    CandidateAssessment,
    ColumnMapping,
    ContinuationCandidate,
    TableFragment,
    TableRegion,
    assess_candidate,
    classify_repeated_boundary_regions,
    collect_table_fragments,
    generate_continuation_candidates,
    infer_active_columns,
    infer_monotonic_column_mapping,
    infer_regions,
    split_markdown_table_row,
)
from app.core.types import Paragraph, Section, Sentence


def make_source_ref(index: int = 0, paragraph_id: str = "paragraph-1") -> SourceRowRef:
    return SourceRowRef("section-1", paragraph_id, index)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("| 12 | drive | 0.5 s 以 |", ("12", "drive", "0.5 s 以")),
        ("| | | 内。 | |", ("", "", "内。", "")),
        ("|---|:---:|---|", ("---", ":---:", "---")),
    ],
)
def test_split_markdown_table_row_preserves_internal_and_trailing_empty_cells(text, expected):
    row = split_markdown_table_row(text, make_source_ref())

    assert row is not None
    assert row.raw_cells == expected


def test_split_markdown_table_row_rejects_plain_paragraph():
    assert split_markdown_table_row("ordinary paragraph", make_source_ref()) is None


def test_split_markdown_table_row_normalizes_only_analysis_cells():
    row = split_markdown_table_row("| **  Alpha  ** | line<br/> break | |", make_source_ref())

    assert row is not None
    assert row.raw_cells == ("**  Alpha  **", "line<br/> break", "")
    assert row.normalized_cells == ("Alpha", "line break", "")
    assert row.occupied == (True, True, False)
    assert row.value_types == ("short_text", "short_text", "empty")


def make_row(cells: tuple[str, ...], index: int, paragraph_id: str):
    row = split_markdown_table_row(
        "|" + "|".join(cells) + "|",
        make_source_ref(index, paragraph_id),
    )
    assert row is not None
    return row


def make_fragment(
    *,
    body_columns: tuple[int, ...] = (0, 1, 2),
    width: int = 3,
    paragraph_id: str = "fragment",
) -> TableFragment:
    values = (
        ("101", "drive", "0.5"),
        ("102", "brake", "0.8"),
        ("103", "coast", "1.1"),
    )
    rows = []
    for row_index, logical_values in enumerate(values):
        cells = [""] * width
        for physical_index, value in zip(body_columns, logical_values):
            cells[physical_index] = value
        rows.append(make_row(tuple(cells), row_index, paragraph_id))
    region = TableRegion(tuple(rows), 0, len(rows), "body")
    return TableFragment(
        section_id="section-1",
        paragraph_id=paragraph_id,
        paragraph_index=0,
        rows=tuple(rows),
        regions=(region,),
        body_region_indexes=(0,),
        active_columns=body_columns,
    )


def test_collect_table_fragments_keeps_source_coordinates_and_infers_structure():
    paragraph = Paragraph(
        paragraph_id="table-paragraph",
        text="",
        sentences=[
            Sentence("| note | note | note | note | note |"),
            Sentence("| 201 | start | 0.2 |"),
            Sentence("| 202 | stop | 0.4 |"),
            Sentence("| note | note | note | note | note |"),
        ],
    )
    section = Section("section-A", "Examples", 1, [paragraph])

    fragments = collect_table_fragments(section)

    assert len(fragments) == 1
    fragment = fragments[0]
    assert [row.source.sentence_index for row in fragment.rows] == [0, 1, 2, 3]
    assert [region.role for region in fragment.regions] == ["boundary", "body", "boundary"]
    assert fragment.body_region_indexes == (1,)
    assert fragment.active_columns == (0, 1, 2)


def test_infer_regions_finds_stable_multirow_body_between_wide_edges():
    paragraph_id = "region-fragment"
    rows = (
        make_row(("alpha", "beta", "alpha", "gamma", "alpha"), 0, paragraph_id),
        make_row(("301", "turn", "1.2"), 1, paragraph_id),
        make_row(("302", "hold", "1.4"), 2, paragraph_id),
        make_row(("omega", "psi", "omega", "chi", "omega"), 3, paragraph_id),
    )
    fragment = TableFragment("section-1", paragraph_id, 0, rows, (), (), ())

    regions = infer_regions(fragment, ())

    assert [(region.start_index, region.end_index, region.role) for region in regions] == [
        (0, 1, "boundary"),
        (1, 3, "body"),
        (3, 4, "boundary"),
    ]


def make_fragments_with_repeated_wide_boundaries(
    boundary_tokens: tuple[str, ...],
) -> tuple[TableFragment, ...]:
    fragments = []
    for fragment_index in range(3):
        paragraph_id = f"boundary-{fragment_index}"
        rows = (
            make_row(boundary_tokens, 0, paragraph_id),
            make_row((str(401 + fragment_index * 2), "open", "2.0"), 1, paragraph_id),
            make_row((str(402 + fragment_index * 2), "close", "2.5"), 2, paragraph_id),
            make_row(boundary_tokens, 3, paragraph_id),
        )
        regions = (
            TableRegion((rows[0],), 0, 1, "boundary"),
            TableRegion(rows[1:3], 1, 3, "body"),
            TableRegion((rows[3],), 3, 4, "boundary"),
        )
        fragments.append(
            TableFragment(
                "section-1",
                paragraph_id,
                fragment_index,
                rows,
                regions,
                (1,),
                (0, 1, 2),
            )
        )
    return tuple(fragments)


def boundary_source_refs(fragments: tuple[TableFragment, ...]) -> set[SourceRowRef]:
    return {row.source for fragment in fragments for row in (fragment.rows[0], fragment.rows[-1])}


def test_repeated_wide_boundary_is_detected_without_matching_fixed_text():
    fragments = make_fragments_with_repeated_wide_boundaries(
        boundary_tokens=("alpha", "beta", "alpha", "gamma", "alpha")
    )

    dropped = classify_repeated_boundary_regions(fragments)

    assert dropped == boundary_source_refs(fragments)


def make_fragments_with_repeated_two_row_boundary() -> tuple[TableFragment, ...]:
    boundary_rows = (
        ("alpha", "beta", "alpha", "gamma", "alpha"),
        ("delta", "epsilon", "delta", "zeta", "delta"),
    )
    fragments = []
    for fragment_index in range(3):
        paragraph_id = f"two-row-boundary-{fragment_index}"
        rows = (
            make_row(boundary_rows[0], 0, paragraph_id),
            make_row(boundary_rows[1], 1, paragraph_id),
            make_row((str(601 + fragment_index * 2), "open", "3.0"), 2, paragraph_id),
            make_row((str(602 + fragment_index * 2), "close", "3.5"), 3, paragraph_id),
        )
        regions = (
            TableRegion(rows[:2], 0, 2, "boundary"),
            TableRegion(rows[2:], 2, 4, "body"),
        )
        fragments.append(
            TableFragment(
                "section-1",
                paragraph_id,
                fragment_index,
                rows,
                regions,
                (1,),
                (0, 1, 2),
            )
        )
    return tuple(fragments)


def test_repeated_multirow_edge_boundary_drops_every_region_row():
    fragments = make_fragments_with_repeated_two_row_boundary()

    dropped = classify_repeated_boundary_regions(fragments)

    assert dropped == {row.source for fragment in fragments for row in fragment.regions[0].rows}


def make_unclassified_fragments_with_trailing_boundary() -> tuple[TableFragment, ...]:
    trailing_rows = (
        ("alpha", "beta", "alpha", "gamma", "alpha"),
        ("delta", "epsilon", "delta", "zeta", "delta"),
    )
    fragments = []
    for fragment_index in range(3):
        paragraph_id = f"inferred-trailing-boundary-{fragment_index}"
        rows = (
            make_row((str(701 + fragment_index * 2), "open", "4.0"), 0, paragraph_id),
            make_row((str(702 + fragment_index * 2), "close", "4.5"), 1, paragraph_id),
            make_row(trailing_rows[0], 2, paragraph_id),
            make_row(trailing_rows[1], 3, paragraph_id),
        )
        fragments.append(
            TableFragment("section-1", paragraph_id, fragment_index, rows, (), (), ())
        )
    return tuple(fragments)


def infer_fragment_structure(fragments: tuple[TableFragment, ...]) -> tuple[TableFragment, ...]:
    with_regions = tuple(
        replace(
            fragment,
            regions=infer_regions(fragment, tuple(peer for peer in fragments if peer is not fragment)),
        )
        for fragment in fragments
    )
    with_body_indexes = tuple(
        replace(
            fragment,
            body_region_indexes=tuple(
                index for index, region in enumerate(fragment.regions) if region.role == "body"
            ),
        )
        for fragment in with_regions
    )
    return tuple(
        replace(
            fragment,
            active_columns=infer_active_columns(
                fragment,
                tuple(peer for peer in with_body_indexes if peer is not fragment),
            ),
        )
        for fragment in with_body_indexes
    )


def test_inferred_trailing_multirow_boundary_keeps_structured_body_and_drops_footer():
    fragments = infer_fragment_structure(make_unclassified_fragments_with_trailing_boundary())

    assert [[region.role for region in fragment.regions] for fragment in fragments] == [
        ["body", "boundary"],
        ["body", "boundary"],
        ["body", "boundary"],
    ]
    assert classify_repeated_boundary_regions(fragments) == {
        row.source for fragment in fragments for row in fragment.regions[1].rows
    }


def make_fragments_with_repeated_body_row() -> tuple[TableFragment, ...]:
    fragments = []
    for fragment_index in range(3):
        fragment = make_fragment(paragraph_id=f"body-{fragment_index}")
        repeated = make_row(("777", "steady", "4.0"), 1, fragment.paragraph_id)
        rows = (fragment.rows[0], repeated, fragment.rows[2])
        fragments.append(
            replace(
                fragment,
                rows=rows,
                regions=(TableRegion(rows, 0, len(rows), "body"),),
            )
        )
    return tuple(fragments)


def test_repetition_alone_does_not_drop_a_real_data_row():
    fragments = make_fragments_with_repeated_body_row()

    dropped = classify_repeated_boundary_regions(fragments)

    for fragment in fragments:
        assert fragment.rows[1].source not in dropped


def test_infer_active_columns_uses_stable_occupied_roles_across_sparse_width():
    fragment = make_fragment(body_columns=(1, 3, 6), width=8)
    fragment = replace(fragment, active_columns=())

    assert infer_active_columns(fragment, ()) == (1, 3, 6)


def test_infer_mapping_projects_different_physical_widths_in_order():
    left = make_fragment(body_columns=(0, 2, 4), width=5, paragraph_id="left")
    right = make_fragment(body_columns=(1, 4, 6), width=7, paragraph_id="right")

    mapping = infer_monotonic_column_mapping(left, right, ())

    assert mapping is not None
    assert mapping.logical_by_physical == {1: 0, 4: 1, 6: 2}
    assert mapping.source_columns == (1, 4, 6)
    assert list(mapping.logical_by_physical.values()) == sorted(mapping.logical_by_physical.values())


def make_numeric_keyed_fragment() -> TableFragment:
    return make_fragment(paragraph_id="numeric-keyed")


def make_unkeyed_matrix_fragment() -> TableFragment:
    paragraph_id = "unkeyed-matrix"
    rows = tuple(
        make_row(values, index, paragraph_id)
        for index, values in enumerate(
            (
                ("north", "active", "ready"),
                ("south", "paused", "waiting"),
                ("east", "active", "ready"),
            )
        )
    )
    return TableFragment(
        "section-1",
        paragraph_id,
        0,
        rows,
        (TableRegion(rows, 0, len(rows), "body"),),
        (0,),
        (0, 1, 2),
    )


def make_fragment_from_rows(
    paragraph_id: str,
    values: tuple[tuple[str, ...], ...],
) -> TableFragment:
    rows = tuple(make_row(row_values, index, paragraph_id) for index, row_values in enumerate(values))
    region = TableRegion(rows, 0, len(rows), "body")
    return TableFragment(
        "section-1",
        paragraph_id,
        0,
        rows,
        (region,),
        (0,),
        tuple(range(len(values[0]))),
    )


def test_infer_mapping_rejects_incompatible_body_schemas():
    left = make_numeric_keyed_fragment()
    right = make_unkeyed_matrix_fragment()

    assert infer_monotonic_column_mapping(left, right, ()) is None


def test_infer_mapping_rejects_alignment_that_skips_both_key_roles():
    left = make_fragment_from_rows(
        "left-reordered-key",
        (
            ("501", "north", "0.1", "ready"),
            ("502", "south", "0.2", "waiting"),
            ("503", "east", "0.3", "paused"),
        ),
    )
    right = make_fragment_from_rows(
        "right-reordered-key",
        (
            ("north", "0.1", "ready", "801"),
            ("south", "0.2", "waiting", "802"),
            ("east", "0.3", "paused", "803"),
        ),
    )

    assert infer_monotonic_column_mapping(left, right, ()) is None


EVIDENCE_CODES = (
    "blank_key_cells",
    "next_row_restores_key_pattern",
    "complementary_content_cells",
    "textual_continuity",
    "boundary_artifacts_only",
    "cross_version_support",
)


def make_candidate(
    *,
    evidence: tuple[str, ...] = (),
    vetoes: tuple[str, ...] = (),
) -> ContinuationCandidate:
    previous = make_row(("101", "lead", "prefix"), 0, "candidate-left")
    continuation = make_row(("", "", "suffix"), 0, "candidate-right")
    return ContinuationCandidate(
        candidate_id="candidate-test",
        side="baseline",
        previous_row=previous,
        continuation_row=continuation,
        next_full_row=None,
        mapping=ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
        evidence=evidence,
        conflicts=(),
        vetoes=vetoes,
        cross_version_rows=(),
    )


def make_candidate_fragments(
    *,
    left_values: tuple[tuple[str, ...], ...] = (
        ("100", "start", "complete"),
        ("101", "lead", "prefix"),
    ),
    right_values: tuple[tuple[str, ...], ...] = (
        ("", "", "suffix"),
        ("102", "next", "complete"),
        ("", "inner", "sparse"),
    ),
    left_columns: tuple[int, ...] = (0, 1, 2),
    right_columns: tuple[int, ...] = (0, 1, 2),
    left_width: int = 3,
    right_width: int = 3,
    left_paragraph_index: int = 0,
    right_paragraph_index: int = 1,
) -> tuple[TableFragment, TableFragment, ColumnMapping]:
    def build(
        paragraph_id: str,
        values: tuple[tuple[str, ...], ...],
        columns: tuple[int, ...],
        width: int,
        paragraph_index: int,
    ) -> TableFragment:
        rows = []
        for index, logical_values in enumerate(values):
            cells = [""] * width
            for physical_index, value in zip(columns, logical_values):
                cells[physical_index] = value
            rows.append(make_row(tuple(cells), index, paragraph_id))
        row_tuple = tuple(rows)
        return TableFragment(
            "section-1",
            paragraph_id,
            paragraph_index,
            row_tuple,
            (TableRegion(row_tuple, 0, len(row_tuple), "body"),),
            (0,),
            columns,
        )

    left = build("candidate-left", left_values, left_columns, left_width, left_paragraph_index)
    right = build("candidate-right", right_values, right_columns, right_width, right_paragraph_index)
    mapping = ColumnMapping(
        right_columns,
        {physical: logical for logical, physical in enumerate(right_columns)},
        1.0,
    )
    return left, right, mapping


@pytest.mark.parametrize(
    "left_values,right_values",
    [
        (
            (("100", "start", "complete"), ("101", "lead", "prefix")),
            (("", "", "suffix"), ("102", "next", "complete"), ("", "inner", "sparse")),
        ),
        (
            (("north", "lead", "prefix"), ("south", "body", "partial")),
            (("", "tail", "suffix"), ("east", "next", "complete"), ("", "inner", "sparse")),
        ),
    ],
    ids=("numbered", "unnumbered"),
)
def test_candidate_is_limited_to_adjacent_fragment_boundary(left_values, right_values):
    left, right, mapping = make_candidate_fragments(
        left_values=left_values,
        right_values=right_values,
    )

    candidates = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")

    assert [candidate.previous_row.source for candidate in candidates] == [left.rows[-1].source]
    assert [candidate.continuation_row.source for candidate in candidates] == [right.rows[0].source]
    assert candidates[0].next_full_row is not None
    assert candidates[0].next_full_row.source == right.rows[1].source


def test_candidate_boundary_rows_are_excluded_before_selecting_adjacent_rows():
    left, right, mapping = make_candidate_fragments()
    boundary_rows = {left.rows[-1].source, right.rows[0].source}

    candidate = generate_continuation_candidates(
        left, right, mapping, boundary_rows, (), "baseline"
    )[0]

    assert candidate.previous_row.source == left.rows[-2].source
    assert candidate.continuation_row.source == right.rows[1].source


@pytest.mark.parametrize(
    "case",
    [
        "new_key_value",
        "header_or_separator",
        "incompatible_schema",
        "new_section_or_table",
        "crosses_real_body_row",
        "conflicting_key_cells",
    ],
)
def test_hard_veto_never_requests_llm(case):
    candidate = make_candidate(evidence=EVIDENCE_CODES, vetoes=(case,))

    assessment = assess_candidate(candidate)

    assert assessment == CandidateAssessment(candidate, "low", "keep_separate")


@pytest.mark.parametrize(
    ("evidence_count", "expected_confidence", "expected_action"),
    [
        (4, "high", "merge"),
        (5, "high", "merge"),
        (6, "high", "merge"),
        (2, "medium", "needs_llm"),
        (3, "medium", "needs_llm"),
        (0, "low", "keep_separate"),
        (1, "low", "keep_separate"),
    ],
)
def test_assess_candidate_uses_approved_confidence_tiers(
    evidence_count, expected_confidence, expected_action
):
    candidate = make_candidate(evidence=EVIDENCE_CODES[:evidence_count])

    assessment = assess_candidate(candidate)

    assert assessment.rule_confidence == expected_confidence
    assert assessment.final_action == expected_action


def test_textual_continuity_alone_remains_low_confidence():
    candidate = make_candidate(evidence=("textual_continuity",))

    assessment = assess_candidate(candidate)

    assert assessment.rule_confidence == "low"
    assert assessment.final_action == "keep_separate"


def test_candidate_conflict_forces_low_confidence():
    candidate = replace(
        make_candidate(evidence=EVIDENCE_CODES),
        conflicts=("ambiguous_content_overlap",),
    )

    assessment = assess_candidate(candidate)

    assert assessment.rule_confidence == "low"
    assert assessment.final_action == "keep_separate"


def test_candidate_collects_blank_key_next_pattern_and_textual_evidence():
    left, right, mapping = make_candidate_fragments()

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]

    assert "blank_key_cells" in candidate.evidence
    assert "next_row_restores_key_pattern" in candidate.evidence
    assert "textual_continuity" in candidate.evidence


def test_candidate_collects_complementary_content_cells_evidence():
    left, right, mapping = make_candidate_fragments(
        left_values=(("100", "start", "complete"), ("101", "lead", "")),
        right_values=(("", "", "tail"), ("102", "next", "complete")),
    )

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]

    assert "complementary_content_cells" in candidate.evidence


def test_candidate_collects_boundary_artifacts_only_evidence():
    left, right, mapping = make_candidate_fragments()
    left_boundary = make_row(("repeated", "repeated", "repeated"), 2, left.paragraph_id)
    right_boundary = make_row(("repeated", "repeated", "repeated"), 99, right.paragraph_id)
    left = replace(left, rows=left.rows + (left_boundary,))
    shifted_right_rows = (right_boundary,) + right.rows
    right = replace(right, rows=shifted_right_rows)

    candidate = generate_continuation_candidates(
        left,
        right,
        mapping,
        {left_boundary.source, right_boundary.source},
        (),
        "baseline",
    )[0]

    assert "boundary_artifacts_only" in candidate.evidence


def test_cross_version_support_evidence_uses_logical_mapping_not_physical_indexes():
    left, right, mapping = make_candidate_fragments(
        left_columns=(0, 2, 4),
        right_columns=(1, 3, 5),
        left_width=5,
        right_width=6,
    )
    peer, _, _ = make_candidate_fragments(
        left_values=(("101", "lead", "prefixsuffix"), ("102", "next", "complete")),
        right_values=(("", "", "unused"), ("103", "later", "complete")),
        left_columns=(1, 4, 6),
        left_width=7,
    )

    candidate = generate_continuation_candidates(
        left, right, mapping, set(), (peer,), "baseline"
    )[0]

    assert "cross_version_support" in candidate.evidence
    assert peer.rows[0] in candidate.cross_version_rows


@pytest.mark.parametrize(
    "case",
    [
        "new_key_value",
        "header_or_separator",
        "incompatible_schema",
        "new_section_or_table",
        "crosses_real_body_row",
        "conflicting_key_cells",
    ],
)
def test_candidate_generation_records_each_hard_veto(case):
    left, right, mapping = make_candidate_fragments()
    if case in {"new_key_value", "conflicting_key_cells"}:
        replacement = make_row(("999", "", "suffix"), 0, right.paragraph_id)
        right = replace(
            right,
            rows=(replacement,) + right.rows[1:],
            regions=(TableRegion((replacement,) + right.rows[1:], 0, len(right.rows), "body"),),
        )
    elif case == "header_or_separator":
        replacement = make_row(("---", "---", "---"), 0, right.paragraph_id)
        right = replace(
            right,
            rows=(replacement,) + right.rows[1:],
            regions=(TableRegion((replacement,) + right.rows[1:], 0, len(right.rows), "body"),),
        )
    elif case == "incompatible_schema":
        mapping = replace(mapping, score=0.1)
    elif case == "new_section_or_table":
        right = replace(right, paragraph_index=3)
    elif case == "crosses_real_body_row":
        intervening = make_row(("150", "real", "row"), 50, left.paragraph_id)
        left = replace(left, rows=left.rows + (intervening,))

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]

    assert case in candidate.vetoes
    assert candidate.evidence == ()


def test_candidate_id_is_stable_for_text_changes_and_mapping_order():
    left, right, mapping = make_candidate_fragments()
    original = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]
    changed_row = make_row(("101", "changed", "content"), 1, left.paragraph_id)
    changed_left = replace(
        left,
        rows=(left.rows[0], changed_row),
        regions=(TableRegion((left.rows[0], changed_row), 0, 2, "body"),),
    )
    reordered_mapping = ColumnMapping((0, 1, 2), {2: 2, 0: 0, 1: 1}, 1.0)

    changed = generate_continuation_candidates(
        changed_left, right, reordered_mapping, set(), (), "baseline"
    )[0]

    assert changed.candidate_id == original.candidate_id


def make_text_key_candidate_fragments(
    continuation_key: str,
) -> tuple[TableFragment, TableFragment, ColumnMapping]:
    return make_candidate_fragments(
        left_values=(
            ("item-a", "phase-a", "complete-a"),
            ("item-b", "phase-b", "complete-b"),
            ("item-c", "phase-c", "prefix"),
        ),
        right_values=(
            (continuation_key, "", "suffix"),
            ("item-d", "phase-d", "complete-d"),
        ),
    )


def test_candidate_vetoes_new_stable_textual_key_value():
    left, right, mapping = make_text_key_candidate_fragments("item-z")

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]
    assessment = assess_candidate(candidate)

    assert "new_key_value" in candidate.vetoes
    assert assessment.rule_confidence == "low"
    assert assessment.final_action == "keep_separate"


def test_candidate_vetoes_conflicting_stable_textual_key_cells():
    left, right, mapping = make_text_key_candidate_fragments("item-z")

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]
    assessment = assess_candidate(candidate)

    assert "conflicting_key_cells" in candidate.vetoes
    assert assessment.rule_confidence == "low"
    assert assessment.final_action == "keep_separate"


def test_candidate_keeps_legitimate_unnumbered_continuation_with_blank_textual_key():
    left, right, mapping = make_text_key_candidate_fragments("")

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]
    assessment = assess_candidate(candidate)

    assert candidate.vetoes == ()
    assert "blank_key_cells" in candidate.evidence
    assert assessment.rule_confidence == "high"
    assert assessment.final_action == "merge"
