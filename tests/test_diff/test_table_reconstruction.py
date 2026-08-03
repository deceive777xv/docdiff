from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from app.core.diff import semantic_matcher
from app.core.normalization import tables as reconstruction
from app.core.normalization.table_trace import ReconstructionOperation, SourceRowRef
from app.core.normalization.tables import (
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
    infer_bounded_rescue_mappings,
    infer_monotonic_column_mapping,
    infer_regions,
    split_markdown_table_row,
)
from app.core.types import DocumentIR, Paragraph, Section, Sentence


def make_source_ref(index: int = 0, paragraph_id: str = "paragraph-1") -> SourceRowRef:
    return SourceRowRef("section-1", paragraph_id, index)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("| unit-cobalt | drive | cedar pre |", ("unit-cobalt", "drive", "cedar pre")),
        ("| | | lude-complete | |", ("", "", "lude-complete", "")),
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


def test_bounded_mapping_rescue_ignores_unoccupied_header_width_without_dropping_business_cells():
    left = make_fragment(body_columns=(0, 1, 2), width=3, paragraph_id="left")
    right = make_fragment(body_columns=(3, 5, 7), width=8, paragraph_id="right")
    right = replace(
        right,
        paragraph_index=1,
        active_columns=tuple(range(8)),
    )

    assert infer_monotonic_column_mapping(left, right, ()) is None

    mappings = infer_bounded_rescue_mappings(left, right, ())

    assert len(mappings) == 1
    assert mappings[0].logical_by_physical == {3: 0, 5: 1, 7: 2}
    assert mappings[0].bounded_rescue is True
    occupied_business_columns = {
        index
        for row in right.rows
        for index, occupied in enumerate(row.occupied)
        if occupied
    }
    assert occupied_business_columns <= set(mappings[0].logical_by_physical)


def test_bounded_mapping_rescue_returns_at_most_three_monotonic_choices():
    left = make_fragment(body_columns=(0, 1, 2, 3), width=4, paragraph_id="left")
    sparse_rows = tuple(
        make_row(("", "", "continuation", "", "", ""), index, "right")
        for index in range(2)
    )
    right = TableFragment(
        "section-1",
        "right",
        1,
        sparse_rows,
        (TableRegion(sparse_rows, 0, len(sparse_rows), "body"),),
        (0,),
        tuple(range(6)),
    )

    mappings = infer_bounded_rescue_mappings(left, right, ())

    assert len(mappings) == 3
    assert all(mapping.bounded_rescue for mapping in mappings)
    assert all(
        list(mapping.logical_by_physical.values())
        == sorted(mapping.logical_by_physical.values())
        for mapping in mappings
    )


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
        previous_mapping=ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
        previous_fragment_rows=(previous,),
        continuation_fragment_rows=(continuation,),
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


def test_candidate_carries_exact_mappings_and_retained_rows_for_both_fragments():
    left, right, mapping = make_candidate_fragments(
        left_columns=(1, 4, 5),
        right_columns=(0, 3, 7),
        left_width=6,
        right_width=8,
    )

    candidate = generate_continuation_candidates(
        left,
        right,
        mapping,
        set(),
        (),
        "baseline",
    )[0]

    assert getattr(candidate, "previous_mapping", None) == ColumnMapping(
        (1, 4, 5),
        {1: 0, 4: 1, 5: 2},
        1.0,
    )
    assert getattr(candidate, "previous_fragment_rows", None) == left.rows
    assert getattr(candidate, "continuation_fragment_rows", None) == right.rows


def test_candidate_projection_rows_exclude_boundary_noise_headers_and_separators():
    left_rows = (
        make_row(("item", "group", "detail"), 0, "projection-left"),
        make_row(("---", "---", "---"), 1, "projection-left"),
        make_row(("100", "start", "complete"), 2, "projection-left"),
        make_row(("101", "lead", "prefix"), 3, "projection-left"),
    )
    right_rows = (
        make_row(("item", "group", "detail"), 0, "projection-right"),
        make_row(("---", "---", "---"), 1, "projection-right"),
        make_row(("", "", "suffix"), 2, "projection-right"),
        make_row(("102", "next", "complete"), 3, "projection-right"),
        make_row(("repeated", "boundary", "noise"), 4, "projection-right"),
    )
    left = TableFragment(
        "section-1",
        "projection-left",
        0,
        left_rows,
        (
            TableRegion(left_rows[:1], 0, 1, "header"),
            TableRegion(left_rows[1:2], 1, 2, "boundary"),
            TableRegion(left_rows[2:], 2, 4, "body"),
        ),
        (2,),
        (0, 1, 2),
    )
    right = TableFragment(
        "section-1",
        "projection-right",
        1,
        right_rows,
        (
            TableRegion(right_rows[:1], 0, 1, "header"),
            TableRegion(right_rows[1:2], 1, 2, "boundary"),
            TableRegion(right_rows[2:], 2, 5, "body"),
        ),
        (2,),
        (0, 1, 2),
    )
    boundary_rows = {right_rows[-1].source}

    candidate = generate_continuation_candidates(
        left,
        right,
        ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
        boundary_rows,
        (),
        "baseline",
    )[0]
    operations = reconstruction.build_reconstruction_operations(
        [CandidateAssessment(candidate, "high", "merge", True, True)],
        {"baseline": boundary_rows, "target": set()},
        {"baseline": set(), "target": set()},
    )

    projected_sources = {
        operation.source_rows[0]
        for operation in operations
        if operation.type == "project_columns"
    }
    assert projected_sources == {
        left_rows[2].source,
        left_rows[3].source,
        right_rows[2].source,
        right_rows[3].source,
    }
    assert not projected_sources.intersection(
        {
            left_rows[0].source,
            left_rows[1].source,
            right_rows[0].source,
            right_rows[1].source,
            right_rows[4].source,
        }
    )


def test_candidate_boundary_rows_are_excluded_before_selecting_adjacent_rows():
    left, right, mapping = make_candidate_fragments()
    boundary_rows = {left.rows[-1].source, right.rows[0].source}

    candidate = generate_continuation_candidates(
        left, right, mapping, boundary_rows, (), "baseline"
    )[0]

    assert candidate.previous_row.source == left.rows[-2].source
    assert candidate.continuation_row.source == right.rows[1].source


def test_candidate_includes_first_retained_right_row_even_when_inferred_as_header():
    left, right, mapping = make_candidate_fragments(
        left_values=(("100", "group-a", "complete"), ("101", "group-b", "prefix")),
        right_values=(("101", "group-b", "suffix"), ("102", "group-c", "complete")),
    )
    right = replace(
        right,
        regions=(
            TableRegion((right.rows[0],), 0, 1, "header"),
            TableRegion((right.rows[1],), 1, 2, "body"),
        ),
        body_region_indexes=(1,),
    )

    candidates = generate_continuation_candidates(
        left, right, mapping, set(), (), "baseline"
    )

    assert len(candidates) == 1
    assert candidates[0].continuation_row.source == right.rows[0].source
    assert "new_key_value" in candidates[0].vetoes
    assert candidates[0].next_full_row == right.rows[1]


def test_fragment_merge_does_not_force_boundary_rows_to_merge():
    left, right, mapping = make_candidate_fragments(
        left_values=(("100", "group-a", "complete"), ("101", "group-b", "prefix")),
        right_values=(("101", "group-b", "suffix"), ("102", "group-c", "complete")),
    )
    candidate = generate_continuation_candidates(
        left, right, mapping, set(), (), "baseline"
    )[0]

    operations = reconstruction.build_reconstruction_operations(
        [
            CandidateAssessment(
                candidate,
                "low",
                "merge",
                merge_rows=False,
                merge_fragments=True,
            )
        ],
        {"baseline": set(), "target": set()},
        {"baseline": set(), "target": set()},
    )

    assert any(operation.type == "merge_fragments" for operation in operations)
    assert not any(operation.type == "merge_rows" for operation in operations)


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
        (4, "high", "needs_llm"),
        (5, "high", "needs_llm"),
        (6, "high", "needs_llm"),
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


@pytest.mark.parametrize(
    "intervening_values",
    [
        ("", "detached-body", "payload"),
        ("foreign-key", "detached-body", "payload"),
    ],
    ids=("blank-key", "different-key-family"),
)
def test_candidate_fails_closed_for_intervening_body_rows(intervening_values):
    left, right, mapping = make_candidate_fragments(
        right_values=(
            intervening_values,
            ("", "", "suffix"),
            ("102", "next", "complete"),
            ("103", "later", "complete"),
        )
    )

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]
    assessment = assess_candidate(candidate)

    assert candidate.continuation_row.source == right.rows[1].source
    assert "crosses_real_body_row" in candidate.vetoes
    assert "boundary_artifacts_only" not in candidate.evidence
    assert assessment.final_action == "keep_separate"


def test_repeated_structural_header_can_precede_a_continuation():
    left, right, mapping = make_candidate_fragments(
        right_values=(
            ("alpha-column", "beta-column", "detail-column"),
            ("", "", "suffix"),
            ("102", "next", "complete"),
            ("103", "later", "complete"),
        )
    )
    peer_header = make_row(
        ("alpha-column", "beta-column", "detail-column"), 0, "peer-fragment"
    )
    peer_body = make_row(("201", "peer", "complete"), 1, "peer-fragment")
    peer = TableFragment(
        "section-1",
        "peer-fragment",
        0,
        (peer_header, peer_body),
        (
            TableRegion((peer_header,), 0, 1, "header"),
            TableRegion((peer_body,), 1, 2, "body"),
        ),
        (1,),
        (0, 1, 2),
    )

    candidate = generate_continuation_candidates(
        left, right, mapping, set(), (peer,), "baseline"
    )[0]
    assessment = assess_candidate(candidate)

    assert candidate.continuation_row.source == right.rows[1].source
    assert candidate.vetoes == ()
    assert "boundary_artifacts_only" in candidate.evidence
    assert assessment.final_action == "needs_llm"


def test_sparse_header_role_row_remains_a_continuation_after_repeated_page_metadata():
    left, right, mapping = make_candidate_fragments(
        right_values=(
            ("page-label", "version-label", "page-number"),
            ("", "", "suffix"),
            ("102", "next", "complete"),
        )
    )
    right = replace(
        right,
        regions=(
            TableRegion(right.rows[:2], 0, 2, "header"),
            TableRegion(right.rows[2:], 2, 3, "body"),
        ),
        body_region_indexes=(1,),
    )
    peer_metadata = make_row(
        ("page-label", "version-label", "page-number"),
        0,
        "peer-fragment",
    )
    peer_body = make_row(("201", "peer", "complete"), 1, "peer-fragment")
    peer = TableFragment(
        "section-1",
        "peer-fragment",
        1,
        (peer_metadata, peer_body),
        (
            TableRegion((peer_metadata,), 0, 1, "header"),
            TableRegion((peer_body,), 1, 2, "body"),
        ),
        (1,),
        (0, 1, 2),
    )

    candidates = generate_continuation_candidates(
        left, right, mapping, {right.rows[0].source}, (peer,), "baseline"
    )

    assert [candidate.continuation_row.source for candidate in candidates] == [
        right.rows[1].source,
    ]


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
    assert assessment.final_action == "needs_llm"


def make_small_left_text_key_candidate_fragments(
    left_row_count: int,
    continuation_key: str,
    *,
    right_complete_rows: int = 2,
) -> tuple[TableFragment, TableFragment, ColumnMapping]:
    left_values = (
        ("key-a", "group-a", "complete-a"),
        ("key-b", "group-b", "prefix"),
    )[-left_row_count:]
    complete_values = (
        ("key-c", "group-c", "complete-c"),
        ("key-d", "group-d", "complete-d"),
    )[:right_complete_rows]
    return make_candidate_fragments(
        left_values=left_values,
        right_values=((continuation_key, "", "suffix"),) + complete_values,
    )


@pytest.mark.parametrize("left_row_count", [1, 2])
def test_candidate_vetoes_new_textual_key_with_small_left_fragment(left_row_count):
    left, right, mapping = make_small_left_text_key_candidate_fragments(
        left_row_count, "key-z"
    )

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]
    assessment = assess_candidate(candidate)

    assert "new_key_value" in candidate.vetoes
    assert assessment.rule_confidence == "low"
    assert assessment.final_action == "keep_separate"


@pytest.mark.parametrize("left_row_count", [1, 2])
def test_candidate_vetoes_conflicting_textual_keys_with_small_left_fragment(left_row_count):
    left, right, mapping = make_small_left_text_key_candidate_fragments(
        left_row_count, "key-z"
    )

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]
    assessment = assess_candidate(candidate)

    assert "conflicting_key_cells" in candidate.vetoes
    assert assessment.rule_confidence == "low"
    assert assessment.final_action == "keep_separate"


@pytest.mark.parametrize("left_row_count", [1, 2])
def test_candidate_keeps_blank_textual_key_eligible_with_small_left_fragment(left_row_count):
    left, right, mapping = make_small_left_text_key_candidate_fragments(left_row_count, "")

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]
    assessment = assess_candidate(candidate)

    assert candidate.vetoes == ()
    assert candidate.conflicts == ()
    assert "blank_key_cells" in candidate.evidence
    assert assessment.rule_confidence == "high"
    assert assessment.final_action == "needs_llm"


def test_candidate_fails_closed_when_pooled_textual_key_profile_is_insufficient():
    left, right, mapping = make_small_left_text_key_candidate_fragments(
        1, "", right_complete_rows=1
    )

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]
    assessment = assess_candidate(candidate)

    assert candidate.vetoes == ()
    assert candidate.evidence == ()
    assert candidate.conflicts == ("insufficient_key_profile",)
    assert assessment.rule_confidence == "low"
    assert assessment.final_action == "keep_separate"


def test_pre_body_continuation_keeps_first_textual_body_row_in_key_profile():
    left_row = make_row(("key-a", "group-a", "prefix"), 0, "text-left")
    left = TableFragment(
        "section-1",
        "text-left",
        0,
        (left_row,),
        (TableRegion((left_row,), 0, 1, "body"),),
        (0,),
        (0, 1, 2),
    )
    continuation = make_row(("", "", "suffix"), 0, "text-right")
    first_body = make_row(("key-b", "group-b", "complete-b"), 1, "text-right")
    second_body = make_row(("key-c", "group-c", "complete-c"), 2, "text-right")
    right = TableFragment(
        "section-1",
        "text-right",
        1,
        (continuation, first_body, second_body),
        (
            TableRegion((continuation,), 0, 1, "header"),
            TableRegion((first_body, second_body), 1, 3, "body"),
        ),
        (1,),
        (0, 1, 2),
    )
    mapping = ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0)

    candidate = generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0]
    assessment = assess_candidate(candidate)

    assert candidate.continuation_row.source == continuation.source
    assert candidate.conflicts == ()
    assert candidate.vetoes == ()
    assert "blank_key_cells" in candidate.evidence
    assert assessment.final_action == "needs_llm"


def make_split_rows(
    previous_content: str,
    continuation_content: str,
) -> tuple:
    previous = make_row(("12", "drive", previous_content), 1, "merge-left")
    continuation = make_row(("", "", "", "", "", continuation_content), 0, "merge-right")
    mapping = ColumnMapping((1, 3, 5), {1: 0, 3: 1, 5: 2}, 1.0)
    return previous, continuation, mapping


def make_conflicting_key_rows() -> tuple:
    previous = make_row(("12", "drive", "prefix"), 1, "merge-left")
    continuation = make_row(("", "14", "", "", "", "suffix"), 0, "merge-right")
    mapping = ColumnMapping((1, 3, 5), {1: 0, 3: 1, 5: 2}, 1.0)
    return previous, continuation, mapping


def test_merge_logical_rows_preserves_raw_text_and_joins_content_only():
    assert hasattr(reconstruction, "merge_logical_rows"), "merge_logical_rows is not implemented"
    previous, continuation, mapping = make_split_rows("cedar pre", "lude-complete")

    cells = reconstruction.merge_logical_rows(previous, continuation, mapping, frozenset({0}))

    assert cells[0] == "12"
    assert cells[2] == "cedar pre<br>lude-complete"


def test_merge_logical_rows_rejects_conflicting_non_empty_key_cells():
    assert hasattr(reconstruction, "merge_logical_rows"), "merge_logical_rows is not implemented"
    previous, continuation, mapping = make_conflicting_key_rows()

    with pytest.raises(ValueError, match="key column conflict"):
        reconstruction.merge_logical_rows(previous, continuation, mapping, frozenset({0}))


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("prefix<br>", "suffix", "prefix<br>suffix"),
        ("prefix", "<br>suffix", "prefix<br>suffix"),
        ("prefix<br/>", "<br />suffix", "prefix<br/><br />suffix"),
    ],
)
def test_merge_logical_rows_does_not_invent_duplicate_breaks(left, right, expected):
    assert hasattr(reconstruction, "merge_logical_rows"), "merge_logical_rows is not implemented"
    previous, continuation, mapping = make_split_rows(left, right)

    cells = reconstruction.merge_logical_rows(previous, continuation, mapping, frozenset({0}))

    assert cells[2] == expected


def _document_from_fixture(data: dict) -> DocumentIR:
    return DocumentIR(
        doc_id=data["doc_id"],
        title=data["title"],
        file_hash=data["file_hash"],
        sections=[
            Section(
                section_id=section["section_id"],
                title=section["title"],
                level=section["level"],
                paragraphs=[
                    Paragraph(
                        paragraph_id=paragraph["paragraph_id"],
                        text=paragraph["text"],
                        sentences=[Sentence(sentence["text"]) for sentence in paragraph["sentences"]],
                    )
                    for paragraph in section["paragraphs"]
                ],
            )
            for section in data["sections"]
        ],
        plain_text=data["plain_text"],
    )


def load_sanitized_fixture() -> tuple[DocumentIR, DocumentIR]:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "cross_page_table_pair.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return _document_from_fixture(payload["baseline"]), _document_from_fixture(payload["target"])


def _source(section_id: str, paragraph_id: str, sentence_index: int) -> SourceRowRef:
    return SourceRowRef(section_id, paragraph_id, sentence_index)


def make_fixture_operations() -> list[ReconstructionOperation]:
    return [
        ReconstructionOperation(
            "baseline-project",
            "baseline",
            "project_columns",
            [_source("baseline-section", "baseline-fragment-b", 0)],
            column_mapping={1: 0, 3: 1, 5: 2},
        ),
        ReconstructionOperation(
            "baseline-drop-rows",
            "baseline",
            "drop_boundary_rows",
            [
                _source("baseline-section", "baseline-boundary-table", 0),
                _source("baseline-section", "baseline-boundary-table", 1),
            ],
        ),
        ReconstructionOperation(
            "baseline-drop-paragraphs",
            "baseline",
            "drop_boundary_paragraphs",
            source_paragraph_ids=["baseline-boundary-table", "baseline-boundary-note"],
        ),
        ReconstructionOperation(
            "baseline-merge-row",
            "baseline",
            "merge_rows",
            [
                _source("baseline-section", "baseline-fragment-a", 1),
                _source("baseline-section", "baseline-fragment-b", 0),
            ],
            decision_id="baseline-decision",
        ),
        ReconstructionOperation(
            "baseline-merge-fragments",
            "baseline",
            "merge_fragments",
            source_paragraph_ids=["baseline-fragment-a", "baseline-fragment-b"],
        ),
        ReconstructionOperation(
            "baseline-project-unnumbered",
            "baseline",
            "project_columns",
            [_source("baseline-section", "baseline-unnumbered-b", 0)],
            column_mapping={1: 0, 3: 1, 5: 2},
        ),
        ReconstructionOperation(
            "baseline-merge-unnumbered-row",
            "baseline",
            "merge_rows",
            [
                _source("baseline-section", "baseline-unnumbered-a", 0),
                _source("baseline-section", "baseline-unnumbered-b", 0),
            ],
            decision_id="baseline-unnumbered-decision",
        ),
        ReconstructionOperation(
            "baseline-merge-unnumbered-fragments",
            "baseline",
            "merge_fragments",
            source_paragraph_ids=["baseline-unnumbered-a", "baseline-unnumbered-b"],
        ),
        ReconstructionOperation(
            "target-project",
            "target",
            "project_columns",
            [_source("target-section", "target-fragment-a", 0)],
            column_mapping={1: 0, 3: 1, 5: 2},
        ),
        ReconstructionOperation(
            "target-drop-rows",
            "target",
            "drop_boundary_rows",
            [
                _source("target-section", "target-boundary-table", 0),
                _source("target-section", "target-boundary-table", 1),
            ],
        ),
        ReconstructionOperation(
            "target-drop-paragraph",
            "target",
            "drop_boundary_paragraphs",
            source_paragraph_ids=["target-boundary-table"],
        ),
        ReconstructionOperation(
            "target-merge-row",
            "target",
            "merge_rows",
            [
                _source("target-section", "target-fragment-a", 0),
                _source("target-section", "target-fragment-b", 0),
            ],
            decision_id="target-decision",
        ),
        ReconstructionOperation(
            "target-merge-fragments",
            "target",
            "merge_fragments",
            source_paragraph_ids=["target-fragment-a", "target-fragment-b"],
        ),
    ]


def repeated_boundary_token() -> str:
    return "repeated-neutral-boundary"


def real_repeated_body_text() -> str:
    return "duplicate-stays"


def target_only_row_text() -> str:
    return "target-only-neutral-row"


def _fixture_row(document: DocumentIR, paragraph_id: str, sentence_index: int):
    section = document.sections[0]
    paragraph = next(
        paragraph for paragraph in section.paragraphs if paragraph.paragraph_id == paragraph_id
    )
    row = split_markdown_table_row(
        paragraph.sentences[sentence_index].text,
        _source(section.section_id, paragraph_id, sentence_index),
    )
    assert row is not None
    return row


def _accepted_fixture_assessment(
    side: str,
    previous,
    continuation,
    previous_mapping: ColumnMapping,
    mapping: ColumnMapping,
    candidate_id: str,
) -> CandidateAssessment:
    candidate = ContinuationCandidate(
        candidate_id=candidate_id,
        side=side,
        previous_row=previous,
        continuation_row=continuation,
        next_full_row=None,
        mapping=mapping,
        previous_mapping=previous_mapping,
        previous_fragment_rows=(previous,),
        continuation_fragment_rows=(continuation,),
        evidence=EVIDENCE_CODES[:4],
        conflicts=(),
        vetoes=(),
        cross_version_rows=(),
    )
    assessment = assess_candidate(candidate)
    assert assessment.final_action == "needs_llm"
    return replace(
        assessment,
        final_action="merge",
        merge_rows=True,
        merge_fragments=True,
    )


def make_fixture_assessments(
    baseline: DocumentIR,
    target: DocumentIR,
) -> list[CandidateAssessment]:
    return [
        _accepted_fixture_assessment(
            "baseline",
            _fixture_row(baseline, "baseline-fragment-a", 1),
            _fixture_row(baseline, "baseline-fragment-b", 0),
            ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
            ColumnMapping((1, 3, 5), {1: 0, 3: 1, 5: 2}, 1.0),
            "fixture-baseline-numbered",
        ),
        _accepted_fixture_assessment(
            "baseline",
            _fixture_row(baseline, "baseline-unnumbered-a", 0),
            _fixture_row(baseline, "baseline-unnumbered-b", 0),
            ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
            ColumnMapping((1, 3, 5), {1: 0, 3: 1, 5: 2}, 1.0),
            "fixture-baseline-unnumbered",
        ),
        _accepted_fixture_assessment(
            "target",
            _fixture_row(target, "target-fragment-a", 0),
            _fixture_row(target, "target-fragment-b", 0),
            ColumnMapping((1, 3, 5), {1: 0, 3: 1, 5: 2}, 1.0),
            ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
            "fixture-target-numbered",
        ),
    ]


def make_fixture_boundaries(
    baseline: DocumentIR,
    target: DocumentIR,
) -> tuple[dict[str, set[SourceRowRef]], dict[str, set[str]]]:
    del baseline, target
    return (
        {
            "baseline": {
                _source("baseline-section", "baseline-boundary-table", 0),
                _source("baseline-section", "baseline-boundary-table", 1),
            },
            "target": {
                _source("target-section", "target-boundary-table", 0),
                _source("target-section", "target-boundary-table", 1),
            },
        },
        {
            "baseline": {"baseline-boundary-table", "baseline-boundary-note"},
            "target": {"target-boundary-table"},
        },
    )


def test_sanitized_fixture_assessment_build_and_replay_preserve_control_rows():
    baseline, target = load_sanitized_fixture()
    analyses = make_fixture_assessments(baseline, target)
    boundary_rows, boundary_paragraphs = make_fixture_boundaries(baseline, target)

    operations = reconstruction.build_reconstruction_operations(
        analyses, boundary_rows, boundary_paragraphs
    )
    normalized_baseline, normalized_target = reconstruction.apply_reconstruction_operations(
        baseline, target, operations
    )

    assert "cedar pre<br>lude-complete" in normalized_baseline.plain_text
    assert "| 14 | violet<br>limit. | within |" in normalized_target.plain_text
    assert "prefix<br>suffix" in normalized_baseline.plain_text
    assert repeated_boundary_token() not in normalized_baseline.plain_text
    assert repeated_boundary_token() not in normalized_target.plain_text
    assert normalized_baseline.plain_text.count(real_repeated_body_text()) == 2
    assert "7.0" in normalized_baseline.plain_text
    assert "7.5" in normalized_target.plain_text
    assert target_only_row_text() in normalized_target.plain_text

    baseline_by_id = {
        paragraph.paragraph_id: paragraph
        for paragraph in normalized_baseline.sections[0].paragraphs
    }
    target_by_id = {
        paragraph.paragraph_id: paragraph
        for paragraph in normalized_target.sections[0].paragraphs
    }
    assert baseline_by_id["baseline-new-table"].text == (
        "| 90 | separate-table | begins |\n| 91 | separate-table | continues |"
    )
    assert target_by_id["target-new-table"].text == "| 90 | separate-table | begins |"
    assert baseline_by_id["baseline-final-incomplete"].text == "| 99 | final | unfinished"
    assert target_by_id["target-final-incomplete"].text == "| 99 | final | unfinished"


def test_builder_projects_sparse_previous_row_before_replay_without_manual_operation():
    baseline, target = load_sanitized_fixture()
    target_assessment = make_fixture_assessments(baseline, target)[2]
    boundary_rows, boundary_paragraphs = make_fixture_boundaries(baseline, target)

    operations = reconstruction.build_reconstruction_operations(
        [target_assessment], boundary_rows, boundary_paragraphs
    )
    projection_by_source = {
        operation.source_rows[0]: operation.column_mapping
        for operation in operations
        if operation.type == "project_columns"
    }

    assert projection_by_source[
        _source("target-section", "target-fragment-a", 0)
    ] == {1: 0, 3: 1, 5: 2}
    normalized_target = reconstruction.apply_reconstruction_operations(
        baseline, target, operations
    )[1]
    assert "| 14 | violet<br>limit. | within |" in normalized_target.plain_text


def test_assessment_builder_and_replay_project_every_retained_body_row_exactly():
    left, right, mapping = make_candidate_fragments(
        left_columns=(1, 4, 5),
        right_columns=(0, 3, 7),
        left_width=6,
        right_width=8,
    )
    candidate = generate_continuation_candidates(
        left,
        right,
        mapping,
        set(),
        (),
        "baseline",
    )[0]
    assessment = assess_candidate(candidate)
    assert assessment.final_action == "needs_llm"
    assessment = replace(
        assessment,
        final_action="merge",
        merge_rows=True,
        merge_fragments=True,
    )
    paragraphs = [
        Paragraph(
            fragment.paragraph_id,
            "\n".join(row.raw_text for row in fragment.rows),
            [Sentence(row.raw_text) for row in fragment.rows],
        )
        for fragment in (left, right)
    ]
    baseline = DocumentIR(
        "projection-document",
        "Projection",
        "projection-hash",
        [Section("section-1", "Projection", 1, paragraphs)],
        "\n".join(paragraph.text for paragraph in paragraphs),
    )
    target = DocumentIR("empty-target", "Empty", "empty-hash")
    baseline_before = deepcopy(baseline)
    target_before = deepcopy(target)

    operations = reconstruction.build_reconstruction_operations(
        [assessment],
        {"baseline": set(), "target": set()},
        {"baseline": set(), "target": set()},
    )
    projections = {
        operation.source_rows[0]: operation.column_mapping
        for operation in operations
        if operation.type == "project_columns"
    }
    assert projections == {
        **{row.source: {1: 0, 4: 1, 5: 2} for row in left.rows},
        **{row.source: {0: 0, 3: 1, 7: 2} for row in right.rows},
    }

    normalized, normalized_target = reconstruction.apply_reconstruction_operations(
        baseline,
        target,
        operations,
    )
    assert baseline == baseline_before
    assert target == target_before
    units = semantic_matcher._expand_paragraphs(normalized.sections[0].paragraphs)
    table_values = [unit.table_values for unit in units if unit.table_values is not None]
    assert table_values == [
        ["100", "start", "complete"],
        ["101", "lead", "prefixsuffix"],
        ["102", "next", "complete"],
        ["", "inner", "sparse"],
    ]
    assert "prefix<br>suffix" in normalized.plain_text
    assert {len(values) for values in table_values} == {3}
    assert reconstruction.apply_reconstruction_operations(
        baseline,
        target,
        operations,
    ) == (normalized, normalized_target)
    assert reconstruction.apply_reconstruction_operations(
        normalized,
        normalized_target,
        operations,
    ) == (normalized, normalized_target)


def test_operation_builder_rejects_conflicting_retained_row_projections():
    left, right, mapping = make_candidate_fragments()
    candidate = generate_continuation_candidates(
        left,
        right,
        mapping,
        set(),
        (),
        "baseline",
    )[0]
    first = CandidateAssessment(candidate, "high", "merge", True, True)
    conflicting = CandidateAssessment(
        replace(
            candidate,
            candidate_id="conflicting-projection",
            mapping=ColumnMapping((0, 1, 2), {0: 1, 1: 0, 2: 2}, 1.0),
        ),
        "high",
        "merge",
        True,
        True,
    )

    with pytest.raises(ValueError, match="conflicting projections"):
        reconstruction.build_reconstruction_operations(
            [first, conflicting],
            {"baseline": set(), "target": set()},
            {"baseline": set(), "target": set()},
        )


def test_operation_builder_rejects_narrow_mapping_that_drops_retained_content():
    left, right, mapping = make_candidate_fragments()
    candidate = generate_continuation_candidates(
        left,
        right,
        mapping,
        set(),
        (),
        "baseline",
    )[0]
    incomplete = CandidateAssessment(
        replace(
            candidate,
            mapping=ColumnMapping((0, 1), {0: 0, 1: 1}, 1.0),
        ),
        "high",
        "merge",
        True,
        True,
    )

    with pytest.raises(ValueError, match="unmapped retained cells"):
        reconstruction.build_reconstruction_operations(
            [incomplete],
            {"baseline": set(), "target": set()},
            {"baseline": set(), "target": set()},
        )


def test_operation_builder_accepts_sparse_continuation_with_fewer_physical_columns():
    previous = make_row(
        ("33", "异响", "开关门过程及动态路试无异响，异响值＜", "/"),
        0,
        "different-width-left",
    )
    continuation = make_row(
        ("", "", "", "", "", "0.1。", "", "", ""),
        0,
        "different-width-right",
    )
    candidate = ContinuationCandidate(
        candidate_id="different-physical-widths",
        side="baseline",
        previous_row=previous,
        continuation_row=continuation,
        next_full_row=None,
        mapping=ColumnMapping((5,), {5: 2}, 0.91),
        previous_mapping=ColumnMapping(
            (0, 1, 2, 3),
            {0: 0, 1: 1, 2: 2, 3: 3},
            1.0,
        ),
        previous_fragment_rows=(previous,),
        continuation_fragment_rows=(continuation,),
        evidence=EVIDENCE_CODES[:4],
        conflicts=(),
        vetoes=(),
        cross_version_rows=(),
    )

    operations = reconstruction.build_reconstruction_operations(
        [CandidateAssessment(candidate, "high", "merge", True, True)],
        {"baseline": set(), "target": set()},
        {"baseline": set(), "target": set()},
    )

    continuation_projection = next(
        operation
        for operation in operations
        if operation.type == "project_columns"
        and operation.source_rows == [continuation.source]
    )
    assert continuation_projection.column_mapping == {5: 2}
    assert any(operation.type == "merge_rows" for operation in operations)
    assert any(operation.type == "merge_fragments" for operation in operations)


def test_operation_builder_rejects_partial_mapping_that_would_drop_retained_cells():
    left, right, _ = make_candidate_fragments(
        right_values=(
            ("", "", "suffix", ""),
            ("102", "next", "complete", "retained-extra"),
        ),
        right_columns=(0, 2, 4, 6),
        right_width=7,
    )
    candidate = ContinuationCandidate(
        candidate_id="partial-retained-projection",
        side="baseline",
        previous_row=left.rows[-1],
        continuation_row=right.rows[0],
        next_full_row=right.rows[1],
        mapping=ColumnMapping((0, 2, 4), {0: 0, 2: 1, 4: 2}, 1.0),
        previous_mapping=ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
        previous_fragment_rows=left.rows,
        continuation_fragment_rows=right.rows,
        evidence=EVIDENCE_CODES[:4],
        conflicts=(),
        vetoes=(),
        cross_version_rows=(),
    )

    with pytest.raises(ValueError, match="unmapped retained cells"):
        reconstruction.build_reconstruction_operations(
            [CandidateAssessment(candidate, "high", "merge", True, True)],
            {"baseline": set(), "target": set()},
            {"baseline": set(), "target": set()},
        )


def test_operation_builder_projects_fully_populated_row_from_shifted_physical_slot():
    left_rows = (
        make_row(("100", "start", "stable", "complete"), 0, "shift-left"),
        make_row(("101", "lead", "stable", "prefix"), 1, "shift-left"),
    )
    right_rows = (
        make_row(("", "", "", "", "", "suffix"), 0, "shift-right"),
        make_row(("", "102", "next", "", "shifted", "complete"), 1, "shift-right"),
    )
    previous_mapping = ColumnMapping(
        (0, 1, 2, 3),
        {0: 0, 1: 1, 2: 2, 3: 3},
        1.0,
    )
    continuation_mapping = ColumnMapping(
        (1, 2, 3, 5),
        {1: 0, 2: 1, 3: 2, 5: 3},
        1.0,
    )
    candidate = ContinuationCandidate(
        candidate_id="shifted-full-row-projection",
        side="baseline",
        previous_row=left_rows[-1],
        continuation_row=right_rows[0],
        next_full_row=right_rows[1],
        mapping=continuation_mapping,
        previous_mapping=previous_mapping,
        previous_fragment_rows=left_rows,
        continuation_fragment_rows=right_rows,
        evidence=EVIDENCE_CODES[:4],
        conflicts=(),
        vetoes=(),
        cross_version_rows=(),
    )

    operations = reconstruction.build_reconstruction_operations(
        [CandidateAssessment(candidate, "high", "merge", True, True)],
        {"baseline": set(), "target": set()},
        {"baseline": set(), "target": set()},
    )
    projection_by_source = {
        operation.source_rows[0]: operation.column_mapping
        for operation in operations
        if operation.type == "project_columns"
    }

    assert projection_by_source[right_rows[1].source] == {
        1: 0,
        2: 1,
        4: 2,
        5: 3,
    }


def test_replay_projects_drops_merges_and_consolidates_without_mutating_sources():
    assert hasattr(reconstruction, "apply_reconstruction_operations"), "replay is not implemented"
    baseline_ir, target_ir = load_sanitized_fixture()
    baseline_before = deepcopy(baseline_ir)
    target_before = deepcopy(target_ir)
    untouched_before = deepcopy(baseline_ir.sections[0].paragraphs[0])

    normalized_baseline, normalized_target = reconstruction.apply_reconstruction_operations(
        baseline_ir, target_ir, make_fixture_operations()
    )

    assert baseline_ir == baseline_before
    assert target_ir == target_before
    assert "cedar pre<br>lude-complete" in normalized_baseline.plain_text
    assert "prefix<br>suffix" in normalized_baseline.plain_text
    assert repeated_boundary_token() not in normalized_baseline.plain_text
    assert normalized_baseline.plain_text.count(real_repeated_body_text()) == 2
    assert target_only_row_text() in normalized_target.plain_text
    assert normalized_baseline.sections[0].paragraphs[0] == untouched_before


def test_replay_is_idempotent_for_same_operations():
    assert hasattr(reconstruction, "apply_reconstruction_operations"), "replay is not implemented"
    baseline_ir, target_ir = load_sanitized_fixture()
    operations = make_fixture_operations()

    first = reconstruction.apply_reconstruction_operations(baseline_ir, target_ir, operations)
    second = reconstruction.apply_reconstruction_operations(first[0], first[1], operations)

    assert second == first


def test_replay_rejects_a_missing_source_without_exact_prior_provenance():
    assert hasattr(reconstruction, "apply_reconstruction_operations"), "replay is not implemented"
    baseline_ir, target_ir = load_sanitized_fixture()
    missing = ReconstructionOperation(
        "missing-source",
        "baseline",
        "drop_boundary_rows",
        [_source("baseline-section", "does-not-exist", 0)],
    )

    with pytest.raises(ValueError, match="missing source row"):
        reconstruction.apply_reconstruction_operations(baseline_ir, target_ir, [missing])


def test_replay_rejects_conflicting_projection_operations_for_the_same_source_row():
    paragraph = Paragraph(
        "projection-source",
        "| | alpha | | beta |",
        [Sentence("| | alpha | | beta |")],
    )
    baseline = DocumentIR(
        "projection-conflict-document",
        "Projection conflict",
        "projection-conflict-hash",
        [Section("section-1", "Projection conflict", 1, [paragraph])],
    )
    baseline_before = deepcopy(baseline)
    source = _source("section-1", "projection-source", 0)
    operations = [
        ReconstructionOperation(
            "projection-conflict-a",
            "baseline",
            "project_columns",
            [source],
            column_mapping={1: 0, 3: 1},
        ),
        ReconstructionOperation(
            "projection-conflict-b",
            "baseline",
            "project_columns",
            [source],
            column_mapping={1: 1, 3: 0},
        ),
    ]

    with pytest.raises(ValueError, match="conflicting projections"):
        reconstruction.apply_reconstruction_operations(
            baseline,
            DocumentIR("empty-target", "Empty", "empty-hash"),
            operations,
        )
    assert baseline == baseline_before


def _make_chain_document() -> DocumentIR:
    paragraphs = [
        Paragraph("chain-a", "| 1 | neutral | part-a |", [Sentence("| 1 | neutral | part-a |")]),
        Paragraph("chain-b", "| | | part-b |", [Sentence("| | | part-b |")]),
        Paragraph("chain-c", "| | | part-c |", [Sentence("| | | part-c |")]),
    ]
    return DocumentIR(
        "chain-document",
        "Chain",
        "chain-hash",
        [Section("chain-section", "Chain", 1, paragraphs)],
        "\n".join(paragraph.text for paragraph in paragraphs),
    )


def _make_chain_operations(document: DocumentIR) -> list[ReconstructionOperation]:
    mapping = ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0)
    row_a = _fixture_row(document, "chain-a", 0)
    row_b = _fixture_row(document, "chain-b", 0)
    row_c = _fixture_row(document, "chain-c", 0)
    assessments = [
        _accepted_fixture_assessment(
            "baseline", row_a, row_b, mapping, mapping, "chain-ab"
        ),
        _accepted_fixture_assessment(
            "baseline", row_b, row_c, mapping, mapping, "chain-bc"
        ),
    ]
    return reconstruction.build_reconstruction_operations(
        assessments,
        {"baseline": set(), "target": set()},
        {"baseline": set(), "target": set()},
    )


def test_chained_overlapping_merges_preserve_provenance_and_are_idempotent():
    baseline = _make_chain_document()
    target = DocumentIR("empty-target", "Empty", "empty-hash")
    operations = _make_chain_operations(baseline)

    first = reconstruction.apply_reconstruction_operations(baseline, target, operations)
    second = reconstruction.apply_reconstruction_operations(first[0], first[1], operations)

    assert "part-a<br>part-b<br>part-c" in first[0].plain_text
    assert len(first[0].sections[0].paragraphs) == 1
    assert second == first


def test_chained_replay_rejects_missing_earlier_provenance():
    baseline = _make_chain_document()
    target = DocumentIR("empty-target", "Empty", "empty-hash")
    operations = _make_chain_operations(baseline)
    normalized, normalized_target = reconstruction.apply_reconstruction_operations(
        baseline, target, operations
    )
    final_sentence = normalized.sections[0].paragraphs[0].sentences[0]
    source_a = _source("chain-section", "chain-a", 0)
    final_sentence._reconstruction_source_rows = frozenset(
        set(final_sentence._reconstruction_source_rows) - {source_a}
    )

    with pytest.raises(ValueError, match="missing source row"):
        reconstruction.apply_reconstruction_operations(normalized, normalized_target, operations)


def test_replay_merge_rows_rejects_conflicting_logical_key_cells():
    previous = Paragraph(
        "conflict-a",
        "| 12 | neutral | prefix |",
        [Sentence("| 12 | neutral | prefix |")],
    )
    continuation = Paragraph(
        "conflict-b",
        "| 14 | | suffix |",
        [Sentence("| 14 | | suffix |")],
    )
    baseline = DocumentIR(
        "conflict-document",
        "Conflict",
        "conflict-hash",
        [Section("conflict-section", "Conflict", 1, [previous, continuation])],
    )
    operation = ReconstructionOperation(
        "conflicting-key-merge",
        "baseline",
        "merge_rows",
        [
            _source("conflict-section", "conflict-a", 0),
            _source("conflict-section", "conflict-b", 0),
        ],
    )

    with pytest.raises(ValueError, match="key column conflict"):
        reconstruction.apply_reconstruction_operations(
            baseline, DocumentIR("empty-target", "Empty", "empty-hash"), [operation]
        )


def test_replay_rejects_fragment_merge_across_a_retained_paragraph():
    paragraphs = [
        Paragraph("left-table", "| 1 | alpha |", [Sentence("| 1 | alpha |")]),
        Paragraph(
            "retained-prose",
            "This explanatory paragraph must remain in place.",
            [Sentence("This explanatory paragraph must remain in place.")],
        ),
        Paragraph("right-table", "| 2 | beta |", [Sentence("| 2 | beta |")]),
    ]
    baseline = DocumentIR(
        "noncontiguous-document",
        "Noncontiguous",
        "noncontiguous-hash",
        [Section("section-1", "Noncontiguous", 1, paragraphs)],
    )
    baseline_before = deepcopy(baseline)
    operation = ReconstructionOperation(
        "noncontiguous-fragment-merge",
        "baseline",
        "merge_fragments",
        source_paragraph_ids=["left-table", "right-table"],
    )

    with pytest.raises(ValueError, match="retained paragraphs"):
        reconstruction.apply_reconstruction_operations(
            baseline,
            DocumentIR("empty-target", "Empty", "empty-hash"),
            [operation],
        )
    assert baseline == baseline_before


def test_operation_builder_is_stable_and_uses_transformation_precedence():
    assert hasattr(reconstruction, "build_reconstruction_operations"), "operation builder is not implemented"
    left, right, mapping = make_candidate_fragments()
    merge = CandidateAssessment(
        generate_continuation_candidates(left, right, mapping, set(), (), "baseline")[0],
        "high",
        "merge",
        True,
        True,
    )
    keep = replace(merge, final_action="keep_separate")
    boundaries = {
        "baseline": {left.rows[0].source},
        "target": set(),
    }
    boundary_paragraphs = {"baseline": {"boundary-note"}, "target": set()}

    first = reconstruction.build_reconstruction_operations(
        [keep, merge], boundaries, boundary_paragraphs
    )
    second = reconstruction.build_reconstruction_operations(
        [merge, keep], boundaries, boundary_paragraphs
    )

    assert first == second
    assert [operation.type for operation in first] == [
        "project_columns",
        "project_columns",
        "project_columns",
        "project_columns",
        "drop_boundary_rows",
        "drop_boundary_paragraphs",
        "merge_rows",
        "merge_fragments",
    ]
    assert all(operation.operation_id for operation in first)
    assert first[6].generated_row_id
    assert first[7].generated_paragraph_id


def test_accepted_repeated_header_merge_keeps_only_the_first_header():
    section_id = "repeated-header-section"

    def table_row(text: str, paragraph_id: str, index: int):
        row = split_markdown_table_row(
            text,
            SourceRowRef(section_id, paragraph_id, index),
        )
        assert row is not None
        return row

    left_header = table_row("| 编号 | 名称 |", "left-table", 0)
    left_separator = table_row("| --- | --- |", "left-table", 1)
    left_body = table_row("| 1 | Alpha |", "left-table", 2)
    repeated_header = table_row("| 编号 | 名称 |", "right-table", 0)
    repeated_separator = table_row("| --- | --- |", "right-table", 1)
    right_body = table_row("| 2 | Beta |", "right-table", 2)
    mapping = ColumnMapping((0, 1), {0: 0, 1: 1}, 1.0)
    candidate = ContinuationCandidate(
        candidate_id="repeated-header-decision",
        side="baseline",
        previous_row=left_body,
        continuation_row=repeated_header,
        next_full_row=right_body,
        mapping=mapping,
        previous_mapping=mapping,
        previous_fragment_rows=(left_body,),
        continuation_fragment_rows=(repeated_header, right_body),
        evidence=("boundary_artifacts_only", "cross_version_support"),
        conflicts=(),
        vetoes=(),
        cross_version_rows=(),
        retained_header_row=left_header,
        repeated_header_rows=(repeated_header, repeated_separator),
    )
    assessment = CandidateAssessment(
        candidate,
        "high",
        "merge",
        merge_fragments=True,
        drop_repeated_header=True,
    )
    operations = reconstruction.build_reconstruction_operations(
        [assessment],
        {"baseline": set(), "target": set()},
        {"baseline": set(), "target": set()},
    )

    drop = next(
        operation
        for operation in operations
        if operation.type == "drop_repeated_table_header"
    )
    assert drop.source_rows == [
        left_header.source,
        repeated_header.source,
        repeated_separator.source,
    ]

    baseline = DocumentIR(
        "repeated-header-doc",
        "Repeated header",
        "repeated-header-hash",
        [
            Section(
                section_id,
                "Table",
                1,
                [
                    Paragraph(
                        "left-table",
                        "\n".join(
                            (left_header.raw_text, left_separator.raw_text, left_body.raw_text)
                        ),
                        [
                            Sentence(left_header.raw_text),
                            Sentence(left_separator.raw_text),
                            Sentence(left_body.raw_text),
                        ],
                    ),
                    Paragraph(
                        "right-table",
                        "\n".join(
                            (
                                repeated_header.raw_text,
                                repeated_separator.raw_text,
                                right_body.raw_text,
                            )
                        ),
                        [
                            Sentence(repeated_header.raw_text),
                            Sentence(repeated_separator.raw_text),
                            Sentence(right_body.raw_text),
                        ],
                    ),
                ],
            )
        ],
    )
    normalized, _ = reconstruction.apply_reconstruction_operations(
        baseline,
        DocumentIR("empty", "Empty", "empty-hash"),
        operations,
    )

    assert normalized.plain_text.count("编号") == 1
    assert normalized.plain_text.count("| --- | --- |") == 1
    assert "Alpha" in normalized.plain_text
    assert "Beta" in normalized.plain_text
    assert len(normalized.sections[0].paragraphs) == 1


def test_candidate_generation_binds_repeated_header_to_retained_header():
    section_id = "candidate-header-section"

    def table_row(text: str, paragraph_id: str, index: int):
        row = split_markdown_table_row(
            text,
            SourceRowRef(section_id, paragraph_id, index),
        )
        assert row is not None
        return row

    left_rows = (
        table_row("| 编号 | 名称 |", "left", 0),
        table_row("| --- | --- |", "left", 1),
        table_row("| 1 | Alpha |", "left", 2),
    )
    right_rows = (
        table_row("| 编号 | 名称 |", "right", 0),
        table_row("| --- | --- |", "right", 1),
        table_row("| 2 | Beta |", "right", 2),
    )
    left = TableFragment(
        section_id,
        "left",
        0,
        left_rows,
        (
            TableRegion((left_rows[0],), 0, 1, "header"),
            TableRegion((left_rows[1],), 1, 2, "boundary"),
            TableRegion((left_rows[2],), 2, 3, "body"),
        ),
        (2,),
        (0, 1),
    )
    right = TableFragment(
        section_id,
        "right",
        1,
        right_rows,
        (
            TableRegion((right_rows[0],), 0, 1, "header"),
            TableRegion((right_rows[1],), 1, 2, "boundary"),
            TableRegion((right_rows[2],), 2, 3, "body"),
        ),
        (2,),
        (0, 1),
    )

    candidates = generate_continuation_candidates(
        left,
        right,
        ColumnMapping((0, 1), {0: 0, 1: 1}, 1.0),
        set(),
        (),
        "baseline",
    )

    assert len(candidates) == 1
    assert candidates[0].continuation_row.source == right_rows[0].source
    assert candidates[0].retained_header_row == left_rows[0]
    assert candidates[0].repeated_header_rows == right_rows[:2]


def test_candidate_generation_keeps_sparse_boundary_role_row_as_continuation():
    section = Section(
        "sparse-continuation-section",
        "耐久性要求",
        1,
        [
            Paragraph(
                "left-table",
                "",
                [
                    Sentence("| 序号 | 项目 | 试验条件 | 技术要求 |"),
                    Sentence("| --- | --- | --- | --- |"),
                    Sentence("| 3 | 强度 | 条件A | 要求A |"),
                    Sentence(
                        "| 4 | 耐久性 | 左门：<br>常温：50000次，<br>40℃-85%：20000次，"
                        "<br>80℃：15000次，<br>-30℃：15000次。 | "
                        "2.1<br>开闭瞬间，车门不能出现卡滞、异响；<br>"
                        "2.3<br>电动开闭左右侧开启或关闭时间差在0.5 s以 |"
                    ),
                ],
                page_no=3,
            ),
            Paragraph(
                "right-table",
                "",
                [
                    Sentence(
                        "||||||内。<br>2.4<br>车门上的各零部件不得有异响、明显变形、"
                        "损坏、污染及失效，车门的塑性下沉量不大于0.5 mm，车门反力较"
                        "耐久前的变化值不大于20%。||||"
                    ),
                    Sentence("||5|防水性|淋雨试验||不得渗水||||"),
                    Sentence("||6|耐腐蚀|盐雾试验||不得锈蚀||||"),
                ],
                page_no=4,
            ),
        ],
    )
    left, right = collect_table_fragments(section)
    mapping = ColumnMapping((1, 2, 3, 5), {1: 0, 2: 1, 3: 2, 5: 3}, 1.0)

    assert right.regions[0].role == "boundary"

    candidates = generate_continuation_candidates(
        left,
        right,
        mapping,
        set(),
        (),
        "baseline",
    )

    assert len(candidates) == 1
    assert candidates[0].previous_row.source.sentence_index == 3
    assert candidates[0].continuation_row.source.sentence_index == 0
    assert candidates[0].next_full_row is not None
    assert candidates[0].next_full_row.source.sentence_index == 1
    assert candidates[0].vetoes == ()
