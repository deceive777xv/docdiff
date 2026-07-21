from __future__ import annotations

from dataclasses import replace

import pytest

from app.core.diff.reconstruction_trace import SourceRowRef
from app.core.diff.table_reconstruction import (
    TableFragment,
    TableRegion,
    classify_repeated_boundary_regions,
    collect_table_fragments,
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


def test_infer_mapping_rejects_incompatible_body_schemas():
    left = make_numeric_keyed_fragment()
    right = make_unkeyed_matrix_fragment()

    assert infer_monotonic_column_mapping(left, right, ()) is None
