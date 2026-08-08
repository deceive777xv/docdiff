"""Tests for narrow parser artifact cleanup."""
from __future__ import annotations


def test_removes_dispimg_from_table_cells_and_preserves_nan_literals():
    from app.core.parser.markdown_cleanup import remove_dispimg_artifacts

    markdown = (
        "| A | B |\n"
        "| --- | --- |\n"
        "| =DISPIMG(\"ID_1\") | NaN |\n"
        "| None | N/A |\n"
        "| nan | NA |"
    )
    cleaned = remove_dispimg_artifacts(markdown)
    assert "DISPIMG" not in cleaned
    for literal in ("NaN", "None", "N/A", "nan", "NA"):
        assert literal in cleaned


def test_does_not_clean_dispimg_outside_markdown_tables():
    from app.core.parser.markdown_cleanup import remove_dispimg_artifacts

    text = '正文 =DISPIMG("ID_1")'
    assert remove_dispimg_artifacts(text) == text
