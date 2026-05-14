"""Verify page_no fields were fully removed from data types."""
from __future__ import annotations
import dataclasses


def test_paragraph_no_page_no():
    from app.core.types import Paragraph
    fields = {f.name for f in dataclasses.fields(Paragraph)}
    assert "page_no" not in fields


def test_diff_item_no_page_fields():
    from app.core.types import DiffItem
    fields = {f.name for f in dataclasses.fields(DiffItem)}
    assert "baseline_page" not in fields
    assert "target_page" not in fields


def test_chunk_page_no_defaults_to_zero():
    from app.core.types import Chunk
    c = Chunk(id="x", version_id="v", chunk_no=0, section_path="sec", text="hello")
    assert c.page_no == 0
