"""Tests for structure_aligner.py"""
from __future__ import annotations
import uuid

import pytest

from app.core.types import DocumentIR, Section, Paragraph, Sentence


def make_ir(section_titles: list[str]) -> DocumentIR:
    sections = []
    for title in section_titles:
        sec = Section(section_id=str(uuid.uuid4()), title=title, level=1)
        sec.paragraphs = [Paragraph(
            paragraph_id=str(uuid.uuid4()), page_no=1,
            text=f"内容：{title}", sentences=[Sentence(text=f"内容：{title}")]
        )]
        sections.append(sec)
    return DocumentIR(doc_id=str(uuid.uuid4()), title="test", file_hash="h", sections=sections)


def test_matching_sections_aligned():
    from app.core.diff.structure_aligner import align_sections

    baseline = make_ir(["第一章 总则", "第二章 规定"])
    target = make_ir(["第一章 总则", "第二章 规定"])

    pairs = align_sections(baseline, target)

    # Both sections should be matched
    matched = [p for p in pairs if p.baseline_section and p.target_section]
    assert len(matched) == 2
    for p in matched:
        assert p.title_similarity > 0


def test_deleted_section():
    from app.core.diff.structure_aligner import align_sections

    baseline = make_ir(["第一章 总则", "第三章 附加条款"])
    target = make_ir(["第一章 总则"])

    pairs = align_sections(baseline, target)

    # "第三章 附加条款" not in target → target_section=None
    deleted = [p for p in pairs if p.baseline_section and p.baseline_section.title == "第三章 附加条款" and p.target_section is None]
    assert len(deleted) == 1


def test_added_section():
    from app.core.diff.structure_aligner import align_sections

    baseline = make_ir(["第一章 总则"])
    target = make_ir(["第一章 总则", "附则"])

    pairs = align_sections(baseline, target)

    # "附则" not in baseline → baseline_section=None
    added = [p for p in pairs if p.target_section and p.target_section.title == "附则" and p.baseline_section is None]
    assert len(added) == 1


def test_all_sections_covered():
    from app.core.diff.structure_aligner import align_sections

    baseline = make_ir(["第一章 总则", "第二章 规定"])
    target = make_ir(["第一章 总则", "第三章 新增章节"])

    pairs = align_sections(baseline, target)

    # Collect all baseline sections covered
    baseline_covered = [p.baseline_section for p in pairs if p.baseline_section is not None]
    target_covered = [p.target_section for p in pairs if p.target_section is not None]

    # All baseline sections appear exactly once
    assert len(baseline_covered) == len(baseline.sections)
    # All target sections appear exactly once
    assert len(target_covered) == len(target.sections)
    # Total pairs == all unique sections from both sides
    unmatched_target = sum(1 for p in pairs if p.baseline_section is None)
    assert len(pairs) == len(baseline.sections) + unmatched_target
