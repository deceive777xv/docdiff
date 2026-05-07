"""Tests for router.evaluate_quality() and app.core.diff.compare()."""
from __future__ import annotations

import uuid

import pytest

from app.core.types import (
    ComparePolicy,
    DocumentIR,
    Paragraph,
    ParseQualityReport,
    Section,
    Sentence,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ir(
    n_sections: int = 2,
    paras_per_section: int = 3,
    para_text: str = "这是一段正常长度的合同条款文字，用于测试解析质量评估函数。",
    page_no: int = 1,
) -> DocumentIR:
    sections = []
    for s in range(n_sections):
        paras = [
            Paragraph(
                paragraph_id=f"p{s}-{p}",
                page_no=page_no,
                text=para_text,
                sentences=[Sentence(text=para_text)],
            )
            for p in range(paras_per_section)
        ]
        sections.append(Section(section_id=str(s), title=f"第{s+1}章", level=1, paragraphs=paras))
    plain = " ".join(para_text for _ in range(n_sections * paras_per_section))
    return DocumentIR(doc_id=str(uuid.uuid4()), title="测试文档", file_hash="abc", sections=sections, plain_text=plain)


# ── evaluate_quality tests ─────────────────────────────────────────────────────

class TestEvaluateQuality:
    def test_good_document_scores_high(self):
        from app.core.parser.router import evaluate_quality
        ir = _make_ir(n_sections=3, paras_per_section=4)
        report = evaluate_quality(ir)
        assert isinstance(report, ParseQualityReport)
        assert report.quality_score >= 0.7
        assert report.needs_ocr is False

    def test_no_sections_triggers_ocr(self):
        from app.core.parser.router import evaluate_quality
        ir = DocumentIR(doc_id="x", title="空", file_hash="0", sections=[], plain_text="")
        report = evaluate_quality(ir)
        assert report.needs_ocr is True
        assert report.quality_score <= 0.2
        assert len(report.warnings) > 0

    def test_no_paragraphs_triggers_ocr(self):
        from app.core.parser.router import evaluate_quality
        ir = DocumentIR(
            doc_id="x", title="空", file_hash="0",
            sections=[Section(section_id="1", title="章", level=1, paragraphs=[])],
            plain_text="",
        )
        report = evaluate_quality(ir)
        assert report.needs_ocr is True

    def test_very_short_paragraphs_lower_score(self):
        from app.core.parser.router import evaluate_quality
        ir = _make_ir(para_text="短", page_no=0)  # 1-char paragraphs, no page numbers
        report = evaluate_quality(ir)
        assert report.quality_score < 0.6
        assert len(report.warnings) > 0

    def test_missing_page_numbers_adds_warning(self):
        from app.core.parser.router import evaluate_quality
        ir = _make_ir(page_no=0)  # page_no=0 → not counted as having page numbers
        report = evaluate_quality(ir)
        assert any("页码" in w for w in report.warnings)

    def test_returns_parse_quality_report_type(self):
        from app.core.parser.router import evaluate_quality
        ir = _make_ir()
        report = evaluate_quality(ir)
        assert isinstance(report, ParseQualityReport)
        assert 0.0 <= report.quality_score <= 1.0
        assert isinstance(report.needs_ocr, bool)
        assert isinstance(report.warnings, list)


# ── compare() tests ────────────────────────────────────────────────────────────

class TestCompare:
    def test_compare_identical_docs_returns_mostly_format_changes(self):
        from app.core.diff import compare
        ir = _make_ir()
        result = compare(ir, ir)
        assert result.task_id
        assert isinstance(result.items, list)
        # Identical docs → all pairs match at high similarity → 格式变化 or 微调
        non_add_del = [i for i in result.items if i.diff_type not in ("新增", "删减")]
        assert len(non_add_del) > 0

    def test_compare_empty_vs_populated_returns_adds(self):
        from app.core.diff import compare
        empty = DocumentIR(doc_id="a", title="空", file_hash="0", sections=[], plain_text="")
        populated = _make_ir()
        result = compare(empty, populated)
        types = {i.diff_type for i in result.items}
        assert "新增" in types

    def test_compare_populated_vs_empty_returns_dels(self):
        from app.core.diff import compare
        populated = _make_ir()
        empty = DocumentIR(doc_id="b", title="空", file_hash="0", sections=[], plain_text="")
        result = compare(populated, empty)
        types = {i.diff_type for i in result.items}
        assert "删减" in types

    def test_compare_accepts_none_embedder(self):
        from app.core.diff import compare
        ir_a = _make_ir()
        ir_b = _make_ir(para_text="本合同付款周期调整为六十天，违约金按日万分之三计算。")
        result = compare(ir_a, ir_b, embedder=None)
        assert result is not None
        assert isinstance(result.items, list)

    def test_compare_with_policy_disables_llm(self):
        from app.core.diff import compare
        ir_a = _make_ir()
        ir_b = _make_ir(para_text="本合同付款周期调整为六十天。")
        policy = ComparePolicy(use_llm_classify=False, rule_strengthen=True)
        result = compare(ir_a, ir_b, policy=policy)
        assert result is not None

    def test_compare_result_has_version_ids(self):
        from app.core.diff import compare
        ir_a = _make_ir()
        ir_b = _make_ir()
        result = compare(ir_a, ir_b)
        assert result.baseline_version_id == ir_a.doc_id
        assert result.target_version_id == ir_b.doc_id

    def test_compare_diff_items_have_required_fields(self):
        from app.core.diff import compare
        ir_a = _make_ir()
        ir_b = _make_ir(para_text="修改后的合同条款，付款周期由三十天调整为九十天。")
        result = compare(ir_a, ir_b)
        for item in result.items:
            assert item.diff_id
            assert item.diff_type in ("新增", "删减", "微调", "实质修改", "重写", "格式变化")
            assert item.risk_level in ("high", "medium", "low")
            assert isinstance(item.similarity_score, float)
