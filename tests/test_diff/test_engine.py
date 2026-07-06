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
) -> DocumentIR:
    sections = []
    for s in range(n_sections):
        paras = [
            Paragraph(
                paragraph_id=f"p{s}-{p}",
                text=para_text,
                sentences=[Sentence(text=para_text)],
            )
            for p in range(paras_per_section)
        ]
        sections.append(Section(section_id=str(s), title=f"第{s+1}章", level=1, paragraphs=paras))
    plain = " ".join(para_text for _ in range(n_sections * paras_per_section))
    return DocumentIR(doc_id=str(uuid.uuid4()), title="测试文档", file_hash="abc", sections=sections, plain_text=plain)


def _make_table_ir(rows: list[str]) -> DocumentIR:
    para = Paragraph(
        paragraph_id="table-1",
        text="\n".join(rows),
        sentences=[Sentence(text=row) for row in rows],
    )
    section = Section(section_id="s1", title="费用表", level=1, paragraphs=[para])
    return DocumentIR(
        doc_id=str(uuid.uuid4()),
        title="表格文档",
        file_hash=str(uuid.uuid4()),
        sections=[section],
        plain_text=para.text,
    )


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
        ir = _make_ir(para_text="短")  # 1-char paragraphs
        report = evaluate_quality(ir)
        assert report.quality_score < 0.6
        assert len(report.warnings) > 0

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
            assert item.risk_level in ("high", "medium", "low", "none")
            assert isinstance(item.similarity_score, float)

    def test_compare_large_table_reports_changed_row_not_whole_table(self):
        """Large table diffs should point at changed rows and hide unchanged rows."""
        from app.core.diff import compare

        unchanged_rows = [f"| 服务项{i} | 保持不变{i} |" for i in range(40)]
        b_rows = [
            "| 项目 | 取值 |",
            "| --- | --- |",
            "| 付款周期 | 30天 |",
            "| 违约金 | 按日万分之三 |",
            *unchanged_rows,
        ]
        t_rows = [
            "| 项目 | 取值 |",
            "| --- | --- |",
            "| 付款周期 | 60天 |",
            "| 违约金 | 按日万分之三 |",
            *unchanged_rows,
        ]

        result = compare(
            _make_table_ir(b_rows),
            _make_table_ir(t_rows),
            policy=ComparePolicy(use_llm_classify=False, rule_strengthen=True),
        )

        assert any("付款周期" in item.baseline_text and "60天" in item.target_text for item in result.items)
        assert all("\n" not in item.baseline_text for item in result.items if item.baseline_text)
        assert all("\n" not in item.target_text for item in result.items if item.target_text)
        assert not any("违约金" in item.baseline_text or "违约金" in item.target_text for item in result.items)

    def test_compare_paragraph_boundary_changes_do_not_create_add_delete_noise(self):
        """Same sentences split into different paragraphs should still match."""
        from app.core.diff import compare

        sentences = [
            "Alpha warranty obligation remains unchanged.",
            "Beta delivery schedule remains unchanged.",
            "Gamma invoice approval remains unchanged.",
        ]
        baseline_para = Paragraph(
            paragraph_id="p-all",
            text="\n".join(sentences),
            sentences=[Sentence(text=text) for text in sentences],
        )
        target_paras = [
            Paragraph(
                paragraph_id=f"p-{index}",
                text=text,
                sentences=[Sentence(text=text)],
            )
            for index, text in enumerate(sentences)
        ]
        baseline = DocumentIR(
            doc_id="baseline",
            title="boundary",
            file_hash="baseline",
            sections=[Section(section_id="s", title="Terms", level=1, paragraphs=[baseline_para])],
        )
        target = DocumentIR(
            doc_id="target",
            title="boundary",
            file_hash="target",
            sections=[Section(section_id="s", title="Terms", level=1, paragraphs=target_paras)],
        )

        result = compare(
            baseline,
            target,
            policy=ComparePolicy(use_llm_classify=False, rule_strengthen=True),
        )

        assert not any(item.diff_type in ("新增", "删减") for item in result.items)

    def test_compare_ordinal_table_reorder_is_not_high_risk_rewrite(self):
        """Rows moved in a table with an ordinal first column are order changes."""
        from app.core.diff import compare

        b_rows = [
            "| 序号 | 区域 | 销售代表 | 一月(万元) |",
            "| --- | --- | --- | --- |",
            "| 4 | 华南 | 赵六 | 87 |",
            "| 5 | 华东 | 周明 | 134 |",
        ]
        t_rows = [
            "| 序号 | 区域 | 销售代表 | 一月(万元) |",
            "| --- | --- | --- | --- |",
            "| 4 | 华东 | 周明 | 134 |",
            "| 5 | 华南 | 赵六 | 87 |",
        ]

        result = compare(
            _make_table_ir(b_rows),
            _make_table_ir(t_rows),
            policy=ComparePolicy(use_llm_classify=False, rule_strengthen=True),
        )

        assert result.items
        assert all(item.diff_type == "格式变化" for item in result.items)
        assert all(item.risk_level == "none" for item in result.items)

    def test_compare_pdf_styled_tables_match_by_business_content_before_sequence(self):
        """PDF Markdown styling should not make ordinal cells dominate matching."""
        from app.core.diff import compare

        b_rows = [
            "|**序**<br>**号**|**区**<br>**域**|**销售**<br>**代表**|**一月**|",
            "|---|---|---|---|",
            "|`4`|华<br>东|周明|`134`|",
            "|`5`|华<br>南|赵六|`87`|",
        ]
        t_rows = [
            "|**序**<br>**号**|**区**<br>**域**|**销售**<br>**代表**|**一月**|",
            "|---|---|---|---|",
            "|`4`|华<br>南|赵六|`87`|",
            "|`5`|华<br>东|周明|`134`|",
        ]

        result = compare(
            _make_table_ir(b_rows),
            _make_table_ir(t_rows),
            policy=ComparePolicy(use_llm_classify=False, rule_strengthen=True),
        )

        assert len(result.items) == 2
        assert all(item.diff_type == "格式变化" for item in result.items)
        assert all(item.risk_level == "none" for item in result.items)
        assert any("周明" in item.baseline_text and "周明" in item.target_text for item in result.items)
        assert any("赵六" in item.baseline_text and "赵六" in item.target_text for item in result.items)

    def test_compare_pdf_key_value_row_matches_plain_row_and_reports_content_change(self):
        """Header-value PDF rows should match by the business key, then compare values."""
        from app.core.diff import compare

        baseline = DocumentIR(
            doc_id="b",
            title="考勤",
            file_hash="b",
            sections=[
                Section(
                    section_id="s",
                    title="员工考勤摘要",
                    level=1,
                    paragraphs=[
                        Paragraph(
                            paragraph_id="p",
                            text=(
                                "|**序号**<br>`3`|**姓名**<br>赵六|**部门**<br>销售|"
                                "**出勤天数**<br>`23`|"
                            ),
                            sentences=[
                                Sentence(text=(
                                    "|**序号**<br>`3`|**姓名**<br>赵六|**部门**<br>销售|"
                                    "**出勤天数**<br>`23`|"
                                )),
                                Sentence(text="|---|---|---|---|"),
                            ],
                        )
                    ],
                )
            ],
        )
        target = _make_table_ir([
            "|**序号**|**姓名**|**部门**|**出勤天数**|",
            "|---|---|---|---|",
            "|`4`|赵六|销售|`20`|",
        ])
        target.sections[0].title = "员工考勤摘要"

        result = compare(
            baseline,
            target,
            policy=ComparePolicy(use_llm_classify=False, rule_strengthen=True),
        )

        assert len(result.items) == 1
        item = result.items[0]
        assert item.diff_type == "实质修改"
        assert item.risk_level == "high"
        assert "赵六" in item.baseline_text and "赵六" in item.target_text

    def test_compare_markdown_table_with_insert_and_reorder_keeps_business_rows_matched(self):
        """Real markdown table syntax should not compare separators or shifted rows."""
        from app.core.diff import compare
        from app.core.parser.markitdown_adapter import _parse_markdown

        baseline_md = """
# 项目数据统计表

## 产品库存状态表

| 序号 | 产品类别 | 产品名称   | 库存数量 | 安全库存 | 状态           | 存放货架 |
| :--: | -------- | ---------- | -------- | -------- | -------------- | -------- |
|  1   | 电子     | 无线鼠标   | 245      | 50       | 充足           | A12      |
|  2   | 电子     | 机械键盘   | 38       | 40       | **低于安全线** | B07      |
|  3   | 办公     | 订书机     | 67       | 40       | 正常           | C09      |
|  4   | 家具     | 人体工学椅 | 12       | 15       | **低于安全线** | D01      |
""".strip()
        target_md = """
# 项目数据统计表

## 产品库存状态表

| 序号 | 产品类别 | 产品名称     | 库存数量 | 安全库存 | 状态           | 存放货架 |
| :--: | -------- | ------------ | -------- | -------- | -------------- | -------- |
|  1   | 电子     | 无线鼠标     | 245      | 50       | 充足           | A12      |
|  2   | 电子     | 机械键盘     | 38       | 40       | **低于安全线** | B07      |
|  3   | 办公     | A4复印纸(箱) | 120      | 100      | 正常           | C03      |
|  4   | 办公     | 订书机       | 67       | 40       | 正常           | C08      |
|  5   | 家具     | 人体工学椅   | 12       | 15       | **低于安全线** | D01      |
""".strip()

        result = compare(
            _parse_markdown(baseline_md, "baseline", "b"),
            _parse_markdown(target_md, "target", "t"),
            policy=ComparePolicy(use_llm_classify=False, rule_strengthen=True),
        )

        assert not any(":--:" in item.baseline_text or ":--:" in item.target_text for item in result.items)
        assert not any(
            "订书机" in item.baseline_text and "A4复印纸" in item.target_text
            for item in result.items
        )
        assert any(
            item.diff_type == "新增" and "A4复印纸" in item.target_text
            for item in result.items
        )
        assert any(
            "订书机" in item.baseline_text
            and "订书机" in item.target_text
            and "C09" in item.baseline_text
            and "C08" in item.target_text
            for item in result.items
        )
