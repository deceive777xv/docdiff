"""Tests for report_service.py — docx and HTML export."""
from __future__ import annotations
import uuid
from pathlib import Path

import pytest

from app.core.types import DiffItem, DiffResult


def _make_result() -> DiffResult:
    items = [
        DiffItem(
            diff_id=str(uuid.uuid4()),
            section_path="第1条",
            diff_type="实质修改",
            risk_level="high",
            baseline_text="合同金额为100万元，付款期限30日。",
            target_text="合同金额为200万元，付款期限60日。",
            similarity_score=0.82,
            explanation="金额和期限均发生变化",
            baseline_page=1,
            target_page=1,
        ),
        DiffItem(
            diff_id=str(uuid.uuid4()),
            section_path="第2条",
            diff_type="新增",
            risk_level="medium",
            baseline_text="",
            target_text="乙方须在交货前提交质量证明文件。",
            similarity_score=0.0,
            explanation="目标文档新增段落",
            baseline_page=0,
            target_page=2,
        ),
        DiffItem(
            diff_id=str(uuid.uuid4()),
            section_path="第3条",
            diff_type="重写",
            risk_level="high",
            baseline_text="甲方负责原材料供应。",
            target_text="乙方承担全部物流及验收责任，费用另行结算。",
            similarity_score=0.12,
            explanation="文本结构大幅调整",
            baseline_page=2,
            target_page=3,
        ),
    ]
    return DiffResult(
        task_id="test-task-001",
        baseline_version_id="baseline-v1",
        target_version_id="target-v1",
        items=items,
    )


def test_export_docx_creates_file(tmp_path):
    from app.services.report_service import export_docx

    out = str(tmp_path / "report.docx")
    export_docx(_make_result(), out)

    assert Path(out).exists()
    assert Path(out).stat().st_size > 1000  # non-trivial file


def test_export_docx_is_valid_docx(tmp_path):
    from docx import Document
    from app.services.report_service import export_docx

    out = str(tmp_path / "report.docx")
    export_docx(_make_result(), out)

    doc = Document(out)
    full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "文档对比报告" in full_text
    assert "实质修改" in full_text
    assert "重写" in full_text


def test_export_html_creates_file(tmp_path):
    from app.services.report_service import export_html

    out = str(tmp_path / "report.html")
    export_html(_make_result(), out)

    assert Path(out).exists()
    assert Path(out).stat().st_size > 500


def test_export_html_contains_required_content(tmp_path):
    from app.services.report_service import export_html

    out = str(tmp_path / "report.html")
    export_html(_make_result(), out)

    content = Path(out).read_text(encoding="utf-8")
    assert "文档对比报告" in content
    assert "实质修改" in content
    assert "新增" in content
    assert "重写" in content
    assert "第1条" in content
    assert "<!DOCTYPE html>" in content


def test_export_html_escapes_html_characters(tmp_path):
    from app.services.report_service import export_html

    items = [
        DiffItem(
            diff_id=str(uuid.uuid4()),
            section_path="<script>alert(1)</script>",
            diff_type="微调",
            risk_level="low",
            baseline_text="text <b>bold</b>",
            target_text="text & more",
            similarity_score=0.9,
            explanation="",
            baseline_page=1,
            target_page=1,
        )
    ]
    result = DiffResult(
        task_id="t",
        baseline_version_id="b",
        target_version_id="v",
        items=items,
    )
    out = str(tmp_path / "report.html")
    export_html(result, out)

    content = Path(out).read_text(encoding="utf-8")
    assert "<script>" not in content  # escaped
    assert "&lt;script&gt;" in content
