"""Document parsing router — format guard + dispatcher to markitdown_adapter."""
from __future__ import annotations

from pathlib import Path

from app.core.parser import markitdown_adapter
from app.core.types import DocumentIR, ParseQualityReport

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls",
    ".html", ".htm", ".csv", ".json", ".xml", ".epub",
    ".txt",
}


def parse_document(
    file_path: str,
    llm_client=None,
    llm_model: str = "",
) -> tuple[DocumentIR, ParseQualityReport]:
    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {suffix!r}")
    ir = markitdown_adapter.extract(file_path, llm_client, llm_model)
    report = evaluate_quality(ir)
    return ir, report


def evaluate_quality(ir: DocumentIR) -> ParseQualityReport:
    all_paras = [p for sec in ir.sections for p in sec.paragraphs]
    warnings: list[str] = []

    if not ir.sections:
        return ParseQualityReport(
            quality_score=0.1, needs_ocr=True,
            warnings=["文档无可识别章节结构，可能是扫描件或解析失败"],
        )

    if not all_paras:
        return ParseQualityReport(
            quality_score=0.1, needs_ocr=True,
            warnings=["文档无段落内容"],
        )

    avg_len = sum(len(p.text) for p in all_paras) / len(all_paras)
    short_ratio = sum(1 for p in all_paras if len(p.text) < 10) / len(all_paras)

    score = 1.0

    if avg_len < 20:
        score -= 0.4
        warnings.append(f"平均段落长度过短（{avg_len:.0f} 字），可能存在解析质量问题")
    elif avg_len < 50:
        score -= 0.2
        warnings.append(f"平均段落长度偏短（{avg_len:.0f} 字）")

    if short_ratio > 0.5:
        score -= 0.3
        warnings.append(f"超过 {short_ratio:.0%} 的段落文本过短，建议检查解析结果")
    elif short_ratio > 0.2:
        score -= 0.1

    score = max(0.0, min(1.0, score))
    needs_ocr = score < 0.4

    return ParseQualityReport(
        quality_score=round(score, 2),
        needs_ocr=needs_ocr,
        warnings=warnings,
    )
