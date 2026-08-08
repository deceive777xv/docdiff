"""Document parsing router with explicit parser ownership by format."""
from __future__ import annotations

from pathlib import Path

from app.core.parser import anydoc_adapter, markitdown_adapter, pymupdf4llm_adapter
from app.core.types import DocumentIR, ParseQualityReport


PDF_EXTENSIONS = frozenset({".pdf"})
ANYDOC_EXTENSIONS = frozenset({
    ".doc", ".docx", ".docm",
    ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".xls", ".xlsx", ".xlsm", ".xlsb",
    ".odt", ".ods", ".odp",
    ".rtf", ".epub", ".csv",
})
MARKITDOWN_EXTENSIONS = frozenset({
    ".html", ".htm", ".json", ".xml", ".txt", ".md", ".markdown",
})
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | ANYDOC_EXTENSIONS | MARKITDOWN_EXTENSIONS


def parse_document(
    file_path: str,
    llm_client=None,
    llm_model: str = "",
) -> tuple[DocumentIR, ParseQualityReport]:
    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {suffix!r}")
    if suffix in PDF_EXTENSIONS:
        ir = pymupdf4llm_adapter.extract(file_path)
    elif suffix in ANYDOC_EXTENSIONS:
        ir = anydoc_adapter.extract(file_path)
    else:
        ir = markitdown_adapter.extract(file_path, llm_client, llm_model)
    return ir, evaluate_quality(ir)


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
    return ParseQualityReport(
        quality_score=round(score, 2),
        needs_ocr=score < 0.4,
        warnings=warnings,
    )
