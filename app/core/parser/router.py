"""Route documents to the appropriate parser based on file type and mode."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Literal

from app.core.types import DocumentIR, ParseQualityReport

logger = logging.getLogger(__name__)

ParseMode = Literal["standard", "fast"]


def parse_document(
    file_path: str,
    mode: ParseMode = "standard",
) -> tuple[DocumentIR, ParseQualityReport]:
    """
    Route a file to the appropriate parser.

    standard mode (PDF):
      1. Try Docling if available
      2. Fall back to PyMuPDF
    fast mode (PDF):
      - PyMuPDF only
    DOCX (any mode):
      - python-docx extractor
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".docx":
        from app.core.parser.docx_extractor import extract
        return extract(file_path)

    if suffix == ".pdf":
        if mode == "standard":
            from app.core.parser.docling_adapter import is_available, extract as docling_extract
            if is_available():
                try:
                    return docling_extract(file_path)
                except Exception as e:
                    logger.warning("Docling failed, falling back to PyMuPDF: %s", e)
        from app.core.parser.pymupdf_extractor import extract as pymupdf_extract
        return pymupdf_extract(file_path)

    raise ValueError(f"Unsupported file format: {suffix!r}. Supported: .pdf, .docx")


def evaluate_quality(ir: DocumentIR) -> ParseQualityReport:
    """Evaluate the quality of a parsed DocumentIR.

    Computes a quality_score in [0, 1] based on:
    - Section structure presence
    - Average paragraph text length
    - Ratio of very-short (likely garbage) paragraphs
    - Presence of page numbers in paragraphs
    Also flags needs_ocr when text density is very low.
    """
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
    has_page_numbers = any(p.page_no > 0 for p in all_paras)

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

    if not has_page_numbers:
        score -= 0.05
        warnings.append("段落缺少页码信息")

    score = max(0.0, min(1.0, score))
    needs_ocr = score < 0.4

    return ParseQualityReport(
        quality_score=round(score, 2),
        needs_ocr=needs_ocr,
        warnings=warnings,
    )
