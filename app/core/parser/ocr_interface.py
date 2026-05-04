"""OCR interface — placeholder for Phase 2 Tesseract integration."""
from __future__ import annotations
from app.core.types import ParseQualityReport


def enhance_pages(
    file_path: str,
    page_numbers: list[int],
    report: ParseQualityReport,
) -> dict[int, str]:
    """
    Phase 2: Run OCR on specified pages and return {page_no: extracted_text}.
    Currently raises NotImplementedError.
    """
    raise NotImplementedError(
        "OCR enhancement is not yet implemented. "
        "It will be added in Phase 2 using Tesseract."
    )
