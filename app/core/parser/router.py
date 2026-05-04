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
