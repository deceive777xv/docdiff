"""Tests for app/core/parser/ocr_interface.py"""
import pytest

from app.core.types import ParseQualityReport
from app.core.parser.ocr_interface import enhance_pages


def test_enhance_pages_raises_not_implemented():
    """enhance_pages should raise NotImplementedError (Phase 2 placeholder)."""
    report = ParseQualityReport(quality_score=0.5, needs_ocr=True)
    with pytest.raises(NotImplementedError):
        enhance_pages("x.pdf", [1], report)
