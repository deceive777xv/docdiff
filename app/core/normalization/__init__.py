"""Unified document-normalization public API."""

from .models import (
    DocumentBoundaryProfile,
    DocumentNormalizationResult,
    DocumentNormalizationTrace,
    NormalizationDepth,
    NormalizationStatus,
)
from .pipeline import (
    ALGORITHM_VERSION,
    SCHEMA_VERSION,
    normalize_document,
)
from .storage import (
    boundary_profile_path,
    normalization_trace_path,
)

__all__ = [
    "ALGORITHM_VERSION",
    "SCHEMA_VERSION",
    "DocumentBoundaryProfile",
    "DocumentNormalizationResult",
    "DocumentNormalizationTrace",
    "NormalizationDepth",
    "NormalizationStatus",
    "boundary_profile_path",
    "normalize_document",
    "normalization_trace_path",
]
