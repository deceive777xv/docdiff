"""Unified document-normalization public API."""

from .models import (
    DeferredTableCandidate,
    DocumentBoundaryProfile,
    DocumentNormalizationResult,
    DocumentNormalizationTrace,
    NormalizationStatus,
)
from .pipeline import (
    ALGORITHM_VERSION,
    SCHEMA_VERSION,
    normalize_document,
    normalize_pair,
)
from .storage import (
    boundary_profile_path,
    load_deferred_table_candidates,
    normalization_trace_path,
)

__all__ = [
    "ALGORITHM_VERSION",
    "SCHEMA_VERSION",
    "DeferredTableCandidate",
    "DocumentBoundaryProfile",
    "DocumentNormalizationResult",
    "DocumentNormalizationTrace",
    "NormalizationStatus",
    "boundary_profile_path",
    "load_deferred_table_candidates",
    "normalize_document",
    "normalize_pair",
    "normalization_trace_path",
]
