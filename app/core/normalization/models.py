"""Public result types for document normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.core.diff.reconstruction_trace import (
    DocumentTraceRef,
    ReconstructionTrace,
    SourceRowRef,
)
from app.core.structure_repair.models import StructureRepairTrace
from app.core.types import DocumentIR


NormalizationStatus = Literal["repaired", "unchanged", "fallback"]


@dataclass(frozen=True)
class DeferredTableCandidate:
    candidate_id: str
    source_rows: list[SourceRowRef]
    column_mapping: dict[int, int]
    failure_code: str


@dataclass(frozen=True)
class DocumentNormalizationTrace:
    scope: Literal["document"]
    schema_version: int
    algorithm_version: str
    source_document_refs: list[DocumentTraceRef]
    status: NormalizationStatus
    structure_trace: StructureRepairTrace
    table_trace: ReconstructionTrace
    deferred_table_candidates: list[DeferredTableCandidate] = field(
        default_factory=list
    )
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentBoundaryProfile:
    schema_version: int
    algorithm_version: str
    doc_id: str
    file_hash: str
    table_candidate_count: int
    deferred_table_candidates: list[DeferredTableCandidate] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class DocumentNormalizationResult:
    document: DocumentIR
    structure_trace: StructureRepairTrace
    table_trace: ReconstructionTrace
    trace: DocumentNormalizationTrace
    boundary_profile: DocumentBoundaryProfile
    status: NormalizationStatus
    warnings: list[str] = field(default_factory=list)
