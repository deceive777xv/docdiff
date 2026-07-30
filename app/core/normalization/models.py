"""Public result types for document normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from app.core.normalization.table_trace import DocumentTraceRef, ReconstructionTrace
from app.core.structure_repair.models import StructureRepairTrace
from app.core.types import DocumentIR


class NormalizationDepth(str, Enum):
    OFF = "off"
    STANDARD = "standard"
    REVIEW = "review"


NormalizationStatus = Literal["skipped", "repaired", "unchanged", "fallback"]


@dataclass(frozen=True)
class DocumentNormalizationTrace:
    scope: Literal["document"]
    schema_version: int
    algorithm_version: str
    source_document_refs: list[DocumentTraceRef]
    status: NormalizationStatus
    structure_trace: StructureRepairTrace
    table_trace: ReconstructionTrace
    warnings: list[str] = field(default_factory=list)
    normalization_depth: str = NormalizationDepth.OFF.value


@dataclass(frozen=True)
class DocumentBoundaryProfile:
    schema_version: int
    algorithm_version: str
    doc_id: str
    file_hash: str
    table_candidate_count: int


@dataclass(frozen=True)
class DocumentNormalizationResult:
    document: DocumentIR
    structure_trace: StructureRepairTrace
    table_trace: ReconstructionTrace
    trace: DocumentNormalizationTrace
    boundary_profile: DocumentBoundaryProfile
    status: NormalizationStatus
    warnings: list[str] = field(default_factory=list)
