"""Typed results and audit records for import-time structure repair."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.core.types import DocumentIR


RepairStatus = Literal["repaired", "unchanged", "fallback"]


@dataclass(frozen=True)
class StructureRepairOperation:
    operation_id: str
    type: str
    source_ids: list[str]
    output_id: str = ""
    target_section_id: str = ""
    reason: str = ""
    actor: Literal["rule", "llm"] = "rule"
    confidence: float | None = None
    source_sentence_indexes: list[int] = field(default_factory=list)
    removed_text: str = ""


@dataclass(frozen=True)
class RejectedStructureCandidate:
    candidate_id: str
    code: str
    reason: str


@dataclass(frozen=True)
class StructureRepairDecision:
    candidate_id: str
    action: str
    source_ids: list[str]
    target_section_id: str
    confidence: float
    reason: str
    actor: Literal["llm"] = "llm"


@dataclass(frozen=True)
class StructureRepairTrace:
    schema_version: int
    algorithm_version: str
    doc_id: str
    raw_hash: str
    normalized_hash: str
    status: RepairStatus
    operations: list[StructureRepairOperation] = field(default_factory=list)
    decisions: list[StructureRepairDecision] = field(default_factory=list)
    rejected: list[RejectedStructureCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StructureRepairResult:
    document: DocumentIR
    trace: StructureRepairTrace
    status: RepairStatus
    warnings: list[str] = field(default_factory=list)
