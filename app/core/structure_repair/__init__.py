"""Import-time structure normalization with conservative failure semantics."""

from .models import (
    RejectedStructureCandidate,
    StructureRepairDecision,
    StructureRepairOperation,
    StructureRepairResult,
    StructureRepairTrace,
)
from .pipeline import ALGORITHM_VERSION, SCHEMA_VERSION, repair_document
from .storage import ImportStructureArtifacts, prepare_import_ir

__all__ = [
    "ALGORITHM_VERSION",
    "SCHEMA_VERSION",
    "RejectedStructureCandidate",
    "ImportStructureArtifacts",
    "StructureRepairOperation",
    "StructureRepairDecision",
    "StructureRepairResult",
    "StructureRepairTrace",
    "repair_document",
    "prepare_import_ir",
]
