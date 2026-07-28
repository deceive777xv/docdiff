"""Import-time structure normalization with conservative failure semantics."""

from typing import TYPE_CHECKING

from .models import (
    RejectedStructureCandidate,
    StructureRepairDecision,
    StructureRepairOperation,
    StructureRepairResult,
    StructureRepairTrace,
)
from .pipeline import ALGORITHM_VERSION, SCHEMA_VERSION, repair_document

if TYPE_CHECKING:
    from .storage import ImportStructureArtifacts


def __getattr__(name: str):
    if name in {"ImportStructureArtifacts", "prepare_import_ir"}:
        from .storage import ImportStructureArtifacts, prepare_import_ir

        return {
            "ImportStructureArtifacts": ImportStructureArtifacts,
            "prepare_import_ir": prepare_import_ir,
        }[name]
    raise AttributeError(name)

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
