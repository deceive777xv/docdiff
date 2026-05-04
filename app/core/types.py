from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


# ── Document IR ────────────────────────────────────────────────────────────────

@dataclass
class Sentence:
    text: str


@dataclass
class Paragraph:
    paragraph_id: str
    page_no: int
    text: str
    sentences: list[Sentence] = field(default_factory=list)


@dataclass
class Section:
    section_id: str
    title: str
    level: int          # 1 / 2 / 3
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class DocumentIR:
    doc_id: str
    title: str
    file_hash: str
    sections: list[Section] = field(default_factory=list)
    plain_text: str = ""


# ── Parsing ────────────────────────────────────────────────────────────────────

@dataclass
class ParseQualityReport:
    quality_score: float        # 0.0–1.0
    needs_ocr: bool
    ocr_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Chunks & Retrieval ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id: str
    version_id: str
    chunk_no: int
    section_path: str
    page_no: int
    text: str
    faiss_index_id: int = -1


@dataclass
class ChunkHit:
    chunk: Chunk
    score: float


class RetrievalScope(Enum):
    CURRENT_DOC  = "current_doc"
    BASELINE     = "baseline"
    TARGET       = "target"
    COMPARE      = "compare"
    STANDARD_LIB = "standard_lib"
    ALL          = "all"


# ── Diff ───────────────────────────────────────────────────────────────────────

DiffType  = Literal["新增", "删减", "微调", "实质修改", "重写", "格式变化"]
RiskLevel = Literal["high", "medium", "low"]


@dataclass
class DiffItem:
    diff_id: str
    section_path: str
    diff_type: DiffType
    risk_level: RiskLevel
    baseline_text: str
    target_text: str
    similarity_score: float
    explanation: str
    baseline_page: int
    target_page: int


@dataclass
class DiffResult:
    task_id: str
    baseline_version_id: str
    target_version_id: str
    items: list[DiffItem] = field(default_factory=list)


@dataclass
class ComparePolicy:
    similarity_threshold: float = 0.75   # below this → 新增/删减
    use_llm_classify: bool = True
    rule_strengthen: bool = True
