# Markitdown Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace docling (and all other parsers) with markitdown as the sole document parsing backend, remove page number logic throughout, and wire markitdown-ocr through the ingest pipeline.

**Architecture:** A single `markitdown_adapter.py` replaces three parser files (docling_adapter, docx_extractor, pymupdf_extractor). The router becomes a thin format-guard + dispatcher. Page number fields are removed from types, ir_builder, compare_repo, diff_classifier, compare_graph, and UI. An `openai.OpenAI` client flows from AppContext through the ingest graph to enable LLM-based OCR via markitdown-ocr.

**Tech Stack:** `markitdown[all]`, `markitdown-ocr`, `openai` SDK (already a dependency).

---

## File Map

| Operation | File | Responsibility |
|---|---|---|
| Create | `app/core/parser/markitdown_adapter.py` | Convert any supported file → DocumentIR via markitdown |
| Create | `tests/test_parser/test_markitdown_adapter.py` | Unit tests for adapter |
| Delete | `app/core/parser/docling_adapter.py` | Replaced by markitdown_adapter |
| Delete | `app/core/parser/docx_extractor.py` | Replaced by markitdown_adapter |
| Delete | `app/core/parser/pymupdf_extractor.py` | Replaced by markitdown_adapter |
| Modify | `app/core/parser/router.py` | Format guard + dispatcher; remove mode/page_no check |
| Modify | `tests/test_parser/test_router.py` | Remove docling monkeypatching; add extension/OCR-client tests |
| Modify | `app/core/types.py` | Remove page_no from Paragraph, ocr_pages from ParseQualityReport, baseline_page/target_page from DiffItem; Chunk.page_no → default 0 |
| Modify | `app/core/parser/ir_builder.py` | Remove `page_no=para.page_no` from Chunk constructors |
| Modify | `app/db/compare_repo.py` | Insert 0,0 for baseline_page/target_page (keep DB cols, remove from DiffItem) |
| Modify | `app/core/diff/diff_classifier.py` | Remove baseline_page/target_page from three DiffItem constructors |
| Modify | `app/agent/compare_graph.py` | Remove `page_no=p["page_no"]` from Paragraph constructor in _load_ir |
| Modify | `app/agent/states.py` | Add llm_client, llm_model to IngestState inputs |
| Modify | `app/agent/ingest_graph.py` | Pass llm_client/llm_model to parse_document; remove ocr_pages log |
| Modify | `app/services/ingest_service.py` | Replace parse_mode with llm_client/llm_model; remove ocr_pages logs |
| Modify | `app/ui/app_context.py` | Add openai_client, openai_model fields |
| Modify | `app/ui/pages/library_page.py` | Add llm_client/llm_model to ingest_graph.invoke() |
| Modify | `app/ui/pages/settings_page.py` | Assign openai_client/openai_model on settings save |
| Modify | `app/ui/pages/compare_page.py` | Remove baseline_page/target_page display from diff items |
| Modify | `requirements.txt` | Replace docling/pymupdf/python-docx with markitdown[all] + markitdown-ocr |
| Modify | `pyproject.toml` | Same as requirements.txt |

---

## Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

- [ ] **Step 1: Update requirements.txt**

Open `requirements.txt`. Remove these three lines:
```
PyMuPDF>=1.24.0
python-docx>=1.1.0
docling>=2.31.0
```

Add these two lines:
```
markitdown[all]>=0.1.0
markitdown-ocr>=0.1.0
```

- [ ] **Step 2: Update pyproject.toml**

In `pyproject.toml`, in the `[project]` `dependencies` list, remove:
```
"PyMuPDF>=1.24.0",
"python-docx>=1.1.0",
"docling>=2.31.0",
```

Add:
```
"markitdown[all]>=0.1.0",
"markitdown-ocr>=0.1.0",
```

- [ ] **Step 3: Install updated dependencies**

Run: `pip install -r requirements.txt`
Expected: markitdown and markitdown-ocr install without error.

- [ ] **Step 4: Verify markitdown is importable**

Run: `python -c "from markitdown import MarkItDown; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "chore: replace docling/pymupdf/python-docx with markitdown"
```

---

## Task 2: Create markitdown_adapter.py (TDD)

**Files:**
- Create: `tests/test_parser/test_markitdown_adapter.py`
- Create: `app/core/parser/markitdown_adapter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_parser/test_markitdown_adapter.py`:

```python
"""Tests for markitdown_adapter."""
from __future__ import annotations


def test_is_available_returns_bool():
    from app.core.parser.markitdown_adapter import is_available
    assert isinstance(is_available(), bool)


def test_headingless_content_creates_default_section():
    """Content with no heading auto-inserts a '正文' section."""
    from app.core.parser.markitdown_adapter import _parse_markdown

    md = "Some text without any heading.\nMore text here."
    ir = _parse_markdown(md, "test_doc", "abc123")

    assert len(ir.sections) == 1
    assert ir.sections[0].title == "正文"
    assert ir.sections[0].level == 1
    assert len(ir.sections[0].paragraphs) >= 1
    assert "Some text" in ir.sections[0].paragraphs[0].text


def test_single_heading_with_body():
    from app.core.parser.markitdown_adapter import _parse_markdown

    md = "# Introduction\n\nThis is the intro text.\n\nMore intro text."
    ir = _parse_markdown(md, "test_doc", "abc123")

    assert len(ir.sections) == 1
    assert ir.sections[0].title == "Introduction"
    assert ir.sections[0].level == 1
    assert len(ir.sections[0].paragraphs) == 2


def test_multi_level_headings():
    from app.core.parser.markitdown_adapter import _parse_markdown

    md = (
        "# Chapter 1\n\nIntro paragraph.\n\n"
        "## Section 1.1\n\nSub content here.\n\n"
        "### Subsection 1.1.1\n\nDeep content.\n\n"
        "## Section 1.2\n\nAnother section."
    )
    ir = _parse_markdown(md, "test_doc", "abc123")

    assert len(ir.sections) == 4
    assert ir.sections[0].title == "Chapter 1"
    assert ir.sections[0].level == 1
    assert ir.sections[1].title == "Section 1.1"
    assert ir.sections[1].level == 2
    assert ir.sections[2].title == "Subsection 1.1.1"
    assert ir.sections[2].level == 3
    assert ir.sections[3].title == "Section 1.2"
    assert ir.sections[3].level == 2


def test_extract_with_no_llm_client(tmp_path):
    """extract() with llm_client=None must not raise."""
    test_file = tmp_path / "test.html"
    test_file.write_text("<h1>Hello</h1><p>World</p>", encoding="utf-8")

    from app.core.parser.markitdown_adapter import extract
    ir = extract(str(test_file), llm_client=None, llm_model="")

    assert ir.title == "test"
    assert len(ir.sections) >= 1


def test_extract_populates_doc_id_and_file_hash(tmp_path):
    test_file = tmp_path / "sample.html"
    test_file.write_text("<h1>Title</h1><p>Content here.</p>", encoding="utf-8")

    from app.core.parser.markitdown_adapter import extract
    ir = extract(str(test_file))

    assert ir.doc_id != ""
    assert ir.file_hash != ""


def test_paragraph_has_no_page_no(tmp_path):
    """After migration Paragraph must not have a page_no attribute."""
    test_file = tmp_path / "test.html"
    test_file.write_text("<h1>Section</h1><p>Text here.</p>", encoding="utf-8")

    from app.core.parser.markitdown_adapter import extract
    ir = extract(str(test_file))

    para = ir.sections[0].paragraphs[0]
    assert not hasattr(para, "page_no"), "Paragraph.page_no must not exist after migration"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parser/test_markitdown_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.parser.markitdown_adapter'`

- [ ] **Step 3: Create markitdown_adapter.py**

Create `app/core/parser/markitdown_adapter.py`:

```python
"""Markitdown-based document parser — converts any supported file to DocumentIR."""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from app.core.types import DocumentIR, Paragraph, Section, Sentence


def is_available() -> bool:
    try:
        import markitdown  # noqa: F401
        return True
    except ImportError:
        return False


def extract(
    file_path: str,
    llm_client=None,
    llm_model: str = "",
) -> DocumentIR:
    if not is_available():
        raise RuntimeError("markitdown is not installed")

    from markitdown import MarkItDown

    md = MarkItDown(
        enable_plugins=bool(llm_client),
        llm_client=llm_client or None,
        llm_model=llm_model or None,
    )
    result = md.convert(file_path)
    title = Path(file_path).stem
    file_hash = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
    return _parse_markdown(result.text_content, title, file_hash)


def _parse_markdown(md_text: str, title: str, doc_hash: str) -> DocumentIR:
    sections: list[Section] = []
    current_section: Section | None = None
    para_buffer: list[str] = []

    def _flush() -> None:
        if current_section is not None and para_buffer:
            joined = " ".join(para_buffer).strip()
            if joined:
                current_section.paragraphs.append(
                    Paragraph(paragraph_id=str(uuid.uuid4()), text=joined)
                )
        para_buffer.clear()

    heading_re = re.compile(r"^(#{1,3})\s+(.+)")

    for line in md_text.splitlines():
        m = heading_re.match(line)
        if m:
            _flush()
            level = len(m.group(1))
            current_section = Section(
                section_id=str(uuid.uuid4()),
                title=m.group(2).strip(),
                level=level,
                paragraphs=[],
            )
            sections.append(current_section)
        elif line.strip() == "":
            _flush()
        else:
            if current_section is None:
                current_section = Section(
                    section_id=str(uuid.uuid4()),
                    title="正文",
                    level=1,
                    paragraphs=[],
                )
                sections.append(current_section)
            para_buffer.append(line.strip())

    _flush()

    plain_text = "\n".join(
        para.text for sec in sections for para in sec.paragraphs
    )
    return DocumentIR(
        doc_id=str(uuid.uuid4()),
        title=title,
        file_hash=doc_hash,
        sections=sections,
        plain_text=plain_text,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parser/test_markitdown_adapter.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core/parser/markitdown_adapter.py tests/test_parser/test_markitdown_adapter.py
git commit -m "feat: add markitdown_adapter — file → DocumentIR via markitdown"
```

---

## Task 3: Rewrite router.py and test_router.py (TDD)

**Files:**
- Modify: `tests/test_parser/test_router.py`
- Modify: `app/core/parser/router.py`

- [ ] **Step 1: Write the new tests**

Replace the entire contents of `tests/test_parser/test_router.py` with:

```python
"""Tests for router.parse_document."""
from __future__ import annotations

import inspect
import dataclasses

import pytest


def test_supported_extensions_set():
    from app.core.parser.router import SUPPORTED_EXTENSIONS
    for ext in (".pdf", ".docx", ".pptx", ".xlsx", ".html", ".csv", ".epub"):
        assert ext in SUPPORTED_EXTENSIONS


def test_unsupported_extension_raises_value_error(tmp_path):
    from app.core.parser.router import parse_document
    bad = tmp_path / "file.xyz"
    bad.write_text("content")
    with pytest.raises(ValueError, match="Unsupported format"):
        parse_document(str(bad))


def test_parse_document_returns_ir_and_report(tmp_path):
    from app.core.parser.router import parse_document
    from app.core.types import DocumentIR, ParseQualityReport

    f = tmp_path / "doc.html"
    f.write_text("<h1>Title</h1><p>Some content here.</p>", encoding="utf-8")

    ir, report = parse_document(str(f))
    assert isinstance(ir, DocumentIR)
    assert isinstance(report, ParseQualityReport)


def test_parse_document_none_llm_client_does_not_raise(tmp_path):
    from app.core.parser.router import parse_document

    f = tmp_path / "doc.html"
    f.write_text("<p>Hello world</p>", encoding="utf-8")

    ir, report = parse_document(str(f), llm_client=None, llm_model="")
    assert ir is not None
    assert report is not None


def test_parse_document_has_no_mode_parameter():
    from app.core.parser import router
    sig = inspect.signature(router.parse_document)
    assert "mode" not in sig.parameters


def test_quality_report_has_no_ocr_pages_field():
    from app.core.types import ParseQualityReport
    fields = {f.name for f in dataclasses.fields(ParseQualityReport)}
    assert "ocr_pages" not in fields
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parser/test_router.py -v`
Expected: Multiple failures — `mode` param still exists, `SUPPORTED_EXTENSIONS` not defined, `ocr_pages` still in ParseQualityReport.

- [ ] **Step 3: Rewrite router.py**

Replace the entire contents of `app/core/parser/router.py` with:

```python
"""Document parsing router — format guard + dispatcher to markitdown_adapter."""
from __future__ import annotations

from pathlib import Path

from app.core.parser import markitdown_adapter
from app.core.types import DocumentIR, ParseQualityReport

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls",
    ".html", ".htm", ".csv", ".json", ".xml", ".epub",
    ".txt",
}


def parse_document(
    file_path: str,
    llm_client=None,
    llm_model: str = "",
) -> tuple[DocumentIR, ParseQualityReport]:
    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {suffix!r}")
    ir = markitdown_adapter.extract(file_path, llm_client, llm_model)
    report = evaluate_quality(ir)
    return ir, report


def evaluate_quality(ir: DocumentIR) -> ParseQualityReport:
    all_paras = [p for sec in ir.sections for p in sec.paragraphs]
    warnings: list[str] = []

    if not ir.sections:
        return ParseQualityReport(
            quality_score=0.1, needs_ocr=True,
            warnings=["文档无可识别章节结构，可能是扫描件或解析失败"],
        )

    if not all_paras:
        return ParseQualityReport(
            quality_score=0.1, needs_ocr=True,
            warnings=["文档无段落内容"],
        )

    avg_len = sum(len(p.text) for p in all_paras) / len(all_paras)
    short_ratio = sum(1 for p in all_paras if len(p.text) < 10) / len(all_paras)

    score = 1.0

    if avg_len < 20:
        score -= 0.4
        warnings.append(f"平均段落长度过短（{avg_len:.0f} 字），可能存在解析质量问题")
    elif avg_len < 50:
        score -= 0.2
        warnings.append(f"平均段落长度偏短（{avg_len:.0f} 字）")

    if short_ratio > 0.5:
        score -= 0.3
        warnings.append(f"超过 {short_ratio:.0%} 的段落文本过短，建议检查解析结果")
    elif short_ratio > 0.2:
        score -= 0.1

    score = max(0.0, min(1.0, score))
    needs_ocr = score < 0.4

    return ParseQualityReport(
        quality_score=round(score, 2),
        needs_ocr=needs_ocr,
        warnings=warnings,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parser/test_router.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core/parser/router.py tests/test_parser/test_router.py
git commit -m "feat: rewrite router to dispatch to markitdown_adapter"
```

---

## Task 4: Remove page_no from Types, ir_builder, compare_repo, diff_classifier, compare_graph

**Files:**
- Modify: `app/core/types.py`
- Modify: `app/core/parser/ir_builder.py`
- Modify: `app/db/compare_repo.py`
- Modify: `app/core/diff/diff_classifier.py`
- Modify: `app/agent/compare_graph.py`

These changes cascade: `Paragraph` loses `page_no`, so every place that reads or writes it must be updated. All page fields are removed from `DiffItem` too; the DB columns are kept but receive hardcoded `0` to avoid a schema migration.

- [ ] **Step 1: Write the failing test**

Add a new file `tests/test_types_page_no_removed.py`:

```python
"""Verify page_no fields were fully removed from data types."""
from __future__ import annotations
import dataclasses


def test_paragraph_no_page_no():
    from app.core.types import Paragraph
    fields = {f.name for f in dataclasses.fields(Paragraph)}
    assert "page_no" not in fields


def test_diff_item_no_page_fields():
    from app.core.types import DiffItem
    fields = {f.name for f in dataclasses.fields(DiffItem)}
    assert "baseline_page" not in fields
    assert "target_page" not in fields


def test_chunk_page_no_defaults_to_zero():
    from app.core.types import Chunk
    c = Chunk(id="x", version_id="v", chunk_no=0, section_path="sec", text="hello")
    assert c.page_no == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_types_page_no_removed.py -v`
Expected: All 3 tests FAIL (fields still exist, Chunk requires page_no).

- [ ] **Step 3: Update app/core/types.py**

**Change 1** — remove `page_no: int` from `Paragraph` (line 17):

Old:
```python
@dataclass
class Paragraph:
    paragraph_id: str
    page_no: int
    text: str
    sentences: list[Sentence] = field(default_factory=list)
```

New:
```python
@dataclass
class Paragraph:
    paragraph_id: str
    text: str
    sentences: list[Sentence] = field(default_factory=list)
```

**Change 2** — remove `ocr_pages` from `ParseQualityReport` (line 45):

Old:
```python
@dataclass
class ParseQualityReport:
    quality_score: float        # 0.0–1.0
    needs_ocr: bool
    ocr_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
```

New:
```python
@dataclass
class ParseQualityReport:
    quality_score: float        # 0.0–1.0
    needs_ocr: bool
    warnings: list[str] = field(default_factory=list)
```

**Change 3** — remove `baseline_page` and `target_page` from `DiffItem` (lines 93–94), and give `Chunk.page_no` a default of 0 (line 57):

Old `DiffItem`:
```python
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
```

New `DiffItem`:
```python
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
```

Old `Chunk`:
```python
@dataclass
class Chunk:
    id: str
    version_id: str
    chunk_no: int
    section_path: str
    page_no: int
    text: str
    faiss_index_id: int = -1
```

New `Chunk`:
```python
@dataclass
class Chunk:
    id: str
    version_id: str
    chunk_no: int
    section_path: str
    text: str
    page_no: int = 0
    faiss_index_id: int = -1
```

- [ ] **Step 4: Update app/core/parser/ir_builder.py**

Remove `page_no=para.page_no,` from both Chunk constructors (lines 25 and 39).

Old (line 19–27):
```python
            if len(para.text) <= max_chars:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    version_id=version_id,
                    chunk_no=chunk_no,
                    section_path=section_path,
                    page_no=para.page_no,
                    text=para.text,
                ))
```

New:
```python
            if len(para.text) <= max_chars:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    version_id=version_id,
                    chunk_no=chunk_no,
                    section_path=section_path,
                    text=para.text,
                ))
```

Old (line 31–41):
```python
                for sent in para.sentences:
                    if not sent.text:
                        continue
                    chunks.append(Chunk(
                        id=str(uuid.uuid4()),
                        version_id=version_id,
                        chunk_no=chunk_no,
                        section_path=section_path,
                        page_no=para.page_no,
                        text=sent.text,
                    ))
```

New:
```python
                for sent in para.sentences:
                    if not sent.text:
                        continue
                    chunks.append(Chunk(
                        id=str(uuid.uuid4()),
                        version_id=version_id,
                        chunk_no=chunk_no,
                        section_path=section_path,
                        text=sent.text,
                    ))
```

- [ ] **Step 5: Update app/db/compare_repo.py**

`DiffItem` no longer has `baseline_page`/`target_page`, but the DB columns exist (no migration needed). Insert hardcoded `0, 0`.

Old `insert_diff_items` function body:
```python
    rows = [
        (
            item.diff_id, task_id, item.section_path, item.diff_type,
            item.risk_level, item.baseline_text, item.target_text,
            item.similarity_score, item.explanation,
            item.baseline_page, item.target_page,
        )
        for item in items
    ]
    conn.executemany(
        """INSERT INTO diff_items
           (id, compare_task_id, section_path, diff_type, risk_level,
            baseline_text, target_text, similarity_score, explanation,
            baseline_page, target_page)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
```

New:
```python
    rows = [
        (
            item.diff_id, task_id, item.section_path, item.diff_type,
            item.risk_level, item.baseline_text, item.target_text,
            item.similarity_score, item.explanation,
            0, 0,
        )
        for item in items
    ]
    conn.executemany(
        """INSERT INTO diff_items
           (id, compare_task_id, section_path, diff_type, risk_level,
            baseline_text, target_text, similarity_score, explanation,
            baseline_page, target_page)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
```

- [ ] **Step 6: Update app/core/diff/diff_classifier.py**

Remove `baseline_page` and `target_page` keyword args from the three `DiffItem(...)` calls.

**First DiffItem** (新增 case, around line 97):

Old:
```python
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type="新增",
                risk_level="medium",
                baseline_text="",
                target_text=pp.target_para.text,
                similarity_score=0.0,
                explanation="目标文档新增段落",
                baseline_page=0,
                target_page=pp.target_para.page_no,
            ))
```

New:
```python
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type="新增",
                risk_level="medium",
                baseline_text="",
                target_text=pp.target_para.text,
                similarity_score=0.0,
                explanation="目标文档新增段落",
            ))
```

**Second DiffItem** (删减 case, around line 110):

Old:
```python
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type="删减",
                risk_level="medium",
                baseline_text=pp.baseline_para.text,
                target_text="",
                similarity_score=0.0,
                explanation="基准文档段落被删除",
                baseline_page=pp.baseline_para.page_no,
                target_page=0,
            ))
```

New:
```python
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type="删减",
                risk_level="medium",
                baseline_text=pp.baseline_para.text,
                target_text="",
                similarity_score=0.0,
                explanation="基准文档段落被删除",
            ))
```

**Third DiffItem** (修改 case, around line 137):

Old:
```python
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type=diff_type,
                risk_level=risk_level,
                baseline_text=pp.baseline_para.text,
                target_text=pp.target_para.text,
                similarity_score=pp.similarity,
                explanation=explanation,
                baseline_page=pp.baseline_para.page_no,
                target_page=pp.target_para.page_no,
            ))
```

New:
```python
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type=diff_type,
                risk_level=risk_level,
                baseline_text=pp.baseline_para.text,
                target_text=pp.target_para.text,
                similarity_score=pp.similarity,
                explanation=explanation,
            ))
```

- [ ] **Step 7: Update app/agent/compare_graph.py**

In `_load_ir`, the `Paragraph` constructor reads `page_no` from the JSON dict. Old JSON files on disk may still have a `page_no` key — just stop passing it to the constructor.

Old (lines 36–43):
```python
        paras = [
            Paragraph(
                paragraph_id=p["paragraph_id"],
                page_no=p["page_no"],
                text=p["text"],
                sentences=[Sentence(text=s["text"]) for s in p.get("sentences", [])],
            )
            for p in sec.get("paragraphs", [])
        ]
```

New:
```python
        paras = [
            Paragraph(
                paragraph_id=p["paragraph_id"],
                text=p["text"],
                sentences=[Sentence(text=s["text"]) for s in p.get("sentences", [])],
            )
            for p in sec.get("paragraphs", [])
        ]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_types_page_no_removed.py tests/test_parser/ -v`
Expected: All tests PASS (including markitdown_adapter and router tests from prior tasks).

- [ ] **Step 9: Commit**

```bash
git add app/core/types.py app/core/parser/ir_builder.py app/db/compare_repo.py
git add app/core/diff/diff_classifier.py app/agent/compare_graph.py
git add tests/test_types_page_no_removed.py
git commit -m "refactor: remove page_no from Paragraph, DiffItem, and all callsites"
```

---

## Task 5: Wire OCR Client Through Ingest Pipeline

**Files:**
- Modify: `app/ui/app_context.py`
- Modify: `app/agent/states.py`
- Modify: `app/agent/ingest_graph.py`
- Modify: `app/services/ingest_service.py`
- Modify: `app/ui/pages/library_page.py`
- Modify: `app/ui/pages/settings_page.py`

- [ ] **Step 1: Update app/ui/app_context.py**

Add `openai_client` and `openai_model` to `AppContext`. These are passed to the ingest graph so markitdown-ocr can use LLM-based OCR when configured.

Old:
```python
@dataclass
class AppContext:
    settings: AppSettings
    conn: sqlite3.Connection
    data_dir: str
    provider: BaseProvider | None = None
    embedder: BaseProvider | None = None
    lc_model: object | None = None  # BaseChatModel, typed as object to avoid hard dep
```

New:
```python
@dataclass
class AppContext:
    settings: AppSettings
    conn: sqlite3.Connection
    data_dir: str
    provider: BaseProvider | None = None
    embedder: BaseProvider | None = None
    lc_model: object | None = None  # BaseChatModel, typed as object to avoid hard dep
    openai_client: object | None = None  # openai.OpenAI, for markitdown-ocr
    openai_model: str = ""
```

- [ ] **Step 2: Update app/agent/states.py**

Add `llm_client` and `llm_model` to the `IngestState` inputs section so `parse_doc` can forward them to the router.

Old (lines 12–18):
```python
class IngestState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    file_path: str
    data_dir: str
    source_type: str           # "standard" | "uploaded"
    document_id: Optional[str] # set when adding new version to existing doc
    embedder: Any
    conn: Any                  # sqlite3.Connection, opened and closed by caller
```

New:
```python
class IngestState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    file_path: str
    data_dir: str
    source_type: str           # "standard" | "uploaded"
    document_id: Optional[str] # set when adding new version to existing doc
    embedder: Any
    conn: Any                  # sqlite3.Connection, opened and closed by caller
    llm_client: Any            # openai.OpenAI, for markitdown-ocr; None → OCR skipped
    llm_model: str
```

- [ ] **Step 3: Update app/agent/ingest_graph.py**

The `parse_doc` node must forward `llm_client` and `llm_model` from state, and drop the now-removed `quality.ocr_pages` log line.

Old `parse_doc` function (lines 52–61):
```python
def parse_doc(state: IngestState) -> dict:
    """Parse the file into a DocumentIR."""
    try:
        ir, quality = parse_document(state["file_path"])
        if quality.needs_ocr:
            logger.warning("Document needs OCR (Phase 2 feature): %s", quality.ocr_pages)
        return {"_ir": ir, "status": "parsed"}
    except Exception as e:
        logger.exception("parse_doc failed")
        return {"error": str(e), "status": "failed"}
```

New:
```python
def parse_doc(state: IngestState) -> dict:
    """Parse the file into a DocumentIR."""
    try:
        ir, quality = parse_document(
            state["file_path"],
            llm_client=state.get("llm_client"),
            llm_model=state.get("llm_model", ""),
        )
        if quality.needs_ocr:
            logger.warning("Low-quality document, OCR may be needed: %s", state["file_path"])
        return {"_ir": ir, "status": "parsed"}
    except Exception as e:
        logger.exception("parse_doc failed")
        return {"error": str(e), "status": "failed"}
```

- [ ] **Step 4: Update app/services/ingest_service.py**

Replace the `parse_mode: str = "standard"` parameter with `llm_client=None, llm_model: str = ""` in both `ingest_document` and `ingest_new_version`, and update the two `parse_document(...)` calls. Also remove the `quality.ocr_pages` log lines.

**`ingest_document` function signature** (line 20–28):

Old:
```python
def ingest_document(
    conn: sqlite3.Connection,
    data_dir: str,
    file_path: str,
    source_type: str = "standard",
    business_category: str = "",
    embedder: BaseProvider | None = None,
    parse_mode: str = "standard",
) -> tuple[str, str]:
```

New:
```python
def ingest_document(
    conn: sqlite3.Connection,
    data_dir: str,
    file_path: str,
    source_type: str = "standard",
    business_category: str = "",
    embedder: BaseProvider | None = None,
    llm_client=None,
    llm_model: str = "",
) -> tuple[str, str]:
```

**`parse_document` call in `ingest_document`** (line 61–63):

Old:
```python
    ir, quality = parse_document(str(path), mode=parse_mode)
    if quality.needs_ocr:
        logger.warning("Document has low-quality pages (OCR needed): %s", quality.ocr_pages)
```

New:
```python
    ir, quality = parse_document(str(path), llm_client=llm_client, llm_model=llm_model)
    if quality.needs_ocr:
        logger.warning("Low-quality document, OCR may be needed: %s", path)
```

**`ingest_new_version` function signature** (line 104–112):

Old:
```python
def ingest_new_version(
    conn: sqlite3.Connection,
    data_dir: str,
    file_path: str,
    document_id: str,
    version_label: str = "",
    embedder: BaseProvider | None = None,
    parse_mode: str = "standard",
) -> str:
```

New:
```python
def ingest_new_version(
    conn: sqlite3.Connection,
    data_dir: str,
    file_path: str,
    document_id: str,
    version_label: str = "",
    embedder: BaseProvider | None = None,
    llm_client=None,
    llm_model: str = "",
) -> str:
```

**`parse_document` call in `ingest_new_version`** (line 120–122):

Old:
```python
    ir, quality = parse_document(str(path), mode=parse_mode)
    if quality.needs_ocr:
        logger.warning("New version has low-quality pages: %s", quality.ocr_pages)
```

New:
```python
    ir, quality = parse_document(str(path), llm_client=llm_client, llm_model=llm_model)
    if quality.needs_ocr:
        logger.warning("Low-quality new version, OCR may be needed: %s", path)
```

- [ ] **Step 5: Update app/ui/pages/library_page.py**

Add `llm_client` and `llm_model` to the `ingest_graph.invoke({...})` call (lines 45–52).

Old:
```python
                result = ingest_graph.invoke({
                    "file_path": self.file_path,
                    "data_dir": self.ctx.data_dir,
                    "source_type": "standard",
                    "document_id": self.document_id,
                    "embedder": self.ctx.embedder,
                    "conn": conn,
                })
```

New:
```python
                result = ingest_graph.invoke({
                    "file_path": self.file_path,
                    "data_dir": self.ctx.data_dir,
                    "source_type": "standard",
                    "document_id": self.document_id,
                    "embedder": self.ctx.embedder,
                    "conn": conn,
                    "llm_client": self.ctx.openai_client,
                    "llm_model": self.ctx.openai_model,
                })
```

- [ ] **Step 6: Update app/ui/pages/settings_page.py**

In `_save()`, after assigning `ctx.provider` and `ctx.embedder` (around line 282), add the `openai_client` and `openai_model` assignments.

Old (lines 281–284):
```python
            self.ctx.provider = build_provider(provider)
            self.ctx.embedder = get_embedder(self.ctx.settings)
            self.provider_changed.emit()
```

New:
```python
            self.ctx.provider = build_provider(provider)
            self.ctx.embedder = get_embedder(self.ctx.settings)
            from openai import OpenAI
            self.ctx.openai_client = OpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url or None,
            )
            self.ctx.openai_model = provider.chat_model
            self.provider_changed.emit()
```

- [ ] **Step 7: Run the existing test suite**

Run: `pytest tests/ -v --ignore=tests/test_parser/test_markitdown_adapter.py -x`
Expected: All tests PASS (no regressions from the OCR wiring changes).

- [ ] **Step 8: Commit**

```bash
git add app/ui/app_context.py app/agent/states.py app/agent/ingest_graph.py
git add app/services/ingest_service.py app/ui/pages/library_page.py app/ui/pages/settings_page.py
git commit -m "feat: wire openai_client/llm_model through ingest pipeline for markitdown-ocr"
```

---

## Task 6: Delete Obsolete Parser Files and Final Cleanup

**Files:**
- Delete: `app/core/parser/docling_adapter.py`
- Delete: `app/core/parser/docx_extractor.py`
- Delete: `app/core/parser/pymupdf_extractor.py`
- Modify: `app/ui/pages/compare_page.py` (remove page number display)

- [ ] **Step 1: Remove page number display from compare_page.py**

Search `app/ui/pages/compare_page.py` for any references to `item.baseline_page`, `item.target_page`, or `p.{n}` page badge display. Remove those UI sections — diff location is now identified by `section_path` only.

Search pattern: `grep -n "baseline_page\|target_page\|page_no\|p\.\{" app/ui/pages/compare_page.py`

Remove any code that reads or displays page numbers from diff items.

- [ ] **Step 2: Delete the three obsolete parser files**

```bash
git rm app/core/parser/docling_adapter.py
git rm app/core/parser/docx_extractor.py
git rm app/core/parser/pymupdf_extractor.py
```

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS. No import errors from deleted files (router.py no longer imports them).

- [ ] **Step 4: Commit**

```bash
git add app/ui/pages/compare_page.py
git commit -m "chore: delete docling/docx/pymupdf parsers; remove page number display from compare_page"
```

---

## Self-Review Checklist

- **Spec coverage:** All 15 files from the spec's "Files Changed" table are addressed across the 6 tasks.
- **page_no removal:** `Paragraph.page_no`, `ParseQualityReport.ocr_pages`, `DiffItem.baseline_page`, `DiffItem.target_page` all removed. `Chunk.page_no` kept with default 0 (DB schema preserved).
- **evaluate_quality:** The `-0.05` page-number check (`if not has_page_numbers`) removed in Task 3's new router.py.
- **Router `mode` param:** Removed in Task 3.
- **OCR plumbing:** `AppContext → IngestState → parse_doc → markitdown_adapter.extract` fully wired in Task 5.
- **compare_repo:** Inserts `0, 0` for `baseline_page, target_page` — DB columns kept, DiffItem fields gone.
- **compare_graph `_load_ir`:** `page_no=p["page_no"]` removed; old JSON files on disk with that key are silently ignored.
- **Tests:** `test_markitdown_adapter.py` (new), `test_router.py` (rewritten), `test_types_page_no_removed.py` (new).
- **Type consistency:** `Paragraph(paragraph_id=..., text=...)` used in both `markitdown_adapter._parse_markdown` and `compare_graph._load_ir`. Matches the updated dataclass definition in Task 4.
