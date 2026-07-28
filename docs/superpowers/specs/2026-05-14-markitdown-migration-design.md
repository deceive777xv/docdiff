# Markitdown Migration Design

**Goal:** Replace docling (and all other parsers) with markitdown as the sole document parsing backend, remove page number logic throughout, and wire markitdown-ocr through the ingest pipeline.

**Architecture:** A single `markitdown_adapter.py` replaces three parser files (docling_adapter, docx_extractor, pymupdf_extractor). The router becomes a thin format-guard + dispatcher. Page number fields are removed from types, ir_builder, chunk_repo, compare_graph, and UI. An `openai.OpenAI` client flows from AppContext through the ingest graph to enable LLM-based OCR via markitdown-ocr.

**Tech Stack:** `markitdown[all]`, `markitdown-ocr`, `openai` SDK (already a dependency).

---

## 1. Files Changed

| Operation | File |
|---|---|
| Create | `app/core/parser/markitdown_adapter.py` |
| Delete | `app/core/parser/docling_adapter.py` |
| Delete | `app/core/parser/docx_extractor.py` |
| Delete | `app/core/parser/pymupdf_extractor.py` |
| Modify | `app/core/parser/router.py` |
| Modify | `app/core/types.py` |
| Modify | `app/core/ir_builder.py` |
| Modify | `app/db/chunk_repo.py` |
| Modify | `app/agent/ingest_graph.py` |
| Modify | `app/core/ingest_service.py` |
| Modify | `app/ui/app_context.py` |
| Modify | `app/agent/compare_graph.py` |
| Modify | `app/ui/pages/compare_page.py` |
| Modify | `assets/diff_template.html` |
| Modify | `requirements.txt` + `pyproject.toml` |

---

## 2. markitdown_adapter.py

### Public API

```python
def is_available() -> bool: ...

def extract(
    file_path: str,
    llm_client=None,   # openai.OpenAI instance, optional
    llm_model: str = "",
) -> DocumentIR: ...   # quality evaluation done in router, avoids circular import
```

### Internals

```python
from markitdown import MarkItDown

def extract(file_path, llm_client=None, llm_model=""):
    if not is_available():
        raise RuntimeError("markitdown not installed")

    md = MarkItDown(
        enable_plugins=bool(llm_client),
        llm_client=llm_client or None,
        llm_model=llm_model or None,
    )
    result = md.convert(file_path)
    return _parse_markdown(result.text_content, Path(file_path).stem, file_path)
```

### Markdown → DocumentIR Parsing

Parse `result.text_content` line by line:

- Lines matching `^(#{1,3})\s+(.+)` → new `Section(level=1/2/3, title=...)`
- Blank lines flush the current paragraph buffer
- Non-empty, non-heading lines accumulate in the buffer; on flush → `Paragraph(text=joined)`
- If content precedes the first heading, a default section "正文" (level 1) is auto-inserted

No `page_no` is set anywhere; `Paragraph` no longer has this field.

---

## 3. router.py

```python
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls",
    ".html", ".htm", ".csv", ".json", ".xml", ".epub",
}

def parse_document(
    file_path: str,
    llm_client=None,
    llm_model: str = "",
) -> tuple[DocumentIR, ParseQualityReport]:
    if Path(file_path).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {Path(file_path).suffix!r}")
    ir = markitdown_adapter.extract(file_path, llm_client, llm_model)
    report = evaluate_quality(ir)
    return ir, report
```

- `mode` parameter removed entirely.
- `evaluate_quality` updated: remove the `-0.05` page-number check and its warning.

---

## 4. Data Model Changes (types.py)

### Paragraph — remove page_no

```python
@dataclass
class Paragraph:
    paragraph_id: str
    text: str
    sentences: list[Sentence] = field(default_factory=list)
```

### ParseQualityReport — remove ocr_pages

```python
@dataclass
class ParseQualityReport:
    quality_score: float
    needs_ocr: bool
    warnings: list[str] = field(default_factory=list)
```

### DiffItem — remove baseline_page / target_page

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

---

## 5. ir_builder.py and chunk_repo.py

`ir_builder.build_chunks()` no longer reads `paragraph.page_no` when constructing `Chunk`. The `Chunk` dataclass retains the `page_no: int = 0` field (DB schema preserved, value always 0) to avoid a schema migration. `chunk_repo` writes `0` for `page_no` on insert and ignores the column on read.

---

## 6. OCR Client Plumbing

### app/ui/app_context.py

```python
@dataclass
class AppContext:
    ...
    openai_client: Any | None = None   # openai.OpenAI, for markitdown-ocr
    openai_model: str = ""
```

Populated when a provider is configured — construct directly from active `ProviderConfig`:

```python
from openai import OpenAI
cfg = settings.get_active_provider()   # returns ProviderConfig
ctx.openai_client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None)
ctx.openai_model  = cfg.chat_model
```

This avoids accessing private internals of `OpenAICompatibleProvider`.

### app/agent/ingest_graph.py

```python
class IngestState(TypedDict):
    ...
    llm_client: Any | None
    llm_model: str
```

`parse_doc` node passes `state["llm_client"]` and `state["llm_model"]` to `router.parse_document()`.

### Callers (library_page, ingest_service)

```python
ingest_graph.invoke({
    ...,
    "llm_client": ctx.openai_client,  # None → OCR silently skipped
    "llm_model":  ctx.openai_model,
})
```

When `llm_client` is `None`, `MarkItDown(enable_plugins=False)` — OCR plugin never loads, no error.

---

## 7. compare_graph.py / compare_page.py / diff_template.html

- `compare_graph.py`: remove `baseline_page` and `target_page` from all `DiffItem(...)` constructor calls.
- `compare_page.py`: remove any UI code that reads or displays `item.baseline_page` / `item.target_page`. Diff location is identified by `section_path` only.
- `diff_template.html`: remove page number display (`p.{n}` badges) from both baseline and target pane headers.

---

## 8. Dependencies

### Remove
- `docling>=2.31.0`
- `pymupdf` (if a direct dependency; markitdown[pdf] includes pdfminer)
- `python-docx` (if a direct dependency; markitdown[docx] includes it)

### Add
- `markitdown[all]`
- `markitdown-ocr`

Apply to both `requirements.txt` and `pyproject.toml`.

---

## 9. Tests

- `tests/test_parser/test_router.py`: update to remove docling monkeypatching, add tests for new supported extensions, add test for unsupported extension raising ValueError, add test that `llm_client=None` path works without error.
- `tests/test_parser/test_markitdown_adapter.py` (new): test Markdown → DocumentIR parsing with headings, headingless content (default section), and multi-level headings.

---

## 10. Error Handling

- `markitdown` raises on unreadable/corrupt files → propagated as-is; existing ingest graph error node handles it.
- Unsupported extension → `ValueError` from router (same behaviour as before).
- OCR failure (LLM error) → markitdown-ocr logs and falls back to non-OCR extraction; does not raise.
