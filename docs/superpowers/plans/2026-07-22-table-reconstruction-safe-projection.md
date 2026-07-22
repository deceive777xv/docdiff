# Table Reconstruction Safe Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent one structurally unsafe cross-page table merge candidate from failing the whole document comparison while preserving every existing no-data-loss guard.

**Architecture:** Keep `build_reconstruction_operations` as the single source of truth for row-level projection safety. Add a deterministic pipeline preflight that incrementally tries accepted merge candidates, downgrades only the candidate that introduces a structural `ValueError`, records `unsafe_fragment_projection`, and then builds final operations from the validated decisions.

**Tech Stack:** Python 3.11+, dataclasses, pytest, existing `DocumentIR` and reconstruction trace models.

## Global Constraints

- Do not modify or copy source PDF/JSON files into the repository.
- Do not call an external Provider, embedding service, FAISS, or network API during reproduction or verification.
- Do not guess column positions, discard unmapped non-empty cells, or weaken `build_reconstruction_operations` validation.
- Do not change persisted `DocumentIR`, trace schema, Provider request format, LLM threshold, chunks, or indexes.
- Preserve deterministic `(side, candidate_id)` decision order and immutable source IR values.

---

### Task 1: Downgrade only the unsafe accepted candidate

**Files:**
- Modify: `tests/test_diff/test_table_reconstruction_pipeline.py:18-220`
- Modify: `app/core/diff/table_reconstruction_pipeline.py:303-415`

**Interfaces:**
- Consumes: `build_reconstruction_operations(analyses, boundary_rows, boundary_paragraph_ids) -> list[ReconstructionOperation]` and resolved `CandidateAssessment` values.
- Produces: `_validate_resolved_assessments(assessments, boundary_rows, boundary_paragraphs) -> list[CandidateAssessment]` and conflict code `unsafe_fragment_projection`.

- [ ] **Step 1: Write the failing pipeline regression test**

Add `build_reconstruction_operations` to the imports from `app.core.diff.table_reconstruction`, then add this test after the existing medium-confidence pipeline tests:

```python
def test_pipeline_downgrades_only_unsafe_llm_merge_after_projection_preflight(
    monkeypatch,
):
    safe = _candidate("safe-high", "high")
    unsafe = _candidate("unsafe-medium", "medium")
    unsafe_extra_row = _row(
        ("102", "next", "complete", "retained-extra"),
        1,
        "unsafe-medium-right",
    )
    unsafe = replace(
        unsafe,
        continuation_fragment_rows=(
            *unsafe.continuation_fragment_rows,
            unsafe_extra_row,
        ),
    )
    _stub_candidates(monkeypatch, [safe, unsafe])
    monkeypatch.setattr(
        pipeline,
        "build_reconstruction_operations",
        build_reconstruction_operations,
    )
    baseline, target, pairs = _documents()
    provider = QueueProvider([_response("unsafe-medium")])

    result = pipeline.reconstruct_table_pairs(
        pairs,
        baseline,
        target,
        provider,
    )

    decisions = {
        decision.candidate_id: decision for decision in result.trace.decisions
    }
    assert decisions["safe-high"].final_action == "merge"
    assert decisions["unsafe-medium"].final_action == "keep_separate"
    assert decisions["unsafe-medium"].llm is not None
    assert decisions["unsafe-medium"].llm.decision == "merge"
    assert "unsafe_fragment_projection" in decisions["unsafe-medium"].rule_conflicts
    assert {
        operation.decision_id
        for operation in result.trace.operations
        if operation.type == "merge_rows"
    } == {"safe-high"}
    assert len(provider.chat_calls) == 1
```

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```powershell
pytest tests/test_diff/test_table_reconstruction_pipeline.py::test_pipeline_downgrades_only_unsafe_llm_merge_after_projection_preflight -v
```

Expected: FAIL with `ValueError: fragment projection contains unmapped retained cells`, proving the test reaches the production failure path.

- [ ] **Step 3: Implement candidate-level incremental preflight**

Add this constant and helper near `_resolve_assessment` in `app/core/diff/table_reconstruction_pipeline.py`:

```python
_UNSAFE_FRAGMENT_PROJECTION = "unsafe_fragment_projection"


def _validate_resolved_assessments(
    assessments: Sequence[CandidateAssessment],
    boundary_rows: Mapping[str, set[SourceRowRef]],
    boundary_paragraphs: Mapping[str, set[str]],
) -> list[CandidateAssessment]:
    validated: list[CandidateAssessment] = []
    build_reconstruction_operations([], boundary_rows, boundary_paragraphs)
    for assessment in assessments:
        if assessment.final_action != "merge":
            validated.append(assessment)
            continue
        try:
            build_reconstruction_operations(
                [*validated, assessment],
                boundary_rows,
                boundary_paragraphs,
            )
        except ValueError:
            candidate = replace(
                assessment.candidate,
                conflicts=tuple(
                    dict.fromkeys(
                        (
                            *assessment.candidate.conflicts,
                            _UNSAFE_FRAGMENT_PROJECTION,
                        )
                    )
                ),
            )
            validated.append(
                replace(
                    assessment,
                    candidate=candidate,
                    final_action="keep_separate",
                )
            )
        else:
            validated.append(assessment)
    return validated
```

After the existing rule/LLM resolution loop in `reconstruct_table_pairs`, validate decisions before final operation construction:

```python
    resolved = _validate_resolved_assessments(
        resolved,
        boundary_rows,
        boundary_paragraphs,
    )
    operations = build_reconstruction_operations(
        resolved,
        boundary_rows,
        boundary_paragraphs,
    )
```

The initial empty build is intentional: a boundary-only failure remains global. Each subsequent build adds exactly one merge candidate, so a new structural `ValueError` is attributable to that candidate or its interaction with earlier accepted candidates. The final build remains outside the fallback path.

- [ ] **Step 4: Run the regression test and verify GREEN**

Run:

```powershell
pytest tests/test_diff/test_table_reconstruction_pipeline.py::test_pipeline_downgrades_only_unsafe_llm_merge_after_projection_preflight -v
```

Expected: PASS. The safe candidate emits `merge_rows`; the unsafe LLM-approved candidate is retained in the trace as `keep_separate` with `unsafe_fragment_projection`.

- [ ] **Step 5: Run focused reconstruction tests**

Run:

```powershell
pytest tests/test_diff/test_table_reconstruction.py tests/test_diff/test_table_reconstruction_pipeline.py tests/test_diff/test_table_reconstruction_llm.py tests/test_diff/test_reconstruction_trace.py -q
```

Expected: all tests pass, including the existing direct builder rejection test and shifted-full-row projection test.

- [ ] **Step 6: Commit the tested production fix**

```powershell
git add -- app/core/diff/table_reconstruction_pipeline.py tests/test_diff/test_table_reconstruction_pipeline.py
git commit -m "fix: downgrade unsafe table projections"
```

---

### Task 2: Verify private local fixtures and complete regression suite

**Files:**
- Read only: `E:\Project\test\*.json`
- Verify: `app/core/diff/table_reconstruction_pipeline.py`
- Verify: `tests/test_diff/test_table_reconstruction_pipeline.py`

**Interfaces:**
- Consumes: public `align_sections` and `reconstruct_table_pairs` functions plus a local in-process `BaseProvider` fake.
- Produces: no repository artifact; only aggregate counts and exit status are printed.

- [ ] **Step 1: Run the natural-pair local acceptance check without network access**

Run this complete read-only script from the repository root:

```powershell
@'
import json
from pathlib import Path

from app.core.diff.structure_aligner import align_sections
from app.core.diff.table_reconstruction_pipeline import reconstruct_table_pairs
from app.core.model.base_provider import BaseProvider
from app.core.types import DocumentIR, Paragraph, Section, Sentence


class LocalAlwaysMergeProvider(BaseProvider):
    chat_model = "local-always-merge"

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        return json.dumps(
            {
                "candidate_id": payload["candidate_id"],
                "decision": "merge",
                "confidence": 0.99,
                "reason": "local deterministic test response",
            }
        )

    def embed(self, texts):
        raise AssertionError("embedding must not be called")

    def health_check(self):
        return True


def load_ir(path: Path) -> DocumentIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    return DocumentIR(
        data["doc_id"],
        data["title"],
        data["file_hash"],
        [
            Section(
                section["section_id"],
                section["title"],
                section["level"],
                [
                    Paragraph(
                        paragraph["paragraph_id"],
                        paragraph["text"],
                        [
                            Sentence(sentence["text"])
                            for sentence in paragraph.get("sentences", [])
                        ],
                    )
                    for paragraph in section.get("paragraphs", [])
                ],
            )
            for section in data.get("sections", [])
        ],
        data.get("plain_text", ""),
    )


root = Path(r"E:\Project\test")
baseline = load_ir(root / "升降器设计校核表-20251212-v1.json")
target = load_ir(root / "升降器设计校核表-20251213-v1.json")
provider = LocalAlwaysMergeProvider()
result = reconstruct_table_pairs(
    align_sections(baseline, target),
    baseline,
    target,
    provider,
)
projection_rejections = [
    decision
    for decision in result.trace.decisions
    if decision.final_action == "keep_separate"
    and "unsafe_fragment_projection" in decision.rule_conflicts
]
assert provider.calls >= 1
assert projection_rejections
assert all(decision.llm is not None for decision in projection_rejections)
print(
    json.dumps(
        {
            "llm_calls": provider.calls,
            "decisions": len(result.trace.decisions),
            "projection_rejections": len(projection_rejections),
            "operations": len(result.trace.operations),
        },
        ensure_ascii=False,
    )
)
'@ | python -
```

Expected: exit code 0, at least one projection rejection, and no HTTP/network log entries. Do not print row text or copy the JSON files.

- [ ] **Step 2: Run the full automated test suite**

Run:

```powershell
pytest -q
```

Expected: all tests pass; only pre-existing environment warnings, if any, may remain.

- [ ] **Step 3: Verify repository hygiene and the exact diff**

Run:

```powershell
git diff --check HEAD^ -- app/core/diff/table_reconstruction_pipeline.py tests/test_diff/test_table_reconstruction_pipeline.py
git status --short
git show --stat --oneline HEAD
```

Expected: no whitespace errors; only the user-owned `.codex-remote-attachments/` remains untracked; the implementation commit touches exactly the pipeline and its test.
