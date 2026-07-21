# Cross-Page Table Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct table rows and compatible table fragments split at page boundaries before semantic comparison, use constrained LLM adjudication for medium-confidence candidates, and replay the exact same deterministic operations in the comparison UI without modifying source JSON or FAISS indexes.

**Architecture:** Add a read-only reconstruction layer between `align_sections` and `match_paragraphs`. Generic structural rules infer logical columns, repeated boundary artifacts, continuation candidates, and hard vetoes from the two parsed `DocumentIR` values. High-confidence candidates are accepted by rules, medium-confidence candidates receive one bounded provider call each, and all accepted transformations are encoded as ordered operations in a versioned sidecar. The graph, synchronous service, and UI share the same reconstruction and replay code; only semantic comparison embeds the reconstructed in-memory text, while retrieval chunks and FAISS remain untouched.

**Tech Stack:** Python 3.11+, dataclasses, standard-library JSON/hash/path utilities, existing `BaseProvider`, LangGraph, PySide6, pytest, pytest-qt.

## Global Constraints

- Use only the existing parsed JSON/`DocumentIR`; do not require the PDF, page numbers, coordinates, or parser changes.
- Do not mutate the loaded source `DocumentIR`, parsed JSON, chunks, FAISS index, database schema, settings model, or settings UI.
- Do not hardcode page-header text, business column names, sample row numbers, page numbers, or fixed physical column indexes.
- Infer logical columns with order-preserving mappings from occupancy, value type, text length, repetition, adjacency, and cross-version evidence.
- Treat repetition as supporting evidence only; it cannot by itself classify a row as boundary noise.
- Generate continuation candidates only at adjacent logical fragment boundaries and apply every hard veto before confidence scoring or LLM invocation.
- Auto-merge high-confidence candidates with at least four support signals, send medium-confidence candidates with two or three signals to the LLM, and keep low-confidence candidates separate.
- The LLM may only return `merge` or `keep_separate` for one fixed candidate; it cannot rewrite text, select rows, alter mappings, or override a hard veto.
- Accept an LLM merge only at confidence `>= 0.75`; invalid JSON, mismatched IDs, provider errors, timeouts, rejections, and lower confidence all keep the candidate separate without failing the task.
- Apply accepted transformations with deterministic cell-by-cell code and make replay idempotent.
- Preserve the existing diff-result JSON list format. Store reconstruction decisions and operations in `exports/<task_id>.reconstruction.json` with schema version 1 and algorithm version `cross-page-table-v1`.
- Stage both export files before replacing either final path, and mark the task `completed` only after both replacements succeed. A sidecar persistence error fails the task.
- The UI validates schema version, document IDs, and file hashes, then replays operations. It never reruns rules or calls the LLM.
- A missing, corrupt, unsupported, or mismatched sidecar falls back to raw IR rendering with a warning; no historical migration or historical-task test matrix is required.
- Use the external JSON files only for local acceptance. Commit a sanitized, minimal fixture without customer document text.
- Add no runtime dependency.

---

### Task 1: Versioned reconstruction trace and atomic artifact I/O

**Files:**
- Create: `app/core/diff/reconstruction_trace.py`
- Create: `tests/test_diff/test_reconstruction_trace.py`

**Interfaces:**
- `SCHEMA_VERSION = 1`
- `ALGORITHM_VERSION = "cross-page-table-v1"`
- `@dataclass(frozen=True) DocumentTraceRef(doc_id: str, file_hash: str)`
- `@dataclass(frozen=True) SourceRowRef(section_id: str, paragraph_id: str, sentence_index: int)`
- `LLMJudgment(model: str, decision: Literal["merge", "keep_separate"], confidence: float, reason: str)`
- `ReconstructionDecision(candidate_id: str, side: Literal["baseline", "target"], source_rows: list[SourceRowRef], column_mapping: dict[int, int], rule_confidence: Literal["high", "medium", "low"], rule_evidence: list[str], rule_conflicts: list[str], llm: LLMJudgment | None, final_action: Literal["merge", "keep_separate"], generated_row_id: str = "")`
- `ReconstructionOperation(operation_id: str, side: Literal["baseline", "target"], type: Literal["project_columns", "drop_boundary_rows", "drop_boundary_paragraphs", "merge_rows", "merge_fragments"], source_rows: list[SourceRowRef] = field(default_factory=list), source_paragraph_ids: list[str] = field(default_factory=list), column_mapping: dict[int, int] = field(default_factory=dict), decision_id: str = "", generated_row_id: str = "", generated_paragraph_id: str = "")`
- `ReconstructionTrace(schema_version: int, algorithm_version: str, baseline: DocumentTraceRef, target: DocumentTraceRef, decisions: list[ReconstructionDecision], operations: list[ReconstructionOperation])`
- `reconstruction_trace_path(data_dir: str | Path, task_id: str) -> Path`
- `trace_to_dict(trace: ReconstructionTrace) -> dict[str, object]`
- `trace_from_dict(data: dict[str, object]) -> ReconstructionTrace`
- `validate_trace_documents(trace: ReconstructionTrace, baseline_ir: DocumentIR, target_ir: DocumentIR) -> None`
- `load_reconstruction_trace(path: Path) -> ReconstructionTrace`
- `write_json_atomic(path: Path, payload: object) -> None`
- `persist_compare_artifacts(data_dir: str | Path, task_id: str, items: list[DiffItem], trace: ReconstructionTrace) -> tuple[Path, Path]`

- [ ] **Step 1: Write failing trace round-trip and validation tests**

Create tests that construct one decision and all five operation types, then assert:

```python
def test_trace_round_trip_preserves_typed_mappings_and_llm_judgment(tmp_path):
    trace = make_trace_with_every_operation()

    path = tmp_path / "task.reconstruction.json"
    write_json_atomic(path, trace_to_dict(trace))
    restored = load_reconstruction_trace(path)

    assert restored == trace
    assert restored.decisions[0].column_mapping == {1: 0, 3: 1}
    assert restored.algorithm_version == "cross-page-table-v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("algorithm_version", "unknown-version"),
    ],
)
def test_trace_rejects_unsupported_versions(field, value):
    payload = trace_to_dict(make_trace_with_every_operation())
    payload[field] = value

    with pytest.raises(ValueError, match="Unsupported reconstruction"):
        trace_from_dict(payload)


def test_trace_rejects_document_identity_mismatch():
    trace = make_trace_with_every_operation()
    baseline_ir, target_ir = make_document_pair()
    target_ir.file_hash = "different-hash"

    with pytest.raises(ValueError, match="target file hash"):
        validate_trace_documents(trace, baseline_ir, target_ir)
```

- [ ] **Step 2: Write failing artifact-staging tests**

Patch `Path.replace` and verify no successful completion can be reported by the caller when either artifact fails. Also assert the existing result file remains a JSON list and the sidecar uses the required filename.

```python
def test_persist_compare_artifacts_writes_existing_result_shape_and_sidecar(tmp_path):
    result_path, trace_path = persist_compare_artifacts(
        tmp_path,
        "task-1",
        [make_diff_item()],
        make_trace_with_every_operation(),
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == [asdict(make_diff_item())]
    assert trace_path.name == "task-1.reconstruction.json"
    assert load_reconstruction_trace(trace_path) == make_trace_with_every_operation()


def test_persist_compare_artifacts_stages_both_files_before_first_replace(tmp_path, monkeypatch):
    observed_temp_files: list[tuple[bool, bool]] = []
    original_replace = Path.replace

    def recording_replace(path: Path, target: Path):
        exports = tmp_path / "exports"
        observed_temp_files.append(
            (
                any(exports.glob("task-1.json.*.tmp")),
                any(exports.glob("task-1.reconstruction.json.*.tmp")),
            )
        )
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    persist_compare_artifacts(tmp_path, "task-1", [make_diff_item()], make_trace_with_every_operation())

    assert observed_temp_files[0] == (True, True)
```

- [ ] **Step 3: Run the focused tests and verify RED**

```powershell
pytest tests/test_diff/test_reconstruction_trace.py -v
```

Expected: collection fails because `app.core.diff.reconstruction_trace` does not exist.

- [ ] **Step 4: Implement strict dataclasses and serialization**

Use dataclasses with explicit defaults created by `field(default_factory=list)` and manually validate every enum-like field, required string, confidence range, non-negative sentence index, integer mapping key/value, schema version, and algorithm version in `trace_from_dict`. JSON object keys for `column_mapping` must convert back to integers.

`validate_trace_documents` must compare both `doc_id` and `file_hash` and raise a side-specific `ValueError` before any operation is applied.

- [ ] **Step 5: Implement staged atomic writes**

`write_json_atomic` writes UTF-8 JSON to a unique sibling temp file, flushes and calls `os.fsync`, then uses `Path.replace`. It removes its own temp file in `finally` if replacement did not consume it.

`persist_compare_artifacts` must:

1. Create `exports`.
2. Serialize the diff item list and trace before filesystem mutation.
3. Write and fsync two unique temp files.
4. Replace `task_id.reconstruction.json`, then `task_id.json`.
5. Clean remaining temp files in `finally`.
6. Return the two final paths only after both replacements succeed.

The task-status update remains outside this helper so graph and service cannot mark completion early.

- [ ] **Step 6: Run the focused tests and verify GREEN**

```powershell
pytest tests/test_diff/test_reconstruction_trace.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add app/core/diff/reconstruction_trace.py tests/test_diff/test_reconstruction_trace.py
git commit -m "feat: add reconstruction trace contract"
```

---

### Task 2: Generic table-row parsing, fragment regions, and logical-column inference

**Files:**
- Create: `app/core/diff/table_reconstruction.py`
- Create: `tests/test_diff/test_table_reconstruction.py`

**Interfaces:**
- `RowKind = Literal["content", "separator", "empty"]`
- `TableRowMatrix(source: SourceRowRef, raw_text: str, raw_cells: tuple[str, ...], normalized_cells: tuple[str, ...], occupied: tuple[bool, ...], value_types: tuple[str, ...], kind: RowKind)`
- `ColumnProfile(physical_index: int, non_empty_ratio: float, type_ratios: dict[str, float], median_length: float, repetition_ratio: float)`
- `TableRegion(rows: tuple[TableRowMatrix, ...], start_index: int, end_index: int, role: Literal["body", "header", "boundary", "unknown"])`
- `TableFragment(section_id: str, paragraph_id: str, paragraph_index: int, rows: tuple[TableRowMatrix, ...], regions: tuple[TableRegion, ...], body_region_indexes: tuple[int, ...], active_columns: tuple[int, ...])`
- `ColumnMapping(source_columns: tuple[int, ...], logical_by_physical: dict[int, int], score: float)`
- `split_markdown_table_row(text: str, source: SourceRowRef) -> TableRowMatrix | None`
- `collect_table_fragments(section: Section) -> list[TableFragment]`
- `infer_regions(fragment: TableFragment, peer_fragments: Sequence[TableFragment]) -> tuple[TableRegion, ...]`
- `infer_active_columns(fragment: TableFragment, peer_fragments: Sequence[TableFragment]) -> tuple[int, ...]`
- `infer_monotonic_column_mapping(left: TableFragment, right: TableFragment, cross_version_fragments: Sequence[TableFragment]) -> ColumnMapping | None`
- `classify_repeated_boundary_regions(fragments: Sequence[TableFragment]) -> set[SourceRowRef]`

- [ ] **Step 1: Write failing row-parser tests that preserve empty cells**

```python
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("| unit-cobalt | drive | cedar pre |", ("unit-cobalt", "drive", "cedar pre")),
        ("| | | lude-complete | |", ("", "", "lude-complete", "")),
        ("|---|:---:|---|", ("---", ":---:", "---")),
    ],
)
def test_split_markdown_table_row_preserves_internal_and_trailing_empty_cells(text, expected):
    row = split_markdown_table_row(text, make_source_ref())

    assert row is not None
    assert row.raw_cells == expected


def test_split_markdown_table_row_rejects_plain_paragraph():
    assert split_markdown_table_row("ordinary paragraph", make_source_ref()) is None
```

Normalize only for analysis: trim outer whitespace, remove Markdown emphasis wrappers, convert `<br>` variants to a single space, and collapse whitespace. Keep `raw_cells` unchanged except outer cell whitespace trimming.

- [ ] **Step 2: Write failing region and boundary-classification tests**

Create synthetic fragments where a repeated five-column boundary region surrounds a stable three-column body, plus a genuine repeated three-column data row. Assert the boundary region is dropped only when position, schema incompatibility, key discontinuity, and abnormal cell repetition jointly support it.

```python
def test_repeated_wide_boundary_is_detected_without_matching_fixed_text():
    fragments = make_fragments_with_repeated_wide_boundaries(
        boundary_tokens=("alpha", "beta", "alpha", "gamma", "alpha")
    )

    dropped = classify_repeated_boundary_regions(fragments)

    assert dropped == boundary_source_refs(fragments)


def test_repetition_alone_does_not_drop_a_real_data_row():
    fragments = make_fragments_with_repeated_body_row()

    dropped = classify_repeated_boundary_regions(fragments)

    assert repeated_body_source_ref(fragments) not in dropped
```

- [ ] **Step 3: Write failing monotonic-column mapping tests**

The test data must place the same three logical body columns at different physical indexes and widths. Do not use the customer sample's physical indexes.

```python
def test_infer_mapping_projects_different_physical_widths_in_order():
    left = make_fragment(body_columns=(0, 2, 4), width=5)
    right = make_fragment(body_columns=(1, 4, 6), width=7)

    mapping = infer_monotonic_column_mapping(left, right, ())

    assert mapping is not None
    assert mapping.logical_by_physical == {1: 0, 4: 1, 6: 2}
    assert list(mapping.logical_by_physical.values()) == sorted(mapping.logical_by_physical.values())


def test_infer_mapping_rejects_incompatible_body_schemas():
    left = make_numeric_keyed_fragment()
    right = make_unkeyed_matrix_fragment()

    assert infer_monotonic_column_mapping(left, right, ()) is None
```

- [ ] **Step 4: Run the focused tests and verify RED**

```powershell
pytest tests/test_diff/test_table_reconstruction.py -k "split_markdown or boundary or mapping" -v
```

Expected: collection fails because the table reconstruction module does not exist.

- [ ] **Step 5: Implement feature extraction and region segmentation**

Implement value types using generic shape categories: `empty`, `integer`, `decimal`, `hierarchical_number`, `short_text`, `long_text`, `placeholder`, and `separator`. Region breaks use weighted changes in row width, occupancy-vector similarity, separator presence, value-type distribution, and the start/end of a stable multi-row pattern.

Use deterministic numeric constants in named module-level configuration dataclasses, not sample text or column indexes. Tests should construct a custom configuration when a threshold boundary must be exercised.

- [ ] **Step 6: Implement active-column and monotonic-mapping inference**

Score columns from non-empty ratio, stable value-type role, median length, repetition ratio, adjacency co-occurrence, and cross-version role similarity. Find the maximum-scoring order-preserving alignment with dynamic programming. Return `None` when the normalized score is below the named compatibility threshold or when key-role conflicts remain.

Boundary classification must require at least three independent signal families, including schema incompatibility and boundary position. Preserve stable body rows even if their text repeats.

- [ ] **Step 7: Run all Task 2 tests and verify GREEN**

```powershell
pytest tests/test_diff/test_table_reconstruction.py -k "split_markdown or region or boundary or active_columns or mapping" -v
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 2**

```powershell
git add app/core/diff/table_reconstruction.py tests/test_diff/test_table_reconstruction.py
git commit -m "feat: infer logical table structure"
```

---

### Task 3: Boundary candidates, hard vetoes, and rule confidence

**Files:**
- Modify: `app/core/diff/table_reconstruction.py`
- Modify: `tests/test_diff/test_table_reconstruction.py`

**Interfaces:**
- `EvidenceCode = Literal["blank_key_cells", "next_row_restores_key_pattern", "complementary_content_cells", "textual_continuity", "boundary_artifacts_only", "cross_version_support"]`
- `VetoCode = Literal["new_key_value", "header_or_separator", "incompatible_schema", "new_section_or_table", "crosses_real_body_row", "conflicting_key_cells"]`
- `ContinuationCandidate(candidate_id: str, side: Literal["baseline", "target"], previous_row: TableRowMatrix, continuation_row: TableRowMatrix, next_full_row: TableRowMatrix | None, mapping: ColumnMapping, evidence: tuple[EvidenceCode, ...], conflicts: tuple[str, ...], vetoes: tuple[VetoCode, ...], cross_version_rows: tuple[TableRowMatrix, ...])`
- `CandidateAssessment(candidate: ContinuationCandidate, rule_confidence: Literal["high", "medium", "low"], final_action: Literal["merge", "keep_separate", "needs_llm"] )`
- `generate_continuation_candidates(left: TableFragment, right: TableFragment, mapping: ColumnMapping, boundary_rows: set[SourceRowRef], cross_version_fragments: Sequence[TableFragment], side: Literal["baseline", "target"]) -> list[ContinuationCandidate]`
- `assess_candidate(candidate: ContinuationCandidate) -> CandidateAssessment`

- [ ] **Step 1: Write failing candidate-locality and hard-veto tests**

Cover numbered and unnumbered body tables. Assert a candidate can only use the final body row of the left fragment and first body row of the right fragment after confirmed boundary rows are excluded.

```python
def test_candidate_is_limited_to_adjacent_fragment_boundary():
    left, right, mapping = make_adjacent_fragments_with_inner_sparse_row()

    candidates = generate_continuation_candidates(
        left, right, mapping, set(), (), "baseline"
    )

    assert [candidate.continuation_row.source for candidate in candidates] == [
        first_body_source(right)
    ]


@pytest.mark.parametrize(
    "case",
    [
        "new_key_value",
        "header_or_separator",
        "incompatible_schema",
        "new_section_or_table",
        "crosses_real_body_row",
        "conflicting_key_cells",
    ],
)
def test_hard_veto_never_requests_llm(case):
    candidate = make_candidate_with_veto(case)

    assessment = assess_candidate(candidate)

    assert assessment.final_action == "keep_separate"
    assert assessment.rule_confidence == "low"
```

- [ ] **Step 2: Write failing evidence and confidence-tier tests**

```python
@pytest.mark.parametrize(
    ("evidence_count", "expected_confidence", "expected_action"),
    [
        (4, "high", "merge"),
        (5, "high", "merge"),
        (6, "high", "merge"),
        (2, "medium", "needs_llm"),
        (3, "medium", "needs_llm"),
        (0, "low", "keep_separate"),
        (1, "low", "keep_separate"),
    ],
)
def test_assess_candidate_uses_approved_confidence_tiers(
    evidence_count, expected_confidence, expected_action
):
    candidate = make_candidate_with_evidence_count(evidence_count)

    assessment = assess_candidate(candidate)

    assert assessment.rule_confidence == expected_confidence
    assert assessment.final_action == expected_action
```

Add focused tests for all six evidence signals. Text continuity alone must remain low confidence. Cross-version support must compare local logical cells after inferred mapping, not physical column indexes.

- [ ] **Step 3: Run the Task 3 tests and verify RED**

```powershell
pytest tests/test_diff/test_table_reconstruction.py -k "candidate or veto or confidence or evidence" -v
```

Expected: tests fail because candidate generation and assessment are not implemented.

- [ ] **Step 4: Implement local candidate generation and stable IDs**

Derive `candidate_id` by hashing algorithm version, side, both source row refs, and the sorted logical mapping. Never include mutable normalized text in the ID. Collect the next complete row only as context; do not cross it for a candidate.

Short ordinary paragraphs may be skipped only when a separate repeated-boundary-paragraph classifier has confirmed same-section repetition and stable boundary position. Record those paragraph IDs later as `drop_boundary_paragraphs` operations.

- [ ] **Step 5: Implement veto-first assessment and six evidence functions**

Evaluate hard vetoes before counting evidence. `assess_candidate` is pure and must not accept a provider. Use exact approved thresholds: at least four signals is high, two or three is medium, zero or one is low. Any hard veto forces low/keep-separate.

- [ ] **Step 6: Run all Task 3 tests and verify GREEN**

```powershell
pytest tests/test_diff/test_table_reconstruction.py -k "candidate or veto or confidence or evidence" -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 3**

```powershell
git add app/core/diff/table_reconstruction.py tests/test_diff/test_table_reconstruction.py
git commit -m "feat: score table continuation candidates"
```

---

### Task 4: Deterministic operations, row merging, and idempotent replay

**Files:**
- Modify: `app/core/diff/table_reconstruction.py`
- Create: `tests/fixtures/cross_page_table_pair.json`
- Modify: `tests/test_diff/test_table_reconstruction.py`

**Interfaces:**
- `merge_logical_rows(previous: TableRowMatrix, continuation: TableRowMatrix, mapping: ColumnMapping, key_logical_columns: frozenset[int]) -> tuple[str, ...]`
- `build_reconstruction_operations(analyses: Sequence[CandidateAssessment], boundary_rows: Mapping[str, set[SourceRowRef]], boundary_paragraph_ids: Mapping[str, set[str]]) -> list[ReconstructionOperation]`
- `apply_reconstruction_operations(baseline_ir: DocumentIR, target_ir: DocumentIR, operations: Sequence[ReconstructionOperation]) -> tuple[DocumentIR, DocumentIR]`
- `derived_row_id(operation: ReconstructionOperation) -> str`
- `derived_paragraph_id(operation: ReconstructionOperation) -> str`

- [ ] **Step 1: Add the sanitized fixture**

Create a two-document fixture containing only invented neutral text with these structural cases:

- an invented baseline row split as `cedar pre` and `lude-complete` around a repeated wider boundary table;
- target row `14` split at a different fragment boundary;
- an unnumbered table continuation;
- a real duplicated body row that must remain;
- a new table after a boundary that must not merge;
- a final incomplete row with no following fragment;
- one real numeric change and one target-only row.

The fixture contains `baseline` and `target` objects in the existing parsed JSON shape. Its physical body columns differ between fragments so fixed-column implementations fail.

- [ ] **Step 2: Write failing cell-merge tests**

```python
def test_merge_logical_rows_preserves_raw_text_and_joins_content_only():
    previous, continuation, mapping = make_split_rows("cedar pre", "lude-complete")

    cells = merge_logical_rows(previous, continuation, mapping, frozenset({0}))

    assert cells[0] == "12"
    assert cells[2] == "cedar pre<br>lude-complete"


def test_merge_logical_rows_rejects_conflicting_non_empty_key_cells():
    previous, continuation, mapping = make_conflicting_key_rows()

    with pytest.raises(ValueError, match="key column conflict"):
        merge_logical_rows(previous, continuation, mapping, frozenset({0}))
```

For two non-empty content cells, insert exactly one `<br>` unless the left cell already ends or the right cell already begins with an explicit line-break marker. Never add semantic text or punctuation.

- [ ] **Step 3: Write failing operation and replay tests**

```python
def test_replay_projects_drops_merges_and_consolidates_without_mutating_sources():
    baseline_ir, target_ir = load_sanitized_fixture()
    baseline_before = deepcopy(baseline_ir)
    target_before = deepcopy(target_ir)
    operations = make_fixture_operations()

    normalized_baseline, normalized_target = apply_reconstruction_operations(
        baseline_ir, target_ir, operations
    )

    assert baseline_ir == baseline_before
    assert target_ir == target_before
    assert "cedar pre<br>lude-complete" in normalized_baseline.plain_text
    assert repeated_boundary_token() not in normalized_baseline.plain_text
    assert real_repeated_body_text() in normalized_baseline.plain_text
    assert target_only_row_text() in normalized_target.plain_text


def test_replay_is_idempotent_for_same_operations():
    baseline_ir, target_ir = load_sanitized_fixture()
    operations = make_fixture_operations()
    first = apply_reconstruction_operations(baseline_ir, target_ir, operations)
    second = apply_reconstruction_operations(first[0], first[1], operations)

    assert second == first
```

- [ ] **Step 4: Run the Task 4 tests and verify RED**

```powershell
pytest tests/test_diff/test_table_reconstruction.py -k "merge_logical or replay or sanitized" -v
```

Expected: tests fail because operation building and replay are not implemented.

- [ ] **Step 5: Implement deterministic operation ordering and replay**

Order operations per side by source section/paragraph/sentence position, with transformation precedence:

1. `project_columns`
2. `drop_boundary_rows`
3. `drop_boundary_paragraphs`
4. `merge_rows`
5. `merge_fragments`

Replay begins from `deepcopy` of both documents. Resolve every source reference against original and already-derived IDs through a replay index. Missing references caused by a prior replay are accepted only when the exact derived ID and source provenance already exist; otherwise raise `ValueError` and abandon the whole trace in the UI caller.

Rebuild each normalized paragraph's `text`, `sentences`, each document's `plain_text`, and stable derived IDs from source references plus operation type. Preserve untouched paragraphs byte-for-byte.

- [ ] **Step 6: Run all Task 4 tests and verify GREEN**

```powershell
pytest tests/test_diff/test_table_reconstruction.py -k "merge_logical or operation or replay or sanitized" -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 4**

```powershell
git add app/core/diff/table_reconstruction.py tests/test_diff/test_table_reconstruction.py tests/fixtures/cross_page_table_pair.json
git commit -m "feat: replay deterministic table reconstruction"
```

---

### Task 5: Constrained LLM adjudication and shared reconstruction pipeline

**Files:**
- Create: `app/core/diff/table_reconstruction_llm.py`
- Create: `app/core/diff/table_reconstruction_pipeline.py`
- Create: `tests/test_diff/test_table_reconstruction_llm.py`
- Create: `tests/test_diff/test_table_reconstruction_pipeline.py`

**Interfaces:**
- `adjudicate_continuation(candidate: ContinuationCandidate, provider: BaseProvider) -> LLMJudgment | None`
- `ReconstructionResult(baseline_ir: DocumentIR, target_ir: DocumentIR, section_pairs: list[SectionPair], trace: ReconstructionTrace)`
- `reconstruct_table_pairs(section_pairs: Sequence[SectionPair], baseline_ir: DocumentIR, target_ir: DocumentIR, provider: BaseProvider | None) -> ReconstructionResult`
- `replay_reconstruction(baseline_ir: DocumentIR, target_ir: DocumentIR, trace: ReconstructionTrace) -> tuple[DocumentIR, DocumentIR]`

- [ ] **Step 1: Write failing strict-LLM-response tests**

Use a recording fake provider whose `chat` returns exact strings. Cover accepted merge, rejection, sub-threshold merge, wrong candidate ID, invalid decision, out-of-range confidence, missing reason, extra operation fields, fenced JSON, exception, and timeout-shaped exception.

```python
def test_adjudicator_parses_matching_valid_response():
    candidate = make_medium_candidate("candidate-1")
    provider = RecordingProvider(
        '{"candidate_id":"candidate-1","decision":"merge","confidence":0.75,"reason":"cells continue"}'
    )

    judgment = adjudicate_continuation(candidate, provider)

    assert judgment is not None
    assert judgment.decision == "merge"
    assert len(provider.chat_calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        '{"candidate_id":"other","decision":"merge","confidence":0.99,"reason":"mismatch"}',
        '{"candidate_id":"candidate-1","decision":"rewrite","confidence":0.99,"reason":"invalid"}',
        '{"candidate_id":"candidate-1","decision":"merge","confidence":1.2,"reason":"invalid"}',
        '{"candidate_id":"candidate-1","decision":"merge","confidence":0.99,"reason":"ok","cells":["new"]}',
        '```json\n{"candidate_id":"candidate-1","decision":"merge","confidence":0.99,"reason":"fenced"}\n```',
    ],
)
def test_adjudicator_rejects_non_strict_responses(response):
    assert adjudicate_continuation(
        make_medium_candidate("candidate-1"), RecordingProvider(response)
    ) is None
```

Add a separate valid response at confidence `0.74`; assert the adjudicator preserves it in `LLMJudgment`, while the pipeline records it and sets `final_action == "keep_separate"`. This keeps the model's auditable reason without allowing a sub-threshold merge.

- [ ] **Step 2: Write failing call-routing tests**

```python
def test_pipeline_calls_llm_once_per_medium_candidate_only():
    provider = QueueProvider(valid_merge_responses(2))
    result = reconstruct_fixture_with_confidences(
        high=1, medium=2, low=1, vetoed=1, provider=provider
    )

    assert len(provider.chat_calls) == 2
    assert accepted_decision_count(result.trace) == 3


def test_pipeline_provider_failure_keeps_candidate_and_continues():
    provider = RaisingProvider(TimeoutError("model timeout"))

    result = reconstruct_fixture_with_confidences(
        high=0, medium=1, low=0, vetoed=0, provider=provider
    )

    assert result.trace.decisions[0].final_action == "keep_separate"
    assert result.trace.decisions[0].llm is None
```

Also assert a hard-veto candidate never reaches the provider and provider `None` keeps medium candidates separate.

- [ ] **Step 3: Write failing pipeline consistency tests**

Assert the pipeline:

- analyzes both sides jointly for cross-version support;
- returns new normalized IR values while originals remain equal to deep copies;
- rebuilds `section_pairs` from normalized IR by calling `align_sections` exactly once after replay;
- produces the same IR values when `replay_reconstruction` applies the emitted trace;
- produces stable trace and derived IDs across two identical runs with the same provider responses;
- never imports or calls `app.core.retrieval.searcher` and never consumes `Chunk.faiss_index_id`.

- [ ] **Step 4: Run the Task 5 tests and verify RED**

```powershell
pytest tests/test_diff/test_table_reconstruction_llm.py tests/test_diff/test_table_reconstruction_pipeline.py -v
```

Expected: collection fails because the LLM adapter and orchestration modules do not exist.

- [ ] **Step 5: Implement the bounded prompt and strict response parser**

Send one system message and one user message through `provider.chat`. The user payload is JSON containing only:

- candidate ID and side;
- previous, continuation, and next logical cell arrays;
- logical column roles and physical mapping summary;
- rule evidence and conflicts;
- a bounded cross-version neighborhood of at most three rows.

The system message states that only one JSON object with exactly `candidate_id`, `decision`, `confidence`, and `reason` is valid. Parse with `json.loads` directly; do not strip Markdown fences or search embedded JSON. Determine the model label with `getattr(provider, "chat_model", provider.__class__.__name__)`.

- [ ] **Step 6: Implement the shared pipeline**

For each aligned section pair:

1. Collect both sides' fragments.
2. Infer regions, body columns, mappings, and repeated boundary artifacts.
3. Generate and assess local candidates jointly.
4. Convert high decisions directly, call the adjudicator only for medium decisions, and keep all other candidates separate.
5. Build one ordered operation list and trace.
6. Replay operations on deep copies.
7. Call `align_sections(normalized_baseline, normalized_target)` to return pairs that reference only normalized documents.

Catch provider exceptions inside each individual adjudication and return `None` for that candidate so the pipeline continues. Do not catch structural/replay errors in the core pipeline.

- [ ] **Step 7: Run Task 5 tests and verify GREEN**

```powershell
pytest tests/test_diff/test_table_reconstruction_llm.py tests/test_diff/test_table_reconstruction_pipeline.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Run the complete reconstruction unit suite**

```powershell
pytest tests/test_diff/test_reconstruction_trace.py tests/test_diff/test_table_reconstruction.py tests/test_diff/test_table_reconstruction_llm.py tests/test_diff/test_table_reconstruction_pipeline.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 5**

```powershell
git add app/core/diff/table_reconstruction_llm.py app/core/diff/table_reconstruction_pipeline.py tests/test_diff/test_table_reconstruction_llm.py tests/test_diff/test_table_reconstruction_pipeline.py
git commit -m "feat: adjudicate table continuations"
```

---

### Task 6: Graph and synchronous service integration with dual-artifact persistence

**Files:**
- Modify: `app/agent/states.py`
- Modify: `app/agent/compare_graph.py`
- Modify: `app/services/compare_service.py`
- Modify: `tests/test_agent/test_compare_graph.py`
- Modify: `tests/test_services/test_compare_service.py`

**Interfaces:**
- Add `CompareState._reconstruction_trace: Any`.
- Add graph node `do_reconstruct_tables(state: CompareState) -> dict` between `do_align` and `do_semantic_compare`.
- `do_semantic_compare` consumes reconstructed `_section_pairs`.
- Both persistence paths call `persist_compare_artifacts` and update task status only after it returns.

- [ ] **Step 1: Write failing graph-order and state-flow tests**

Extend the happy-path graph test with a mocked reconstruction result:

```python
def test_compare_graph_reconstructs_before_matching_and_persists_trace(base_state, monkeypatch):
    calls: list[str] = []
    reconstruction = make_reconstruction_result()

    monkeypatch.setattr(compare_graph_module, "align_sections", recording("align", calls, []))
    monkeypatch.setattr(
        compare_graph_module,
        "reconstruct_table_pairs",
        recording("reconstruct", calls, reconstruction),
    )
    monkeypatch.setattr(
        compare_graph_module,
        "match_paragraphs",
        recording("match", calls, []),
    )
    monkeypatch.setattr(
        compare_graph_module,
        "persist_compare_artifacts",
        recording("persist", calls, make_artifact_paths()),
    )

    final_state = compare_graph_module.compare_graph.invoke(base_state)

    assert calls.index("align") < calls.index("reconstruct") < calls.index("match")
    assert calls.index("match") < calls.index("persist")
    assert final_state["status"] == "completed"
```

Assert `match_paragraphs` receives `reconstruction.section_pairs`, not the initial aligned pairs.

- [ ] **Step 2: Write failing persistence-failure tests**

For graph and service, make `persist_compare_artifacts` raise `OSError("sidecar write failed")`; assert task status becomes `failed`, never `completed`, and the service re-raises. Add a provider failure inside one medium decision and assert the task still completes with a keep-separate decision.

- [ ] **Step 3: Write failing graph/service equivalence test**

Run both paths over the sanitized fixture with deterministic fake embedder/provider responses. Assert equal trace dictionaries, equal normalized texts passed to `match_paragraphs`, and unchanged source JSON files.

- [ ] **Step 4: Run integration tests and verify RED**

```powershell
pytest tests/test_agent/test_compare_graph.py tests/test_services/test_compare_service.py -k "reconstruct or sidecar or equivalence" -v
```

Expected: tests fail because neither compare path invokes reconstruction or writes a sidecar.

- [ ] **Step 5: Add graph state and node**

In `CompareState`, add `_reconstruction_trace`. Implement `do_reconstruct_tables` with the same error/status pattern as other nodes:

```python
def do_reconstruct_tables(state: CompareState) -> dict:
    try:
        reconstructed = reconstruct_table_pairs(
            state["_section_pairs"],
            state["_baseline_ir"],
            state["_target_ir"],
            state.get("provider"),
        )
        return {
            "_baseline_ir": reconstructed.baseline_ir,
            "_target_ir": reconstructed.target_ir,
            "_section_pairs": reconstructed.section_pairs,
            "_reconstruction_trace": reconstructed.trace,
            "status": "tables_reconstructed",
        }
    except Exception as exc:
        logger.exception("do_reconstruct_tables failed")
        compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
        return {"error": str(exc), "status": "failed"}
```

Insert the node immediately after `do_align` in the graph node list.

- [ ] **Step 6: Integrate the service through the same entry point**

After initial alignment, call `reconstruct_table_pairs` once and replace local IR/pair variables with its result. Pass only reconstructed pairs to `match_paragraphs`. Do not add a second implementation or reconstruct chunks.

- [ ] **Step 7: Replace direct export writes in both paths**

After DB diff-item insertion, call:

```python
result_path, trace_path = persist_compare_artifacts(
    state["data_dir"],
    task_id,
    result.items,
    state["_reconstruction_trace"],
)
```

The service uses the equivalent local variables. Only then call `compare_repo.update_task_status(conn, task_id, "completed", str(result_path))`. The graph calls the same repository function with `state["conn"]`. Log both artifact paths at debug level. Leave the existing result JSON schema unchanged.

- [ ] **Step 8: Run integration tests and verify GREEN**

```powershell
pytest tests/test_agent/test_compare_graph.py tests/test_services/test_compare_service.py -v
```

Expected: all graph and service tests pass.

- [ ] **Step 9: Commit Task 6**

```powershell
git add app/agent/states.py app/agent/compare_graph.py app/services/compare_service.py tests/test_agent/test_compare_graph.py tests/test_services/test_compare_service.py
git commit -m "feat: reconstruct tables before comparison"
```

---

### Task 7: Comparison-page sidecar validation and replay

**Files:**
- Modify: `app/ui/pages/compare_page.py`
- Modify: `tests/ui/test_compare_page.py`

**Interfaces:**
- Add `ComparePage._load_reconstructed_ir_pair(task_id: str, baseline_version_id: str, target_version_id: str) -> tuple[DocumentIR | None, DocumentIR | None]`.
- `_render_diff(result: DiffResult)` consumes the replayed pair for full-document rendering.
- Missing or invalid trace returns the two raw IR values and logs one warning.

- [ ] **Step 1: Write failing valid-replay UI test**

Build raw IR files and a trace sidecar under the test `data_dir`, then invoke `_render_diff`:

```python
def test_render_diff_replays_task_trace_before_rendering(compare_page, tmp_path):
    baseline_ir, target_ir, trace = make_ui_replay_fixture(tmp_path)
    write_source_irs(compare_page, baseline_ir, target_ir)
    write_json_atomic(
        reconstruction_trace_path(tmp_path, "task-replay"),
        trace_to_dict(trace),
    )

    compare_page._render_diff(make_result("task-replay"))

    script = compare_page._web_view.page().runJavaScript.call_args.args[0]
    assert "cedar pre<br>lude-complete" in decode_injected_html(script)
    assert boundary_fixture_token() not in decode_injected_html(script)
```

Assert the rendered normalized row receives the same diff marker as the corresponding `DiffItem.baseline_text`/`target_text`.

- [ ] **Step 2: Write failing safe-fallback tests**

Parameterize missing sidecar, invalid JSON, schema version 2, wrong doc ID, and wrong file hash. Assert raw table text is rendered, no partial operation is applied, no provider method is called, and `logger.warning` is called once with task ID and failure category.

- [ ] **Step 3: Run focused UI tests and verify RED**

```powershell
pytest tests/ui/test_compare_page.py -k "replay_task_trace or reconstruction_sidecar" -v
```

Expected: tests fail because `_render_diff` still loads raw IR directly.

- [ ] **Step 4: Implement task-sidecar loading and all-or-nothing replay**

Use the existing `_load_version_ir` for both raw documents. If either load returns `None`, return the raw pair immediately. Build the trace path from `self.ctx.data_dir` and result task ID. In one `try` block:

1. Load strict trace.
2. Validate both document identities and hashes.
3. Call `replay_reconstruction`.
4. Return both normalized IR values.

Catch `FileNotFoundError`, `json.JSONDecodeError`, `TypeError`, `ValueError`, and `OSError`, log one warning, and return the untouched pair. Do not retain a partially reconstructed side.

- [ ] **Step 5: Render and match against replayed IR**

Change `_render_diff` to call `_load_reconstructed_ir_pair`. Keep existing full-document rendering, exact normalized row lookup, first-page initial view, focus behavior, and synchronized scrolling. Do not call candidate discovery, the provider, the embedder, or FAISS from the page.

- [ ] **Step 6: Run UI tests and verify GREEN**

```powershell
pytest tests/ui/test_compare_page.py -v
```

Expected: all compare-page tests pass.

- [ ] **Step 7: Commit Task 7**

```powershell
git add app/ui/pages/compare_page.py tests/ui/test_compare_page.py
git commit -m "feat: replay table reconstruction in compare page"
```

---

### Task 8: Documentation, external-sample acceptance, and full verification

**Files:**
- Modify: `TECHNICAL.md`
- Verify: `app/core/diff/reconstruction_trace.py`
- Verify: `app/core/diff/table_reconstruction.py`
- Verify: `app/core/diff/table_reconstruction_llm.py`
- Verify: `app/core/diff/table_reconstruction_pipeline.py`
- Verify: `app/agent/compare_graph.py`
- Verify: `app/services/compare_service.py`
- Verify: `app/ui/pages/compare_page.py`

**Interfaces:**
- Document the new pipeline stage, sidecar contract, failure behavior, and explicit non-dependency on retrieval chunks/FAISS.
- Produce a local acceptance report from the two user-provided JSON files without copying them into the repository.

- [ ] **Step 1: Add technical documentation**

Update `TECHNICAL.md` with this exact data flow:

```text
parsed DocumentIR pair
  -> align_sections
  -> reconstruct_table_pairs
  -> match_paragraphs (fresh embeddings of reconstructed in-memory text)
  -> classify
  -> persist diff JSON + reconstruction sidecar
```

Document that `app/core/retrieval/searcher.py`, `Chunk.faiss_index_id`, and stored FAISS indexes remain exclusive to retrieval/QA and are not read or rebuilt by comparison reconstruction.

- [ ] **Step 2: Run static non-hardcoding checks**

```powershell
rg -n "<known-external-label-patterns>|<known-external-id-prefixes>" app tests docs
rg -n "faiss|faiss_index_id|core\.retrieval" app/core/diff -g 'table_reconstruction*.py' -g 'reconstruction_trace.py'
```

Expected: both commands print no matches. Generic tests may use invented neutral boundary tokens only.

- [ ] **Step 3: Run the external JSON acceptance through a temporary local harness**

Create a temporary, untracked script at `$env:TEMP\verify_cross_page_reconstruction.py` by copying the following complete code into the shell. It reads the two supplied files, performs alignment and reconstruction, and writes only a compact report under `$env:TEMP`:

```python
import json
import re
import tempfile
from copy import deepcopy
from pathlib import Path

from app.config.settings import load
from app.core.diff.structure_aligner import align_sections
from app.core.diff.table_reconstruction_pipeline import reconstruct_table_pairs
from app.core.model.factory import get_provider
from app.core.types import DocumentIR, Paragraph, Section, Sentence


def load_ir(path: Path) -> DocumentIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    return DocumentIR(
        doc_id=data["doc_id"],
        title=data["title"],
        file_hash=data["file_hash"],
        sections=[
            Section(
                section_id=section["section_id"],
                title=section["title"],
                level=section["level"],
                paragraphs=[
                    Paragraph(
                        paragraph_id=paragraph["paragraph_id"],
                        text=paragraph["text"],
                        sentences=[
                            Sentence(text=sentence["text"])
                            for sentence in paragraph.get("sentences", [])
                        ],
                    )
                    for paragraph in section.get("paragraphs", [])
                ],
            )
            for section in data.get("sections", [])
        ],
        plain_text=data.get("plain_text", ""),
    )


def compact(text: str) -> str:
    return re.sub(r"\s+|<br\s*/?>", "", text)


baseline_path = Path(r"<external-baseline-json>")
target_path = Path(r"<external-target-json>")
baseline_ir = load_ir(baseline_path)
target_ir = load_ir(target_path)
baseline_before = deepcopy(baseline_ir)
target_before = deepcopy(target_ir)
provider = get_provider(load())
result = reconstruct_table_pairs(
    align_sections(baseline_ir, target_ir),
    baseline_ir,
    target_ir,
    provider,
)
baseline_text = compact(result.baseline_ir.plain_text)
target_text = compact(result.target_ir.plain_text)
combined_text = baseline_text + target_text

assert any(operation.type == "merge_rows" for operation in result.trace.operations)
assert baseline_ir == baseline_before
assert target_ir == target_before
assert all(
    decision.candidate_id and decision.final_action in {"merge", "keep_separate"}
    for decision in result.trace.decisions
)
report = {
    "baseline_doc_id": baseline_ir.doc_id,
    "target_doc_id": target_ir.doc_id,
    "decision_count": len(result.trace.decisions),
    "merge_count": sum(
        decision.final_action == "merge" for decision in result.trace.decisions
    ),
    "operation_counts": {
        operation_type: sum(
            operation.type == operation_type for operation in result.trace.operations
        )
        for operation_type in {
            "project_columns",
            "drop_boundary_rows",
            "drop_boundary_paragraphs",
            "merge_rows",
            "merge_fragments",
        }
    },
    "contains_joined_split_text": any(
        "<br>" in sentence.text
        for section in (*result.baseline_ir.sections, *result.target_ir.sections)
        for paragraph in section.paragraphs
        for sentence in paragraph.sentences
    ),
}
Path(tempfile.gettempdir(), "cross_page_reconstruction_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=False, indent=2))
```

Run the script from the repository root:

```powershell
python "$env:TEMP\verify_cross_page_reconstruction.py"
```

Expected: exit code 0; the report shows at least one merge and includes nonzero `drop_boundary_rows`, `merge_rows`, and `merge_fragments` counts.

Manually inspect the trace and normalized row texts for the seven approved acceptance points:

1. Logical columns are inferred from varying physical layouts.
2. Repeated wide page-header rows do not appear as business rows.
3. The approved split row is joined without invented wording; tracked examples use only `cedar pre` and `lude-complete`.
4. Each approved split business row is whole in the version where a continuation exists.
5. The terminal incomplete row remains incomplete when no continuation exists.
6. Real numeric changes and the target-only business row remain present.
7. Replay of the emitted trace equals both normalized IR values.

If a medium-confidence candidate invokes the configured provider, record the candidate ID, bounded input size, decision, and confidence in the local report; never copy source row text into the repository.

- [ ] **Step 4: Run focused and regression suites**

```powershell
pytest tests/test_diff/test_reconstruction_trace.py tests/test_diff/test_table_reconstruction.py tests/test_diff/test_table_reconstruction_llm.py tests/test_diff/test_table_reconstruction_pipeline.py tests/test_agent/test_compare_graph.py tests/test_services/test_compare_service.py tests/ui/test_compare_page.py -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Run the complete test suite**

```powershell
pytest -q
```

Expected: all tests pass. Report Windows sandbox-only warnings separately; do not change product code to hide environmental failures.

- [ ] **Step 6: Verify formatting, source immutability, and dependency boundaries**

```powershell
git diff --check
git status --short
git diff -- app/core/types.py app/core/parser app/core/retrieval
```

Expected: `git diff --check` has no output; the final command has no output; status contains only intended feature files before the documentation commit.

- [ ] **Step 7: Commit Task 8**

```powershell
git add TECHNICAL.md
git commit -m "docs: explain table reconstruction pipeline"
```

- [ ] **Step 8: Final branch review**

```powershell
git log --oneline -8
git status --short
```

Expected: eight focused implementation commits follow the approved design/plan commits, and the worktree is clean.
