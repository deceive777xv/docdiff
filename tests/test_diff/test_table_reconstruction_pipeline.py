from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import inspect
import json

from app.core.diff.reconstruction_trace import (
    ALGORITHM_VERSION,
    SCHEMA_VERSION,
    SourceRowRef,
)
from app.core.diff.structure_aligner import SectionPair
from app.core.diff.table_reconstruction import (
    CandidateAssessment,
    ColumnMapping,
    ContinuationCandidate,
    TableFragment,
    TableRegion,
    split_markdown_table_row,
)
from app.core.diff import table_reconstruction_pipeline as pipeline
from app.core.model.base_provider import BaseProvider
from app.core.types import DocumentIR, Paragraph, Section, Sentence


class QueueProvider(BaseProvider):
    chat_model = "queue-model"

    def __init__(self, responses: list[str | BaseException]):
        self.responses = list(responses)
        self.chat_calls: list[list[dict]] = []

    def chat(self, messages: list[dict], **kwargs) -> str:
        self.chat_calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("retrieval is outside reconstruction")

    def health_check(self) -> bool:
        return True


def _row(cells: tuple[str, ...], index: int, paragraph_id: str, section_id: str = "section-1"):
    row = split_markdown_table_row(
        "|" + "|".join(cells) + "|",
        SourceRowRef(section_id, paragraph_id, index),
    )
    assert row is not None
    return row


def _fragment(paragraph_id: str, paragraph_index: int, section_id: str = "section-1") -> TableFragment:
    rows = (
        _row((str(paragraph_index + 10), "body", "prefix"), 0, paragraph_id, section_id),
        _row((str(paragraph_index + 11), "body", "complete"), 1, paragraph_id, section_id),
    )
    return TableFragment(
        section_id,
        paragraph_id,
        paragraph_index,
        rows,
        (TableRegion(rows, 0, len(rows), "body"),),
        (0,),
        (0, 1, 2),
    )


def _candidate(
    candidate_id: str,
    confidence: str,
    *,
    side: str = "baseline",
    vetoed: bool = False,
) -> ContinuationCandidate:
    evidence_count = {"high": 4, "medium": 2, "low": 1}[confidence]
    evidence = (
        "blank_key_cells",
        "next_row_restores_key_pattern",
        "complementary_content_cells",
        "textual_continuity",
    )[:evidence_count]
    return ContinuationCandidate(
        candidate_id=candidate_id,
        side=side,
        previous_row=_row(("12", "drive", "prefix"), 0, f"{candidate_id}-left"),
        continuation_row=_row(("", "", "suffix"), 0, f"{candidate_id}-right"),
        next_full_row=None,
        mapping=ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
        evidence=evidence,
        conflicts=(),
        vetoes=("new_key_value",) if vetoed else (),
        cross_version_rows=(),
    )


def _documents() -> tuple[DocumentIR, DocumentIR, list[SectionPair]]:
    baseline_section = Section("baseline-section", "Tables", 1, [])
    target_section = Section("target-section", "Tables", 1, [])
    baseline = DocumentIR("baseline-doc", "Baseline", "baseline-hash", [baseline_section])
    target = DocumentIR("target-doc", "Target", "target-hash", [target_section])
    return baseline, target, [SectionPair(baseline_section, target_section, 1.0)]


def _stub_candidates(monkeypatch, candidates: list[ContinuationCandidate]):
    baseline_fragments = (_fragment("baseline-a", 0, "baseline-section"), _fragment("baseline-b", 1, "baseline-section"))
    target_fragments = (_fragment("target-a", 0, "target-section"), _fragment("target-b", 1, "target-section"))

    def collect(section):
        return list(baseline_fragments if section.section_id == "baseline-section" else target_fragments)

    seen_cross_version: list[tuple[str, tuple[str, ...]]] = []

    def generate(
        left,
        right,
        mapping,
        boundary_rows,
        cross_version_fragments,
        side,
        allow_non_table_gap=False,
    ):
        del allow_non_table_gap
        seen_cross_version.append((side, tuple(fragment.section_id for fragment in cross_version_fragments)))
        if side == "target":
            return []
        return candidates

    monkeypatch.setattr(pipeline, "collect_table_fragments", collect)
    monkeypatch.setattr(pipeline, "infer_monotonic_column_mapping", lambda *args: ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0))
    monkeypatch.setattr(pipeline, "generate_continuation_candidates", generate)
    monkeypatch.setattr(pipeline, "classify_repeated_boundary_regions", lambda fragments: set())
    monkeypatch.setattr(pipeline, "build_reconstruction_operations", lambda *args: [])
    monkeypatch.setattr(pipeline, "apply_reconstruction_operations", lambda baseline, target, operations: (deepcopy(baseline), deepcopy(target)))
    return seen_cross_version


def _response(candidate_id: str, decision: str = "merge", confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "candidate_id": candidate_id,
            "decision": decision,
            "confidence": confidence,
            "reason": "bounded structural evidence",
        }
    )


def test_pipeline_calls_llm_once_per_medium_candidate_only(monkeypatch):
    candidates = [
        _candidate("high", "high"),
        _candidate("medium-1", "medium"),
        _candidate("medium-2", "medium"),
        _candidate("low", "low"),
        _candidate("vetoed", "high", vetoed=True),
    ]
    _stub_candidates(monkeypatch, candidates)
    baseline, target, pairs = _documents()
    provider = QueueProvider([_response("medium-1"), _response("medium-2")])

    result = pipeline.reconstruct_table_pairs(pairs, baseline, target, provider)

    assert len(provider.chat_calls) == 2
    assert [(decision.candidate_id, decision.final_action) for decision in result.trace.decisions] == [
        ("high", "merge"),
        ("low", "keep_separate"),
        ("medium-1", "merge"),
        ("medium-2", "merge"),
        ("vetoed", "keep_separate"),
    ]
    decisions = {decision.candidate_id: decision for decision in result.trace.decisions}
    assert decisions["high"].llm is None
    assert decisions["low"].llm is None
    assert decisions["vetoed"].llm is None
    assert decisions["medium-1"].llm is not None
    assert decisions["medium-2"].llm is not None


def test_pipeline_provider_failure_is_nonfatal_and_continues(monkeypatch):
    candidates = [_candidate("medium-1", "medium"), _candidate("medium-2", "medium")]
    _stub_candidates(monkeypatch, candidates)
    baseline, target, pairs = _documents()
    provider = QueueProvider([TimeoutError("model timeout"), _response("medium-2")])

    result = pipeline.reconstruct_table_pairs(pairs, baseline, target, provider)

    decisions = {decision.candidate_id: decision for decision in result.trace.decisions}
    assert decisions["medium-1"].final_action == "keep_separate"
    assert decisions["medium-1"].llm is None
    assert decisions["medium-2"].final_action == "merge"
    assert len(provider.chat_calls) == 2


def test_pipeline_keeps_medium_separate_without_provider(monkeypatch):
    _stub_candidates(monkeypatch, [_candidate("medium", "medium")])
    baseline, target, pairs = _documents()

    result = pipeline.reconstruct_table_pairs(pairs, baseline, target, None)

    assert result.trace.decisions[0].final_action == "keep_separate"
    assert result.trace.decisions[0].llm is None


def test_pipeline_records_sub_threshold_llm_merge_but_keeps_separate(monkeypatch):
    _stub_candidates(monkeypatch, [_candidate("medium", "medium")])
    baseline, target, pairs = _documents()

    result = pipeline.reconstruct_table_pairs(
        pairs, baseline, target, QueueProvider([_response("medium", confidence=0.74)])
    )

    decision = result.trace.decisions[0]
    assert decision.llm is not None
    assert decision.llm.confidence == 0.74
    assert decision.final_action == "keep_separate"


def test_pipeline_keeps_medium_separate_for_duplicate_response_member(monkeypatch):
    _stub_candidates(monkeypatch, [_candidate("medium", "medium")])
    baseline, target, pairs = _documents()
    duplicate_response = (
        '{"candidate_id":"other","candidate_id":"medium","decision":"merge",'
        '"confidence":0.99,"reason":"duplicate"}'
    )

    result = pipeline.reconstruct_table_pairs(
        pairs,
        baseline,
        target,
        QueueProvider([duplicate_response]),
    )

    decision = result.trace.decisions[0]
    assert decision.llm is None
    assert decision.final_action == "keep_separate"


def test_pipeline_supplies_opposite_version_fragments_as_joint_context(monkeypatch):
    seen = _stub_candidates(monkeypatch, [])
    baseline, target, pairs = _documents()

    pipeline.reconstruct_table_pairs(pairs, baseline, target, None)

    assert seen == [
        ("baseline", ("target-section", "target-section")),
        ("target", ("baseline-section", "baseline-section")),
    ]


def test_pipeline_skips_fully_dropped_boundary_fragment_when_pairing_neighbors(monkeypatch):
    fragments = (
        _fragment("table-a", 0, "baseline-section"),
        _fragment("repeated-boundary", 1, "baseline-section"),
        _fragment("table-b", 2, "baseline-section"),
    )
    paired: list[tuple[str, str]] = []

    monkeypatch.setattr(
        pipeline,
        "collect_table_fragments",
        lambda section: list(fragments) if section.section_id == "baseline-section" else [],
    )
    monkeypatch.setattr(
        pipeline,
        "classify_repeated_boundary_regions",
        lambda found: {row.source for row in fragments[1].rows} if found else set(),
    )
    monkeypatch.setattr(
        pipeline,
        "infer_monotonic_column_mapping",
        lambda *args: ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
    )

    def record_pair(left, right, *args, **kwargs):
        del kwargs
        paired.append((left.paragraph_id, right.paragraph_id))
        return []

    monkeypatch.setattr(pipeline, "generate_continuation_candidates", record_pair)
    monkeypatch.setattr(pipeline, "build_reconstruction_operations", lambda *args: [])
    monkeypatch.setattr(
        pipeline,
        "apply_reconstruction_operations",
        lambda baseline, target, operations: (deepcopy(baseline), deepcopy(target)),
    )
    baseline, target, pairs = _documents()

    pipeline.reconstruct_table_pairs(pairs, baseline, target, None)

    assert paired == [("table-a", "table-b")]


def test_pipeline_replays_on_copies_then_aligns_normalized_documents_once(monkeypatch):
    _stub_candidates(monkeypatch, [])
    baseline, target, pairs = _documents()
    baseline_before = deepcopy(baseline)
    target_before = deepcopy(target)
    calls: list[tuple[DocumentIR, DocumentIR]] = []

    def recording_align(normalized_baseline, normalized_target):
        calls.append((normalized_baseline, normalized_target))
        return [SectionPair(normalized_baseline.sections[0], normalized_target.sections[0], 1.0)]

    monkeypatch.setattr(pipeline, "align_sections", recording_align)

    result = pipeline.reconstruct_table_pairs(pairs, baseline, target, None)

    assert baseline == baseline_before
    assert target == target_before
    assert result.baseline_ir is not baseline
    assert result.target_ir is not target
    assert calls == [(result.baseline_ir, result.target_ir)]
    assert result.section_pairs[0].baseline_section is result.baseline_ir.sections[0]
    assert result.section_pairs[0].target_section is result.target_ir.sections[0]


def test_replay_matches_emitted_normalized_ir_and_trace_is_stable(monkeypatch):
    _stub_candidates(monkeypatch, [])
    baseline, target, pairs = _documents()

    first = pipeline.reconstruct_table_pairs(pairs, baseline, target, None)
    second = pipeline.reconstruct_table_pairs(pairs, baseline, target, None)
    replayed = pipeline.replay_reconstruction(baseline, target, first.trace)

    assert replayed == (first.baseline_ir, first.target_ir)
    assert first.trace == second.trace
    assert first.trace.schema_version == SCHEMA_VERSION
    assert first.trace.algorithm_version == ALGORITHM_VERSION


def test_pipeline_has_no_retrieval_or_faiss_dependency():
    source = inspect.getsource(pipeline)

    assert "app.core.retrieval.searcher" not in source
    assert "faiss_index_id" not in source


def test_pipeline_reconstructs_leading_continuation_before_body_across_non_table_gap():
    left_lines = [
        "| item | group | detail | limit |",
        "| --- | --- | --- | --- |",
        "| 70101 | amber | stable | ready |",
        "| 70102 | blue | prefix | cedar pre |",
    ]
    right_lines = [
        "| neutral-boundary | neutral-boundary | neutral-boundary | neutral-boundary | neutral-boundary | neutral-boundary | neutral-boundary | neutral-boundary | neutral-boundary |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        "| | item | group | detail | | limit | | | |",
        "| | | | | | lude-complete | | | |",
        "| | 70103 | cyan | next | | complete | | | |",
        "| | 70104 | green | later | | complete | | | |",
    ]

    def paragraph(paragraph_id: str, lines: list[str]) -> Paragraph:
        return Paragraph(
            paragraph_id,
            "\n".join(lines),
            [Sentence(line) for line in lines],
        )

    section = Section(
        "section-neutral",
        "Neutral table",
        1,
        [
            paragraph("fragment-left", left_lines),
            paragraph("neutral-gap", ["Neutral non-table marker."]),
            paragraph("fragment-right", right_lines),
        ],
    )
    baseline = DocumentIR(
        "baseline-neutral",
        "Baseline neutral",
        "baseline-neutral-hash",
        [section],
        "\n".join(paragraph.text for paragraph in section.paragraphs),
    )
    target = DocumentIR("target-neutral", "Target neutral", "target-neutral-hash", [])

    result = pipeline.reconstruct_table_pairs(
        [SectionPair(section, None, 0.0)], baseline, target, None
    )
    assert "cedar pre<br>lude-complete" in result.baseline_ir.plain_text
    assert any(decision.final_action == "merge" for decision in result.trace.decisions)
