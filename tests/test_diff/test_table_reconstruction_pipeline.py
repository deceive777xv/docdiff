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
    build_reconstruction_operations,
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


class EchoMergeProvider(BaseProvider):
    chat_model = "echo-merge"

    def chat(self, messages: list[dict], **kwargs) -> str:
        payload = json.loads(messages[-1]["content"])
        return _response(
            payload["candidate_id"],
            boundary_id=payload["boundary_id"],
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("retrieval is outside reconstruction")

    def health_check(self) -> bool:
        return True


class TextChoiceProvider(BaseProvider):
    chat_model = "text-choice"

    def __init__(self, continuation_needle: str):
        self.continuation_needle = continuation_needle

    def chat(self, messages: list[dict], **kwargs) -> str:
        payload = json.loads(messages[-1]["content"])
        continuation = " ".join(payload["continuation_cells"])
        return _response(
            payload["candidate_id"],
            decision=(
                "merge"
                if self.continuation_needle in continuation
                else "keep"
            ),
            boundary_id=payload["boundary_id"],
        )

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
    previous = _row(("12", "drive", "prefix"), 0, f"{candidate_id}-left")
    continuation = _row(("", "", "suffix"), 0, f"{candidate_id}-right")
    return ContinuationCandidate(
        candidate_id=candidate_id,
        side=side,
        previous_row=previous,
        continuation_row=continuation,
        next_full_row=None,
        mapping=ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
        previous_mapping=ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
        previous_fragment_rows=(previous,),
        continuation_fragment_rows=(continuation,),
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


def _response(
    candidate_id: str,
    decision: str = "merge",
    confidence: float = 0.9,
    *,
    boundary_id: str | None = None,
) -> str:
    merge = decision == "merge"
    return json.dumps(
        {
            "boundary_id": boundary_id or candidate_id,
            "candidate_id": candidate_id,
            "roles": {
                "previous_row": "body_row",
                "continuation_row": "continuation_row",
            },
            "action": "merge_row" if merge else "keep",
            "mapping_id": f"{candidate_id}:mapping:0",
            "confidence": confidence,
            "reason": "bounded structural evidence",
        }
    )


def test_pipeline_calls_llm_once_per_nontrivial_candidate(monkeypatch):
    candidates = [
        _candidate("high", "high"),
        _candidate("medium-1", "medium"),
        _candidate("medium-2", "medium"),
        _candidate("low", "low"),
        _candidate("vetoed", "high", vetoed=True),
    ]
    _stub_candidates(monkeypatch, candidates)
    baseline, target, pairs = _documents()
    provider = QueueProvider(
        [_response("high"), _response("medium-1"), _response("medium-2")]
    )

    result = pipeline.reconstruct_table_pairs(pairs, baseline, target, provider)

    assert len(provider.chat_calls) == 3
    assert [(decision.candidate_id, decision.final_action) for decision in result.trace.decisions] == [
        ("high", "merge"),
        ("low", "keep_separate"),
        ("medium-1", "merge"),
        ("medium-2", "merge"),
        ("vetoed", "keep_separate"),
    ]
    decisions = {decision.candidate_id: decision for decision in result.trace.decisions}
    assert decisions["high"].llm is not None
    assert decisions["low"].llm is None
    assert decisions["vetoed"].llm is None
    assert decisions["medium-1"].llm is not None
    assert decisions["medium-2"].llm is not None


def test_pipeline_rejects_known_non_adjacent_page_boundary_before_llm(monkeypatch):
    previous = _row(("12", "drive", "prefix"), 0, "left", "baseline-section")
    continuation = _row(("", "", "suffix"), 0, "right", "baseline-section")
    candidate = replace(
        _candidate("non-adjacent", "high"),
        previous_row=previous,
        continuation_row=continuation,
        previous_fragment_rows=(previous,),
        continuation_fragment_rows=(continuation,),
    )
    _stub_candidates(monkeypatch, [candidate])
    baseline, target, pairs = _documents()
    baseline.sections[0].paragraphs = [
        Paragraph("left", previous.raw_text, [Sentence(previous.raw_text)], page_no=1),
        Paragraph("right", continuation.raw_text, [Sentence(continuation.raw_text)], page_no=3),
    ]
    provider = QueueProvider([])

    result = pipeline.reconstruct_table_pairs(pairs, baseline, target, provider)

    assert provider.chat_calls == []
    assert result.trace.decisions[0].final_action == "keep_separate"


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
    provider = QueueProvider([_response("safe-high"), _response("unsafe-medium")])

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
    assert (
        "unsafe_fragment_projection"
        in decisions["unsafe-medium"].rule_conflicts
    )
    assert {
        operation.decision_id
        for operation in result.trace.operations
        if operation.type == "merge_rows"
    } == {"safe-high"}
    assert len(provider.chat_calls) == 2


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


def test_pipeline_fails_closed_when_llm_accepts_two_choices_for_same_previous_row(
    monkeypatch,
):
    first = _candidate("choice-a", "medium")
    second = _candidate("choice-b", "medium")
    second = replace(
        second,
        previous_row=first.previous_row,
        previous_fragment_rows=first.previous_fragment_rows,
    )
    _stub_candidates(monkeypatch, [first, second])
    baseline, target, pairs = _documents()
    provider = QueueProvider([_response("choice-a"), _response("choice-b")])

    result = pipeline.reconstruct_table_pairs(pairs, baseline, target, provider)

    decisions = {decision.candidate_id: decision for decision in result.trace.decisions}
    assert {decision.final_action for decision in decisions.values()} == {
        "keep_separate"
    }
    assert all(
        "ambiguous_continuation_choices" in decision.rule_conflicts
        for decision in decisions.values()
    )
    assert not any(
        operation.type == "merge_rows" for operation in result.trace.operations
    )


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


def _paragraph(
    paragraph_id: str,
    lines: list[str],
    *,
    page_no: int | None = None,
) -> Paragraph:
    return Paragraph(
        paragraph_id,
        "\n".join(lines),
        [Sentence(line) for line in lines],
        page_no=page_no,
    )


def _continuation_gap_fragments() -> tuple[list[str], list[str]]:
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
    return left_lines, right_lines


def _supporting_table_lines(start: int) -> list[str]:
    return [
        "| item | group | detail | limit |",
        "| --- | --- | --- | --- |",
        f"| {start} | bronze | steady | ready |",
        f"| {start + 1} | silver | steady | complete |",
    ]


def test_pipeline_recovers_sparse_continuation_columns_from_complete_peer_version():
    baseline_section = Section(
        "baseline-regulation",
        "2.3 法规要求",
        1,
        [
            _paragraph(
                "baseline-regulation-left",
                [
                    "|序号|法规编号|法规名称|条款|要求|",
                    "|---|---|---|---|---|",
                    "|5|GB/T 15086|汽车门锁|3.2.3|完整要求|",
                    "|6|GB/T30512|汽车禁用物质要求|/|完整要求|",
                    "|7|GB/T 25985-2010|汽车防盗装置的保护|4.6|如果一年中制造的车辆少于1000辆，则其组合数应与车辆数相等。|",
                ],
                page_no=6,
            ),
            _paragraph(
                "baseline-regulation-right",
                [
                    "|||||文件名称|文件名称|文件名称|文件名称|文件名称|文件名称|",
                    "|---|---|---|---|---|---|---|---|---|---|",
                    "|||||文件编号|||版本：001|第6页||",
                    "|||||||在一种车型的所有汽车中，同一组合的出现率应不大于1/1000。||||",
                ],
                page_no=7,
            ),
        ],
    )
    target_section = Section(
        "target-regulation",
        "2.3 法规要求",
        1,
        [
            _paragraph(
                "target-regulation-left",
                [
                    "|序号|法规编号|法规名称|条款|要求|",
                    "|---|---|---|---|---|",
                    "|5|GB/T 15086|汽车门锁|3.2.3|完整要求|",
                    "|6|GB/T30512|汽车禁用物质要求|/|完整要求|",
                ],
                page_no=5,
            ),
            _paragraph(
                "target-regulation-footer",
                ["FMA-272-A19-V01（20171013）"],
                page_no=5,
            ),
            _paragraph(
                "target-regulation-complete",
                [
                    "|||||文件名称|文件名称|文件名称|文件名称|文件名称|文件名称|",
                    "|---|---|---|---|---|---|---|---|---|---|",
                    "|||||文件编号|||版本：001|第6页||",
                    "||6|GB/T30512||汽车禁用物质要求|/|||||",
                    "||7|GB/T 25985-2010||汽车防盗装置的保护|4.6|如果一年中制造的车辆少于1000辆，则其组合数应与车辆数相等。<br>在一种车型的所有汽车中，同一组合的出现率应不大于1/1000。||||",
                ],
                page_no=6,
            )
        ],
    )
    baseline = DocumentIR(
        "baseline-regulation-doc",
        "Baseline",
        "baseline-regulation-hash",
        [baseline_section],
    )
    target = DocumentIR(
        "target-regulation-doc",
        "Target",
        "target-regulation-hash",
        [target_section],
    )

    result = pipeline.reconstruct_table_pairs(
        [SectionPair(baseline_section, target_section, 1.0)],
        baseline,
        target,
        TextChoiceProvider("在一种车型"),
    )

    compact = result.baseline_ir.plain_text.replace(" ", "").replace("|", "")
    assert "与车辆数相等。<br>在一种车型" in compact


def test_pipeline_recovers_sparse_continuation_columns_from_next_complete_row():
    section = Section(
        "baseline-function",
        "3.1 系统功能",
        1,
        [
            _paragraph(
                "baseline-function-left",
                [
                    "||序号|功能|要求||备注||||",
                    "|---|---|---|---|---|---|---|---|---|",
                    "||20|开关门舒适性|关门反力：300 ± 30 N。||根据分析结果微调。||||",
                    "||21|防夹要求|关门过程：①电撑杆高位、中位、低位（峰值）≤100 N；开门过程：③电撑杆高位、中位、低位（峰值）≤120 N；||/||||",
                ],
                page_no=7,
            ),
            _paragraph(
                "baseline-function-right",
                [
                    "|||||文件名称|文件名称|文件名称|文件名称|文件名称|",
                    "|---|---|---|---|---|---|---|---|---|",
                    "|||||文件编号||版本：001|第8页||",
                    "||||电撑杆高位、中位、低位（有效值）≤120 N。<br>2、防夹响应时间：≤0.5 s；||||||",
                    "||22|障碍物检测|障碍物前停止距离：100-150 mm。||/||||",
                ],
                page_no=8,
            ),
        ],
    )
    baseline = DocumentIR(
        "baseline-function-doc",
        "Baseline",
        "baseline-function-hash",
        [section],
    )
    target = DocumentIR("target-empty", "Target", "target-empty-hash", [])

    result = pipeline.reconstruct_table_pairs(
        [SectionPair(section, None, 0.0)],
        baseline,
        target,
        EchoMergeProvider(),
    )

    merged_row = next(
        sentence.text
        for output_section in result.baseline_ir.sections
        for paragraph in output_section.paragraphs
        for sentence in paragraph.sentences
        if "21" in sentence.text and "防夹要求" in sentence.text
    )
    compact = merged_row.replace(" ", "").replace("|", "")
    assert "（有效值）≤120N。" in compact


def _run_single_side_section(section: Section):
    baseline = DocumentIR(
        "baseline-neutral",
        "Baseline neutral",
        "baseline-neutral-hash",
        [section],
        "\n".join(paragraph.text for paragraph in section.paragraphs),
    )
    target = DocumentIR("target-neutral", "Target neutral", "target-neutral-hash", [])
    return baseline, pipeline.reconstruct_table_pairs(
        [SectionPair(section, None, 0.0)], baseline, target, EchoMergeProvider()
    )


def test_pipeline_allows_repeated_stable_boundary_paragraphs_and_replays_their_removal():
    left_lines, right_lines = _continuation_gap_fragments()
    section = Section(
        "section-neutral",
        "Neutral table",
        1,
        [
            _paragraph("fragment-left", left_lines),
            _paragraph("boundary-note-a", ["Repeated boundary note."]),
            _paragraph("fragment-right", right_lines),
            _paragraph("support-left", _supporting_table_lines(80101)),
            _paragraph("boundary-note-b", ["  repeated   BOUNDARY note.  "]),
            _paragraph("support-right", _supporting_table_lines(80103)),
        ],
    )
    baseline, result = _run_single_side_section(section)

    assert "cedar pre<br>lude-complete" in result.baseline_ir.plain_text
    dropped_paragraph_ids = {
        paragraph_id
        for operation in result.trace.operations
        if operation.type == "drop_boundary_paragraphs"
        for paragraph_id in operation.source_paragraph_ids
    }
    assert dropped_paragraph_ids == {"boundary-note-a", "boundary-note-b"}
    assert all(
        paragraph_id not in {
            paragraph.paragraph_id
            for paragraph in result.baseline_ir.sections[0].paragraphs
        }
        for paragraph_id in dropped_paragraph_ids
    )
    assert dropped_paragraph_ids.issubset(
        {
            paragraph.paragraph_id
            for paragraph in baseline.sections[0].paragraphs
        }
    )
    replayed = pipeline.replay_reconstruction(
        baseline,
        DocumentIR("target-neutral", "Target neutral", "target-neutral-hash", []),
        result.trace,
    )
    assert replayed == (result.baseline_ir, result.target_ir)


def test_pipeline_keeps_fragments_separate_across_unique_prose_and_preserves_order():
    left_lines, right_lines = _continuation_gap_fragments()
    section = Section(
        "section-neutral",
        "Neutral table",
        1,
        [
            _paragraph("fragment-left", left_lines),
            _paragraph("unique-prose", ["A unique explanatory sentence remains here."]),
            _paragraph("fragment-right", right_lines),
        ],
    )
    expected_order = [paragraph.paragraph_id for paragraph in section.paragraphs]
    _, result = _run_single_side_section(section)

    assert "cedar pre<br>lude-complete" not in result.baseline_ir.plain_text
    assert any(
        decision.final_action == "keep_separate"
        and "new_section_or_table" in decision.rule_conflicts
        for decision in result.trace.decisions
    )
    assert not any(
        operation.type == "drop_boundary_paragraphs"
        for operation in result.trace.operations
    )
    assert [
        paragraph.paragraph_id
        for paragraph in result.baseline_ir.sections[0].paragraphs
    ] == expected_order


def test_pipeline_keeps_fragments_separate_when_repeated_prose_has_unstable_position():
    left_lines, right_lines = _continuation_gap_fragments()
    section = Section(
        "section-neutral",
        "Neutral table",
        1,
        [
            _paragraph("unstable-prose-outside", ["Repeated explanatory prose."]),
            _paragraph("fragment-left", left_lines),
            _paragraph("unstable-prose-gap", [" repeated   explanatory PROSE. "]),
            _paragraph("fragment-right", right_lines),
        ],
    )
    _, result = _run_single_side_section(section)

    assert "cedar pre<br>lude-complete" not in result.baseline_ir.plain_text
    assert any(
        decision.final_action == "keep_separate"
        and "new_section_or_table" in decision.rule_conflicts
        for decision in result.trace.decisions
    )
    assert not any(
        operation.type == "drop_boundary_paragraphs"
        for operation in result.trace.operations
    )


def test_pipeline_rejects_repeated_prose_at_different_positions_inside_table_gaps():
    section = Section(
        "section-neutral",
        "Neutral table",
        1,
        [
            _paragraph("support-a", _supporting_table_lines(81101)),
            _paragraph("unstable-note-a", ["Repeated positional note."]),
            _paragraph("support-b", _supporting_table_lines(81103)),
            _paragraph("support-c", _supporting_table_lines(81105)),
            _paragraph("gap-filler", ["Unrelated retained paragraph."]),
            _paragraph("unstable-note-b", [" repeated   positional NOTE. "]),
            _paragraph("support-d", _supporting_table_lines(81107)),
        ],
    )
    _, result = _run_single_side_section(section)

    dropped_paragraph_ids = {
        paragraph_id
        for operation in result.trace.operations
        if operation.type == "drop_boundary_paragraphs"
        for paragraph_id in operation.source_paragraph_ids
    }
    assert dropped_paragraph_ids.isdisjoint(
        {"unstable-note-a", "unstable-note-b"}
    )


def test_pipeline_rejects_repeated_boundary_paragraphs_over_the_shortness_limit():
    long_note = "x" * (
        pipeline._MAX_BOUNDARY_PARAGRAPH_NORMALIZED_LENGTH + 1
    )
    section = Section(
        "section-neutral",
        "Neutral table",
        1,
        [
            _paragraph("support-a", _supporting_table_lines(82101)),
            _paragraph("long-note-a", [long_note]),
            _paragraph("support-b", _supporting_table_lines(82103)),
            _paragraph("support-c", _supporting_table_lines(82105)),
            _paragraph("long-note-b", [long_note]),
            _paragraph("support-d", _supporting_table_lines(82107)),
        ],
    )
    _, result = _run_single_side_section(section)

    dropped_paragraph_ids = {
        paragraph_id
        for operation in result.trace.operations
        if operation.type == "drop_boundary_paragraphs"
        for paragraph_id in operation.source_paragraph_ids
    }
    assert dropped_paragraph_ids.isdisjoint({"long-note-a", "long-note-b"})


def test_pipeline_requires_every_paragraph_in_a_gap_to_be_confirmed_boundary_material():
    left_lines, right_lines = _continuation_gap_fragments()
    section = Section(
        "section-neutral",
        "Neutral table",
        1,
        [
            _paragraph("fragment-left", left_lines),
            _paragraph("boundary-note-a", ["Repeated boundary note."]),
            _paragraph("unknown-gap-prose", ["An unrelated body paragraph blocks merging."]),
            _paragraph("fragment-right", right_lines),
            _paragraph("support-left", _supporting_table_lines(80101)),
            _paragraph("boundary-note-b", ["repeated boundary note."]),
            _paragraph("support-right", _supporting_table_lines(80103)),
        ],
    )
    _, result = _run_single_side_section(section)

    merge_by_source = {
        tuple(source.paragraph_id for source in decision.source_rows): decision.final_action
        for decision in result.trace.decisions
    }
    assert merge_by_source[("fragment-left", "fragment-right")] == "keep_separate"
    assert "cedar pre<br>lude-complete" not in result.baseline_ir.plain_text
