from __future__ import annotations

from dataclasses import replace
import json

import pytest

from app.core.normalization.table_trace import SourceRowRef
from app.core.normalization.table_boundary_context import (
    BoundaryContextItem,
    TableBoundaryContext,
)
from app.core.normalization.tables import (
    ColumnMapping,
    ContinuationCandidate,
    split_markdown_table_row,
)
from app.core.normalization.table_resolver import adjudicate_continuation
from app.core.model.base_provider import BaseProvider


class RecordingProvider(BaseProvider):
    chat_model = "recording-model"

    def __init__(self, responses: str | BaseException | list[str | BaseException]):
        self.responses = list(responses) if isinstance(responses, list) else [responses]
        self.chat_calls: list[list[dict]] = []

    def chat(self, messages: list[dict], **kwargs) -> str:
        self.chat_calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embedding is outside reconstruction adjudication")

    def health_check(self) -> bool:
        return True


def _response(
    candidate_id: str = "candidate-1",
    *,
    continuation_role: str = "continuation_row",
    row_action: str = "merge",
    table_action: str = "merge_fragments",
    confidence: float = 0.75,
    reason: str = "bounded structural evidence",
) -> str:
    return json.dumps(
        {
            "candidate_id": candidate_id,
            "continuation_role": continuation_role,
            "row_action": row_action,
            "table_action": table_action,
            "confidence": confidence,
            "reason": reason,
        }
    )


def _row(cells: tuple[str, ...], index: int, paragraph_id: str):
    row = split_markdown_table_row(
        "|" + "|".join(cells) + "|",
        SourceRowRef("section-1", paragraph_id, index),
    )
    assert row is not None
    return row


def make_medium_candidate(candidate_id: str = "candidate-1") -> ContinuationCandidate:
    cross_version_rows = tuple(
        _row((str(index + 20), "peer", f"context-{index}"), index, "peer")
        for index in range(5)
    )
    previous = _row(("12", "drive", "0.5 s"), 0, "left")
    continuation = _row(("", "", "within"), 0, "right")
    return ContinuationCandidate(
        candidate_id=candidate_id,
        side="baseline",
        previous_row=previous,
        continuation_row=continuation,
        next_full_row=_row(("13", "stop", "0.8 s"), 1, "right"),
        mapping=ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 0.9),
        previous_mapping=ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 1.0),
        previous_fragment_rows=(previous,),
        continuation_fragment_rows=(continuation,),
        evidence=("blank_key_cells", "textual_continuity"),
        conflicts=("weak_cross_version_match",),
        vetoes=(),
        cross_version_rows=cross_version_rows,
    )


def _context(candidate: ContinuationCandidate) -> TableBoundaryContext:
    def item(
        item_id: str,
        paragraph_id: str,
        sentence_index: int | None,
        page_no: int,
        kind: str,
        text: str,
    ) -> BoundaryContextItem:
        return BoundaryContextItem(
            item_id,
            "section-1",
            paragraph_id,
            sentence_index,
            page_no,
            kind,
            text,
        )

    return TableBoundaryContext(
        boundary_id="opaque-boundary",
        side="baseline",
        previous_page_no=1,
        next_page_no=2,
        inferred=False,
        items=(
            item("before-0", "before-0", None, 1, "paragraph", "before text 0"),
            item("before-1", "before-1", None, 1, "paragraph", "before text 1"),
            item("before-2", "before-2", None, 1, "paragraph", "before text 2"),
            item("before-3", "before-3", None, 1, "paragraph", "before text 3"),
            item("before-4", "before-4", None, 1, "paragraph", "before text 4"),
            item("previous", "left", 0, 1, "table_row", candidate.previous_row.raw_text),
            item("header-0", "header-0", 0, 2, "table_row", "|page header 0|"),
            item("header-1", "header-1", None, 2, "paragraph", "page header 1"),
            item("continuation", "right", 0, 2, "table_row", candidate.continuation_row.raw_text),
            item("next", "right", 1, 2, "table_row", candidate.next_full_row.raw_text),
            item("footer", "footer", None, 2, "paragraph", "page footer"),
            item("after", "after", 0, 2, "table_row", "|after row|"),
        ),
    )


@pytest.mark.parametrize(
    ("row_action", "table_action", "continuation_role", "decision"),
    [
        ("merge", "merge_fragments", "continuation_row", "merge"),
        ("keep", "merge_fragments", "new_business_row", "merge"),
        ("keep", "keep", "new_table", "keep_separate"),
        ("keep", "keep", "continuation_row", "keep_separate"),
    ],
)
def test_adjudicator_parses_candidate_bound_response(
    row_action,
    table_action,
    continuation_role,
    decision,
):
    provider = RecordingProvider(
        _response(
            row_action=row_action,
            table_action=table_action,
            continuation_role=continuation_role,
        )
    )

    judgment = adjudicate_continuation(make_medium_candidate(), provider)

    assert judgment is not None
    assert judgment.model == "recording-model"
    assert judgment.decision == decision
    assert judgment.roles == {"continuation": continuation_role}
    assert judgment.row_action == row_action
    assert judgment.table_action == table_action
    assert len(provider.chat_calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        _response("other"),
        _response(continuation_role="body_row"),
        _response(row_action="rewrite"),
        _response(table_action="rewrite"),
        _response(continuation_role="page_header", row_action="merge"),
        _response(continuation_role="new_table", table_action="merge_fragments"),
        _response(confidence=1.2),
        '{"candidate_id":"candidate-1","continuation_role":"continuation_row",'
        '"row_action":"merge","table_action":"merge_fragments","confidence":true,"reason":"invalid"}',
        '{"candidate_id":"candidate-1","continuation_role":"continuation_row",'
        '"row_action":"merge","table_action":"merge_fragments","confidence":"0.9","reason":"invalid"}',
        '{"candidate_id":"candidate-1","continuation_role":"continuation_row",'
        '"row_action":"merge","table_action":"merge_fragments","confidence":0.9}',
        '{"candidate_id":"candidate-1","continuation_role":"continuation_row",'
        '"row_action":"merge","table_action":"merge_fragments","confidence":0.9,"reason":"ok","extra":true}',
        '{"candidate_id":"other","candidate_id":"candidate-1",'
        '"continuation_role":"continuation_row","row_action":"merge",'
        '"table_action":"merge_fragments",'
        '"confidence":0.9,"reason":"duplicate"}',
        _response(reason=""),
        _response(reason="x" * 201),
        '```json\n' + _response() + '\n```',
        "prefix " + _response(),
        "[" + _response() + "]",
    ],
)
def test_adjudicator_rejects_non_strict_or_unbound_responses(response):
    provider = RecordingProvider([response, response])

    assert adjudicate_continuation(make_medium_candidate(), provider) is None
    assert len(provider.chat_calls) == 2


@pytest.mark.parametrize("error", [RuntimeError("offline"), TimeoutError("model timeout")])
def test_adjudicator_treats_provider_exceptions_as_candidate_local_failure(error):
    provider = RecordingProvider(error)

    assert adjudicate_continuation(make_medium_candidate(), provider) is None
    assert len(provider.chat_calls) == 1


def test_adjudicator_preserves_valid_sub_threshold_judgment():
    provider = RecordingProvider(_response(confidence=0.74))

    judgment = adjudicate_continuation(make_medium_candidate(), provider)

    assert judgment is not None
    assert judgment.decision == "merge"
    assert judgment.confidence == 0.74
    assert judgment.reason == "bounded structural evidence"


def test_adjudicator_sends_minimized_candidate_bound_payload():
    candidate = make_medium_candidate()
    provider = RecordingProvider(_response())

    adjudicate_continuation(candidate, provider, _context(candidate))

    assert len(provider.chat_calls) == 1
    messages = provider.chat_calls[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    payload = json.loads(messages[1]["content"])
    assert payload == {
        "candidate_id": "candidate-1",
        "candidate": {
            "previous": ["12", "drive", "0.5 s"],
            "continuation": ["", "", "within"],
            "next": ["13", "stop", "0.8 s"],
        },
        "nearby_context": {
            "before": [
                "before text 0",
                "before text 1",
                "before text 2",
                "before text 3",
                "before text 4",
            ],
            "after": [
                "|page header 0|",
                "page header 1",
                "|13|stop|0.8 s|",
                "page footer",
                "|after row|",
            ],
        },
        "peer_rows": [
            ["20", "peer", "context-0"],
            ["21", "peer", "context-1"],
        ],
    }
    serialized = messages[1]["content"]
    for removed_field in (
        "boundary_id",
        "side",
        "kind",
        "page_no",
        "physical_mapping",
        "mapping_candidates",
        "logical_column_roles",
        "rule_evidence",
        "rule_conflicts",
        "context_items",
    ):
        assert removed_field not in serialized


def test_adjudicator_omits_empty_optional_payload_fields():
    candidate = replace(
        make_medium_candidate(),
        next_full_row=None,
        cross_version_rows=(),
    )
    provider = RecordingProvider(_response())

    adjudicate_continuation(candidate, provider)

    payload = json.loads(provider.chat_calls[0][1]["content"])
    assert payload == {
        "candidate_id": "candidate-1",
        "candidate": {
            "previous": ["12", "drive", "0.5 s"],
            "continuation": ["", "", "within"],
        },
    }


def test_adjudicator_projects_sparse_previous_row_without_mapping_metadata():
    candidate = make_medium_candidate()
    sparse_previous = _row(("", "beta", "", "", "", "prefix"), 0, "left")
    candidate = replace(
        candidate,
        previous_row=sparse_previous,
        previous_mapping=ColumnMapping(
            (0, 1, 5),
            {0: 0, 1: 1, 5: 2},
            1.0,
        ),
        previous_fragment_rows=(sparse_previous,),
    )
    provider = RecordingProvider(_response())

    adjudicate_continuation(candidate, provider)

    payload = json.loads(provider.chat_calls[0][1]["content"])
    assert payload["candidate"]["previous"] == ["", "beta", "prefix"]
    assert "physical_mapping" not in payload


def test_adjudicator_sends_detailed_strict_system_contract():
    provider = RecordingProvider(_response())

    adjudicate_continuation(make_medium_candidate(), provider)

    prompt = provider.chat_calls[0][0]["content"]
    for required_instruction in (
        "candidate_id",
        "continuation_role",
        "row_action",
        "table_action",
        "confidence",
        "reason",
    ):
        assert required_instruction in prompt
    payload = json.loads(provider.chat_calls[0][-1]["content"])
    assert set(payload["candidate"]) >= {"previous", "continuation"}
    assert payload["candidate_id"] == make_medium_candidate().candidate_id


def test_adjudicator_retries_invalid_response_once():
    provider = RecordingProvider(
        [
            '{"row_action":"merge"}',
            _response(confidence=0.91, reason="corrected strict response"),
        ]
    )

    judgment = adjudicate_continuation(make_medium_candidate(), provider)

    assert judgment is not None
    assert judgment.confidence == 0.91
    assert len(provider.chat_calls) == 2


def test_adjudicator_retries_invalid_json_once():
    provider = RecordingProvider(
        [
            '{"candidate_id":',
            _response(confidence=0.91, reason="corrected strict response"),
        ]
    )

    judgment = adjudicate_continuation(make_medium_candidate(), provider)

    assert judgment is not None
    assert judgment.confidence == 0.91
    assert len(provider.chat_calls) == 2
