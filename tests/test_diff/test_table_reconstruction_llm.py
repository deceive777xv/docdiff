from __future__ import annotations

import json
from dataclasses import replace

import pytest

from app.core.diff.reconstruction_trace import SourceRowRef
from app.core.diff.table_reconstruction import (
    ColumnMapping,
    ContinuationCandidate,
    split_markdown_table_row,
)
from app.core.diff.table_reconstruction_llm import adjudicate_continuation
from app.core.model.base_provider import BaseProvider


class RecordingProvider(BaseProvider):
    chat_model = "recording-model"

    def __init__(self, response: str | BaseException):
        self.response = response
        self.chat_calls: list[list[dict]] = []

    def chat(self, messages: list[dict], **kwargs) -> str:
        self.chat_calls.append(messages)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embedding is outside reconstruction adjudication")

    def health_check(self) -> bool:
        return True


def _response(
    boundary_id: str = "candidate-1",
    *,
    row_action: str = "merge",
    table_action: str = "merge_fragments",
    confidence: float = 0.75,
    roles: dict[str, str] | None = None,
) -> str:
    return json.dumps(
        {
            "boundary_id": boundary_id,
            "roles": roles
            or {
                "previous_row": "body_row",
                "continuation_row": "continuation_row",
            },
            "row_action": row_action,
            "table_action": table_action,
            "confidence": confidence,
            "reason": "bounded structural evidence",
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


@pytest.mark.parametrize(
    ("decision", "confidence"),
    [("merge", 0.75), ("keep_separate", 0.82)],
)
def test_adjudicator_parses_matching_valid_response(decision, confidence):
    candidate = make_medium_candidate("candidate-1")
    provider = RecordingProvider(
        _response(
            row_action="merge" if decision == "merge" else "keep",
            table_action="merge_fragments" if decision == "merge" else "keep",
            confidence=confidence,
        )
    )

    judgment = adjudicate_continuation(candidate, provider)

    assert judgment is not None
    assert judgment.model == "recording-model"
    assert judgment.decision == decision
    assert judgment.confidence == confidence
    assert judgment.roles["continuation_row"] == "continuation_row"
    assert len(provider.chat_calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        '{"candidate_id":"other","decision":"merge","confidence":0.99,"reason":"mismatch"}',
        '{"candidate_id":"candidate-1","decision":"rewrite","confidence":0.99,"reason":"invalid"}',
        '{"candidate_id":"candidate-1","decision":"merge","confidence":1.2,"reason":"invalid"}',
        '{"candidate_id":"candidate-1","decision":"merge","confidence":true,"reason":"invalid"}',
        '{"candidate_id":"candidate-1","decision":"merge","confidence":0.99}',
        '{"candidate_id":"candidate-1","decision":"merge","confidence":0.99,"reason":"ok","cells":["new"]}',
        '{"candidate_id":"other","candidate_id":"candidate-1","decision":"merge","confidence":0.99,"reason":"duplicate"}',
        '{"candidate_id":"candidate-1","decision":"keep_separate","decision":"merge","confidence":0.99,"reason":"duplicate"}',
        '{"candidate_id":"candidate-1","decision":"merge","confidence":1.2,"confidence":0.99,"reason":"duplicate"}',
        '{"candidate_id":"candidate-1","decision":"merge","confidence":0.99,"reason":"","reason":"duplicate"}',
        '```json\n{"candidate_id":"candidate-1","decision":"merge","confidence":0.99,"reason":"fenced"}\n```',
        'prefix {"candidate_id":"candidate-1","decision":"merge","confidence":0.99,"reason":"embedded"}',
        '[{"candidate_id":"candidate-1","decision":"merge","confidence":0.99,"reason":"array"}]',
    ],
)
def test_adjudicator_rejects_non_strict_responses(response):
    provider = RecordingProvider(response)

    assert adjudicate_continuation(make_medium_candidate(), provider) is None
    assert len(provider.chat_calls) in {1, 2}


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


def test_adjudicator_sends_only_bounded_structural_context():
    provider = RecordingProvider(_response())

    adjudicate_continuation(make_medium_candidate(), provider)

    assert len(provider.chat_calls) == 1
    messages = provider.chat_calls[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    payload = json.loads(messages[1]["content"])
    assert set(payload) == {
        "boundary_id",
        "candidate_id",
        "side",
        "previous_cells",
        "continuation_cells",
        "next_cells",
        "logical_column_roles",
        "physical_mapping",
        "mapping_candidates",
        "rule_evidence",
        "rule_conflicts",
        "cross_version_rows",
        "context_items",
    }
    assert len(payload["cross_version_rows"]) == 3
    assert "paragraph_id" not in messages[1]["content"]


def test_adjudicator_projects_sparse_previous_row_to_logical_cells():
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
    assert payload["previous_cells"] == ["", "beta", "prefix"]
    assert payload["physical_mapping"]["previous_logical_by_physical"] == [
        [0, 0],
        [1, 1],
        [5, 2],
    ]
    assert len(payload["logical_column_roles"]) == 3


def test_adjudicator_rejects_roles_outside_the_bounded_context():
    provider = RecordingProvider(
        _response(
            roles={
                "previous_row": "body_row",
                "continuation_row": "continuation_row",
                "invented-row": "ordinary_text",
            }
        )
    )

    assert adjudicate_continuation(make_medium_candidate(), provider) is None


def test_adjudicator_accepts_atomic_row_action_with_supplied_mapping_id():
    class AtomicProvider(RecordingProvider):
        def __init__(self):
            super().__init__("")

        def chat(self, messages: list[dict], **kwargs) -> str:
            self.chat_calls.append(messages)
            payload = json.loads(messages[-1]["content"])
            mapping_id = payload["mapping_candidates"][0]["mapping_id"]
            return json.dumps(
                {
                    "boundary_id": payload["boundary_id"],
                    "candidate_id": payload["candidate_id"],
                    "action": "merge_row",
                    "mapping_id": mapping_id,
                    "roles": {
                        "previous_row": "body_row",
                        "continuation_row": "continuation_row",
                    },
                    "confidence": 0.93,
                    "reason": "continuation content belongs to the previous logical row",
                }
            )

    provider = AtomicProvider()

    judgment = adjudicate_continuation(make_medium_candidate(), provider)

    assert judgment is not None
    assert judgment.decision == "merge"
    assert judgment.row_action == "merge"
    assert judgment.table_action == "merge_fragments"
    payload = json.loads(provider.chat_calls[0][-1]["content"])
    assert judgment.mapping_id == payload["mapping_candidates"][0]["mapping_id"]
    assert payload["mapping_candidates"][0]["logical_by_physical"] == [
        [0, 0],
        [1, 1],
        [2, 2],
    ]


def test_adjudicator_retries_once_after_invalid_atomic_response():
    class RetryProvider(RecordingProvider):
        def __init__(self):
            super().__init__("")

        def chat(self, messages: list[dict], **kwargs) -> str:
            self.chat_calls.append(messages)
            if len(self.chat_calls) == 1:
                return '{"action":"merge_row"}'
            payload = json.loads(messages[-1]["content"])
            return json.dumps(
                {
                    "boundary_id": payload["boundary_id"],
                    "candidate_id": payload["candidate_id"],
                    "action": "merge_row",
                    "mapping_id": payload["mapping_candidates"][0]["mapping_id"],
                    "roles": {
                        "previous_row": "body_row",
                        "continuation_row": "continuation_row",
                    },
                    "confidence": 0.91,
                    "reason": "corrected strict response",
                }
            )

    provider = RetryProvider()

    judgment = adjudicate_continuation(make_medium_candidate(), provider)

    assert judgment is not None
    assert judgment.decision == "merge"
    assert len(provider.chat_calls) == 2
    assert "validation" in provider.chat_calls[1][0]["content"].lower()
