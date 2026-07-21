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
    return ContinuationCandidate(
        candidate_id=candidate_id,
        side="baseline",
        previous_row=_row(("12", "drive", "0.5 s"), 0, "left"),
        continuation_row=_row(("", "", "within"), 0, "right"),
        next_full_row=_row(("13", "stop", "0.8 s"), 1, "right"),
        mapping=ColumnMapping((0, 1, 2), {0: 0, 1: 1, 2: 2}, 0.9),
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
        json.dumps(
            {
                "candidate_id": "candidate-1",
                "decision": decision,
                "confidence": confidence,
                "reason": "cells continue",
            }
        )
    )

    judgment = adjudicate_continuation(candidate, provider)

    assert judgment is not None
    assert judgment.model == "recording-model"
    assert judgment.decision == decision
    assert judgment.confidence == confidence
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
    assert len(provider.chat_calls) == 1


@pytest.mark.parametrize("error", [RuntimeError("offline"), TimeoutError("model timeout")])
def test_adjudicator_treats_provider_exceptions_as_candidate_local_failure(error):
    provider = RecordingProvider(error)

    assert adjudicate_continuation(make_medium_candidate(), provider) is None
    assert len(provider.chat_calls) == 1


def test_adjudicator_preserves_valid_sub_threshold_judgment():
    provider = RecordingProvider(
        '{"candidate_id":"candidate-1","decision":"merge","confidence":0.74,"reason":"plausible"}'
    )

    judgment = adjudicate_continuation(make_medium_candidate(), provider)

    assert judgment is not None
    assert judgment.decision == "merge"
    assert judgment.confidence == 0.74
    assert judgment.reason == "plausible"


def test_adjudicator_sends_only_bounded_structural_context():
    provider = RecordingProvider(
        '{"candidate_id":"candidate-1","decision":"merge","confidence":0.75,"reason":"continues"}'
    )

    adjudicate_continuation(make_medium_candidate(), provider)

    assert len(provider.chat_calls) == 1
    messages = provider.chat_calls[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    payload = json.loads(messages[1]["content"])
    assert set(payload) == {
        "candidate_id",
        "side",
        "previous_cells",
        "continuation_cells",
        "next_cells",
        "logical_column_roles",
        "physical_mapping",
        "rule_evidence",
        "rule_conflicts",
        "cross_version_rows",
    }
    assert len(payload["cross_version_rows"]) == 3
    assert "paragraph_id" not in messages[1]["content"]


def test_adjudicator_projects_sparse_previous_row_to_logical_cells():
    candidate = make_medium_candidate()
    sparse_previous = _row(("", "12", "", "drive", "", "0.5 s"), 0, "left")
    candidate = replace(candidate, previous_row=sparse_previous)
    provider = RecordingProvider(
        '{"candidate_id":"candidate-1","decision":"merge","confidence":0.75,"reason":"continues"}'
    )

    adjudicate_continuation(candidate, provider)

    payload = json.loads(provider.chat_calls[0][1]["content"])
    assert payload["previous_cells"] == ["12", "drive", "0.5 s"]
    assert len(payload["logical_column_roles"]) == 3
