"""Strict local LLM adjudication for ambiguous structure candidates."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core.types import Paragraph, Section


_SECTION_RESPONSE_FIELDS = {
    "candidate_id",
    "action",
    "source_ids",
    "target_section_id",
    "confidence",
    "reason",
}
_PARAGRAPH_RESPONSE_FIELDS = {
    "candidate_id",
    "action",
    "confidence",
    "reason",
}
_TABLE_FRAGMENT_RESPONSE_FIELDS = {
    "candidate_id",
    "action",
    "confidence",
    "reason",
}
_ALLOWED_ACTIONS = {"keep", "move_to_section"}
_MIN_CONFIDENCE = 0.85
_MAX_REASON_LENGTH = 200
_PARAGRAPH_SYSTEM_MESSAGE = """You are a document paragraph-continuation classifier.

Task boundary:
- Judge only whether candidate.continuation is the continuation of candidate.previous
  and both belong to one original paragraph.
- candidate.previous and candidate.continuation are fixed slots. Never replace either
  slot with text from nearby_context.
- nearby_context and rule_evidence are background evidence only.
- Do not rewrite, summarize, correct, remove, or generate document text.

Return exactly one JSON object with exactly these four fields:
candidate_id, action, confidence, reason.

Field contract:
- candidate_id: JSON string. Copy the supplied candidate_id exactly.
- action: JSON string. It must be exactly merge_paragraphs or keep.
- confidence: JSON number from 0.0 through 1.0 inclusive. Do not return a quoted
  number, boolean, null, NaN, or infinity.
- reason: non-empty JSON string of at most 200 characters. Explain only the
  relationship between the fixed previous and continuation slots.

Strict JSON rules:
- Output one bare JSON object and nothing before or after it.
- Do not use Markdown fences or prose outside the object.
- Do not omit, add, or duplicate fields.
- Field names and enum values are case-sensitive.
- Never invent or modify candidate_id.

Valid output example:
{"candidate_id":"copy-exactly","action":"merge_paragraphs","confidence":0.92,"reason":"The continuation completes the unfinished sentence."}
"""
_TABLE_FRAGMENT_SYSTEM_MESSAGE = """You are a cross-page table-fragment classifier.

Task boundary:
- Judge only the two fixed table fragments in candidate.previous and
  candidate.continuation.
- merge_fragments means the continuation fragment starts with one or more new logical
  data rows of the same table after a repeated header.
- Return keep when they are separate tables, or when the first business row of the
  continuation fragment may continue the last business row of the previous fragment;
  row continuation is handled by a separate row-level classifier.
- Never replace either slot, rewrite cells, invent rows, or choose a column mapping.

Return exactly one JSON object with exactly these four fields:
candidate_id, action, confidence, reason.

Field contract:
- candidate_id: JSON string. Copy the supplied candidate_id exactly.
- action: JSON string. It must be exactly merge_fragments or keep.
- confidence: JSON number from 0.0 through 1.0 inclusive. Do not return a quoted
  number, boolean, null, NaN, or infinity.
- reason: non-empty JSON string of at most 200 characters. Explain only the
  relationship between the two fixed fragments.

Strict JSON rules:
- Output one bare JSON object and nothing before or after it.
- Do not use Markdown fences or prose outside the object.
- Do not omit, add, or duplicate fields.
- Field names and enum values are case-sensitive.
- Never invent or modify candidate_id.

Valid output example:
{"candidate_id":"copy-exactly","action":"merge_fragments","confidence":0.94,"reason":"The repeated header is followed by new rows of the same table."}
"""


@dataclass(frozen=True)
class SectionMoveJudgment:
    action: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class ParagraphMergeJudgment:
    action: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class TableFragmentMergeJudgment:
    action: str
    confidence: float
    reason: str


def _reject_duplicate_members(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    data: dict[str, object] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"duplicate JSON member: {key}")
        data[key] = value
    return data


def _chat(provider: object, messages: list[dict[str, str]], model: str) -> str:
    chat = getattr(provider, "chat", None)
    if callable(chat):
        return str(chat(messages))

    completions = getattr(getattr(provider, "chat", None), "completions", None)
    create = getattr(completions, "create", None)
    if not callable(create):
        raise TypeError("structure repair provider does not expose a supported chat API")
    response = create(model=model, messages=messages, temperature=0)
    return str(response.choices[0].message.content or "")


def adjudicate_section_parent(
    section: Section,
    previous: Section,
    provider: object,
    model: str = "",
) -> tuple[SectionMoveJudgment | None, str]:
    """Return a validated move judgment and a stable rejection code."""
    candidate_id = f"section:{section.section_id}"
    payload = {
        "candidate_id": candidate_id,
        "allowed_actions": sorted(_ALLOWED_ACTIONS),
        "candidate": {
            "section_id": section.section_id,
            "title": section.title,
            "level": section.level,
            "paragraphs": [paragraph.text[:800] for paragraph in section.paragraphs[:3]],
        },
        "previous_section": {
            "section_id": previous.section_id,
            "title": previous.title,
            "level": previous.level,
            "paragraphs": [paragraph.text[:800] for paragraph in previous.paragraphs[-3:]],
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Judge whether the candidate unnumbered section belongs under the previous "
                "numbered section. Return exactly one JSON object with these fields:\n"
                "- candidate_id: the candidate identifier, e.g. \"section:xxx\" (string)\n"
                "- action: exactly \"keep\" or \"move_to_section\"\n"
                "- source_ids: a JSON array containing exactly one element, the candidate_id value\n"
                "- target_section_id: the previous section's section_id (string, without prefix)\n"
                "- confidence: a number between 0.0 and 1.0\n"
                "- reason: a non-empty string explaining your decision\n"
                "Use only supplied IDs. Never rewrite or generate document text."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    try:
        raw = _chat(provider, messages, model)
        data: Any = json.loads(raw)
    except Exception:
        return None, "llm_error"
    if not isinstance(data, dict) or set(data) != _SECTION_RESPONSE_FIELDS:
        return None, "invalid_response"
    if (
        data["candidate_id"] != candidate_id
        or data["action"] not in _ALLOWED_ACTIONS
        or data["source_ids"] != [section.section_id]
        or data["target_section_id"] != previous.section_id
        or not isinstance(data["reason"], str)
        or not data["reason"].strip()
    ):
        return None, "invalid_response"
    confidence = data["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None, "invalid_response"
    if not 0.0 <= float(confidence) <= 1.0:
        return None, "invalid_response"
    if float(confidence) < _MIN_CONFIDENCE:
        return None, "low_confidence"
    return (
        SectionMoveJudgment(
            action=str(data["action"]),
            confidence=float(confidence),
            reason=str(data["reason"]),
        ),
        "",
    )


def adjudicate_paragraph_merge(
    section: Section,
    previous: Paragraph,
    following: Paragraph,
    context: list[Paragraph],
    provider: object,
    model: str = "",
    rule_evidence: tuple[str, ...] = (),
) -> tuple[ParagraphMergeJudgment | None, str]:
    """Return a validated merge judgment for two existing paragraph IDs."""
    candidate_id = f"paragraphs:{previous.paragraph_id}:{following.paragraph_id}"
    payload = {
        "candidate_id": candidate_id,
        "candidate": {
            "previous": previous.text[:800],
            "continuation": following.text[:800],
        },
    }
    if previous.page_no is not None or following.page_no is not None:
        payload["pages"] = [previous.page_no, following.page_no]
    previous_index = next(
        (
            index
            for index, paragraph in enumerate(context)
            if paragraph.paragraph_id == previous.paragraph_id
        ),
        None,
    )
    following_index = next(
        (
            index
            for index, paragraph in enumerate(context)
            if paragraph.paragraph_id == following.paragraph_id
        ),
        None,
    )
    nearby: dict[str, list[str]] = {}
    if previous_index is not None:
        before = [paragraph.text[:800] for paragraph in context[:previous_index]][-6:]
        if before:
            nearby["before"] = before
    if following_index is not None:
        after = [paragraph.text[:800] for paragraph in context[following_index + 1 :]][:6]
        if after:
            nearby["after"] = after
    if nearby:
        payload["nearby_context"] = nearby
    if rule_evidence:
        payload["rule_evidence"] = list(rule_evidence)
    messages = [
        {"role": "system", "content": _PARAGRAPH_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    def parse(raw: str) -> tuple[ParagraphMergeJudgment | None, str]:
        try:
            data: Any = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_members,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, "invalid_json"
        if not isinstance(data, dict) or set(data) != _PARAGRAPH_RESPONSE_FIELDS:
            return None, "invalid_fields"
        if data["candidate_id"] != candidate_id:
            return None, "candidate_id_mismatch"
        if data["action"] not in {"keep", "merge_paragraphs"}:
            return None, "invalid_action"
        confidence = data["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None, "invalid_confidence"
        if not 0.0 <= float(confidence) <= 1.0:
            return None, "invalid_confidence"
        reason = data["reason"]
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > _MAX_REASON_LENGTH
        ):
            return None, "invalid_reason"
        if float(confidence) < _MIN_CONFIDENCE:
            return None, "low_confidence"
        return (
            ParagraphMergeJudgment(
                action=str(data["action"]),
                confidence=float(confidence),
                reason=reason,
            ),
            "",
        )

    try:
        raw = _chat(provider, messages, model)
    except Exception:
        return None, "llm_error"
    judgment, failure_code = parse(raw)
    if judgment is not None or failure_code == "low_confidence":
        return judgment, failure_code

    retry_messages = [
        {
            "role": "system",
            "content": (
                _PARAGRAPH_SYSTEM_MESSAGE
                + "\nThe previous response failed strict validation with code: "
                + failure_code
                + ". Return a complete replacement JSON object for the same fixed slots."
            ),
        },
        messages[1],
    ]
    try:
        raw = _chat(provider, retry_messages, model)
    except Exception:
        return None, "llm_error"
    return parse(raw)


def adjudicate_table_fragment_merge(
    previous: Paragraph,
    following: Paragraph,
    provider: object,
    model: str = "",
) -> tuple[TableFragmentMergeJudgment | None, str]:
    """Return a strict judgment for one fixed same-schema fragment pair."""
    candidate_id = f"tables:{previous.paragraph_id}:{following.paragraph_id}"
    previous_rows = [line.strip() for line in previous.text.splitlines() if line.strip()]
    following_rows = [line.strip() for line in following.text.splitlines() if line.strip()]
    payload = {
        "candidate_id": candidate_id,
        "candidate": {
            "previous": previous_rows[-3:],
            "continuation": following_rows[:4],
        },
    }
    if previous.page_no is not None or following.page_no is not None:
        payload["pages"] = [previous.page_no, following.page_no]
    user_message = {
        "role": "user",
        "content": json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }

    def parse(raw: str) -> tuple[TableFragmentMergeJudgment | None, str]:
        try:
            data: Any = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_members,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, "invalid_json"
        if not isinstance(data, dict) or set(data) != _TABLE_FRAGMENT_RESPONSE_FIELDS:
            return None, "invalid_fields"
        if data["candidate_id"] != candidate_id:
            return None, "candidate_id_mismatch"
        if data["action"] not in {"keep", "merge_fragments"}:
            return None, "invalid_action"
        confidence = data["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None, "invalid_confidence"
        if not 0.0 <= float(confidence) <= 1.0:
            return None, "invalid_confidence"
        reason = data["reason"]
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > _MAX_REASON_LENGTH
        ):
            return None, "invalid_reason"
        if float(confidence) < _MIN_CONFIDENCE:
            return None, "low_confidence"
        return (
            TableFragmentMergeJudgment(
                action=str(data["action"]),
                confidence=float(confidence),
                reason=reason,
            ),
            "",
        )

    messages = [
        {"role": "system", "content": _TABLE_FRAGMENT_SYSTEM_MESSAGE},
        user_message,
    ]
    try:
        raw = _chat(provider, messages, model)
    except Exception:
        return None, "llm_error"
    judgment, failure_code = parse(raw)
    if judgment is not None or failure_code == "low_confidence":
        return judgment, failure_code
    retry_messages = [
        {
            "role": "system",
            "content": (
                _TABLE_FRAGMENT_SYSTEM_MESSAGE
                + "\nThe previous response failed strict validation with code: "
                + failure_code
                + ". Return a complete replacement JSON object for the same fixed slots."
            ),
        },
        user_message,
    ]
    try:
        raw = _chat(provider, retry_messages, model)
    except Exception:
        return None, "llm_error"
    return parse(raw)
