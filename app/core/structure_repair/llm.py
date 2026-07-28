"""Strict local LLM adjudication for ambiguous structure candidates."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core.types import Paragraph, Section


_RESPONSE_FIELDS = {
    "candidate_id",
    "action",
    "source_ids",
    "target_section_id",
    "confidence",
    "reason",
}
_ALLOWED_ACTIONS = {"keep", "move_to_section"}
_MIN_CONFIDENCE = 0.85


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
    if not isinstance(data, dict) or set(data) != _RESPONSE_FIELDS:
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
) -> tuple[ParagraphMergeJudgment | None, str]:
    """Return a validated merge judgment for two existing paragraph IDs."""
    candidate_id = f"paragraphs:{previous.paragraph_id}:{following.paragraph_id}"
    payload = {
        "candidate_id": candidate_id,
        "allowed_actions": ["keep", "merge_paragraphs"],
        "section": {
            "section_id": section.section_id,
            "title": section.title,
            "level": section.level,
        },
        "source_paragraphs": [
            {
                "paragraph_id": previous.paragraph_id,
                "page_no": previous.page_no,
                "text": previous.text[:800],
            },
            {
                "paragraph_id": following.paragraph_id,
                "page_no": following.page_no,
                "text": following.text[:800],
            },
        ],
        "context": [
            {
                "paragraph_id": paragraph.paragraph_id,
                "page_no": paragraph.page_no,
                "text": paragraph.text[:800],
            }
            for paragraph in context[:6]
        ],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Judge whether two adjacent parsed paragraphs are fragments of one original "
                "paragraph. Return exactly one JSON object with these fields:\n"
                "- candidate_id: the candidate identifier, e.g. \"paragraphs:xxx:yyy\" (string)\n"
                "- action: exactly \"keep\" or \"merge_paragraphs\"\n"
                "- source_ids: a JSON array containing exactly two elements: the previous paragraph_id "
                "and the following paragraph_id (without prefix)\n"
                "- target_section_id: the section_id of the section containing these paragraphs (string, without prefix)\n"
                "- confidence: a number between 0.0 and 1.0\n"
                "- reason: a non-empty string explaining your decision\n"
                "Use only supplied IDs. Never rewrite, summarize, or generate document text."
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
    if not isinstance(data, dict) or set(data) != _RESPONSE_FIELDS:
        return None, "invalid_response"
    if (
        data["candidate_id"] != candidate_id
        or data["action"] not in {"keep", "merge_paragraphs"}
        or data["source_ids"] != [previous.paragraph_id, following.paragraph_id]
        or data["target_section_id"] != section.section_id
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
        ParagraphMergeJudgment(
            action=str(data["action"]),
            confidence=float(confidence),
            reason=str(data["reason"]),
        ),
        "",
    )
