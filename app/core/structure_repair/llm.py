"""Strict local LLM adjudication for ambiguous structure candidates."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core.llm_call_budget import LLMCallBudget
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
_PAGE_NOISE_RESPONSE_FIELDS = {"boundary_id", "labels"}
_PAGE_NOISE_LABEL_FIELDS = {"id", "action", "confidence", "reason"}
_ALLOWED_ACTIONS = {"keep", "move_to_section"}
_MIN_CONFIDENCE = 0.85
_PAGE_NOISE_MIN_CONFIDENCE = 0.90
_MAX_REASON_LENGTH = 200
_PAGE_NOISE_SYSTEM_MESSAGE = """You classify fixed text items near one physical page boundary.

Task definition:
- For every supplied item, decide whether it is printed page-header or page-footer
  furniture that should be removed from a normalized document copy.
- Page furniture is position-dependent printed material outside the document's logical
  content. It commonly has a repeated role across pages. Boundary position alone is
  not sufficient evidence.
- Judge the complete boundary batch together. Each item is an action target and must
  receive exactly one label. Never omit, add, reorder, replace, merge, or split items.
- Do not rewrite, summarize, correct, generate, merge, or relocate document text.

Content safety rules:
- Digits, standards, or document identifiers are NOT noise if they appear within the main body rows, 
  section headings, or table cells. However, if they appear only at the extreme top or bottom edges 
  and are repeated, they are classic page furniture and should be removed.
- A row with an ordinal or key column plus requirement descriptions, performance
  targets, standards, references, or attachment filenames is strong business content.
  Keep it even when it contains numbers, .docx, Markdown pipes, empty cells, HTML
  <br> tags, or manual line breaks.
- Keep real section headings, item headings, body text, notes, business column-name
  headers, business table data, and cross-page continuations.
- Repeated printed metadata laid out as a table may still be page noise, but table
  syntax, short length, or empty cells alone do not make it page noise.
- Repetition + Boundary proximity (top/bottom) + Generic descriptive text (e.g., 'Page', 'Doc ID', 'Confidential', company name)
  is STRONG evidence for page noise, regardless of whether it contains numbers or identifiers.
  Remove these unless they are part of a continuous data table.
- When evidence is strictly balanced between content and furniture, 
  favor keep. However, if an item is clearly identified as repeated positioning metadata
  (e.g., page numbers, document titles, copyright notices, or dates at the exact top/bottom),
  the evidence is NOT ambiguous. You MUST remove it.
- Empty-cell handling for data rows: When a table row contains many empty cells but the populated cells contain specific nouns,
  codes, or unique values (e.g., "Product-A", "John Doe", "100kg"),
  it is a legitimate data record with blank optional fields.
  Its content itself may be related to the previous text. Keep it.

Input contract:
- boundary_id is an opaque fixed string. Copy it exactly.
- items is the complete fixed list. Every item has exactly id, position, and text.
- id is request-local and opaque. Copy every id exactly and in the same order.
- position is only boundary evidence. It is one of previous_page_end,
  next_page_start, document_start, or document_end.

Output contract:
- Return exactly one bare JSON object with exactly boundary_id and labels.
- labels must contain exactly one object per input item in the same order.
- Every label must contain exactly id, action, confidence, and reason.
- action must be exactly remove_as_page_noise or keep.
- confidence must be a JSON number from 0.0 through 1.0 inclusive. It cannot be a
  quoted number, boolean, null, NaN, or infinity.
- reason must be a non-empty JSON string of at most 200 characters and discuss only
  the corresponding fixed item.
- Do not use Markdown fences or prose outside the JSON object. Do not add fields.

Valid output shape:
{"boundary_id":"copy-exactly","labels":[{"id":"L01","action":"keep","confidence":0.99,"reason":"The numbered row contains a business requirement and target."}]}
"""
_PAGE_NOISE_REVIEW_SYSTEM_MESSAGE = _PAGE_NOISE_SYSTEM_MESSAGE + """

Review task:
- This is the final safety review of the supplied initial_labels for the same fixed
  items. Re-evaluate every item against the original text and full boundary batch.
- Pay special attention to false deletions of numbered business rows, requirements,
  targets, standards, references, and attachment filenames.
- Return a complete replacement label list. Do not merely approve the initial result.
- A candidate is removed only when both rounds independently return a valid,
  high-confidence remove_as_page_noise label. When uncertain, return keep.
"""
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
class PageNoiseLabel:
    item_id: str
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


def _parse_page_noise_batch_response(
    raw: str,
    boundary_id: str,
    items: list[dict[str, str]],
) -> tuple[list[PageNoiseLabel] | None, str]:
    try:
        data: Any = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_members,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(data, dict) or set(data) != _PAGE_NOISE_RESPONSE_FIELDS:
        return None, "invalid_fields"
    if data["boundary_id"] != boundary_id:
        return None, "boundary_id_mismatch"
    labels = data["labels"]
    if not isinstance(labels, list) or len(labels) != len(items):
        return None, "label_set_mismatch"
    expected_ids = [item["id"] for item in items]
    parsed: list[PageNoiseLabel] = []
    for expected_id, label in zip(expected_ids, labels, strict=True):
        if not isinstance(label, dict) or set(label) != _PAGE_NOISE_LABEL_FIELDS:
            return None, "invalid_label_fields"
        if label["id"] != expected_id:
            return None, "label_set_mismatch"
        if label["action"] not in {"keep", "remove_as_page_noise"}:
            return None, "invalid_action"
        confidence = label["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None, "invalid_confidence"
        if not 0.0 <= float(confidence) <= 1.0:
            return None, "invalid_confidence"
        reason = label["reason"]
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > _MAX_REASON_LENGTH
        ):
            return None, "invalid_reason"
        action = str(label["action"])
        if (
            action == "remove_as_page_noise"
            and float(confidence) < _PAGE_NOISE_MIN_CONFIDENCE
        ):
            action = "keep"
        parsed.append(
            PageNoiseLabel(
                item_id=expected_id,
                action=action,
                confidence=float(confidence),
                reason=reason,
            )
        )
    return parsed, ""


def _chat_and_parse_with_validation_retry(
    provider: object,
    messages: list[dict[str, str]],
    model: str,
    parse: Any,
    call_budget: LLMCallBudget | None = None,
) -> tuple[Any | None, str]:
    budget = call_budget or LLMCallBudget()
    for attempt in range(2):
        if not budget.consume():
            return None, "llm_call_budget_exhausted"
        try:
            raw = _chat(provider, messages, model)
        except Exception:
            return None, "llm_error"
        parsed, failure_code = parse(raw)
        if parsed is not None or failure_code == "low_confidence":
            return parsed, failure_code
        if attempt == 1 or budget.remaining <= 0:
            return None, failure_code
    return None, "invalid_response"


def adjudicate_page_noise_batch(
    boundary_id: str,
    items: list[dict[str, str]],
    provider: object,
    model: str = "",
    call_budget: LLMCallBudget | None = None,
) -> tuple[list[PageNoiseLabel] | None, str]:
    """Label one fixed page-boundary batch with strict, fail-closed validation."""
    payload: dict[str, object] = {
        "boundary_id": boundary_id,
        "items": [
            {
                "id": item["id"],
                "position": item["position"],
                "text": item["text"][:1200],
            }
            for item in items
        ],
    }

    messages = [
        {"role": "system", "content": _PAGE_NOISE_SYSTEM_MESSAGE},
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
    return _chat_and_parse_with_validation_retry(
        provider,
        messages,
        model,
        lambda raw: _parse_page_noise_batch_response(raw, boundary_id, items),
        call_budget,
    )


def review_page_noise_batch(
    boundary_id: str,
    items: list[dict[str, str]],
    initial_labels: list[PageNoiseLabel],
    provider: object,
    model: str = "",
    call_budget: LLMCallBudget | None = None,
) -> tuple[list[PageNoiseLabel] | None, str]:
    """Review one fixed batch within the caller's shared LLM call budget."""
    payload: dict[str, object] = {
        "boundary_id": boundary_id,
        "items": [
            {
                "id": item["id"],
                "position": item["position"],
                "text": item["text"][:1200],
            }
            for item in items
        ],
        "initial_labels": [
            {
                "id": label.item_id,
                "action": label.action,
                "confidence": label.confidence,
                "reason": label.reason,
            }
            for label in initial_labels
        ],
    }
    messages = [
        {
            "role": "system",
            "content": _PAGE_NOISE_REVIEW_SYSTEM_MESSAGE,
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
    review_items = [
        {"id": item["id"], "position": item["position"], "text": item["text"]}
        for item in items
    ]
    return _chat_and_parse_with_validation_retry(
        provider,
        messages,
        model,
        lambda raw: _parse_page_noise_batch_response(
            raw,
            boundary_id,
            review_items,
        ),
        call_budget,
    )


def adjudicate_section_parent(
    section: Section,
    previous: Section,
    provider: object,
    model: str = "",
    call_budget: LLMCallBudget | None = None,
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
    def parse(raw: str) -> tuple[SectionMoveJudgment | None, str]:
        try:
            data: Any = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_members,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, "invalid_json"
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

    return _chat_and_parse_with_validation_retry(
        provider,
        messages,
        model,
        parse,
        call_budget,
    )


def adjudicate_paragraph_merge(
    section: Section,
    previous: Paragraph,
    following: Paragraph,
    context: list[Paragraph],
    provider: object,
    model: str = "",
    rule_evidence: tuple[str, ...] = (),
    document_title: str = "",
    section_path: tuple[str, ...] = (),
    call_budget: LLMCallBudget | None = None,
) -> tuple[ParagraphMergeJudgment | None, str]:
    """Return a validated merge judgment for two existing paragraph IDs."""
    candidate_id = f"paragraphs:{previous.paragraph_id}:{following.paragraph_id}"
    payload = {
        "candidate_id": candidate_id,
        "candidate": {
            "previous": previous.text[:800],
            "continuation": following.text[:800],
        },
        "paragraph_context": {
            "document_title": document_title,
            "section_path": list(section_path) or [section.title],
            "current_section_title": section.title,
            "section_level": section.level,
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

    return _chat_and_parse_with_validation_retry(
        provider,
        messages,
        model,
        parse,
        call_budget,
    )
