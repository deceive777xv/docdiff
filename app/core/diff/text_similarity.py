"""Shared lexical and rule-aware text similarity helpers for compare."""
from __future__ import annotations

import re


_RULE_PATTERNS = (
    re.compile(r"\d+[\.,]\d*"),
    re.compile(r"[不无未没]"),
    re.compile(r"(?:应|须|必须|不得|禁止)"),
)
_LEXICAL_TOKEN_RE = re.compile(
    r"[a-z]+|\d+(?:[\.,]\d+)?%?|[\u4e00-\u9fff]",
    re.IGNORECASE,
)
_NUMERIC_TOKEN_RE = re.compile(r"\d+(?:[\.,]\d+)?%?")


def rule_score_delta(text_a: str, text_b: str) -> float:
    """Return a small penalty when important rule patterns differ."""
    score = 0.0
    for pattern in _RULE_PATTERNS:
        if set(pattern.findall(text_a)) != set(pattern.findall(text_b)):
            score += 0.067
    return score


def _set_cosine(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / ((len(left) * len(right)) ** 0.5)


def lexical_similarity(text_a: str, text_b: str) -> float:
    """Return lexical overlap using Latin words, numbers, and CJK characters."""
    tokens_a = set(_LEXICAL_TOKEN_RE.findall(text_a.lower()))
    tokens_b = set(_LEXICAL_TOKEN_RE.findall(text_b.lower()))
    token_similarity = _set_cosine(tokens_a, tokens_b)

    non_numeric_a = {
        token for token in tokens_a if not _NUMERIC_TOKEN_RE.fullmatch(token)
    }
    non_numeric_b = {
        token for token in tokens_b if not _NUMERIC_TOKEN_RE.fullmatch(token)
    }
    if len(non_numeric_a) >= 2 and len(non_numeric_b) >= 2:
        token_similarity = max(
            token_similarity,
            _set_cosine(non_numeric_a, non_numeric_b) * 0.95,
        )
    return token_similarity
