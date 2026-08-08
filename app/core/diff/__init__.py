"""Diff engine public interface.

Top-level entry point matching the design spec:
    compare(baseline, target, policy) -> DiffResult
"""
from __future__ import annotations

import re
import uuid

from app.core.types import ComparePolicy, DiffResult, DocumentIR
from app.core.model.base_provider import BaseProvider


class _JaccardEmbedder:
    """Fallback embedder using lightweight lexical overlap vectors.

    Used when no real embedder is provided so compare() remains callable in
    tests and offline scenarios without a running model server.
    """

    def _features(self, text: str) -> set[tuple[str, str]]:
        compact = re.sub(r"\s+", "", text.lower())
        useful_chars = [ch for ch in compact if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"]
        features: set[tuple[str, str]] = {("char", ch) for ch in useful_chars}
        features.update(
            ("bigram", "".join(pair))
            for pair in zip(useful_chars, useful_chars[1:])
        )
        features.update(
            ("token", token)
            for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower())
        )
        return features

    def embed(self, texts: list[str]) -> list[list[float]]:
        feature_sets: list[set[tuple[str, str]]] = []
        vocab: set = set()
        for text in texts:
            features = self._features(text)
            feature_sets.append(features)
            vocab |= features

        if not vocab:
            return [[0.0]] * len(texts)

        vocab_list = sorted(vocab)
        return [
            [1.0 if feature in features else 0.0 for feature in vocab_list]
            for features in feature_sets
        ]

    def chat(self, messages, **kwargs) -> str:  # pragma: no cover
        raise NotImplementedError

    def health_check(self) -> bool:  # pragma: no cover
        return True


def compare(
    baseline: DocumentIR,
    target: DocumentIR,
    policy: ComparePolicy | None = None,
    *,
    embedder: BaseProvider | None = None,
    provider: BaseProvider | None = None,
) -> DiffResult:
    """Run the full three-stage semantic diff pipeline.

    Stage 1 — align scopes:     compare-only logical section alignment
    Stage 2 — match_paragraphs: scoped semantic paragraph matching
    Stage 3 — classify:         LLM + rule-based diff type classification

    When embedder is None a character-bigram Jaccard embedder is used so the
    function is callable in tests without a real model server (lower accuracy).
    When provider is None or policy.use_llm_classify is False, rule-based
    classification is used for Stage 3.
    """
    from app.core.diff.section_scope_aligner import align_compare_scopes
    from app.core.diff.semantic_matcher import match_paragraphs
    from app.core.diff.diff_classifier import classify
    from dataclasses import replace as dc_replace

    if policy is None:
        policy = ComparePolicy()

    effective_embedder: BaseProvider = embedder if embedder is not None else _JaccardEmbedder()  # type: ignore[assignment]

    # Disable LLM classification when no provider is available
    effective_policy = policy
    if provider is None and policy.use_llm_classify:
        effective_policy = dc_replace(policy, use_llm_classify=False)

    alignment_plan = align_compare_scopes(
        baseline,
        target,
        effective_embedder,
        similarity_threshold=effective_policy.similarity_threshold,
    )
    para_pairs = match_paragraphs(
        alignment_plan,
        effective_embedder,
        effective_policy.similarity_threshold,
        rerank_provider=provider if effective_policy.use_llm_match else None,
        use_llm_rerank=effective_policy.use_llm_match,
    )
    return classify(
        para_pairs,
        policy=effective_policy,
        provider=provider,
        task_id=str(uuid.uuid4()),
        baseline_version_id=baseline.doc_id,
        target_version_id=target.doc_id,
    )
