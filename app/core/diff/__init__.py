"""Diff engine public interface.

Top-level entry point matching the design spec:
    compare(baseline, target, policy) -> DiffResult
"""
from __future__ import annotations

import uuid

from app.core.types import ComparePolicy, DiffResult, DocumentIR
from app.core.model.base_provider import BaseProvider


class _JaccardEmbedder:
    """Fallback embedder using character-bigram overlap vectors.

    Used when no real embedder is provided so compare() remains callable in
    tests and offline scenarios without a running model server.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        bigram_sets: list[set] = []
        vocab: set = set()
        for text in texts:
            chars = list(text.replace(" ", ""))
            bgs = set(zip(chars, chars[1:]))
            bigram_sets.append(bgs)
            vocab |= bgs

        if not vocab:
            return [[0.0]] * len(texts)

        vocab_list = sorted(vocab)
        return [
            [1.0 if bg in bgs else 0.0 for bg in vocab_list]
            for bgs in bigram_sets
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

    Stage 1 — align_sections:   title-similarity section alignment
    Stage 2 — match_paragraphs: embedding cosine paragraph matching
    Stage 3 — classify:         LLM + rule-based diff type classification

    When embedder is None a character-bigram Jaccard embedder is used so the
    function is callable in tests without a real model server (lower accuracy).
    When provider is None or policy.use_llm_classify is False, rule-based
    classification is used for Stage 3.
    """
    from app.core.diff.structure_aligner import align_sections
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

    section_pairs = align_sections(baseline, target)
    para_pairs = match_paragraphs(
        section_pairs,
        effective_embedder,
        effective_policy.similarity_threshold,
    )
    return classify(
        para_pairs,
        policy=effective_policy,
        provider=provider,
        task_id=str(uuid.uuid4()),
        baseline_version_id=baseline.doc_id,
        target_version_id=target.doc_id,
    )
