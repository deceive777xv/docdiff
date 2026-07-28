# QA Enhancement: Hybrid Retrieval, Streaming, Session Memory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the QA pipeline with hybrid BM25+FAISS retrieval, token-level streaming via LangGraph `astream_events`, and per-session conversation memory via `MemorySaver`.

**Architecture:** BM25 is added as a query-time in-memory index alongside the existing FAISS search; results are merged via Reciprocal Rank Fusion. A LangChain `BaseChatModel` replaces the blocking `provider.chat()` call in `generate_answer`, which becomes an async node. Non-serializable runtime objects (`conn`, `embedder`, `lc_model`) move from `QAState` into `config["configurable"]` to avoid MemorySaver's `deepcopy` incompatibility. Session memory is keyed by a per-session `thread_id` UUID managed in `qa_page.py`.

**Tech Stack:** Python 3.11, LangGraph ≥ 0.2, `langchain-openai` ≥ 0.3, `langchain-core` ≥ 0.3, `rank_bm25` ≥ 0.2 (latest PyPI release is 0.2.2), PySide6.

**Spec:** `docs/superpowers/specs/2026-05-06-qa-hybrid-streaming-memory-design.md`

---

## File Map

| Action | File |
|--------|------|
| Modify | `pyproject.toml` |
| **New** | `app/core/retrieval/bm25_searcher.py` |
| Modify | `app/core/retrieval/searcher.py` |
| **New** | `app/core/model/lc_factory.py` |
| Modify | `app/agent/states.py` |
| Modify | `app/agent/qa_graph.py` |
| **New** | `tests/test_retrieval/test_hybrid_search.py` |
| Modify | `tests/test_agent/test_qa_graph.py` |
| **New** | `tests/test_agent/test_qa_stream.py` |
| Modify | `app/ui/app_context.py` |
| Modify | `main.py` |
| Modify | `app/ui/pages/qa_page.py` |

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add three new dependencies**

Edit `pyproject.toml` — add to the `dependencies` list:

```toml
[project]
name = "doc-diff-agent"
version = "0.1.0"
description = "Windows desktop document diff and QA agent"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "PySide6>=6.7.0",
    "PyMuPDF>=1.24.0",
    "python-docx>=1.1.0",
    "docling>=2.31.0",
    "faiss-cpu>=1.9.0",
    "sqlalchemy>=2.0.0",
    "sentence-transformers>=3.0.0",
    "openai>=1.0.0",
    "cryptography>=42.0.0",
    "numpy>=1.26.0",
    "langgraph>=0.2.0",
    "rank_bm25>=0.4",
    "langchain-openai>=0.3",
    "langchain-core>=0.3",
]
```

- [ ] **Step 2: Install new dependencies**

Run: `uv sync`

Expected: installs `rank_bm25`, `langchain-openai`, `langchain-core` without errors.

If `uv` is not available: `pip install "rank_bm25>=0.4" "langchain-openai>=0.3" "langchain-core>=0.3"`

- [ ] **Step 3: Verify imports work**

Run:
```bash
python -c "from rank_bm25 import BM25Okapi; from langchain_openai import ChatOpenAI; from langchain_core.messages import HumanMessage; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add rank_bm25, langchain-openai, langchain-core dependencies"
```

---

## Task 2: BM25 Searcher Module

**Files:**
- Create: `app/core/retrieval/bm25_searcher.py`
- Create: `tests/test_retrieval/test_bm25_searcher.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_retrieval/test_bm25_searcher.py`:

```python
"""Tests for BM25 lexical search."""
from __future__ import annotations

import pytest

from app.core.types import Chunk


def _make_chunk(idx: int, text: str) -> Chunk:
    return Chunk(id=f"c{idx}", version_id="v", chunk_no=idx,
                 section_path="", page_no=idx + 1, text=text)


def test_bm25_search_returns_relevant_chunk():
    from app.core.retrieval.bm25_searcher import bm25_search

    chunks = [
        _make_chunk(0, "付款周期为三十天"),
        _make_chunk(1, "违约金按日万分之五计算"),
        _make_chunk(2, "交货期为六十个工作日"),
    ]
    results = bm25_search(chunks, "付款", top_k=2)

    assert len(results) == 2
    top_idx, top_score = results[0]
    assert top_idx == 0  # "付款周期" should rank highest for "付款" query
    assert top_score > 0


def test_bm25_search_empty_chunks_returns_empty():
    from app.core.retrieval.bm25_searcher import bm25_search

    results = bm25_search([], "付款", top_k=5)
    assert results == []


def test_bm25_search_top_k_limits_results():
    from app.core.retrieval.bm25_searcher import bm25_search

    chunks = [_make_chunk(i, f"文本内容{i}付款") for i in range(10)]
    results = bm25_search(chunks, "付款", top_k=3)
    assert len(results) <= 3


def test_bm25_search_returns_chunk_index_not_id():
    """Return values are (chunk_index_in_list, score), not chunk ids."""
    from app.core.retrieval.bm25_searcher import bm25_search

    chunks = [
        _make_chunk(0, "完全无关的内容"),
        _make_chunk(1, "付款方式和条款"),
    ]
    results = bm25_search(chunks, "付款", top_k=2)
    # index 1 should score higher
    assert results[0][0] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retrieval/test_bm25_searcher.py -v`

Expected: `ImportError` or `ModuleNotFoundError` — `bm25_searcher` doesn't exist yet.

- [ ] **Step 3: Create `app/core/retrieval/bm25_searcher.py`**

```python
"""BM25 lexical search over a list of Chunk objects."""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from app.core.types import Chunk


def bm25_search(chunks: list[Chunk], query: str, top_k: int) -> list[tuple[int, float]]:
    """Return (chunk_index, bm25_score) pairs sorted by score descending.

    chunk_index is the position in the input list, not the chunk's own id.
    Character-level tokenization — effective for Chinese text.
    """
    if not chunks:
        return []

    tokenized_corpus = [list(c.text.replace(" ", "")) for c in chunks]
    tokenized_query = list(query.replace(" ", ""))

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in ranked[:top_k]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retrieval/test_bm25_searcher.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core/retrieval/bm25_searcher.py tests/test_retrieval/test_bm25_searcher.py
git commit -m "feat: add BM25 lexical searcher with character-level Chinese tokenization"
```

---

## Task 3: Hybrid RRF Retrieval in `searcher.py`

**Files:**
- Modify: `app/core/retrieval/searcher.py`
- Create: `tests/test_retrieval/test_hybrid_search.py`

**Context:** Current `search()` in `searcher.py:49` does pure FAISS (L2 distance, lower = better). Replace with hybrid BM25+FAISS merged via RRF. Signature stays the same; `ChunkHit.score` now means "RRF score, higher = better" — callers only use scores for ordering so this is a safe change.

- [ ] **Step 1: Write failing RRF merge test**

Create `tests/test_retrieval/test_hybrid_search.py`:

```python
"""Tests for hybrid BM25+FAISS retrieval with RRF merging."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from app.core.types import Chunk


def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE chunks (
            id TEXT, version_id TEXT, chunk_no INTEGER,
            section_path TEXT, page_no INTEGER, text TEXT, faiss_index_id INTEGER
        )
    """)
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", [
        ("c1", "v1", 0, "", 1, "付款周期三十天", 0),
        ("c2", "v1", 1, "", 2, "违约金计算方式", 1),
        ("c3", "v1", 2, "", 3, "交货期六十天", 2),
    ])
    conn.commit()
    return conn


def test_rrf_doc_in_both_ranks_higher_than_faiss_only():
    """A doc in both FAISS and BM25 must rank above a doc in only FAISS."""
    from app.core.retrieval.searcher import search

    conn = _make_conn()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 4]

    # FAISS returns c1 (faiss_id=0, rank 0) and c2 (faiss_id=1, rank 1)
    # BM25 query "付款" → c1 ranks highest (text "付款周期三十天"), c3 ranks next
    # c1 appears in BOTH → highest RRF; c2 only in FAISS; c3 only in BM25
    with patch("app.core.retrieval.searcher.faiss_store.index_exists", return_value=True), \
         patch("app.core.retrieval.searcher.faiss_store.search",
               return_value=[(0, 0.1), (1, 0.2)]):
        hits = search("/tmp", conn, "付款", mock_embedder, ["v1"], top_k=3)

    assert len(hits) > 0
    # c1 must be the top result (in both lists)
    assert hits[0].chunk.id == "c1"
    conn.close()


def test_rrf_score_is_higher_better():
    """RRF scores should be in descending order (higher = better)."""
    from app.core.retrieval.searcher import search

    conn = _make_conn()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 4]

    with patch("app.core.retrieval.searcher.faiss_store.index_exists", return_value=True), \
         patch("app.core.retrieval.searcher.faiss_store.search",
               return_value=[(0, 0.1), (1, 0.2)]):
        hits = search("/tmp", conn, "付款", mock_embedder, ["v1"], top_k=3)

    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    conn.close()


def test_no_faiss_index_uses_bm25_only():
    """When FAISS index is absent, results still come from BM25 alone."""
    from app.core.retrieval.searcher import search

    conn = _make_conn()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 4]

    with patch("app.core.retrieval.searcher.faiss_store.index_exists", return_value=False):
        hits = search("/tmp", conn, "付款", mock_embedder, ["v1"], top_k=3)

    assert len(hits) > 0
    assert hits[0].chunk.id == "c1"  # BM25 ranks "付款周期三十天" first
    conn.close()


def test_empty_version_ids_returns_empty():
    from app.core.retrieval.searcher import search

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 4]
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    hits = search("/tmp", conn, "付款", mock_embedder, [], top_k=5)
    assert hits == []
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retrieval/test_hybrid_search.py -v`

Expected: tests fail — current `search()` returns L2-sorted results, not RRF.

- [ ] **Step 3: Rewrite `app/core/retrieval/searcher.py`**

Replace the entire file:

```python
"""Hybrid BM25+FAISS retrieval with Reciprocal Rank Fusion."""
from __future__ import annotations

import logging
import sqlite3

import numpy as np

from app.core.model.base_provider import BaseProvider
from app.core.retrieval.bm25_searcher import bm25_search
from app.core.types import Chunk, ChunkHit
from app.db import chunk_repo, faiss_store

logger = logging.getLogger(__name__)

_RRF_K = 60


def _row_to_chunk(row) -> Chunk:
    return Chunk(
        id=row["id"],
        version_id=row["version_id"],
        chunk_no=row["chunk_no"],
        section_path=row["section_path"] or "",
        page_no=row["page_no"] or 0,
        text=row["text"],
        faiss_index_id=row["faiss_index_id"],
    )


def search(
    data_dir: str,
    conn: sqlite3.Connection,
    query: str,
    embedder: BaseProvider,
    version_ids: list[str],
    top_k: int = 5,
) -> list[ChunkHit]:
    """Hybrid BM25+FAISS search with RRF merge.

    Returns top_k hits sorted by RRF score descending (higher = better).
    """
    if not version_ids:
        return []

    query_embedding = embedder.embed([query])[0]
    query_vec = np.array(query_embedding, dtype=np.float32)

    # chunk_id → {"faiss": rank, "bm25": rank}
    ranks: dict[str, dict[str, int]] = {}
    chunk_map: dict[str, Chunk] = {}

    for vid in version_ids:
        all_rows = chunk_repo.get_chunks_by_version(conn, vid)
        if not all_rows:
            continue
        all_chunks = [_row_to_chunk(r) for r in all_rows]
        for c in all_chunks:
            chunk_map[c.id] = c

        # FAISS branch
        if faiss_store.index_exists(data_dir, vid):
            faiss_hits = faiss_store.search(data_dir, vid, query_vec, top_k)
            for rank, (faiss_id, _dist) in enumerate(faiss_hits):
                row = chunk_repo.get_chunk_by_faiss_id(conn, vid, faiss_id)
                if row:
                    cid = row["id"]
                    ranks.setdefault(cid, {})["faiss"] = rank

        # BM25 branch
        bm25_hits = bm25_search(all_chunks, query, top_k)
        for rank, (chunk_idx, _score) in enumerate(bm25_hits):
            cid = all_chunks[chunk_idx].id
            ranks.setdefault(cid, {})["bm25"] = rank

    def _rrf(rank_dict: dict[str, int]) -> float:
        score = 0.0
        if "faiss" in rank_dict:
            score += 1.0 / (_RRF_K + rank_dict["faiss"])
        if "bm25" in rank_dict:
            score += 1.0 / (_RRF_K + rank_dict["bm25"])
        return score

    scored = [(cid, _rrf(r)) for cid, r in ranks.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [
        ChunkHit(chunk=chunk_map[cid], score=score)
        for cid, score in scored[:top_k]
        if cid in chunk_map
    ]
```

- [ ] **Step 4: Run new and existing retrieval tests**

Run: `pytest tests/test_retrieval/ -v`

Expected: all tests in `test_hybrid_search.py` and `test_bm25_searcher.py` PASS. `test_searcher.py` may need inspection — check if it still passes.

If `test_searcher.py` fails due to the score ordering change (old tests expected ascending L2 distance, new returns descending RRF): update those assertions to check descending order instead.

- [ ] **Step 5: Run full test suite to catch regressions**

Run: `pytest -x -q`

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add app/core/retrieval/searcher.py app/core/retrieval/bm25_searcher.py tests/test_retrieval/test_hybrid_search.py
git commit -m "feat: replace pure FAISS retrieval with hybrid BM25+FAISS RRF search"
```

---

## Task 4: LangChain Model Factory

**Files:**
- Create: `app/core/model/lc_factory.py`
- Create: `tests/test_model/test_lc_factory.py`

**Context:** This factory builds a `ChatOpenAI` (from `langchain_openai`) from the existing `AppSettings`. It is a parallel to `get_provider()` in `factory.py` — both read the same provider config, but `get_chat_model()` returns a LangChain model for streaming QA, while `get_provider()` returns a `BaseProvider` for diff classification and embedding.

- [ ] **Step 1: Write failing tests**

Create `tests/test_model/test_lc_factory.py`:

```python
"""Tests for LangChain model factory."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config.settings import AppSettings, ProviderConfig


def test_get_chat_model_returns_none_when_no_providers():
    from app.core.model.lc_factory import get_chat_model

    settings = AppSettings(providers=[], active_provider="")
    assert get_chat_model(settings) is None


def test_get_chat_model_returns_none_when_providers_empty_list():
    from app.core.model.lc_factory import get_chat_model

    settings = AppSettings(providers=[], active_provider="default")
    assert get_chat_model(settings) is None


def test_get_chat_model_returns_model_when_configured():
    from app.core.model.lc_factory import get_chat_model

    settings = AppSettings(
        providers=[
            ProviderConfig(
                name="default",
                api_key="sk-test-key",
                base_url="https://api.example.com/v1",
                chat_model="gpt-4o",
            )
        ],
        active_provider="default",
    )
    model = get_chat_model(settings)
    assert model is not None


def test_get_chat_model_uses_active_provider_chat_model():
    from app.core.model.lc_factory import get_chat_model

    settings = AppSettings(
        providers=[
            ProviderConfig(
                name="default",
                api_key="sk-key",
                base_url="https://api.example.com/v1",
                chat_model="deepseek-chat",
            )
        ],
        active_provider="default",
    )
    model = get_chat_model(settings)
    assert model is not None
    # LangChain ChatOpenAI stores model name in .model_name or .model
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
    assert model_name == "deepseek-chat"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model/test_lc_factory.py -v`

Expected: `ImportError` — `lc_factory` doesn't exist yet.

- [ ] **Step 3: Create `app/core/model/lc_factory.py`**

```python
"""LangChain ChatOpenAI factory for streaming QA generation."""
from __future__ import annotations

from app.config.settings import AppSettings, get_active_provider


def get_chat_model(settings: AppSettings):
    """Create a LangChain ChatOpenAI from active provider config, or None.

    Returns None when no provider is configured. Called at startup and on
    provider_changed signal.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    config = get_active_provider(settings)
    if config is None:
        return None

    return ChatOpenAI(
        model=config.chat_model,
        api_key=config.api_key,
        base_url=config.base_url or None,
        streaming=True,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_model/test_lc_factory.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core/model/lc_factory.py tests/test_model/test_lc_factory.py
git commit -m "feat: add LangChain ChatOpenAI factory for streaming QA"
```

---

## Task 5: Update `QAState` — Add `messages`, Remove Runtime Objects

**Files:**
- Modify: `app/agent/states.py`

**Context:** Current `QAState` (line 56) has `provider: Any`, `embedder: Any`, `conn: Any`. These are NOT deepcopy-safe and will crash `MemorySaver`. They must be removed from state and passed via `config["configurable"]` instead. Add `messages: Annotated[list[BaseMessage], add_messages]` for session history.

- [ ] **Step 1: Replace `QAState` in `app/agent/states.py`**

The file currently imports `Any` and `Optional` from `typing`. Add `Annotated` to that import. Add two new imports for LangChain types. Replace only the `QAState` class — `IngestState` and `CompareState` are untouched.

Full replacement for the `QAState` section (lines 56–77):

```python
from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class QAState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    data_dir: str
    question: str
    scope: str                 # "current_doc" | "standard_lib" | "all"
    current_version_ids: list  # version IDs in scope for "current_doc"

    # ── Session memory (accumulated via add_messages reducer) ────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Node-internal ────────────────────────────────────────────────────────
    _version_ids: list
    _hits: list                # list[ChunkHit]

    # ── Node outputs ────────────────────────────────────────────────────────
    answer: str
    citations: list

    # ── Status ──────────────────────────────────────────────────────────────
    error: Optional[str]
    status: str
```

The complete updated `app/agent/states.py` file:

```python
"""TypedDict state definitions for LangGraph workflows."""
from __future__ import annotations

from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class IngestState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    file_path: str
    data_dir: str
    source_type: str           # "standard" | "uploaded"
    document_id: Optional[str] # set when adding new version to existing doc
    embedder: Any
    conn: Any                  # sqlite3.Connection, opened and closed by caller

    # ── Node-internal intermediate values ───────────────────────────────────
    _file_hash: str
    _ir: Any                   # DocumentIR
    _chunks: list

    # ── Node outputs ────────────────────────────────────────────────────────
    doc_id: str
    version_id: str

    # ── Status ──────────────────────────────────────────────────────────────
    error: Optional[str]
    status: str


class CompareState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    data_dir: str
    baseline_version_id: str
    target_version_id: str
    provider: Any
    embedder: Any
    conn: Any

    # ── Node-internal ────────────────────────────────────────────────────────
    _baseline_ir: Any          # DocumentIR
    _target_ir: Any            # DocumentIR
    _section_pairs: list
    _para_pairs: list

    # ── Node outputs ────────────────────────────────────────────────────────
    task_id: str
    result: Any                # DiffResult

    # ── Status ──────────────────────────────────────────────────────────────
    error: Optional[str]
    status: str


class QAState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    data_dir: str
    question: str
    scope: str                 # "current_doc" | "standard_lib" | "all"
    current_version_ids: list  # version IDs in scope for "current_doc"

    # ── Session memory (accumulated via add_messages reducer) ────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Node-internal ────────────────────────────────────────────────────────
    _version_ids: list
    _hits: list                # list[ChunkHit]

    # ── Node outputs ────────────────────────────────────────────────────────
    answer: str
    citations: list

    # ── Status ──────────────────────────────────────────────────────────────
    error: Optional[str]
    status: str
```

- [ ] **Step 2: Verify the import works**

Run:
```bash
python -c "from app.agent.states import QAState, IngestState, CompareState; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/agent/states.py
git commit -m "feat: add messages field to QAState, remove non-serializable runtime objects"
```

---

## Task 6: Refactor `qa_graph.py` — Async, MemorySaver, RunnableConfig

**Files:**
- Modify: `app/agent/qa_graph.py`

**Context:** Three changes happen together here because they are tightly coupled:
1. `generate_answer` becomes async and uses `lc_model` from config
2. `resolve_scope` and `retrieve_chunks` gain `config: RunnableConfig` to read `conn`/`embedder`
3. The graph is compiled with `MemorySaver` checkpointer

**Critical:** `MemorySaver` deepcopies `QAState` after each step. `conn`, `embedder`, `lc_model` are NOT in QAState — they travel only in `config["configurable"]` which MemorySaver never touches.

- [ ] **Step 1: Rewrite `app/agent/qa_graph.py`**

Replace the entire file:

```python
"""LangGraph StateGraph for the QA (retrieval-augmented answering) workflow."""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessageChunk, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agent.states import QAState
from app.core.retrieval.searcher import search
from app.db import document_repo

logger = logging.getLogger(__name__)

_checkpointer = MemorySaver()

_QA_SYSTEM_PROMPT = """你是一个专业的文档问答助手。请根据以下参考资料回答用户问题。

参考资料：
{context}

回答要求：
1. 只根据参考资料中的内容回答，不要编造信息
2. 如果参考资料中找不到答案，请明确说明"文档中未找到相关内容"
3. 引用具体章节或页码（如资料中有）
4. 回答简洁、准确
"""


def _route(state: QAState) -> str:
    return "end" if state.get("error") else "continue"


def _format_context(hits: list) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        chunk = hit.chunk
        ref = f"[{i}] "
        if chunk.section_path:
            ref += f"章节：{chunk.section_path}，"
        if chunk.page_no:
            ref += f"第{chunk.page_no}页，"
        ref += f"内容：{chunk.text}"
        parts.append(ref)
    return "\n\n".join(parts)


def resolve_scope(state: QAState, config: RunnableConfig) -> dict:
    """Map scope string to concrete version_id list."""
    try:
        scope = state.get("scope", "current_doc")
        conn = config["configurable"]["conn"]

        if scope in ("current_doc", "compare"):
            ids = list(state.get("current_version_ids") or [])
            if not ids:
                label = "对比文档" if scope == "compare" else "当前文档"
                return {"error": f"{label}范围未指定版本，请先选择文档。", "status": "failed"}
            return {"_version_ids": ids, "status": "scope_resolved"}

        if scope == "standard_lib":
            docs = document_repo.list_documents(conn, source_type="standard")
            ids = [document_repo.list_versions(conn, d["id"])[0]["id"]
                   for d in docs
                   if document_repo.list_versions(conn, d["id"])]
            if not ids:
                return {"error": "标准文档库中没有可检索的文档。", "status": "failed"}
            return {"_version_ids": ids, "status": "scope_resolved"}

        # "all"
        ids = list(state.get("current_version_ids") or [])
        for doc in document_repo.list_documents(conn, source_type="standard"):
            versions = document_repo.list_versions(conn, doc["id"])
            if versions and versions[0]["id"] not in ids:
                ids.append(versions[0]["id"])
        if not ids:
            return {"error": "没有可检索的文档。", "status": "failed"}
        return {"_version_ids": ids, "status": "scope_resolved"}

    except Exception as e:
        logger.exception("resolve_scope failed")
        return {"error": str(e), "status": "failed"}


def retrieve_chunks(state: QAState, config: RunnableConfig) -> dict:
    """Vector search for relevant chunks."""
    try:
        conn = config["configurable"]["conn"]
        embedder = config["configurable"]["embedder"]
        hits = search(
            state["data_dir"],
            conn,
            state["question"],
            embedder,
            state["_version_ids"],
            top_k=5,
        )
        return {"_hits": hits, "status": "retrieved"}
    except Exception as e:
        logger.exception("retrieve_chunks failed")
        return {"error": str(e), "status": "failed"}


async def generate_answer(state: QAState, config: RunnableConfig) -> dict:
    """Generate answer via streaming LangChain model, accumulate into session messages."""
    try:
        hits = state.get("_hits", [])
        if not hits:
            return {"answer": "文档中未找到与问题相关的内容。", "status": "answered"}

        lc_model = config["configurable"].get("lc_model")
        if not lc_model:
            return {"answer": "请先在设置页面配置模型", "status": "answered"}

        context = _format_context(hits)
        system_msg = SystemMessage(content=_QA_SYSTEM_PROMPT.format(context=context))

        history = list(state.get("messages", []))[-6:]
        messages_to_send = [system_msg] + history

        chunks: list[BaseMessageChunk] = []
        async for chunk in lc_model.astream(messages_to_send):
            chunks.append(chunk)
        answer = "".join(c.content for c in chunks if isinstance(c.content, str))

        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "status": "answered",
        }
    except Exception as e:
        logger.exception("generate_answer failed")
        return {"error": str(e), "status": "failed"}


def attach_citations(state: QAState) -> dict:
    """Package chunk hits as citation list."""
    return {"citations": list(state.get("_hits", [])), "status": "completed"}


def _build_qa_graph():
    graph = StateGraph(QAState)
    graph.add_node("resolve_scope",    resolve_scope)
    graph.add_node("retrieve_chunks",  retrieve_chunks)
    graph.add_node("generate_answer",  generate_answer)
    graph.add_node("attach_citations", attach_citations)

    graph.set_entry_point("resolve_scope")
    graph.add_conditional_edges("resolve_scope",   _route, {"continue": "retrieve_chunks", "end": END})
    graph.add_conditional_edges("retrieve_chunks", _route, {"continue": "generate_answer",  "end": END})
    graph.add_conditional_edges("generate_answer", _route, {"continue": "attach_citations", "end": END})
    graph.add_edge("attach_citations", END)
    return graph.compile(checkpointer=_checkpointer)


qa_graph = _build_qa_graph()
```

- [ ] **Step 2: Verify import**

Run:
```bash
python -c "from app.agent.qa_graph import qa_graph; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/agent/qa_graph.py
git commit -m "feat: add MemorySaver checkpointer, async generate_answer, RunnableConfig on QA nodes"
```

---

## Task 7: Update `test_qa_graph.py` + Create `test_qa_stream.py`

**Files:**
- Modify: `tests/test_agent/test_qa_graph.py`
- Create: `tests/test_agent/test_qa_stream.py`

**Context:** The existing tests in `test_qa_graph.py` call `resolve_scope(state)` and `qa_graph.invoke(state)`. Both now require a `config` parameter. `generate_answer` is async so graph tests must use `asyncio.run(qa_graph.ainvoke(...))`.

- [ ] **Step 1: Rewrite `tests/test_agent/test_qa_graph.py`**

Replace the entire file:

```python
"""Tests for qa_graph LangGraph workflow."""
from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import MagicMock

import pytest

from langchain_core.messages import AIMessageChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE documents (
            id TEXT PRIMARY KEY, doc_name TEXT, doc_type TEXT,
            file_path TEXT, file_hash TEXT, source_type TEXT,
            business_category TEXT, created_at TEXT, updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE document_versions (
            id TEXT PRIMARY KEY, document_id TEXT, version_no INTEGER,
            version_label TEXT, status TEXT, parsed_json_path TEXT,
            summary TEXT, created_at TEXT
        )
    """)
    conn.commit()
    return conn


async def _mock_astream(messages):
    yield AIMessageChunk(content="答案是X。")


def _make_config(conn=None, embedder=None, lc_model=None, thread_id="t1"):
    mock_lc = MagicMock()
    mock_lc.astream = _mock_astream
    return {
        "configurable": {
            "thread_id": thread_id,
            "conn": conn or _make_conn(),
            "embedder": embedder or MagicMock(),
            "lc_model": lc_model if lc_model is not None else mock_lc,
        }
    }


# ── resolve_scope direct tests ─────────────────────────────────────────────────

def test_resolve_scope_current_doc():
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {"scope": "current_doc", "current_version_ids": ["ver-1"]}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert result["_version_ids"] == ["ver-1"]
    assert result.get("error") is None
    conn.close()


def test_resolve_scope_empty_current_doc_returns_error():
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {"scope": "current_doc", "current_version_ids": []}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert result.get("error") is not None
    assert result.get("status") == "failed"
    conn.close()


def test_resolve_scope_compare_returns_provided_version_ids():
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {"scope": "compare", "current_version_ids": ["baseline-v1", "target-v1"]}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert set(result["_version_ids"]) == {"baseline-v1", "target-v1"}
    conn.close()


def test_resolve_scope_compare_error_when_no_ids():
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {"scope": "compare", "current_version_ids": []}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert "error" in result
    conn.close()


def test_resolve_scope_current_doc_unchanged():
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {"scope": "current_doc", "current_version_ids": ["v-abc"]}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert result["_version_ids"] == ["v-abc"]
    conn.close()


# ── Full graph integration tests ───────────────────────────────────────────────

def test_graph_happy_path():
    from unittest.mock import patch
    from app.agent.qa_graph import qa_graph

    mock_hit = MagicMock()
    mock_hit.chunk.section_path = "第一章"
    mock_hit.chunk.page_no = 1
    mock_hit.chunk.text = "付款周期为30天。"

    conn = _make_conn()
    config = _make_config(conn=conn, thread_id="happy-path-1")
    state_input = {
        "data_dir": "/tmp",
        "question": "付款周期是多少天？",
        "scope": "current_doc",
        "current_version_ids": ["ver-1"],
    }

    with patch("app.agent.qa_graph.search", return_value=[mock_hit]):
        result = asyncio.run(qa_graph.ainvoke(state_input, config))

    assert result.get("error") is None
    assert result["answer"] == "答案是X。"
    assert len(result["citations"]) == 1
    assert result["status"] == "completed"
    conn.close()


def test_graph_no_hits_returns_default_message():
    from unittest.mock import patch
    from app.agent.qa_graph import qa_graph

    conn = _make_conn()
    # lc_model=None is fine here — early return before model call when no hits
    config = {"configurable": {
        "thread_id": "no-hits-1",
        "conn": conn,
        "embedder": MagicMock(),
        "lc_model": None,
    }}
    state_input = {
        "data_dir": "/tmp",
        "question": "付款周期是多少天？",
        "scope": "current_doc",
        "current_version_ids": ["ver-1"],
    }

    with patch("app.agent.qa_graph.search", return_value=[]):
        result = asyncio.run(qa_graph.ainvoke(state_input, config))

    assert result.get("error") is None
    assert "未找到" in result["answer"]
    assert result["citations"] == []
    conn.close()
```

- [ ] **Step 2: Run updated tests**

Run: `pytest tests/test_agent/test_qa_graph.py -v`

Expected: all tests PASS.

- [ ] **Step 3: Create `tests/test_agent/test_qa_stream.py`**

```python
"""Tests for async generate_answer and streaming graph behavior."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage


def _make_lc_model(response_text: str = "流式回答"):
    async def _astream(messages):
        yield AIMessageChunk(content=response_text)

    model = MagicMock()
    model.astream = _astream
    return model


def _make_config(lc_model=None, conn=None):
    return {
        "configurable": {
            "thread_id": "t1",
            "conn": conn or MagicMock(),
            "embedder": MagicMock(),
            "lc_model": lc_model,
        }
    }


# ── generate_answer unit tests ─────────────────────────────────────────────────

def test_generate_answer_streams_tokens_from_lc_model():
    from app.agent.qa_graph import generate_answer

    mock_hit = MagicMock()
    mock_hit.chunk.section_path = "第一章"
    mock_hit.chunk.page_no = 1
    mock_hit.chunk.text = "付款周期为三十天。"

    state = {
        "_hits": [mock_hit],
        "messages": [HumanMessage(content="付款周期是多少天？")],
    }
    config = _make_config(lc_model=_make_lc_model("三十天"))

    result = asyncio.run(generate_answer(state, config))
    assert result["answer"] == "三十天"
    assert result["status"] == "answered"


def test_generate_answer_appends_ai_message_to_messages():
    from app.agent.qa_graph import generate_answer

    mock_hit = MagicMock()
    mock_hit.chunk.text = "内容"
    mock_hit.chunk.section_path = ""
    mock_hit.chunk.page_no = 0

    state = {
        "_hits": [mock_hit],
        "messages": [HumanMessage(content="问题")],
    }
    config = _make_config(lc_model=_make_lc_model("答案"))

    result = asyncio.run(generate_answer(state, config))
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "答案"


def test_generate_answer_no_lc_model_returns_prompt():
    from app.agent.qa_graph import generate_answer

    mock_hit = MagicMock()
    mock_hit.chunk.text = "内容"
    mock_hit.chunk.section_path = ""
    mock_hit.chunk.page_no = 0

    state = {"_hits": [mock_hit], "messages": []}
    config = _make_config(lc_model=None)

    result = asyncio.run(generate_answer(state, config))
    assert "配置" in result["answer"]
    assert result["status"] == "answered"


def test_generate_answer_no_hits_returns_not_found():
    from app.agent.qa_graph import generate_answer

    state = {"_hits": [], "messages": [HumanMessage(content="问题")]}
    config = _make_config(lc_model=None)  # lc_model irrelevant, exits before model call

    result = asyncio.run(generate_answer(state, config))
    assert "未找到" in result["answer"]
    assert result["status"] == "answered"


def test_generate_answer_truncates_history_to_6_messages():
    """Only last 6 messages are sent to model, not the full history."""
    from app.agent.qa_graph import generate_answer

    sent_messages = []

    async def capture_astream(messages):
        sent_messages.extend(messages)
        yield AIMessageChunk(content="ok")

    model = MagicMock()
    model.astream = capture_astream

    mock_hit = MagicMock()
    mock_hit.chunk.text = "内容"
    mock_hit.chunk.section_path = ""
    mock_hit.chunk.page_no = 0

    history = [HumanMessage(content=f"问题{i}") for i in range(10)]
    state = {"_hits": [mock_hit], "messages": history}
    config = _make_config(lc_model=model)

    asyncio.run(generate_answer(state, config))
    # messages_to_send = [system_msg] + last 6 history messages
    # so total = 1 system + 6 history = 7
    assert len(sent_messages) == 7


# ── resolve_scope with new config signature ────────────────────────────────────

def test_resolve_scope_reads_conn_from_config():
    import sqlite3
    from app.agent.qa_graph import resolve_scope

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE documents (id TEXT PRIMARY KEY, doc_name TEXT,
        doc_type TEXT, file_path TEXT, file_hash TEXT, source_type TEXT,
        business_category TEXT, created_at TEXT, updated_at TEXT)""")
    conn.execute("""CREATE TABLE document_versions (id TEXT PRIMARY KEY,
        document_id TEXT, version_no INTEGER, version_label TEXT, status TEXT,
        parsed_json_path TEXT, summary TEXT, created_at TEXT)""")
    conn.commit()

    state = {"scope": "current_doc", "current_version_ids": ["v1"]}
    config = {"configurable": {"conn": conn}}
    result = resolve_scope(state, config)
    assert result["_version_ids"] == ["v1"]
    conn.close()
```

- [ ] **Step 4: Run new tests**

Run: `pytest tests/test_agent/test_qa_stream.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Run full agent test suite**

Run: `pytest tests/test_agent/ -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_agent/test_qa_graph.py tests/test_agent/test_qa_stream.py
git commit -m "test: update qa_graph tests for RunnableConfig signature, add streaming tests"
```

---

## Task 8: Update `app_context.py` + `main.py`

**Files:**
- Modify: `app/ui/app_context.py`
- Modify: `main.py`

**Context:** `AppContext` needs an `lc_model` field. `main.py` needs to call `get_chat_model()` at startup and rebuild it when `provider_changed` fires — same pattern as `get_provider()` / `get_embedder()`.

- [ ] **Step 1: Update `app/ui/app_context.py`**

Current file is 17 lines. Replace with:

```python
"""Application context — shared state for all UI pages."""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass

from app.config.settings import AppSettings
from app.core.model.base_provider import BaseProvider


@dataclass
class AppContext:
    settings: AppSettings
    conn: sqlite3.Connection
    data_dir: str
    provider: BaseProvider | None = None
    embedder: BaseProvider | None = None
    lc_model: object | None = None  # BaseChatModel, typed as object to avoid hard dep
```

- [ ] **Step 2: Update `main.py` — add `get_chat_model` at startup**

Current `main.py:62–78` builds provider and embedder. Add `lc_model` after:

```python
    from app.core.model.factory import get_embedder, get_provider
    from app.core.model.lc_factory import get_chat_model

    try:
        provider = get_provider(settings)
    except Exception:
        provider = None
    try:
        embedder = get_embedder(settings)
    except Exception:
        embedder = None
    try:
        lc_model = get_chat_model(settings)
    except Exception:
        lc_model = None

    ctx = AppContext(
        settings=settings,
        conn=conn,
        data_dir=str(data_dir),
        provider=provider,
        embedder=embedder,
        lc_model=lc_model,
    )
```

- [ ] **Step 3: Update `_rebuild_providers` in `main.py` — also rebuild `lc_model`**

Current `_rebuild_providers` (lines 22–37) rebuilds provider and embedder. Add lc_model:

```python
def _rebuild_providers(ctx: AppContext, compare, qa) -> None:
    """Reload provider, embedder, and lc_model from current settings."""
    from app.core.model.factory import get_embedder, get_provider
    from app.core.model.lc_factory import get_chat_model

    try:
        ctx.provider = get_provider(ctx.settings)
    except Exception:
        ctx.provider = None
    try:
        ctx.embedder = get_embedder(ctx.settings)
    except Exception:
        ctx.embedder = None
    try:
        ctx.lc_model = get_chat_model(ctx.settings)
    except Exception:
        ctx.lc_model = None

    compare.refresh_versions()
    qa.refresh_documents()
    qa.refresh_compare_tasks()
```

- [ ] **Step 4: Verify app starts without import errors**

Run:
```bash
python -c "from app.ui.app_context import AppContext; from app.core.model.lc_factory import get_chat_model; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run full test suite**

Run: `pytest -x -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/ui/app_context.py main.py
git commit -m "feat: add lc_model to AppContext, build and rebuild on provider_changed"
```

---

## Task 9: Refactor `qa_page.py` — Streaming Worker + Session Management

**Files:**
- Modify: `app/ui/pages/qa_page.py`

**Context:** The current worker (`_QaWorker`) uses a blocking `qa_graph.invoke()`. Replace with an async worker using `astream_events`. The `QaPage` class gains: `_thread_id` (UUID for MemorySaver session isolation), `_current_bubble` (reference to the streaming assistant bubble), `_accumulated` (token accumulator), a 「新会话」button, and auto-reset on scope/doc change.

The new `_add_message` returns `(QLabel, QWidget)` so `send_question` can hold a reference to the empty bubble being streamed into.

- [ ] **Step 1: Replace the entire `app/ui/pages/qa_page.py`**

```python
"""QA page — chat-style retrieval-augmented question answering with streaming."""
from __future__ import annotations

import asyncio
import logging
import uuid

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.types import RetrievalScope
from app.db import document_repo
from app.ui.app_context import AppContext
from app.ui.theme import Theme

logger = logging.getLogger(__name__)

_SCOPE_MAP: dict[str, RetrievalScope] = {
    "当前文档": RetrievalScope.CURRENT_DOC,
    "对比文档": RetrievalScope.COMPARE,
    "标准文档库": RetrievalScope.STANDARD_LIB,
    "全部": RetrievalScope.ALL,
}

_USER_BUBBLE_STYLE = (
    f"background:{Theme.COLOR_PRIMARY};color:white;"
    "border-radius:12px;padding:10px;margin:4px 0;"
)
_ASST_BUBBLE_STYLE = (
    f"background:{Theme.BG_CARD};border:1px solid {Theme.BORDER};"
    "border-radius:12px;padding:10px;margin:4px 0;"
)


class _QaWorker(QObject):
    """Run qa_graph via astream_events in a background thread."""

    token_received = Signal(str)
    citations_ready = Signal(list)
    error = Signal(str)
    done = Signal()

    def __init__(
        self,
        data_dir: str,
        question: str,
        embedder,
        lc_model,
        scope: RetrievalScope,
        current_version_ids: list[str],
        thread_id: str,
        parent=None,
    ):
        super().__init__(parent)
        self._data_dir = data_dir
        self._question = question
        self._embedder = embedder
        self._lc_model = lc_model
        self._scope = scope
        self._current_version_ids = current_version_ids
        self._thread_id = thread_id

    def run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        from app.agent.qa_graph import qa_graph
        from app.db.schema import open_db
        from langchain_core.messages import HumanMessage

        conn = open_db(self._data_dir)
        try:
            config = {
                "configurable": {
                    "thread_id": self._thread_id,
                    "conn": conn,
                    "embedder": self._embedder,
                    "lc_model": self._lc_model,
                }
            }
            state_input = {
                "messages": [HumanMessage(content=self._question)],
                "question": self._question,
                "scope": self._scope.value,
                "current_version_ids": self._current_version_ids,
                "data_dir": self._data_dir,
            }
            try:
                async for event in qa_graph.astream_events(state_input, config, version="v2"):
                    if event["event"] == "on_chat_model_stream":
                        token = event["data"]["chunk"].content
                        if token:
                            self.token_received.emit(token)
                    elif event["event"] == "on_chain_error":
                        self.error.emit(str(event["data"].get("error", "未知错误")))
                        return
                final = await qa_graph.aget_state(config)
                self.citations_ready.emit(final.values.get("citations", []))
            except Exception as exc:
                logger.exception("QA worker failed")
                self.error.emit(str(exc))
        finally:
            conn.close()
            self.done.emit()


class QaPage(QWidget):
    """Chat-style QA page with streaming RAG backend and session memory."""

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._threads: set[QThread] = set()
        self._thread_id: str = str(uuid.uuid4())
        self._accumulated: str = ""
        self._current_bubble: QLabel | None = None
        self._build_ui()
        self.refresh_documents()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)
        self.setStyleSheet(f"background-color:{Theme.BG_CARD};")

        # ── Top: scope/document selectors + 新会话 button ──────────────────────
        top_group = QGroupBox()
        top_layout = QHBoxLayout(top_group)
        top_layout.setSpacing(10)

        top_layout.addWidget(QLabel("检索范围："))
        self._scope_combo = QComboBox()
        self._scope_combo.addItems(list(_SCOPE_MAP.keys()))
        self._scope_combo.currentTextChanged.connect(self._on_scope_changed)
        top_layout.addWidget(self._scope_combo)

        top_layout.addWidget(QLabel("文档："))
        self._doc_combo = QComboBox()
        self._doc_combo.setMinimumWidth(200)
        self._doc_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._doc_combo.currentIndexChanged.connect(self._on_doc_changed)
        top_layout.addWidget(self._doc_combo)

        self._compare_task_label = QLabel("对比任务：")
        top_layout.addWidget(self._compare_task_label)
        self._compare_task_combo = QComboBox()
        self._compare_task_combo.setMinimumWidth(280)
        self._compare_task_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_layout.addWidget(self._compare_task_combo)

        top_layout.addStretch()

        new_session_btn = QPushButton("新会话")
        new_session_btn.setStyleSheet(Theme.btn_primary())
        new_session_btn.setFixedWidth(72)
        new_session_btn.clicked.connect(self._new_session)
        top_layout.addWidget(new_session_btn)

        root.addWidget(top_group)

        # ── Middle: chat scroll area ───────────────────────────────────────────
        self._chat_scroll = QScrollArea()
        self._chat_scroll.setWidgetResizable(True)
        self._chat_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._chat_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #dde1ea;
                border-radius: 4px;
            }
        """)
        self._chat_content = QWidget()
        self._chat_layout = QVBoxLayout(self._chat_content)
        self._chat_layout.setSpacing(4)
        self._chat_layout.addStretch()

        self._chat_scroll.setWidget(self._chat_content)
        root.addWidget(self._chat_scroll, 1)

        # ── Bottom: input area ─────────────────────────────────────────────────
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self._input = QTextEdit()
        self._input.setMaximumHeight(40)
        self._input.setPlaceholderText("输入问题…")
        input_row.addWidget(self._input, 1)

        send_btn = QPushButton("发送")
        send_btn.setStyleSheet(Theme.btn_primary())
        send_btn.setFixedWidth(72)
        send_btn.clicked.connect(self.send_question)
        input_row.addWidget(send_btn)

        root.addLayout(input_row)

        self._on_scope_changed(self._scope_combo.currentText())

    # ── Public API ─────────────────────────────────────────────────────────────

    def refresh_documents(self) -> None:
        self._doc_combo.blockSignals(True)
        try:
            self._doc_combo.clear()
            docs = document_repo.list_documents(self.ctx.conn)
            for doc in docs:
                versions = document_repo.list_versions(self.ctx.conn, doc["id"])
                for ver in versions:
                    label = f"{doc['doc_name']} — v{ver['version_no']}"
                    if ver["version_label"]:
                        label += f"  ({ver['version_label']})"
                    self._doc_combo.addItem(label, ver["id"])
        except Exception as exc:
            logger.warning("refresh_documents failed: %s", exc)
        finally:
            self._doc_combo.blockSignals(False)

    def refresh_compare_tasks(self) -> None:
        self._compare_task_combo.blockSignals(True)
        try:
            self._compare_task_combo.clear()
            rows = self.ctx.conn.execute("""
                SELECT ct.baseline_version_id, ct.target_version_id,
                       bd.doc_name AS b_name, bv.version_no AS b_ver,
                       td.doc_name AS t_name, tv.version_no AS t_ver
                FROM compare_tasks ct
                JOIN document_versions bv ON ct.baseline_version_id = bv.id
                JOIN documents bd ON bv.document_id = bd.id
                JOIN document_versions tv ON ct.target_version_id = tv.id
                JOIN documents td ON tv.document_id = td.id
                WHERE ct.status = 'completed'
                ORDER BY ct.created_at DESC
                LIMIT 20
            """).fetchall()
            for row in rows:
                label = (
                    f"{row['b_name']} v{row['b_ver']}"
                    f" ↔ {row['t_name']} v{row['t_ver']}"
                )
                self._compare_task_combo.addItem(
                    label,
                    (row["baseline_version_id"], row["target_version_id"]),
                )
        except Exception as exc:
            logger.warning("refresh_compare_tasks failed: %s", exc)
        finally:
            self._compare_task_combo.blockSignals(False)

    def send_question(self) -> None:
        question = self._input.toPlainText().strip()
        if not question:
            return

        if self.ctx.embedder is None or self.ctx.lc_model is None:
            self._add_message("assistant", "请先在设置页面配置模型")
            return

        self._add_message("user", question)
        self._input.clear()

        scope_text = self._scope_combo.currentText()
        scope = _SCOPE_MAP.get(scope_text, RetrievalScope.ALL)

        current_version_ids: list[str] = []
        if scope == RetrievalScope.CURRENT_DOC:
            vid = self._doc_combo.currentData()
            if vid:
                current_version_ids = [vid]
        elif scope == RetrievalScope.COMPARE:
            task_data = self._compare_task_combo.currentData()
            if task_data:
                current_version_ids = list(task_data)

        bubble_label, _ = self._add_message("assistant", "")
        self._current_bubble = bubble_label
        self._accumulated = ""

        thread = QThread()
        worker = _QaWorker(
            data_dir=self.ctx.data_dir,
            question=question,
            embedder=self.ctx.embedder,
            lc_model=self.ctx.lc_model,
            scope=scope,
            current_version_ids=current_version_ids,
            thread_id=self._thread_id,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.token_received.connect(self._on_token)
        worker.citations_ready.connect(self._on_citations)
        worker.error.connect(self._on_error)
        worker.done.connect(self._on_done)
        worker.done.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._threads.discard(thread))
        self._threads.add(thread)
        thread.start()

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_scope_changed(self, text: str) -> None:
        self._doc_combo.setVisible(text == "当前文档")
        self._compare_task_label.setVisible(text == "对比文档")
        self._compare_task_combo.setVisible(text == "对比文档")
        self._thread_id = str(uuid.uuid4())

    def _on_doc_changed(self) -> None:
        self._thread_id = str(uuid.uuid4())

    def _new_session(self) -> None:
        self._thread_id = str(uuid.uuid4())
        self._clear_chat()

    def _clear_chat(self) -> None:
        while self._chat_layout.count() > 1:
            item = self._chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_token(self, token: str) -> None:
        self._accumulated += token
        if self._current_bubble is not None:
            self._current_bubble.setText(self._accumulated)
        self._chat_scroll.verticalScrollBar().setValue(
            self._chat_scroll.verticalScrollBar().maximum()
        )

    def _on_citations(self, hits: list) -> None:
        if not hits:
            return
        cit_outer = QWidget()
        cit_layout = QHBoxLayout(cit_outer)
        cit_layout.setContentsMargins(0, 0, 0, 0)

        cit_parts: list[str] = []
        for hit in hits:
            chunk = hit.chunk
            parts: list[str] = []
            if chunk.section_path:
                parts.append(chunk.section_path)
            if chunk.page_no:
                parts.append(f"p.{chunk.page_no}")
            cit_parts.append("  ".join(parts))

        cit_lbl = QLabel(f"引用：{' | '.join(cit_parts)}")
        cit_lbl.setStyleSheet(
            f"color:{Theme.TEXT_PLACEHOLDER};font-size:11px;margin-left:4px;"
        )
        cit_lbl.setWordWrap(True)
        cit_layout.addWidget(cit_lbl)
        cit_layout.addStretch()

        self._chat_layout.insertWidget(self._chat_layout.count() - 1, cit_outer)

    def _on_error(self, msg: str) -> None:
        if self._current_bubble is not None:
            self._current_bubble.setText(f"错误：{msg}")
        else:
            self._add_message("assistant", f"错误：{msg}")

    def _on_done(self) -> None:
        self._current_bubble = None
        self._accumulated = ""

    # ── Message rendering ──────────────────────────────────────────────────────

    def _add_message(self, role: str, text: str, citations: list | None = None) -> tuple[QLabel, QWidget]:
        """Add a chat bubble. Returns (bubble_label, outer_widget)."""
        is_user = (role == "user")

        outer = QWidget()
        outer_layout = QHBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setStyleSheet(_USER_BUBBLE_STYLE if is_user else _ASST_BUBBLE_STYLE)
        bubble.setMaximumWidth(600)

        if is_user:
            outer_layout.addStretch()
            outer_layout.addWidget(bubble)
        else:
            outer_layout.addWidget(bubble)
            outer_layout.addStretch()

        self._chat_layout.insertWidget(self._chat_layout.count() - 1, outer)
        self._chat_scroll.verticalScrollBar().setValue(
            self._chat_scroll.verticalScrollBar().maximum()
        )

        return bubble, outer
```

- [ ] **Step 2: Verify import**

Run:
```bash
python -c "from app.ui.pages.qa_page import QaPage; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run full test suite**

Run: `pytest -x -q`

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add app/ui/pages/qa_page.py
git commit -m "feat: streaming QA worker, session memory management, 新会话 button"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| BM25+FAISS hybrid with RRF | Tasks 2, 3 |
| Character-level Chinese tokenization | Task 2 |
| `bm25_search()` returns `(chunk_index, score)` | Task 2 |
| `search()` signature unchanged | Task 3 |
| `ChunkHit.score` → RRF score | Task 3 |
| `lc_factory.get_chat_model()` | Task 4 |
| `AppContext.lc_model` field | Task 8 |
| `main.py` builds/rebuilds `lc_model` | Task 8 |
| `QAState.messages` with `add_messages` reducer | Task 5 |
| `conn`/`embedder`/`lc_model` NOT in QAState | Task 5 |
| All nodes accept `config: RunnableConfig` | Task 6 |
| `MemorySaver` checkpointer on graph | Task 6 |
| `generate_answer` async, uses `lc_model` from config | Task 6 |
| History truncated to last 6 messages | Task 6 |
| `astream_events` worker with `token_received` signal | Task 9 |
| `_thread_id` UUID per session | Task 9 |
| 「新会话」button clears chat + resets thread_id | Task 9 |
| Auto-reset thread_id on scope/doc change | Task 9 |
| Auto-reset does NOT clear bubbles | Task 9 |
| `test_hybrid_search.py` | Tasks 2, 3 |
| `test_qa_stream.py` | Task 7 |
| `test_qa_graph.py` updated | Task 7 |
