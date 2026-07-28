# QA Enhancement: Hybrid Retrieval, Streaming Output, Session Memory

## Goal

Upgrade the QA pipeline with three improvements: (A) hybrid BM25+FAISS retrieval with RRF merging for better recall, (B) token-level streaming output via LangGraph `astream_events`, (C) per-session conversation memory via LangGraph `MemorySaver`.

## Architecture

The QA pipeline stays LangGraph-based. Feature A replaces the retrieval function in-place. Features B and C require replacing the blocking LLM call with a LangChain `BaseChatModel` and compiling the graph with a `MemorySaver` checkpointer. The existing `BaseProvider` is **not replaced** — it continues to serve diff classification; `lc_model` is a parallel addition for QA generation only.

**Critical constraint — MemorySaver serialization:** `MemorySaver` calls `copy.deepcopy()` on the full `QAState` after each superstep. `sqlite3.Connection`, `BaseProvider` (wraps openai client), and `BaseChatModel` (wraps openai client with connection pools) are NOT deepcopy-safe. These objects must **never** appear in `QAState`. Instead, pass them via `config["configurable"]` — LangGraph's config dict is per-call metadata that MemorySaver does NOT persist or deepcopy.

## Tech Stack

Python 3.11, LangGraph ≥ 0.2, `langchain-openai` ≥ 0.3, `langchain-core` ≥ 0.3, `rank_bm25` ≥ 0.4, PySide6, FAISS-cpu, SQLite

---

## Feature A: Hybrid BM25 + FAISS Retrieval

### Design

Current retrieval is pure dense search (FAISS flat L2). BM25 adds lexical recall — especially useful for exact legal terms, article numbers, and party names that dense vectors may miss. The two rankings are merged with **Reciprocal Rank Fusion** (RRF), which requires no score normalization:

```
rrf_score(doc) = 1/(60 + rank_faiss) + 1/(60 + rank_bm25)
```

Documents appearing in only one ranking contribute `1/(60 + rank)` from that side and 0 from the other.

### Tokenization

Character-level splitting (`list(text.replace(" ", ""))`) — no extra dependency, effective for Chinese contract clauses where individual characters carry semantic weight.

### New file: `app/core/retrieval/bm25_searcher.py`

```python
def bm25_search(chunks: list[Chunk], query: str, top_k: int) -> list[tuple[int, float]]:
    """Return list of (chunk_index, score) sorted by BM25 score descending."""
```

- Builds a `BM25Okapi` index from the chunks' texts on every call (query-time, in-memory).
- Returns `(chunk_index, score)` pairs — chunk_index is the position in the input list.

### Modified: `app/core/retrieval/searcher.py`

`search()` signature is unchanged. Internally:

1. Fetch all chunks for each version from SQLite (needed for BM25; FAISS already does this implicitly via `get_chunk_by_faiss_id`).
2. Run FAISS search → `list[(faiss_id, distance)]` per version → convert to `(chunk_id, rank)`.
3. Run `bm25_search(all_chunks, query, top_k)` per version → `(chunk_index, score)` → convert to `(chunk_id, rank)`.
4. RRF merge across all versions, sort by RRF score descending, slice to `top_k`.
5. Return `list[ChunkHit]` with `score = rrf_score` (higher = better, range roughly 0–0.03).

The `ChunkHit.score` interpretation changes from "L2 distance (lower = better)" to "RRF score (higher = better)". Callers only use scores for ordering, so no downstream changes are needed.

### Performance

For typical document versions (100–500 chunks, texts ≤ 300 chars each), building `BM25Okapi` takes < 10 ms. FAISS index is still loaded from disk per query (existing behavior — caching is out of scope).

### Tests

`tests/test_retrieval/test_hybrid_search.py`:
- Mock `faiss_store.search` to return fixed results.
- Use real `bm25_search` with simple Chinese sentences.
- Verify RRF merge ranking is correct (docs in both lists rank higher than docs in only one).

---

## Feature B: Streaming Output via LangGraph `astream_events`

### New file: `app/core/model/lc_factory.py`

```python
def get_chat_model(settings: AppSettings) -> BaseChatModel | None:
    """Create a LangChain ChatOpenAI from the first provider config, or None."""
```

- Uses `langchain_openai.ChatOpenAI(model=..., api_key=..., base_url=..., streaming=True)`.
- Returns `None` if no provider is configured.
- Called at startup and whenever `provider_changed` fires.

### Modified: `app/ui/app_context.py`

Add field:
```python
lc_model: BaseChatModel | None = None
```

### Modified: `app/agent/states.py`

Add to `QAState`:
```python
messages: Annotated[list[BaseMessage], add_messages]  # session history, accumulated by MemorySaver
```

`conn`, `embedder`, and `lc_model` are **not** fields of `QAState`. They are passed per-call via `config["configurable"]` and never persisted.

### Modified: `app/agent/qa_graph.py`

**MemorySaver:**
```python
from langgraph.checkpoint.memory import MemorySaver
_checkpointer = MemorySaver()  # module-level singleton
```

**All node functions accept `config: RunnableConfig`:**

`resolve_scope` (existing node, signature change):
```python
from langchain_core.runnables import RunnableConfig

def resolve_scope(state: QAState, config: RunnableConfig) -> dict:
    conn = config["configurable"]["conn"]
    # rest of function unchanged, replaces local conn parameter
```

`retrieve_chunks` (existing node, signature change):
```python
def retrieve_chunks(state: QAState, config: RunnableConfig) -> dict:
    conn = config["configurable"]["conn"]
    embedder = config["configurable"]["embedder"]
    # rest of function unchanged
```

**`generate_answer` becomes async:**
```python
async def generate_answer(state: QAState, config: RunnableConfig) -> dict:
    lc_model = config["configurable"].get("lc_model")
    if not lc_model:
        return {"answer": "请先在设置页面配置模型", "status": "answered"}

    context = _format_context(state.get("_hits", []))
    system_msg = SystemMessage(content=_QA_SYSTEM_PROMPT.format(context=context))

    # Recent history (last 6 messages = 3 turns) + inject system
    history = list(state.get("messages", []))[-6:]
    messages_to_send = [system_msg] + history

    chunks: list[BaseMessageChunk] = []
    async for chunk in lc_model.astream(messages_to_send):
        chunks.append(chunk)
    answer = "".join(c.content for c in chunks if isinstance(c.content, str))

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],  # appended via add_messages reducer
        "status": "answered",
    }
```

**Graph compilation:**
```python
qa_graph = builder.compile(checkpointer=_checkpointer)
```

**Input per call** includes `messages: [HumanMessage(question)]` in `state_input` — the `add_messages` reducer appends this to history. Runtime objects go in `config["configurable"]`.

### Modified: `app/ui/pages/qa_page.py` — Worker

```python
class _QaWorker(QObject):
    token_received = Signal(str)   # replaces result_ready str part
    citations_ready = Signal(list) # replaces result_ready list part
    error = Signal(str)
    done = Signal()

    def run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        from langchain_core.messages import HumanMessage
        import sqlite3
        conn = sqlite3.connect(self._db_path)
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
                self.error.emit(str(exc))
        finally:
            conn.close()
            self.done.emit()
```

`_QaWorker.__init__` gains `lc_model`, `thread_id`, and `db_path` parameters (replacing the `conn` parameter — the worker opens and closes its own connection inside `_run_async`).

### UI bubble update

`send_question` creates the assistant bubble immediately (empty text) and holds a reference:
```python
bubble_label, _ = self._add_message("assistant", "")
self._current_bubble = bubble_label
```

`token_received` slot:
```python
def _on_token(self, token: str) -> None:
    self._accumulated += token
    self._current_bubble.setText(self._accumulated)
```

`done` slot clears `_accumulated` and `_current_bubble`.

---

## Feature C: Session Memory via MemorySaver

### How memory works

`MemorySaver` persists the full `QAState` (minus config) after each graph run, keyed by `thread_id`. On the next call with the same `thread_id`, LangGraph restores the checkpoint and merges with the new input:
- `messages` field: new `HumanMessage` is **appended** (via `add_messages` reducer) to the stored history.
- All other fields (`question`, `scope`, etc.): new input values **overwrite** stored values.
- `conn`, `embedder`, `lc_model`: never persisted — provided fresh on every call via `config["configurable"]`.

The result: `generate_answer` receives `state["messages"]` containing the full conversation history from all previous turns in this session, plus the current user message.

### Session isolation in `qa_page.py`

```python
self._thread_id: str = str(uuid.uuid4())  # initialized in __init__
```

**New「新会话」button** (top bar, right side):
```python
def _new_session(self) -> None:
    self._thread_id = str(uuid.uuid4())
    self._clear_chat()  # remove all bubble widgets from _chat_layout
```

**Auto-reset on scope/document change:**
```python
def _on_scope_changed(self, text: str) -> None:
    self._thread_id = str(uuid.uuid4())  # new memory context
    # existing visibility logic unchanged
    ...

# doc_combo.currentIndexChanged connected to:
def _on_doc_changed(self) -> None:
    self._thread_id = str(uuid.uuid4())
```

Auto-reset does **not** clear the chat bubbles — only 「新会话」does. This allows the user to review prior answers while the LLM no longer has access to them.

### History truncation

`generate_answer` takes only `list(state["messages"])[-6:]` before building the prompt. This caps the context at 3 full turns regardless of how long the MemorySaver history grows. The full history remains in MemorySaver (not pruned), so a future "load checkpoint" feature could restore it.

---

## File Map

| Action | File |
|--------|------|
| **New** | `app/core/retrieval/bm25_searcher.py` |
| **New** | `app/core/model/lc_factory.py` |
| **New** | `tests/test_retrieval/test_hybrid_search.py` |
| **New** | `tests/test_agent/test_qa_stream.py` |
| Modify | `pyproject.toml` — add `rank_bm25>=0.4`, `langchain-openai>=0.3`, `langchain-core>=0.3` |
| Modify | `app/core/retrieval/searcher.py` — hybrid RRF |
| Modify | `app/agent/states.py` — add `messages` field only |
| Modify | `app/agent/qa_graph.py` — async generate, MemorySaver, RunnableConfig on all nodes |
| Modify | `app/ui/app_context.py` — `lc_model` field |
| Modify | `app/ui/pages/qa_page.py` — async worker, token signals, session management |
| Modify | `main.py` — `get_chat_model()`, rebuild on `provider_changed` |
| Modify | `tests/test_agent/test_qa_graph.py` — update `resolve_scope`/`retrieve_chunks` calls to pass `config` |

---

## Error Handling

- `lc_model` is `None` (provider not configured): `generate_answer` returns the prompt message immediately; `token_received` never fires.
- LLM API error during streaming: caught in `_run_async`, `error` signal emitted, `done` signal still fires.
- BM25 search on empty chunks list: returns empty list; FAISS result is used alone.
- `aget_state` failure after streaming: citations default to `[]`, answer is still displayed.

## Testing

| Test file | What it verifies |
|-----------|-----------------|
| `tests/test_retrieval/test_hybrid_search.py` | RRF merge correctness with mocked FAISS, real BM25 |
| `tests/test_agent/test_qa_stream.py` | `resolve_scope` + `retrieve_chunks` node functions (call with `(state, config)` signature); `generate_answer` node with mock `lc_model` in config returning a coroutine |
| `tests/test_agent/test_qa_graph.py` | Must continue to pass — update call sites to pass config dict `{"configurable": {"conn": ..., "embedder": ...}}` |

## Dependencies Added

```toml
"rank_bm25>=0.4",
"langchain-openai>=0.3",
"langchain-core>=0.3",
```
