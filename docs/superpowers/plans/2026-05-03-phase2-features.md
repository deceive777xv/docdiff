# Phase 2 Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the three remaining Phase 2 items from the design doc: (A) enhance 实质修改/重写 diff classification using similarity scores, (B) add 对比文档 as a fourth QA retrieval scope, (C) add diff report export to Word and HTML.

**Architecture:** Three independent feature sets sharing no runtime dependencies — each can be executed and committed separately. Feature A modifies one diff module; Feature B extends types + qa_graph + qa_page; Feature C adds a new service module and a UI export button.

**Tech Stack:** Python 3.11, PySide6, python-docx (already in deps), LangGraph StateGraph, SQLite, FAISS

> **Note on independence:** These three features are fully independent subsystems. They are grouped in one plan for convenience but can be executed in any order. If you hit trouble on one, skip to the next.

---

## File Map

### Feature A — 实质修改/重写分类
| Action | File |
|--------|------|
| Modify | `app/core/diff/diff_classifier.py` |
| Extend | `tests/test_diff/test_diff_classifier.py` |

### Feature B — 对比文档 QA 模式
| Action | File |
|--------|------|
| Modify | `app/core/types.py` — add `COMPARE` to `RetrievalScope` |
| Modify | `app/agent/qa_graph.py` — handle `"compare"` scope in `resolve_scope` |
| Modify | `app/ui/pages/qa_page.py` — add compare-task selector |
| Modify | `main.py` — call `qa.refresh_compare_tasks()` |
| Create | `tests/test_agent/test_qa_graph.py` |
| Create | `tests/test_agent/__init__.py` |

### Feature C — 差异报告导出
| Action | File |
|--------|------|
| Create | `app/services/report_service.py` |
| Create | `tests/test_services/test_report_service.py` |
| Modify | `app/ui/pages/compare_page.py` — add export button |

---

## Feature A: 实质修改/重写分类

The current `_rule_classify` never returns "重写"; the LLM prompt doesn't know the similarity score. Adding `similarity: float` to classifiers lets us detect structural rewrites (similarity < 0.3) by rule and guide the LLM with a numeric hint.

---

### Task 1: Write failing tests for similarity-aware classification

**Files:**
- Extend: `tests/test_diff/test_diff_classifier.py`

- [ ] **Step 1: Append the following tests to `tests/test_diff/test_diff_classifier.py`**

```python
def test_rule_classify_rewrite_on_low_similarity():
    """similarity < 0.3 → '重写', risk 'high' regardless of text patterns."""
    from app.core.diff.diff_classifier import _rule_classify

    dtype, risk, _ = _rule_classify(
        "甲方负责提供全部原材料及质检。",
        "乙方承担所有运输费用并负责到货验收。",
        similarity=0.15,
    )
    assert dtype == "重写"
    assert risk == "high"


def test_rule_classify_substantial_above_rewrite_threshold():
    """similarity >= 0.3 still detects 实质修改 when numbers differ."""
    from app.core.diff.diff_classifier import _rule_classify

    dtype, risk, _ = _rule_classify(
        "合同金额为100万元，付款期限30日。",
        "合同金额为200万元，付款期限60日。",
        similarity=0.85,
    )
    assert dtype == "实质修改"
    assert risk == "high"


def test_rule_classify_minor_above_rewrite_threshold():
    """similarity >= 0.3 and no structural triggers → '微调'."""
    from app.core.diff.diff_classifier import _rule_classify

    dtype, _, _ = _rule_classify(
        "甲方应当按时履行义务。",
        "甲方须按期完成履约。",
        similarity=0.85,
    )
    assert dtype == "微调"


def test_classify_passes_similarity_to_rule_classifier():
    """classify() with low similarity → DiffItem.diff_type == '重写'."""
    from app.core.diff.diff_classifier import classify
    from app.core.diff.semantic_matcher import ParagraphPair

    b_para = make_para("甲方负责原材料供应及质量控制，费用由甲方承担。")
    t_para = make_para("乙方承担所有物流运输及到货验收责任，费用另行结算。")
    pp = ParagraphPair(baseline_para=b_para, target_para=t_para, similarity=0.1)

    result = classify(
        para_pairs=[pp],
        policy=_NO_LLM_POLICY,
        provider=None,  # type: ignore[arg-type]
        task_id="t2",
        baseline_version_id="b2",
        target_version_id="v2",
    )

    assert result.items[0].diff_type == "重写"
    assert result.items[0].risk_level == "high"
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_diff/test_diff_classifier.py::test_rule_classify_rewrite_on_low_similarity -v
```
Expected: `FAILED` — `TypeError: _rule_classify() takes 2 positional arguments but 3 were given`

---

### Task 2: Implement similarity-aware classification

**Files:**
- Modify: `app/core/diff/diff_classifier.py`

- [ ] **Step 1: Replace the entire file with the updated implementation**

```python
"""Classify paragraph pairs into structured DiffItems."""
from __future__ import annotations
import json
import logging
import re
import uuid

from app.core.diff.semantic_matcher import ParagraphPair
from app.core.model.base_provider import BaseProvider
from app.core.types import ComparePolicy, DiffItem, DiffResult

logger = logging.getLogger(__name__)

_CLASSIFY_PROMPT = """你是一个专业的文档差异分析助手。请分析以下两段文本之间的差异，并给出结构化判断。

原文：
{baseline}

修改后：
{target}

向量相似度（0~1，越低越不同）：{similarity:.2f}

请以JSON格式回答，只输出JSON，不要有任何其他内容：
{{
  "diff_type": "微调|实质修改|重写|格式变化",
  "risk_level": "high|medium|low",
  "explanation": "简短的差异说明（30字以内）"
}}

判断规则：
- 格式变化：仅排版、标点、序号变化，语义完全相同
- 微调：措辞调整，核心意思不变（相似度通常 > 0.8）
- 实质修改：金额、日期、责任主体、权利义务等核心内容变化（相似度通常 0.3~0.8）
- 重写：段落大幅改写，原结构基本不保留（相似度通常 < 0.4）
"""


def _rule_classify(baseline: str, target: str, similarity: float = 1.0) -> tuple[str, str, str]:
    """Quick rule-based classification as fallback or supplement."""
    if re.sub(r'\s+', '', baseline) == re.sub(r'\s+', '', target):
        return "格式变化", "low", "仅格式变化"

    if similarity < 0.3:
        return "重写", "high", "文本结构大幅调整"

    numbers_b = set(re.findall(r'\d+[\.,]?\d*', baseline))
    numbers_t = set(re.findall(r'\d+[\.,]?\d*', target))
    neg_b = set(re.findall(r'[不无未没]', baseline))
    neg_t = set(re.findall(r'[不无未没]', target))
    oblig_b = set(re.findall(r'(?:应|须|必须|不得|禁止)', baseline))
    oblig_t = set(re.findall(r'(?:应|须|必须|不得|禁止)', target))

    if numbers_b != numbers_t or neg_b != neg_t or oblig_b != oblig_t:
        return "实质修改", "high", "关键数值或义务条款发生变化"

    return "微调", "medium", "措辞有所调整"


def _llm_classify(
    baseline: str,
    target: str,
    provider: BaseProvider,
    similarity: float = 1.0,
) -> tuple[str, str, str]:
    prompt = _CLASSIFY_PROMPT.format(
        baseline=baseline[:500],
        target=target[:500],
        similarity=similarity,
    )
    try:
        response = provider.chat([{"role": "user", "content": prompt}])
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return (
                data.get("diff_type", "微调"),
                data.get("risk_level", "medium"),
                data.get("explanation", ""),
            )
    except Exception as e:
        logger.warning("LLM classification failed, using rules: %s", e)
    return _rule_classify(baseline, target, similarity)


def classify(
    para_pairs: list[ParagraphPair],
    policy: ComparePolicy,
    provider: BaseProvider | None,
    task_id: str,
    baseline_version_id: str,
    target_version_id: str,
) -> DiffResult:
    items: list[DiffItem] = []
    for pp in para_pairs:
        if pp.baseline_para is None and pp.target_para is not None:
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type="新增",
                risk_level="medium",
                baseline_text="",
                target_text=pp.target_para.text,
                similarity_score=0.0,
                explanation="目标文档新增段落",
                baseline_page=0,
                target_page=pp.target_para.page_no,
            ))
        elif pp.baseline_para is not None and pp.target_para is None:
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type="删减",
                risk_level="medium",
                baseline_text=pp.baseline_para.text,
                target_text="",
                similarity_score=0.0,
                explanation="基准文档段落被删除",
                baseline_page=pp.baseline_para.page_no,
                target_page=0,
            ))
        elif pp.baseline_para is not None and pp.target_para is not None:
            if policy.use_llm_classify and provider is not None:
                diff_type, risk_level, explanation = _llm_classify(
                    pp.baseline_para.text, pp.target_para.text, provider, pp.similarity
                )
            else:
                diff_type, risk_level, explanation = _rule_classify(
                    pp.baseline_para.text, pp.target_para.text, pp.similarity
                )
            if policy.rule_strengthen:
                _, rule_risk, _ = _rule_classify(
                    pp.baseline_para.text, pp.target_para.text, pp.similarity
                )
                if rule_risk == "high" and risk_level != "high":
                    risk_level = "high"
            items.append(DiffItem(
                diff_id=str(uuid.uuid4()),
                section_path=pp.section_path,
                diff_type=diff_type,
                risk_level=risk_level,
                baseline_text=pp.baseline_para.text,
                target_text=pp.target_para.text,
                similarity_score=pp.similarity,
                explanation=explanation,
                baseline_page=pp.baseline_para.page_no,
                target_page=pp.target_para.page_no,
            ))
    return DiffResult(
        task_id=task_id,
        baseline_version_id=baseline_version_id,
        target_version_id=target_version_id,
        items=items,
    )
```

- [ ] **Step 2: Run the new tests**

```
uv run pytest tests/test_diff/test_diff_classifier.py -v
```
Expected: all tests PASS (4 existing + 4 new = 8 total)

- [ ] **Step 3: Run full suite to check for regressions**

```
uv run pytest tests/ -q
```
Expected: all 116+ tests pass

- [ ] **Step 4: Commit**

```bash
git add app/core/diff/diff_classifier.py tests/test_diff/test_diff_classifier.py
git commit -m "feat: similarity-aware 重写/实质修改 classification in diff_classifier"
```

---

## Feature B: 对比文档 QA 模式

Adds a fourth QA retrieval scope — "对比文档" — that searches across both documents of a completed compare task. Requires: a new enum value, a graph node branch, and a new combo box in the UI.

---

### Task 3: Add COMPARE to RetrievalScope

**Files:**
- Modify: `app/core/types.py`

- [ ] **Step 1: Open `app/core/types.py` and find the `RetrievalScope` class. Add `COMPARE = "compare"` after `TARGET`**

Current (find this block):
```python
class RetrievalScope(str, Enum):
    CURRENT_DOC = "current_doc"
    BASELINE = "baseline"
    TARGET = "target"
    STANDARD_LIB = "standard_lib"
    ALL = "all"
```

Replace with:
```python
class RetrievalScope(str, Enum):
    CURRENT_DOC = "current_doc"
    BASELINE = "baseline"
    TARGET = "target"
    COMPARE = "compare"
    STANDARD_LIB = "standard_lib"
    ALL = "all"
```

- [ ] **Step 2: Verify the change doesn't break imports**

```
uv run pytest tests/test_types.py -v
```
Expected: PASS

---

### Task 4: Write failing test for compare scope in qa_graph

**Files:**
- Create: `tests/test_agent/__init__.py`
- Create: `tests/test_agent/test_qa_graph.py`

- [ ] **Step 1: Create `tests/test_agent/__init__.py`** (empty file)

- [ ] **Step 2: Create `tests/test_agent/test_qa_graph.py`**

```python
"""Tests for qa_graph node functions."""
from __future__ import annotations
import sqlite3


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


def test_resolve_scope_compare_returns_provided_version_ids():
    """scope='compare' with current_version_ids → _version_ids equals input list."""
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {
        "scope": "compare",
        "current_version_ids": ["baseline-v1", "target-v1"],
        "conn": conn,
    }
    result = resolve_scope(state)
    assert set(result["_version_ids"]) == {"baseline-v1", "target-v1"}
    conn.close()


def test_resolve_scope_compare_error_when_no_ids():
    """scope='compare' with empty current_version_ids → error."""
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {
        "scope": "compare",
        "current_version_ids": [],
        "conn": conn,
    }
    result = resolve_scope(state)
    assert "error" in result
    conn.close()


def test_resolve_scope_current_doc_unchanged():
    """Existing 'current_doc' scope still works after the compare branch is added."""
    from app.agent.qa_graph import resolve_scope

    conn = _make_conn()
    state = {
        "scope": "current_doc",
        "current_version_ids": ["v-abc"],
        "conn": conn,
    }
    result = resolve_scope(state)
    assert result["_version_ids"] == ["v-abc"]
    conn.close()
```

- [ ] **Step 3: Run to verify tests fail**

```
uv run pytest tests/test_agent/test_qa_graph.py -v
```
Expected: `test_resolve_scope_compare_*` FAIL — `KeyError: '_version_ids'` or assertion error

---

### Task 5: Update qa_graph.py to handle "compare" scope

**Files:**
- Modify: `app/agent/qa_graph.py`

- [ ] **Step 1: In `app/agent/qa_graph.py`, replace the `resolve_scope` function**

Find:
```python
def resolve_scope(state: QAState) -> dict:
    """Map scope string to concrete version_id list."""
    try:
        scope = state.get("scope", "current_doc")
        conn = state["conn"]

        if scope == "current_doc":
            ids = list(state.get("current_version_ids") or [])
            if not ids:
                return {"error": "当前文档范围未指定版本，请先选择文档。", "status": "failed"}
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
```

Replace with:
```python
def resolve_scope(state: QAState) -> dict:
    """Map scope string to concrete version_id list."""
    try:
        scope = state.get("scope", "current_doc")
        conn = state["conn"]

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
```

- [ ] **Step 2: Run the agent tests**

```
uv run pytest tests/test_agent/test_qa_graph.py -v
```
Expected: all 3 tests PASS

---

### Task 6: Update qa_page.py to add compare-task selector

**Files:**
- Modify: `app/ui/pages/qa_page.py`
- Modify: `main.py`

- [ ] **Step 1: Replace the entire `app/ui/pages/qa_page.py` with the updated version**

Key changes: add `"对比文档"` to `_SCOPE_MAP`, add `_compare_task_combo`, add `refresh_compare_tasks()`, update `_on_scope_changed`, update `send_question`.

```python
"""QA page — chat-style retrieval-augmented question answering."""
from __future__ import annotations

import logging

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
    """Run qa_service.answer in a background thread."""

    result_ready = Signal(str, list)
    error = Signal(str)

    def __init__(
        self,
        data_dir: str,
        question: str,
        provider,
        embedder,
        scope: RetrievalScope,
        current_version_ids: list[str],
        parent=None,
    ):
        super().__init__(parent)
        self._data_dir = data_dir
        self._question = question
        self._provider = provider
        self._embedder = embedder
        self._scope = scope
        self._current_version_ids = current_version_ids

    def run(self) -> None:
        try:
            from app.agent.qa_graph import qa_graph
            from app.db.schema import open_db

            conn = open_db(self._data_dir)
            try:
                result = qa_graph.invoke({
                    "data_dir": self._data_dir,
                    "question": self._question,
                    "scope": self._scope.value,
                    "current_version_ids": self._current_version_ids,
                    "provider": self._provider,
                    "embedder": self._embedder,
                    "conn": conn,
                })
            finally:
                conn.close()

            if result.get("error"):
                self.error.emit(result["error"])
            else:
                self.result_ready.emit(result["answer"], result.get("citations", []))
        except Exception as exc:
            logger.exception("QA worker failed")
            self.error.emit(str(exc))


class QaPage(QWidget):
    """Chat-style QA page with RAG backend."""

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._threads: set[QThread] = set()
        self._build_ui()
        self.refresh_documents()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")

        # ── Top: scope and document selectors ─────────────────────────────────
        top_group = QGroupBox("检索配置")
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
        top_layout.addWidget(self._doc_combo)

        top_layout.addWidget(QLabel("对比任务："))
        self._compare_task_label = top_layout.itemAt(top_layout.count() - 1).widget()
        self._compare_task_combo = QComboBox()
        self._compare_task_combo.setMinimumWidth(280)
        self._compare_task_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_layout.addWidget(self._compare_task_combo)

        top_layout.addStretch()
        root.addWidget(top_group)

        # ── Middle: chat scroll area ───────────────────────────────────────────
        self._chat_scroll = QScrollArea()
        self._chat_scroll.setWidgetResizable(True)
        self._chat_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

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
        self._input.setMaximumHeight(80)
        self._input.setPlaceholderText("输入问题…")
        input_row.addWidget(self._input, 1)

        send_btn = QPushButton("发送")
        send_btn.setStyleSheet(Theme.btn_primary())
        send_btn.setFixedWidth(72)
        send_btn.clicked.connect(self.send_question)
        input_row.addWidget(send_btn)

        root.addLayout(input_row)

        # Initial visibility based on default scope
        self._on_scope_changed(self._scope_combo.currentText())

    # ── Public API ─────────────────────────────────────────────────────────────

    def refresh_documents(self) -> None:
        """Repopulate document combo from DB."""
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
        """Repopulate compare task combo from DB (completed tasks only)."""
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
        """Read input text, validate, and start QA worker."""
        question = self._input.toPlainText().strip()
        if not question:
            return

        if self.ctx.provider is None or self.ctx.embedder is None:
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
                current_version_ids = list(task_data)  # [baseline_id, target_id]

        thread = QThread()
        worker = _QaWorker(
            self.ctx.data_dir,
            question,
            self.ctx.provider,
            self.ctx.embedder,
            scope,
            current_version_ids,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result_ready.connect(self._on_answer)
        worker.result_ready.connect(thread.quit)
        worker.error.connect(self._on_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._threads.discard(thread))
        self._threads.add(thread)
        thread.start()

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_scope_changed(self, text: str) -> None:
        self._doc_combo.setVisible(text == "当前文档")
        self._compare_task_combo.setVisible(text == "对比文档")

    def _on_answer(self, answer_text: str, hits: list) -> None:
        self._add_message("assistant", answer_text, citations=hits)

    def _on_error(self, msg: str) -> None:
        self._add_message("assistant", f"错误：{msg}")

    # ── Message rendering ──────────────────────────────────────────────────────

    def _add_message(self, role: str, text: str, citations: list | None = None) -> None:
        """Add a chat bubble widget for user or assistant."""
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

        if not is_user and citations:
            cit_outer = QWidget()
            cit_layout = QHBoxLayout(cit_outer)
            cit_layout.setContentsMargins(0, 0, 0, 0)

            cit_parts: list[str] = []
            for hit in citations:
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

        self._chat_scroll.verticalScrollBar().setValue(
            self._chat_scroll.verticalScrollBar().maximum()
        )
```

> **Note on the "对比任务：" label widget:** The `QLabel("对比任务：")` is retrieved via `top_layout.itemAt(top_layout.count() - 1).widget()` — this is fragile. For clean code, keep a reference. After pasting, find `self._compare_task_label = top_layout.itemAt(top_layout.count() - 1).widget()` and replace the whole label+combo block with:

```python
        self._compare_task_label = QLabel("对比任务：")
        top_layout.addWidget(self._compare_task_label)
        self._compare_task_combo = QComboBox()
        self._compare_task_combo.setMinimumWidth(280)
        self._compare_task_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_layout.addWidget(self._compare_task_combo)
```

Also update `_on_scope_changed` to hide the label together with the combo:
```python
    def _on_scope_changed(self, text: str) -> None:
        self._doc_combo.setVisible(text == "当前文档")
        self._compare_task_label.setVisible(text == "对比文档")
        self._compare_task_combo.setVisible(text == "对比文档")
```

- [ ] **Step 2: Update `main.py` — add `refresh_compare_tasks()` call**

Find in `main.py`:
```python
    window.navigate_to(0)

    home.refresh()
    library.refresh()
    compare.refresh_versions()
    qa.refresh_documents()
```

Replace with:
```python
    window.navigate_to(0)

    home.refresh()
    library.refresh()
    compare.refresh_versions()
    qa.refresh_documents()
    qa.refresh_compare_tasks()
```

Also find `_rebuild_providers`:
```python
def _rebuild_providers(ctx: AppContext, compare, qa) -> None:
    ...
    compare.refresh_versions()
    qa.refresh_documents()
```

Replace the last two lines with:
```python
    compare.refresh_versions()
    qa.refresh_documents()
    qa.refresh_compare_tasks()
```

- [ ] **Step 3: Run all tests**

```
uv run pytest tests/ -q
```
Expected: all tests pass (plus 3 new agent tests)

- [ ] **Step 4: Commit**

```bash
git add app/core/types.py app/agent/qa_graph.py app/ui/pages/qa_page.py main.py \
        tests/test_agent/__init__.py tests/test_agent/test_qa_graph.py
git commit -m "feat: add 对比文档 QA scope for cross-document retrieval"
```

---

## Feature C: 差异报告导出

Adds Word (.docx) and HTML export of any completed comparison result. A new service module handles formatting; the compare page gets an "导出报告" button enabled after a comparison completes.

---

### Task 7: Write failing tests for report_service

**Files:**
- Create: `tests/test_services/test_report_service.py`

- [ ] **Step 1: Create `tests/test_services/test_report_service.py`**

```python
"""Tests for report_service.py — docx and HTML export."""
from __future__ import annotations
import uuid
from pathlib import Path

import pytest

from app.core.types import DiffItem, DiffResult


def _make_result() -> DiffResult:
    items = [
        DiffItem(
            diff_id=str(uuid.uuid4()),
            section_path="第1条",
            diff_type="实质修改",
            risk_level="high",
            baseline_text="合同金额为100万元，付款期限30日。",
            target_text="合同金额为200万元，付款期限60日。",
            similarity_score=0.82,
            explanation="金额和期限均发生变化",
            baseline_page=1,
            target_page=1,
        ),
        DiffItem(
            diff_id=str(uuid.uuid4()),
            section_path="第2条",
            diff_type="新增",
            risk_level="medium",
            baseline_text="",
            target_text="乙方须在交货前提交质量证明文件。",
            similarity_score=0.0,
            explanation="目标文档新增段落",
            baseline_page=0,
            target_page=2,
        ),
        DiffItem(
            diff_id=str(uuid.uuid4()),
            section_path="第3条",
            diff_type="重写",
            risk_level="high",
            baseline_text="甲方负责原材料供应。",
            target_text="乙方承担全部物流及验收责任，费用另行结算。",
            similarity_score=0.12,
            explanation="文本结构大幅调整",
            baseline_page=2,
            target_page=3,
        ),
    ]
    return DiffResult(
        task_id="test-task-001",
        baseline_version_id="baseline-v1",
        target_version_id="target-v1",
        items=items,
    )


def test_export_docx_creates_file(tmp_path):
    from app.services.report_service import export_docx

    out = str(tmp_path / "report.docx")
    export_docx(_make_result(), out)

    assert Path(out).exists()
    assert Path(out).stat().st_size > 1000  # non-trivial file


def test_export_docx_is_valid_docx(tmp_path):
    from docx import Document
    from app.services.report_service import export_docx

    out = str(tmp_path / "report.docx")
    export_docx(_make_result(), out)

    doc = Document(out)
    full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "文档对比报告" in full_text
    assert "实质修改" in full_text
    assert "重写" in full_text


def test_export_html_creates_file(tmp_path):
    from app.services.report_service import export_html

    out = str(tmp_path / "report.html")
    export_html(_make_result(), out)

    assert Path(out).exists()
    assert Path(out).stat().st_size > 500


def test_export_html_contains_required_content(tmp_path):
    from app.services.report_service import export_html

    out = str(tmp_path / "report.html")
    export_html(_make_result(), out)

    content = Path(out).read_text(encoding="utf-8")
    assert "文档对比报告" in content
    assert "实质修改" in content
    assert "新增" in content
    assert "重写" in content
    assert "第1条" in content
    assert "<!DOCTYPE html>" in content


def test_export_html_escapes_html_characters(tmp_path):
    from app.services.report_service import export_html

    items = [
        DiffItem(
            diff_id=str(uuid.uuid4()),
            section_path="<script>alert(1)</script>",
            diff_type="微调",
            risk_level="low",
            baseline_text="text <b>bold</b>",
            target_text="text & more",
            similarity_score=0.9,
            explanation="",
            baseline_page=1,
            target_page=1,
        )
    ]
    result = DiffResult(
        task_id="t",
        baseline_version_id="b",
        target_version_id="v",
        items=items,
    )
    out = str(tmp_path / "report.html")
    export_html(result, out)

    content = Path(out).read_text(encoding="utf-8")
    assert "<script>" not in content  # escaped
    assert "&lt;script&gt;" in content
```

- [ ] **Step 2: Run to verify they fail**

```
uv run pytest tests/test_services/test_report_service.py -v
```
Expected: all 5 FAIL — `ModuleNotFoundError: No module named 'app.services.report_service'`

---

### Task 8: Implement report_service.py

**Files:**
- Create: `app/services/report_service.py`

- [ ] **Step 1: Create `app/services/report_service.py`**

```python
"""Export diff results to Word (.docx) or HTML."""
from __future__ import annotations

import html as html_mod
from collections import Counter
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

from app.core.types import DiffResult

_RISK_LABELS = {"high": "高风险", "medium": "中风险", "low": "低风险"}

_DIFF_COLORS_HEX: dict[str, str] = {
    "新增":     "22c55e",
    "删减":     "ef4444",
    "微调":     "eab308",
    "实质修改": "f97316",
    "重写":     "a855f7",
    "格式变化": "9ca3af",
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def export_docx(result: DiffResult, output_path: str) -> None:
    """Export DiffResult to a Word document at output_path."""
    doc = Document()

    doc.add_heading("文档对比报告", 0)
    doc.add_paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph(f"差异总数：{len(result.items)}")

    counts = Counter(item.diff_type for item in result.items)
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Shading"
    hdr = table.rows[0].cells
    hdr[0].text = "差异类型"
    hdr[1].text = "数量"
    for dtype, count in counts.most_common():
        row = table.add_row().cells
        row[0].text = dtype
        row[1].text = str(count)

    doc.add_page_break()
    doc.add_heading("差异详情", 1)

    for i, item in enumerate(result.items, 1):
        doc.add_heading(f"{i}. [{item.diff_type}] {item.section_path}", 2)
        risk_label = _RISK_LABELS.get(item.risk_level, item.risk_level)
        doc.add_paragraph(f"风险等级：{risk_label}  |  相似度：{item.similarity_score:.3f}")
        if item.baseline_text:
            p = doc.add_paragraph()
            run = p.add_run("基准：")
            run.bold = True
            p.add_run(item.baseline_text[:400])
        if item.target_text:
            p = doc.add_paragraph()
            run = p.add_run("目标：")
            run.bold = True
            p.add_run(item.target_text[:400])
        if item.explanation:
            p = doc.add_paragraph(f"说明：{item.explanation}")
            p.runs[0].font.size = Pt(10)
            r, g, b = _hex_to_rgb("6b7280")
            p.runs[0].font.color.rgb = RGBColor(r, g, b)

    doc.save(output_path)


def export_html(result: DiffResult, output_path: str) -> None:
    """Export DiffResult to a standalone HTML report at output_path."""
    counts = Counter(item.diff_type for item in result.items)
    stats_rows = "".join(
        f"<tr><td>{html_mod.escape(dt)}</td><td>{cnt}</td></tr>"
        for dt, cnt in counts.most_common()
    )

    items_html_parts: list[str] = []
    for item in result.items:
        color = _DIFF_COLORS_HEX.get(item.diff_type, "9ca3af")
        risk = _RISK_LABELS.get(item.risk_level, item.risk_level)
        section = html_mod.escape(item.section_path)
        b_text = html_mod.escape(item.baseline_text[:400]) if item.baseline_text else ""
        t_text = html_mod.escape(item.target_text[:400]) if item.target_text else ""
        exp = html_mod.escape(item.explanation) if item.explanation else ""

        baseline_block = (
            f'<div class="baseline"><strong>基准：</strong>{b_text}</div>' if b_text else ""
        )
        target_block = (
            f'<div class="target"><strong>目标：</strong>{t_text}</div>' if t_text else ""
        )
        explain_block = (
            f'<div class="explain">说明：{exp}</div>' if exp else ""
        )

        items_html_parts.append(f"""
    <div class="diff-card" style="border-left:4px solid #{color}">
      <div class="header">
        <span class="badge" style="background:#{color}">{html_mod.escape(item.diff_type)}</span>
        <span class="risk">{html_mod.escape(risk)}</span>
        <span class="section">{section}</span>
        <span class="sim">相似度：{item.similarity_score:.3f}</span>
      </div>
      {baseline_block}
      {target_block}
      {explain_block}
    </div>""")

    items_html = "\n".join(items_html_parts)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    html_content = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>文档对比报告</title>
  <style>
    body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; max-width: 1000px;
           margin: 40px auto; color: #1f2937; background: #f9fafb; }}
    h1 {{ color: #1e3a5f; margin-bottom: 4px; }}
    .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 24px; }}
    h2 {{ color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; margin: 16px 0; background: white;
             border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
    td, th {{ border: 1px solid #e5e7eb; padding: 8px 16px; text-align: left; }}
    th {{ background: #f3f4f6; font-weight: 600; }}
    .diff-card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px;
                  padding: 12px 16px; margin: 12px 0;
                  box-shadow: 0 1px 2px rgba(0,0,0,.05); }}
    .header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
    .badge {{ color: white; border-radius: 4px; padding: 2px 8px;
              font-size: 12px; font-weight: bold; }}
    .risk {{ font-size: 12px; color: #6b7280; }}
    .section {{ font-size: 13px; font-weight: 600; }}
    .sim {{ font-size: 11px; color: #9ca3af; margin-left: auto; }}
    .baseline {{ background: #fef2f2; border-radius: 4px; padding: 6px 10px;
                 margin: 6px 0; font-size: 13px; line-height: 1.6; }}
    .target {{ background: #f0fdf4; border-radius: 4px; padding: 6px 10px;
               margin: 6px 0; font-size: 13px; line-height: 1.6; }}
    .explain {{ color: #6b7280; font-size: 12px; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>文档对比报告</h1>
  <p class="meta">生成时间：{generated_at} &nbsp;|&nbsp; 差异总数：{len(result.items)}</p>
  <h2>差异统计</h2>
  <table>
    <tr><th>差异类型</th><th>数量</th></tr>
    {stats_rows}
  </table>
  <h2>差异详情</h2>
  {items_html}
</body>
</html>"""

    Path(output_path).write_text(html_content, encoding="utf-8")
```

- [ ] **Step 2: Run the report tests**

```
uv run pytest tests/test_services/test_report_service.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 3: Run full suite**

```
uv run pytest tests/ -q
```
Expected: all tests pass

---

### Task 9: Add export button to compare_page.py

**Files:**
- Modify: `app/ui/pages/compare_page.py`

The compare page already has `self._run_btn` and `self._loading_label` in the top bar group (`top_group`). We add an "导出报告" button in the same row, disabled until a result is available.

- [ ] **Step 1: In `_build_ui()`, find this block in the top bar section**

```python
        self._loading_label = QLabel("")
        self._loading_label.setStyleSheet(Theme.label_secondary())
        top_layout.addWidget(self._loading_label)

        root.addWidget(top_group)
```

Replace with:
```python
        self._loading_label = QLabel("")
        self._loading_label.setStyleSheet(Theme.label_secondary())
        top_layout.addWidget(self._loading_label)

        self._export_btn = QPushButton("导出报告")
        self._export_btn.setStyleSheet(
            f"background-color:transparent;color:{Theme.COLOR_PRIMARY};"
            f"border:1px solid {Theme.COLOR_PRIMARY};padding:6px 14px;"
            f"border-radius:{Theme.CARD_RADIUS}px;font-size:13px;"
        )
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_report)
        top_layout.addWidget(self._export_btn)

        root.addWidget(top_group)
```

- [ ] **Step 2: In `_on_compare_done`, add a line to enable the export button**

Find:
```python
    def _on_compare_done(self, result: DiffResult) -> None:
        self._current_result = result
        self._diff_items_by_id = {item.diff_id: item for item in result.items}
        self._loading_label.setText(f"完成！发现 {len(result.items)} 处差异。")
        self._run_btn.setEnabled(True)
```

Replace with:
```python
    def _on_compare_done(self, result: DiffResult) -> None:
        self._current_result = result
        self._diff_items_by_id = {item.diff_id: item for item in result.items}
        self._loading_label.setText(f"完成！发现 {len(result.items)} 处差异。")
        self._run_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
```

- [ ] **Step 3: Add the `_export_report` method to `ComparePage`**

Add after `_on_compare_error`:
```python
    def _export_report(self) -> None:
        """Open save dialog and export the current diff result."""
        if self._current_result is None:
            return
        from PySide6.QtWidgets import QFileDialog
        default_name = f"report_{self._current_result.task_id[:8]}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出报告",
            default_name,
            "Word 文件 (*.docx);;HTML 文件 (*.html)",
        )
        if not path:
            return
        try:
            from app.services.report_service import export_docx, export_html
            if path.lower().endswith(".html"):
                export_html(self._current_result, path)
            else:
                if not path.lower().endswith(".docx"):
                    path += ".docx"
                export_docx(self._current_result, path)
            QMessageBox.information(self, "导出成功", f"报告已保存至：\n{path}")
        except Exception as exc:
            logger.exception("Export failed")
            QMessageBox.critical(self, "导出失败", str(exc))
```

- [ ] **Step 4: Run full test suite**

```
uv run pytest tests/ -q
```
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add app/services/report_service.py \
        tests/test_services/test_report_service.py \
        app/ui/pages/compare_page.py
git commit -m "feat: diff report export to Word and HTML"
```

---

## Final verification

- [ ] **Run complete test suite one more time**

```
uv run pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: all tests green, no regressions.

- [ ] **Update design doc Phase 2 checklist**

In `docs/superpowers/specs/2026-05-01-doc-diff-agent-design.md`, update the Phase 2 section:

```markdown
### 第二阶段

- [x] LangGraph 编排层接入（2026-05-03 完成）
- [ ] OCR 增强（Tesseract）← 跳过，按需增加
- [x] 实质修改/重写分类（相似度感知，规则+LLM）
- [x] 对比/标准库/混合问答模式
- [x] 差异报告导出（Word/HTML）
```

- [ ] **Final commit**

```bash
git add docs/superpowers/specs/2026-05-01-doc-diff-agent-design.md
git commit -m "docs: mark Phase 2 features complete (skip OCR)"
```
