# UI Optimization + LangGraph Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded UI styles with a centralized Theme module (low-saturation palette, app icons), then migrate the three direct-call service workflows (ingest / compare / QA) to LangGraph StateGraphs, and add the missing "新增版本" button in LibraryPage (I5).

**Architecture:** A new `app/ui/theme.py` exports color constants consumed by all pages. Three new LangGraph graphs live in `app/agent/` and are invoked by the existing QThread workers instead of calling service functions directly; service functions are preserved unchanged and called from within graph nodes. Each graph receives an open `sqlite3.Connection` from the worker via state to avoid cross-thread connection sharing.

**Tech Stack:** PySide6 6.x, LangGraph ≥0.2.0, existing SQLite/FAISS/service layer unchanged.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `app/ui/theme.py` | **Create** | All color constants, QSS helper classmethods |
| `app/agent/__init__.py` | **Create** | Package marker |
| `app/agent/states.py` | **Create** | TypedDict state definitions (IngestState, CompareState, QAState) |
| `app/agent/ingest_graph.py` | **Create** | LangGraph StateGraph: file_check → parse_doc → save_document → build_embeddings |
| `app/agent/compare_graph.py` | **Create** | LangGraph StateGraph: create_task → ensure_parsed → do_align → do_semantic_compare → do_classify → persist_result |
| `app/agent/qa_graph.py` | **Create** | LangGraph StateGraph: resolve_scope → retrieve_chunks → generate_answer → attach_citations |
| `tests/test_agent/__init__.py` | **Create** | Package marker |
| `tests/test_agent/test_ingest_graph.py` | **Create** | Tests for ingest graph nodes and happy-path flow |
| `tests/test_agent/test_compare_graph.py` | **Create** | Tests for compare graph nodes and error propagation |
| `tests/test_agent/test_qa_graph.py` | **Create** | Tests for QA graph scope resolution and answer generation |
| `pyproject.toml` | **Modify** | Add `langgraph>=0.2.0` dependency |
| `main.py` | **Modify** | Set `QApplication.setWindowIcon` using `docdiff.ico` |
| `app/ui/main_window.py` | **Modify** | Import Theme; sidebar uses Theme colors + docdiff.png logo |
| `app/ui/pages/home_page.py` | **Modify** | Import Theme; replace all hardcoded styles and emoji |
| `app/ui/pages/library_page.py` | **Modify** | Import Theme; I5 patch (add-version button); wire worker to ingest_graph |
| `app/ui/pages/compare_page.py` | **Modify** | Import Theme; update diff colors; wire worker to compare_graph |
| `app/ui/pages/qa_page.py` | **Modify** | Import Theme; wire worker to qa_graph |
| `app/ui/pages/settings_page.py` | **Modify** | Import Theme; replace hardcoded styles |
| `assets/diff_template.html` | **Modify** | Update CSS diff highlight colors to match Theme |

---

## Task 1: Add langgraph dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Add langgraph to dependencies**

  Open `pyproject.toml` and replace the `dependencies` list:

  ```toml
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
  ]
  ```

- [ ] **Install**

  ```bash
  uv sync --dev
  ```

- [ ] **Verify**

  ```bash
  uv run python -c "import langgraph; print(langgraph.__version__)"
  ```

  Expected: prints a version string like `0.2.x`

- [ ] **Commit**

  ```bash
  git add pyproject.toml uv.lock
  git commit -m "deps: add langgraph>=0.2.0"
  ```

---

## Task 2: Create Theme module

**Files:**
- Create: `app/ui/theme.py`

- [ ] **Create `app/ui/theme.py`**

  ```python
  """Centralized UI theme constants — import this instead of hardcoding colors."""


  class Theme:
      # Layout
      SIDEBAR_WIDTH = 140
      PAGE_MARGIN = 24
      CARD_RADIUS = 8

      # Sidebar
      BG_SIDEBAR = "#1e2736"
      NAV_ACTIVE_BG = "#3b5080"
      NAV_ACTIVE_TEXT = "#ffffff"
      NAV_TEXT = "#c4cad8"
      LOGO_COLOR = "#a8b8d8"

      # Page areas
      BG_PAGE = "#f4f5f7"
      BG_CARD = "#ffffff"
      BG_HEADER = "#edf0f5"

      # Text
      TEXT_PRIMARY = "#2c3a52"
      TEXT_SECONDARY = "#6b7a99"
      TEXT_PLACEHOLDER = "#9ca3af"

      # Borders
      BORDER = "#dde1ea"

      # Action colors
      COLOR_PRIMARY = "#3d5fa0"
      COLOR_SUCCESS = "#3e7d6a"
      COLOR_DANGER = "#a04040"
      COLOR_WARNING = "#b09830"

      # Diff highlight colors — also used in assets/diff_template.html
      DIFF_ADDED = "#4a9e72"
      DIFF_DELETED = "#c05050"
      DIFF_MINOR = "#b09830"
      DIFF_MAJOR = "#c07840"
      DIFF_REWRITE = "#7a58c0"
      DIFF_FORMAT = "#9ca3af"

      # QSS helper classmethods
      @classmethod
      def btn_primary(cls) -> str:
          return (
              f"background-color:{cls.COLOR_PRIMARY};color:white;"
              f"border:none;border-radius:6px;padding:8px 16px;font-size:13px;"
          )

      @classmethod
      def btn_success(cls) -> str:
          return (
              f"background-color:{cls.COLOR_SUCCESS};color:white;"
              f"border:none;border-radius:6px;padding:8px 16px;font-size:13px;"
          )

      @classmethod
      def btn_danger(cls) -> str:
          return (
              f"background-color:{cls.COLOR_DANGER};color:white;"
              f"border:none;border-radius:6px;padding:8px 16px;font-size:13px;"
          )

      @classmethod
      def card(cls) -> str:
          return (
              f"background:{cls.BG_CARD};border:1px solid {cls.BORDER};"
              f"border-radius:{cls.CARD_RADIUS}px;"
          )

      @classmethod
      def label_primary(cls) -> str:
          return f"color:{cls.TEXT_PRIMARY};font-size:13px;"

      @classmethod
      def label_secondary(cls) -> str:
          return f"color:{cls.TEXT_SECONDARY};font-size:12px;"

      @classmethod
      def page_title(cls) -> str:
          return f"color:{cls.TEXT_PRIMARY};font-size:22px;font-weight:bold;"
  ```

- [ ] **Verify import**

  ```bash
  uv run python -c "from app.ui.theme import Theme; print(Theme.COLOR_PRIMARY)"
  ```

  Expected: `#3d5fa0`

- [ ] **Commit**

  ```bash
  git add app/ui/theme.py
  git commit -m "feat: add centralized UI theme module"
  ```

---

## Task 3: Update main_window.py and main.py

**Files:**
- Modify: `app/ui/main_window.py`
- Modify: `main.py`

- [ ] **Rewrite `app/ui/main_window.py`**

  ```python
  """Main application window with sidebar navigation."""
  from __future__ import annotations

  from pathlib import Path

  from PySide6.QtCore import Qt
  from PySide6.QtGui import QFont, QPixmap
  from PySide6.QtWidgets import (
      QHBoxLayout,
      QLabel,
      QMainWindow,
      QPushButton,
      QStackedWidget,
      QVBoxLayout,
      QWidget,
  )

  from app.ui.app_context import AppContext
  from app.ui.theme import Theme

  _NAV_ITEMS = [
      ("首页",     0),
      ("文档对比", 1),
      ("标准库",   2),
      ("智能问答", 3),
      ("设置",     4),
  ]

  _WINDOW_TITLE = "Doc-Diff-Agent"
  _WINDOW_SIZE = (1280, 800)
  _ICON_PATH = Path(__file__).parent.parent.parent / "assets" / "icons" / "docdiff.png"


  class NavButton(QPushButton):
      _ACTIVE_STYLE = (
          f"background-color:{Theme.NAV_ACTIVE_BG};color:{Theme.NAV_ACTIVE_TEXT};"
          "border:none;padding:12px 8px;text-align:left;font-size:14px;border-radius:6px;"
      )
      _INACTIVE_STYLE = (
          f"background-color:transparent;color:{Theme.NAV_TEXT};"
          "border:none;padding:12px 8px;text-align:left;font-size:14px;border-radius:6px;"
      )

      def __init__(self, label: str, parent=None):
          super().__init__(label, parent)
          self.setStyleSheet(self._INACTIVE_STYLE)
          self.setCursor(Qt.CursorShape.PointingHandCursor)
          self.setFixedWidth(Theme.SIDEBAR_WIDTH - 16)

      def set_active(self, active: bool) -> None:
          self.setStyleSheet(self._ACTIVE_STYLE if active else self._INACTIVE_STYLE)


  class SideBar(QWidget):
      def __init__(self, on_navigate, parent=None):
          super().__init__(parent)
          self.setFixedWidth(Theme.SIDEBAR_WIDTH)
          self.setStyleSheet(
              f"background-color:{Theme.BG_SIDEBAR};"
              f"border-right:1px solid {Theme.BORDER};"
          )

          layout = QVBoxLayout(self)
          layout.setContentsMargins(8, 16, 8, 16)
          layout.setSpacing(4)

          # Logo row: icon + text
          logo_row = QWidget()
          logo_layout = QHBoxLayout(logo_row)
          logo_layout.setContentsMargins(4, 0, 4, 0)
          logo_layout.setSpacing(8)

          logo_img = QLabel()
          if _ICON_PATH.exists():
              pix = QPixmap(str(_ICON_PATH)).scaled(
                  28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation
              )
              logo_img.setPixmap(pix)
          logo_layout.addWidget(logo_img)

          logo_text = QLabel("DocDiff")
          logo_text.setFont(QFont("", 14, QFont.Weight.Bold))
          logo_text.setStyleSheet(f"color:{Theme.LOGO_COLOR};")
          logo_layout.addWidget(logo_text)
          logo_layout.addStretch()
          layout.addWidget(logo_row)
          layout.addSpacing(16)

          self._buttons: list[NavButton] = []
          for label, idx in _NAV_ITEMS:
              btn = NavButton(label)
              btn.clicked.connect(lambda checked, i=idx: on_navigate(i))
              self._buttons.append(btn)
              layout.addWidget(btn)

          layout.addStretch()
          self._set_active(0)

      def _set_active(self, index: int) -> None:
          for i, btn in enumerate(self._buttons):
              btn.set_active(i == index)

      def navigate(self, index: int) -> None:
          self._set_active(index)


  class MainWindow(QMainWindow):
      def __init__(self, ctx: AppContext):
          super().__init__()
          self.ctx = ctx
          self.setWindowTitle(_WINDOW_TITLE)
          self.resize(*_WINDOW_SIZE)
          self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")

          central = QWidget()
          self.setCentralWidget(central)
          root_layout = QHBoxLayout(central)
          root_layout.setContentsMargins(0, 0, 0, 0)
          root_layout.setSpacing(0)

          self._sidebar = SideBar(self._on_navigate)
          root_layout.addWidget(self._sidebar)

          self._stack = QStackedWidget()
          self._stack.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
          root_layout.addWidget(self._stack, 1)

          for label, _ in _NAV_ITEMS:
              placeholder = QLabel(f"{label} 页面")
              placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
              placeholder.setStyleSheet(f"font-size:24px;color:{Theme.TEXT_PLACEHOLDER};")
              self._stack.addWidget(placeholder)

      def add_page(self, index: int, widget: QWidget) -> None:
          old = self._stack.widget(index)
          self._stack.insertWidget(index, widget)
          if old is not None:
              self._stack.removeWidget(old)
              old.deleteLater()

      def _on_navigate(self, index: int) -> None:
          self._sidebar.navigate(index)
          self._stack.setCurrentIndex(index)

      def navigate_to(self, index: int) -> None:
          self._on_navigate(index)
  ```

- [ ] **Add window icon in `main.py`**

  In `main.py`, after `app = QApplication(sys.argv)` add:

  ```python
  from PySide6.QtGui import QIcon
  _ico_path = Path(__file__).parent / "assets" / "icons" / "docdiff.ico"
  if _ico_path.exists():
      app.setWindowIcon(QIcon(str(_ico_path)))
  ```

- [ ] **Verify import**

  ```bash
  uv run python -c "from app.ui.main_window import MainWindow; print('OK')"
  ```

  Expected: `OK`

- [ ] **Commit**

  ```bash
  git add app/ui/main_window.py main.py
  git commit -m "feat: apply theme to main window, add app and sidebar icons"
  ```

---

## Task 4: Update home_page.py

**Files:**
- Modify: `app/ui/pages/home_page.py`

- [ ] **Add import at top**

  Add after `from app.ui.app_context import AppContext`:
  ```python
  from app.ui.theme import Theme
  ```

- [ ] **Replace `_StatCard.__init__` styles**

  Find the `_StatCard.__init__` method and replace:
  ```python
  # OLD
  self.setStyleSheet(
      f"background:{color}15;border:1px solid {color}40;"
      "border-radius:10px;padding:12px;"
  )
  val_lbl.setStyleSheet(f"font-size:28px;font-weight:bold;color:{color};")
  lbl.setStyleSheet("font-size:12px;color:#6b7280;")

  # NEW
  self.setStyleSheet(
      f"background:{color}20;border:1px solid {color}50;"
      f"border-radius:{Theme.CARD_RADIUS}px;padding:12px;"
  )
  val_lbl.setStyleSheet(f"font-size:26px;font-weight:bold;color:{color};")
  lbl.setStyleSheet(Theme.label_secondary())
  ```

- [ ] **Replace `HomePage._build_ui` hardcoded values**

  ```python
  # At start of _build_ui, add:
  self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
  layout.setContentsMargins(
      Theme.PAGE_MARGIN, Theme.PAGE_MARGIN, Theme.PAGE_MARGIN, Theme.PAGE_MARGIN
  )

  # Title:
  title.setStyleSheet(Theme.page_title())

  # Subtitle:
  subtitle.setStyleSheet(Theme.label_secondary())

  # Stat card colors:
  self._card_docs  = _StatCard("标准文档", "0", Theme.COLOR_PRIMARY)
  self._card_tasks = _StatCard("对比任务", "0", Theme.COLOR_SUCCESS)
  self._card_done  = _StatCard("已完成",   "0", "#7a58c0")

  # Quick actions — replace (label, page_idx, color) tuples:
  actions = [
      ("导入标准文档", 2, Theme.COLOR_PRIMARY),
      ("开始文档对比", 1, Theme.COLOR_SUCCESS),
      ("智能问答",     3, "#7a58c0"),
  ]

  # Action button style:
  btn.setStyleSheet(
      f"background-color:{color};color:white;padding:10px 20px;"
      f"border:none;border-radius:{Theme.CARD_RADIUS}px;font-size:13px;"
  )
  ```

- [ ] **Commit**

  ```bash
  git add app/ui/pages/home_page.py
  git commit -m "feat: apply theme to home page, remove emoji"
  ```

---

## Task 5: Update library_page.py (UI only)

**Files:**
- Modify: `app/ui/pages/library_page.py`

- [ ] **Add import**

  Add after existing imports:
  ```python
  from app.ui.theme import Theme
  ```

- [ ] **Replace styles in `_build_ui`**

  ```python
  # At start of _build_ui:
  self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
  layout.setContentsMargins(
      Theme.PAGE_MARGIN, Theme.PAGE_MARGIN, Theme.PAGE_MARGIN, Theme.PAGE_MARGIN
  )

  # Title:
  title.setStyleSheet(Theme.page_title())

  # Import button:
  import_btn.setStyleSheet(Theme.btn_primary())

  # Status label:
  self._status.setStyleSheet(Theme.label_secondary())

  # Table stylesheet (add after table widget creation):
  self._table.setStyleSheet(
      f"QTableWidget {{ background:{Theme.BG_CARD};gridline-color:{Theme.BORDER}; }}"
      f"QHeaderView::section {{ background:{Theme.BG_HEADER};color:{Theme.TEXT_PRIMARY};"
      f"border:1px solid {Theme.BORDER};padding:4px; }}"
  )
  ```

- [ ] **Commit**

  ```bash
  git add app/ui/pages/library_page.py
  git commit -m "feat: apply theme to library page (UI only)"
  ```

---

## Task 6: Update compare_page.py (UI only)

**Files:**
- Modify: `app/ui/pages/compare_page.py`

- [ ] **Add import**

  ```python
  from app.ui.theme import Theme
  ```

- [ ] **Replace `_DIFF_CSS` dict** (near top of file after imports):

  ```python
  _DIFF_CSS: dict[str, tuple[str, str]] = {
      "新增":     ("added",   Theme.DIFF_ADDED),
      "删减":     ("deleted", Theme.DIFF_DELETED),
      "微调":     ("minor",   Theme.DIFF_MINOR),
      "实质修改": ("major",   Theme.DIFF_MAJOR),
      "重写":     ("rewrite", Theme.DIFF_REWRITE),
      "格式变化": ("format",  Theme.DIFF_FORMAT),
  }
  ```

- [ ] **Replace `_RISK_COLORS` dict**:

  ```python
  _RISK_COLORS: dict[str, str] = {
      "high":   Theme.DIFF_DELETED,
      "medium": Theme.DIFF_MAJOR,
      "low":    Theme.DIFF_ADDED,
  }
  ```

- [ ] **Update `_build_ui`**:

  ```python
  # At start of _build_ui:
  self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")

  # Run button:
  self._run_btn.setStyleSheet(Theme.btn_primary())

  # Loading label:
  self._loading_label.setStyleSheet(Theme.label_secondary())
  ```

- [ ] **Update `_make_diff_card` baseline/target text backgrounds**:

  ```python
  # baseline text label:
  b_lbl.setStyleSheet(
      f"color:{Theme.TEXT_PRIMARY};font-size:12px;"
      f"background:{Theme.DIFF_DELETED}20;border-radius:3px;padding:3px 5px;"
  )
  # target text label:
  t_lbl.setStyleSheet(
      f"color:{Theme.TEXT_PRIMARY};font-size:12px;"
      f"background:{Theme.DIFF_ADDED}20;border-radius:3px;padding:3px 5px;"
  )
  # similarity label:
  sim_lbl.setStyleSheet(Theme.label_secondary())
  # section label:
  section_lbl.setStyleSheet(Theme.label_secondary())
  ```

- [ ] **Commit**

  ```bash
  git add app/ui/pages/compare_page.py
  git commit -m "feat: apply theme to compare page"
  ```

---

## Task 7: Update qa_page.py (UI only)

**Files:**
- Modify: `app/ui/pages/qa_page.py`

- [ ] **Add import**

  ```python
  from app.ui.theme import Theme
  ```

- [ ] **Replace module-level bubble style constants**:

  ```python
  _USER_BUBBLE_STYLE = (
      f"background:{Theme.COLOR_PRIMARY};color:white;"
      "border-radius:12px;padding:10px;margin:4px 0;"
  )
  _ASST_BUBBLE_STYLE = (
      f"background:{Theme.BG_CARD};border:1px solid {Theme.BORDER};"
      "border-radius:12px;padding:10px;margin:4px 0;"
  )
  ```

- [ ] **Update `_build_ui`**:

  ```python
  # At start of _build_ui:
  self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")

  # Send button:
  send_btn.setStyleSheet(Theme.btn_primary())
  ```

- [ ] **Update citation label in `_add_message`**:

  ```python
  cit_lbl.setStyleSheet(f"color:{Theme.TEXT_PLACEHOLDER};font-size:11px;margin-left:4px;")
  ```

- [ ] **Commit**

  ```bash
  git add app/ui/pages/qa_page.py
  git commit -m "feat: apply theme to QA page"
  ```

---

## Task 8: Update settings_page.py

**Files:**
- Modify: `app/ui/pages/settings_page.py`

- [ ] **Add import**

  ```python
  from app.ui.theme import Theme
  ```

- [ ] **Update `_build_ui`**:

  ```python
  # At start of _build_ui:
  self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
  outer.setContentsMargins(
      Theme.PAGE_MARGIN, Theme.PAGE_MARGIN, Theme.PAGE_MARGIN, Theme.PAGE_MARGIN
  )

  # Title:
  title.setStyleSheet(Theme.page_title())

  # Save button:
  save_btn.setStyleSheet(
      f"background-color:{Theme.COLOR_PRIMARY};color:white;padding:10px 24px;"
      f"border:none;border-radius:6px;font-size:14px;"
  )
  ```

- [ ] **Commit**

  ```bash
  git add app/ui/pages/settings_page.py
  git commit -m "feat: apply theme to settings page"
  ```

---

## Task 9: Update diff_template.html colors

**Files:**
- Modify: `assets/diff_template.html`

- [ ] **Update CSS diff class colors**

  In `assets/diff_template.html`, find the `<style>` block and replace all diff-type color rules. The new rules must match `Theme.DIFF_*` values:

  ```css
  .diff-item.added   { background: #4a9e7220; border-left: 3px solid #4a9e72; }
  .diff-item.deleted { background: #c0505020; border-left: 3px solid #c05050; }
  .diff-item.minor   { background: #b0983020; border-left: 3px solid #b09830; }
  .diff-item.major   { background: #c0784020; border-left: 3px solid #c07840; }
  .diff-item.rewrite { background: #7a58c020; border-left: 3px solid #7a58c0; }
  .diff-item.format  { background: #9ca3af20; border-left: 3px solid #9ca3af; }
  ```

- [ ] **Commit**

  ```bash
  git add assets/diff_template.html
  git commit -m "feat: update diff template with low-saturation colors"
  ```

---

## Task 10: Verify UI changes don't break existing tests

- [ ] **Run full test suite**

  ```bash
  uv run pytest tests/ -x -q
  ```

  Expected: 105 tests pass. Fix any failures before continuing.

---

## Task 11: Create agent package and states

**Files:**
- Create: `app/agent/__init__.py`
- Create: `app/agent/states.py`
- Create: `tests/test_agent/__init__.py`

- [ ] **Create `app/agent/__init__.py`** (empty file)

- [ ] **Create `tests/test_agent/__init__.py`** (empty file)

- [ ] **Create `app/agent/states.py`**

  ```python
  """TypedDict state definitions for LangGraph workflows."""
  from __future__ import annotations

  from typing import Any, Optional

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
      provider: Any
      embedder: Any
      conn: Any

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

- [ ] **Verify import**

  ```bash
  uv run python -c "from app.agent.states import IngestState, CompareState, QAState; print('OK')"
  ```

  Expected: `OK`

- [ ] **Commit**

  ```bash
  git add app/agent/ tests/test_agent/
  git commit -m "feat: add agent package and TypedDict state definitions"
  ```

---

## Task 12: Create ingest_graph with tests (TDD)

**Files:**
- Create: `tests/test_agent/test_ingest_graph.py`
- Create: `app/agent/ingest_graph.py`

- [ ] **Write failing tests** (`tests/test_agent/test_ingest_graph.py`):

  ```python
  """Tests for ingest_graph LangGraph workflow."""
  from __future__ import annotations

  from unittest.mock import MagicMock, patch

  import pytest


  def test_file_check_missing_file():
      """file_check sets error when file does not exist."""
      from app.agent.ingest_graph import file_check

      result = file_check({
          "file_path": "/nonexistent/file.pdf",
          "data_dir": "/tmp",
          "source_type": "standard",
          "conn": MagicMock(),
      })
      assert result.get("error") is not None
      assert result.get("status") == "failed"


  def test_file_check_duplicate(tmp_path):
      """file_check sets error when doc hash already exists in DB."""
      from app.agent.ingest_graph import file_check

      doc = tmp_path / "test.pdf"
      doc.write_bytes(b"%PDF-1.4 test")

      with patch(
          "app.agent.ingest_graph.document_repo.get_document_by_hash",
          return_value={"id": "existing-id"},
      ):
          result = file_check({
              "file_path": str(doc),
              "data_dir": str(tmp_path),
              "source_type": "standard",
              "document_id": None,
              "conn": MagicMock(),
          })
      assert result.get("error") is not None
      assert result.get("status") == "failed"


  def test_graph_propagates_error_to_end():
      """Graph reaches END immediately when file_check sets error."""
      from app.agent.ingest_graph import ingest_graph

      result = ingest_graph.invoke({
          "file_path": "/nonexistent/file.pdf",
          "data_dir": "/tmp",
          "source_type": "standard",
          "conn": MagicMock(),
      })
      assert result.get("error") is not None
      assert result.get("status") == "failed"
      assert not result.get("doc_id")


  def test_graph_happy_path(tmp_path):
      """Full happy path: file exists, no duplicate, parse succeeds."""
      from app.agent.ingest_graph import ingest_graph
      from app.core.types import DocumentIR, ParseQualityReport

      doc = tmp_path / "test.pdf"
      doc.write_bytes(b"%PDF-1.4 test")

      mock_ir = DocumentIR(
          doc_id="doc-uuid",
          title="Test",
          file_hash="abc123",
          sections=[],
          plain_text="",
      )
      mock_quality = ParseQualityReport(needs_ocr=False, ocr_pages=[])

      with (
          patch("app.agent.ingest_graph.document_repo.get_document_by_hash", return_value=None),
          patch("app.agent.ingest_graph.parse_document", return_value=(mock_ir, mock_quality)),
          patch("app.agent.ingest_graph.document_repo.insert_document", return_value="doc-123"),
          patch("app.agent.ingest_graph.document_repo.insert_version", return_value="ver-456"),
          patch("app.agent.ingest_graph.chunk_repo.insert_chunks"),
          patch("app.agent.ingest_graph.build_chunks", return_value=[]),
      ):
          result = ingest_graph.invoke({
              "file_path": str(doc),
              "data_dir": str(tmp_path),
              "source_type": "standard",
              "document_id": None,
              "embedder": None,
              "conn": MagicMock(),
          })

      assert result.get("error") is None
      assert result["doc_id"] == "doc-123"
      assert result["version_id"] == "ver-456"
      assert result["status"] == "completed"
  ```

- [ ] **Run to verify FAIL**

  ```bash
  uv run pytest tests/test_agent/test_ingest_graph.py -v
  ```

  Expected: `ImportError` or `ModuleNotFoundError` for `app.agent.ingest_graph`

- [ ] **Create `app/agent/ingest_graph.py`**

  ```python
  """LangGraph StateGraph for the document ingest workflow."""
  from __future__ import annotations

  import json
  import logging
  import shutil
  from dataclasses import asdict
  from pathlib import Path

  from langgraph.graph import END, StateGraph

  from app.agent.states import IngestState
  from app.core.parser.ir_builder import build_chunks
  from app.core.parser.router import parse_document
  from app.core.utils import file_hash as compute_file_hash
  from app.db import chunk_repo, document_repo

  logger = logging.getLogger(__name__)


  def _route(state: IngestState) -> str:
      return "end" if state.get("error") else "continue"


  def file_check(state: IngestState) -> dict:
      """Verify file exists, compute hash, detect duplicates."""
      try:
          path = Path(state["file_path"])
          if not path.exists():
              return {"error": f"文件不存在：{path}", "status": "failed"}

          file_hash = compute_file_hash(path)
          conn = state["conn"]

          if not state.get("document_id"):
              existing = document_repo.get_document_by_hash(conn, file_hash)
              if existing:
                  return {
                      "error": (
                          f"文档已存在（hash {file_hash[:8]}…）。"
                          "如需新增版本，请选择文档后点击"新增版本"。"
                      ),
                      "status": "failed",
                  }

          return {"_file_hash": file_hash, "status": "file_checked"}
      except Exception as e:
          logger.exception("file_check failed")
          return {"error": str(e), "status": "failed"}


  def parse_doc(state: IngestState) -> dict:
      """Parse the file into a DocumentIR."""
      try:
          ir, quality = parse_document(state["file_path"])
          if quality.needs_ocr:
              logger.warning("Document needs OCR (Phase 2 feature): %s", quality.ocr_pages)
          return {"_ir": ir, "status": "parsed"}
      except Exception as e:
          logger.exception("parse_doc failed")
          return {"error": str(e), "status": "failed"}


  def save_document(state: IngestState) -> dict:
      """Copy file, persist IR JSON, insert DB rows, insert chunks."""
      try:
          conn = state["conn"]
          data_dir = state["data_dir"]
          ir = state["_ir"]
          path = Path(state["file_path"])
          file_hash = state["_file_hash"]

          docs_dir = Path(data_dir) / "docs"
          docs_dir.mkdir(parents=True, exist_ok=True)
          dest = docs_dir / f"{file_hash}{path.suffix}"
          if not dest.exists():
              shutil.copy2(str(path), str(dest))

          parsed_dir = Path(data_dir) / "parsed"
          parsed_dir.mkdir(parents=True, exist_ok=True)
          ir_path = parsed_dir / f"{ir.doc_id}.json"
          ir_path.write_text(
              json.dumps(asdict(ir), ensure_ascii=False, indent=2), encoding="utf-8"
          )

          document_id = state.get("document_id")
          if document_id:
              versions = document_repo.list_versions(conn, document_id)
              version_no = (max(v["version_no"] for v in versions) + 1) if versions else 1
              version_id = document_repo.insert_version(
                  conn,
                  document_id=document_id,
                  version_no=version_no,
                  parsed_json_path=str(ir_path),
                  summary=ir.title,
              )
              doc_id = document_id
          else:
              doc_id = document_repo.insert_document(
                  conn,
                  doc_name=path.stem,
                  doc_type=path.suffix.lstrip(".").lower(),
                  file_path=str(dest),
                  file_hash=file_hash,
                  source_type=state.get("source_type", "standard"),
                  business_category="",
              )
              version_id = document_repo.insert_version(
                  conn,
                  document_id=doc_id,
                  version_no=1,
                  parsed_json_path=str(ir_path),
                  summary=ir.title,
              )

          chunks = build_chunks(ir, version_id)
          chunk_repo.insert_chunks(conn, chunks)

          return {"doc_id": doc_id, "version_id": version_id, "_chunks": chunks, "status": "saved"}
      except Exception as e:
          logger.exception("save_document failed")
          return {"error": str(e), "status": "failed"}


  def build_embeddings(state: IngestState) -> dict:
      """Build FAISS index for the ingested version (skipped if no embedder)."""
      try:
          embedder = state.get("embedder")
          chunks = state.get("_chunks", [])
          if embedder and chunks:
              from app.core.retrieval.indexer import build_index
              build_index(state["data_dir"], state["conn"], state["version_id"], chunks, embedder)
          return {"status": "completed"}
      except Exception as e:
          logger.exception("build_embeddings failed")
          return {"error": str(e), "status": "failed"}


  def _build_ingest_graph():
      graph = StateGraph(IngestState)
      graph.add_node("file_check", file_check)
      graph.add_node("parse_doc", parse_doc)
      graph.add_node("save_document", save_document)
      graph.add_node("build_embeddings", build_embeddings)

      graph.set_entry_point("file_check")
      graph.add_conditional_edges("file_check",       _route, {"continue": "parse_doc",       "end": END})
      graph.add_conditional_edges("parse_doc",        _route, {"continue": "save_document",    "end": END})
      graph.add_conditional_edges("save_document",    _route, {"continue": "build_embeddings", "end": END})
      graph.add_edge("build_embeddings", END)
      return graph.compile()


  ingest_graph = _build_ingest_graph()
  ```

- [ ] **Run tests — must all pass**

  ```bash
  uv run pytest tests/test_agent/test_ingest_graph.py -v
  ```

  Expected: 4 tests PASSED

- [ ] **Commit**

  ```bash
  git add app/agent/ingest_graph.py tests/test_agent/test_ingest_graph.py
  git commit -m "feat: add ingest LangGraph workflow with tests"
  ```

---

## Task 13: Create compare_graph with tests (TDD)

**Files:**
- Create: `tests/test_agent/test_compare_graph.py`
- Create: `app/agent/compare_graph.py`

- [ ] **Write failing tests** (`tests/test_agent/test_compare_graph.py`):

  ```python
  """Tests for compare_graph LangGraph workflow."""
  from __future__ import annotations

  from unittest.mock import MagicMock, patch

  import pytest


  @pytest.fixture
  def base_state():
      return {
          "data_dir": "/tmp",
          "baseline_version_id": "ver-1",
          "target_version_id": "ver-2",
          "provider": MagicMock(),
          "embedder": MagicMock(),
          "conn": MagicMock(),
      }


  def test_create_task_node(base_state):
      """create_task inserts a compare_tasks record and returns task_id."""
      from app.agent.compare_graph import create_task

      with (
          patch("app.agent.compare_graph.compare_repo.create_compare_task", return_value="task-001"),
          patch("app.agent.compare_graph.compare_repo.update_task_status"),
      ):
          result = create_task(base_state)

      assert result["task_id"] == "task-001"
      assert result.get("error") is None


  def test_graph_sets_error_on_missing_version(base_state):
      """Graph sets error when a version's IR file is missing."""
      from app.agent.compare_graph import compare_graph

      with (
          patch("app.agent.compare_graph.compare_repo.create_compare_task", return_value="task-err"),
          patch("app.agent.compare_graph.compare_repo.update_task_status"),
          patch("app.agent.compare_graph.document_repo.get_version_by_id", return_value=None),
      ):
          result = compare_graph.invoke(base_state)

      assert result.get("error") is not None
      assert result.get("status") == "failed"


  def test_graph_happy_path(base_state, tmp_path):
      """Full happy path: both IRs load, align, compare, classify, persist."""
      from app.agent.compare_graph import compare_graph
      from app.core.types import DiffResult, DocumentIR

      mock_ir = DocumentIR(doc_id="d1", title="T", file_hash="h", sections=[], plain_text="")
      mock_result = DiffResult(task_id="task-001", items=[])
      base_state["data_dir"] = str(tmp_path)

      with (
          patch("app.agent.compare_graph.compare_repo.create_compare_task", return_value="task-001"),
          patch("app.agent.compare_graph.compare_repo.update_task_status"),
          patch("app.agent.compare_graph.compare_repo.insert_diff_items"),
          patch("app.agent.compare_graph._load_ir", return_value=mock_ir),
          patch("app.agent.compare_graph.align_sections", return_value=[]),
          patch("app.agent.compare_graph.match_paragraphs", return_value=[]),
          patch("app.agent.compare_graph.classify", return_value=mock_result),
      ):
          result = compare_graph.invoke(base_state)

      assert result.get("error") is None
      assert result["task_id"] == "task-001"
      assert result["status"] == "completed"
  ```

- [ ] **Run to verify FAIL**

  ```bash
  uv run pytest tests/test_agent/test_compare_graph.py -v
  ```

  Expected: `ImportError`

- [ ] **Create `app/agent/compare_graph.py`**

  ```python
  """LangGraph StateGraph for the document comparison workflow."""
  from __future__ import annotations

  import json
  import logging
  from dataclasses import asdict
  from pathlib import Path

  from langgraph.graph import END, StateGraph

  from app.agent.states import CompareState
  from app.core.diff.diff_classifier import classify
  from app.core.diff.semantic_matcher import match_paragraphs
  from app.core.diff.structure_aligner import align_sections
  from app.core.types import ComparePolicy, DocumentIR, Paragraph, Section, Sentence
  from app.db import compare_repo, document_repo

  logger = logging.getLogger(__name__)


  def _route(state: CompareState) -> str:
      return "end" if state.get("error") else "continue"


  def _load_ir(version_id: str, conn) -> DocumentIR:
      """Load DocumentIR from the parsed JSON path stored in DB."""
      row = document_repo.get_version_by_id(conn, version_id)
      if not row:
          raise ValueError(f"Version not found: {version_id}")
      ir_path = row["parsed_json_path"]
      if not ir_path or not Path(ir_path).exists():
          raise FileNotFoundError(f"Parsed IR not found: {ir_path}")
      data = json.loads(Path(ir_path).read_text(encoding="utf-8"))
      sections = []
      for sec in data.get("sections", []):
          paras = [
              Paragraph(
                  paragraph_id=p["paragraph_id"],
                  page_no=p["page_no"],
                  text=p["text"],
                  sentences=[Sentence(text=s["text"]) for s in p.get("sentences", [])],
              )
              for p in sec.get("paragraphs", [])
          ]
          sections.append(Section(
              section_id=sec["section_id"],
              title=sec["title"],
              level=sec["level"],
              paragraphs=paras,
          ))
      return DocumentIR(
          doc_id=data["doc_id"],
          title=data["title"],
          file_hash=data["file_hash"],
          sections=sections,
          plain_text=data.get("plain_text", ""),
      )


  def create_task(state: CompareState) -> dict:
      """Insert compare_tasks record and mark as running."""
      try:
          task_id = compare_repo.create_compare_task(
              state["conn"],
              baseline_version_id=state["baseline_version_id"],
              target_version_id=state["target_version_id"],
          )
          compare_repo.update_task_status(state["conn"], task_id, "running")
          return {"task_id": task_id, "status": "task_created"}
      except Exception as e:
          logger.exception("create_task failed")
          return {"error": str(e), "status": "failed"}


  def ensure_parsed(state: CompareState) -> dict:
      """Load both DocumentIRs from DB-stored JSON paths."""
      try:
          baseline_ir = _load_ir(state["baseline_version_id"], state["conn"])
          target_ir = _load_ir(state["target_version_id"], state["conn"])
          return {"_baseline_ir": baseline_ir, "_target_ir": target_ir, "status": "irs_loaded"}
      except Exception as e:
          logger.exception("ensure_parsed failed")
          compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
          return {"error": str(e), "status": "failed"}


  def do_align(state: CompareState) -> dict:
      """Align document sections using title similarity."""
      try:
          pairs = align_sections(state["_baseline_ir"], state["_target_ir"])
          return {"_section_pairs": pairs, "status": "aligned"}
      except Exception as e:
          logger.exception("do_align failed")
          compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
          return {"error": str(e), "status": "failed"}


  def do_semantic_compare(state: CompareState) -> dict:
      """Match paragraphs by embedding cosine similarity."""
      try:
          policy = ComparePolicy()
          para_pairs = match_paragraphs(
              state["_section_pairs"], state["embedder"], policy.similarity_threshold
          )
          return {"_para_pairs": para_pairs, "status": "matched"}
      except Exception as e:
          logger.exception("do_semantic_compare failed")
          compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
          return {"error": str(e), "status": "failed"}


  def do_classify(state: CompareState) -> dict:
      """Classify paragraph pairs with LLM and rule-based strengthening."""
      try:
          policy = ComparePolicy()
          result = classify(
              state["_para_pairs"],
              policy=policy,
              provider=state["provider"],
              task_id=state["task_id"],
              baseline_version_id=state["baseline_version_id"],
              target_version_id=state["target_version_id"],
          )
          return {"result": result, "status": "classified"}
      except Exception as e:
          logger.exception("do_classify failed")
          compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
          return {"error": str(e), "status": "failed"}


  def persist_result(state: CompareState) -> dict:
      """Write diff_items to DB and save JSON export."""
      try:
          result = state["result"]
          conn = state["conn"]
          task_id = state["task_id"]

          compare_repo.insert_diff_items(conn, task_id, result.items)

          exports_dir = Path(state["data_dir"]) / "exports"
          exports_dir.mkdir(parents=True, exist_ok=True)
          result_path = exports_dir / f"{task_id}.json"
          result_path.write_text(
              json.dumps([asdict(i) for i in result.items], ensure_ascii=False, indent=2),
              encoding="utf-8",
          )
          compare_repo.update_task_status(conn, task_id, "completed", str(result_path))
          logger.info("Compare task %s completed: %d items", task_id, len(result.items))
          return {"status": "completed"}
      except Exception as e:
          logger.exception("persist_result failed")
          compare_repo.update_task_status(state["conn"], state["task_id"], "failed")
          return {"error": str(e), "status": "failed"}


  def _build_compare_graph():
      graph = StateGraph(CompareState)
      nodes = [
          ("create_task",         create_task),
          ("ensure_parsed",       ensure_parsed),
          ("do_align",            do_align),
          ("do_semantic_compare", do_semantic_compare),
          ("do_classify",         do_classify),
          ("persist_result",      persist_result),
      ]
      for name, fn in nodes:
          graph.add_node(name, fn)

      graph.set_entry_point("create_task")
      sequence = [n for n, _ in nodes]
      for i, src in enumerate(sequence[:-1]):
          dst = sequence[i + 1]
          graph.add_conditional_edges(src, _route, {"continue": dst, "end": END})
      graph.add_edge("persist_result", END)
      return graph.compile()


  compare_graph = _build_compare_graph()
  ```

- [ ] **Run tests — must all pass**

  ```bash
  uv run pytest tests/test_agent/test_compare_graph.py -v
  ```

  Expected: 3 tests PASSED

- [ ] **Commit**

  ```bash
  git add app/agent/compare_graph.py tests/test_agent/test_compare_graph.py
  git commit -m "feat: add compare LangGraph workflow with tests"
  ```

---

## Task 14: Create qa_graph with tests (TDD)

**Files:**
- Create: `tests/test_agent/test_qa_graph.py`
- Create: `app/agent/qa_graph.py`

- [ ] **Write failing tests** (`tests/test_agent/test_qa_graph.py`):

  ```python
  """Tests for qa_graph LangGraph workflow."""
  from __future__ import annotations

  from unittest.mock import MagicMock, patch

  import pytest


  @pytest.fixture
  def base_state():
      mock_provider = MagicMock()
      mock_provider.chat.return_value = "答案是X。"
      return {
          "data_dir": "/tmp",
          "question": "付款周期是多少天？",
          "scope": "current_doc",
          "current_version_ids": ["ver-1"],
          "provider": mock_provider,
          "embedder": MagicMock(),
          "conn": MagicMock(),
      }


  def test_resolve_scope_current_doc(base_state):
      from app.agent.qa_graph import resolve_scope
      result = resolve_scope(base_state)
      assert result["_version_ids"] == ["ver-1"]
      assert result.get("error") is None


  def test_resolve_scope_empty_current_doc_returns_error(base_state):
      from app.agent.qa_graph import resolve_scope
      base_state["current_version_ids"] = []
      result = resolve_scope(base_state)
      assert result.get("error") is not None
      assert result.get("status") == "failed"


  def test_graph_happy_path(base_state):
      from app.agent.qa_graph import qa_graph

      mock_hit = MagicMock()
      mock_hit.chunk.section_path = "第一章"
      mock_hit.chunk.page_no = 1
      mock_hit.chunk.text = "付款周期为30天。"

      with patch("app.agent.qa_graph.search", return_value=[mock_hit]):
          result = qa_graph.invoke(base_state)

      assert result.get("error") is None
      assert result["answer"] == "答案是X。"
      assert len(result["citations"]) == 1
      assert result["status"] == "completed"


  def test_graph_no_hits_returns_default_message(base_state):
      from app.agent.qa_graph import qa_graph

      with patch("app.agent.qa_graph.search", return_value=[]):
          result = qa_graph.invoke(base_state)

      assert result.get("error") is None
      assert "未找到" in result["answer"]
      assert result["citations"] == []
  ```

- [ ] **Run to verify FAIL**

  ```bash
  uv run pytest tests/test_agent/test_qa_graph.py -v
  ```

  Expected: `ImportError`

- [ ] **Create `app/agent/qa_graph.py`**

  ```python
  """LangGraph StateGraph for the QA (retrieval-augmented answering) workflow."""
  from __future__ import annotations

  import logging

  from langgraph.graph import END, StateGraph

  from app.agent.states import QAState
  from app.core.retrieval.searcher import search
  from app.db import document_repo

  logger = logging.getLogger(__name__)

  _QA_PROMPT = """你是一个专业的文档问答助手。请根据以下参考资料回答用户问题。

  参考资料：
  {context}

  用户问题：{question}

  回答要求：
  1. 只根据参考资料中的内容回答，不要编造信息
  2. 如果参考资料中找不到答案，请明确说明"文档中未找到相关内容"
  3. 引用具体章节或页码（如资料中有）
  4. 回答简洁、准确
  """


  def _route(state: QAState) -> str:
      return "end" if state.get("error") else "continue"


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


  def retrieve_chunks(state: QAState) -> dict:
      """Vector search for relevant chunks."""
      try:
          hits = search(
              state["data_dir"],
              state["conn"],
              state["question"],
              state["embedder"],
              state["_version_ids"],
              top_k=5,
          )
          return {"_hits": hits, "status": "retrieved"}
      except Exception as e:
          logger.exception("retrieve_chunks failed")
          return {"error": str(e), "status": "failed"}


  def generate_answer(state: QAState) -> dict:
      """Generate answer from retrieved chunks using LLM."""
      try:
          hits = state.get("_hits", [])
          if not hits:
              return {"answer": "文档中未找到与问题相关的内容。", "status": "answered"}

          context_parts = []
          for i, hit in enumerate(hits, 1):
              chunk = hit.chunk
              ref = f"[{i}] "
              if chunk.section_path:
                  ref += f"章节：{chunk.section_path}，"
              if chunk.page_no:
                  ref += f"第{chunk.page_no}页，"
              ref += f"内容：{chunk.text}"
              context_parts.append(ref)

          prompt = _QA_PROMPT.format(
              context="\n\n".join(context_parts),
              question=state["question"],
          )
          answer_text = state["provider"].chat([{"role": "user", "content": prompt}])
          return {"answer": answer_text, "status": "answered"}
      except Exception as e:
          logger.exception("generate_answer failed")
          return {"error": str(e), "status": "failed"}


  def attach_citations(state: QAState) -> dict:
      """Package chunk hits as citation list."""
      return {"citations": list(state.get("_hits", [])), "status": "completed"}


  def _build_qa_graph():
      graph = StateGraph(QAState)
      graph.add_node("resolve_scope",   resolve_scope)
      graph.add_node("retrieve_chunks", retrieve_chunks)
      graph.add_node("generate_answer", generate_answer)
      graph.add_node("attach_citations", attach_citations)

      graph.set_entry_point("resolve_scope")
      graph.add_conditional_edges("resolve_scope",   _route, {"continue": "retrieve_chunks", "end": END})
      graph.add_conditional_edges("retrieve_chunks", _route, {"continue": "generate_answer",  "end": END})
      graph.add_conditional_edges("generate_answer", _route, {"continue": "attach_citations", "end": END})
      graph.add_edge("attach_citations", END)
      return graph.compile()


  qa_graph = _build_qa_graph()
  ```

- [ ] **Run tests — must all pass**

  ```bash
  uv run pytest tests/test_agent/test_qa_graph.py -v
  ```

  Expected: 4 tests PASSED

- [ ] **Commit**

  ```bash
  git add app/agent/qa_graph.py tests/test_agent/test_qa_graph.py
  git commit -m "feat: add QA LangGraph workflow with tests"
  ```

---

## Task 15: Wire library_page.py to ingest_graph + I5 patch

**Files:**
- Modify: `app/ui/pages/library_page.py`

- [ ] **Update `_IngestWorker.__init__` to accept optional `document_id`**

  ```python
  def __init__(self, ctx: AppContext, file_path: str, document_id: str | None = None):
      super().__init__()
      self.ctx = ctx
      self.file_path = file_path
      self.document_id = document_id
  ```

- [ ] **Replace `_IngestWorker.run()` to use ingest_graph**

  ```python
  def run(self) -> None:
      try:
          from app.agent.ingest_graph import ingest_graph
          from app.db.schema import open_db

          conn = open_db(self.ctx.data_dir)
          try:
              result = ingest_graph.invoke({
                  "file_path": self.file_path,
                  "data_dir": self.ctx.data_dir,
                  "source_type": "standard",
                  "document_id": self.document_id,
                  "embedder": self.ctx.embedder,
                  "conn": conn,
              })
          finally:
              conn.close()

          if result.get("error"):
              self.error.emit(result["error"])
          else:
              self.finished.emit(result["doc_id"], result["version_id"])
      except Exception as e:
          logger.exception("Ingest worker failed")
          self.error.emit(f"导入失败：{e}")
  ```

- [ ] **Add `_add_version_btn` to `_build_ui` — insert after `import_btn`**

  After `header.addWidget(import_btn)` add:
  ```python
  self._add_version_btn = QPushButton("新增版本")
  self._add_version_btn.setStyleSheet(Theme.btn_success())
  self._add_version_btn.setEnabled(False)
  self._add_version_btn.clicked.connect(self._add_version)
  header.addWidget(self._add_version_btn)
  ```

- [ ] **Store doc_id on row items in `refresh()`**

  In the `refresh()` method, replace:
  ```python
  self._table.setItem(row, 0, QTableWidgetItem(doc["doc_name"]))
  ```
  with:
  ```python
  item0 = QTableWidgetItem(doc["doc_name"])
  item0.setData(Qt.UserRole, doc["id"])
  self._table.setItem(row, 0, item0)
  ```

- [ ] **Wire selection change in `_build_ui`**

  After `layout.addWidget(self._table, 1)` add:
  ```python
  self._table.itemSelectionChanged.connect(self._on_selection_changed)
  ```

- [ ] **Add `_on_selection_changed` method**

  ```python
  def _on_selection_changed(self) -> None:
      self._add_version_btn.setEnabled(len(self._table.selectedItems()) > 0)
  ```

- [ ] **Add `_add_version` method**

  ```python
  def _add_version(self) -> None:
      rows = self._table.selectionModel().selectedRows()
      if not rows:
          return
      row = rows[0].row()
      doc_id = self._table.item(row, 0).data(Qt.UserRole)
      doc_name = self._table.item(row, 0).text()
      paths, _ = QFileDialog.getOpenFileNames(
          self,
          f"为《{doc_name}》选择新版本文件",
          "",
          "文档文件 (*.pdf *.docx);;PDF 文件 (*.pdf);;Word 文档 (*.docx)",
      )
      for path in paths:
          self._run_ingest(path, document_id=doc_id)
  ```

- [ ] **Update `_run_ingest` signature to forward `document_id`**

  ```python
  def _run_ingest(self, file_path: str, document_id: str | None = None) -> None:
      thread = QThread()
      worker = _IngestWorker(self.ctx, file_path, document_id=document_id)
      # ... rest of method unchanged
  ```

- [ ] **Run tests**

  ```bash
  uv run pytest tests/ -x -q
  ```

  Expected: all tests pass

- [ ] **Commit**

  ```bash
  git add app/ui/pages/library_page.py
  git commit -m "feat: wire library page to ingest_graph, add new-version button (I5)"
  ```

---

## Task 16: Wire compare_page.py to compare_graph

**Files:**
- Modify: `app/ui/pages/compare_page.py`

- [ ] **Replace `_CompareWorker.run()`**

  ```python
  def run(self) -> None:
      try:
          from app.agent.compare_graph import compare_graph
          from app.db.schema import open_db

          conn = open_db(self._data_dir)
          try:
              result = compare_graph.invoke({
                  "data_dir": self._data_dir,
                  "baseline_version_id": self._baseline_version_id,
                  "target_version_id": self._target_version_id,
                  "provider": self._provider,
                  "embedder": self._embedder,
                  "conn": conn,
              })
          finally:
              conn.close()

          if result.get("error"):
              self.error.emit(result["error"])
          else:
              self.result_ready.emit(result["result"])
      except Exception as exc:
          logger.exception("Compare worker failed")
          self.error.emit(str(exc))
  ```

- [ ] **Run tests**

  ```bash
  uv run pytest tests/ -x -q
  ```

- [ ] **Commit**

  ```bash
  git add app/ui/pages/compare_page.py
  git commit -m "feat: wire compare page worker to compare_graph"
  ```

---

## Task 17: Wire qa_page.py to qa_graph

**Files:**
- Modify: `app/ui/pages/qa_page.py`

- [ ] **Replace `_QaWorker.run()`**

  `RetrievalScope` is a string enum (`CURRENT_DOC.value == "current_doc"` etc.), so pass `.value` to the graph which uses string comparisons.

  ```python
  def run(self) -> None:
      try:
          from app.agent.qa_graph import qa_graph
          from app.db.schema import open_db

          conn = open_db(self._data_dir)
          try:
              result = qa_graph.invoke({
                  "data_dir": self._data_dir,
                  "question": self._question,
                  "scope": self._scope.value,        # RetrievalScope → string
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
  ```

- [ ] **Run full test suite**

  ```bash
  uv run pytest tests/ -v
  ```

  Expected: all original 105 tests + new agent tests pass

- [ ] **Commit**

  ```bash
  git add app/ui/pages/qa_page.py
  git commit -m "feat: wire QA page worker to qa_graph"
  ```

---

## Task 18: Final verification

- [ ] **All graphs import and compile cleanly**

  ```bash
  uv run python -c "
  from app.agent.ingest_graph import ingest_graph
  from app.agent.compare_graph import compare_graph
  from app.agent.qa_graph import qa_graph
  print('All graphs OK')
  "
  ```

  Expected: `All graphs OK`

- [ ] **Full test suite passes**

  ```bash
  uv run pytest tests/ -v --tb=short
  ```

  Expected: all tests pass (105 original + 11 new agent tests = 116+)

- [ ] **Update design doc Phase 2 status**

  In `docs/superpowers/specs/2026-05-01-doc-diff-agent-design.md`, mark as in-progress:
  ```markdown
  ### 第二阶段
  - [x] LangGraph 编排层接入
  - [ ] OCR 增强（Tesseract）
  - [ ] 实质修改/重写分类
  - [ ] 对比/标准库/混合问答模式
  - [ ] 差异报告导出（Word/HTML）
  ```

- [ ] **Final commit**

  ```bash
  git add docs/
  git commit -m "docs: mark LangGraph migration complete in phase 2 checklist"
  ```
