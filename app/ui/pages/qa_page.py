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

def _user_bubble_style() -> str:
    return (
        f"background:{Theme.COLOR_PRIMARY};color:{Theme.NAV_ACTIVE_TEXT};"
        "border-radius:12px;padding:10px;margin:4px 0;"
    )


def _asst_bubble_style() -> str:
    return (
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
        self._apply_theme()
        from app.ui.theme_manager import ThemeManager
        ThemeManager.instance().theme_changed.connect(self._apply_theme)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        # ── Top: scope/document selectors + 新会话 button ──────────────────────
        top_group = QGroupBox()
        top_layout = QHBoxLayout(top_group)
        top_layout.setSpacing(10)

        tmp_label = QLabel("检索范围：")
        tmp_label.setStyleSheet(Theme.form_label_large())
        top_layout.addWidget(tmp_label)
        self._scope_combo = QComboBox()
        self._scope_combo.addItems(list(_SCOPE_MAP.keys()))
        self._scope_combo.currentTextChanged.connect(self._on_scope_changed)
        top_layout.addWidget(self._scope_combo)

        tmp_label = QLabel("文档：")
        tmp_label.setStyleSheet(Theme.form_label_large())
        top_layout.addWidget(tmp_label)
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

        self._new_session_btn = QPushButton("新会话")
        self._new_session_btn.setStyleSheet(Theme.btn_primary())
        self._new_session_btn.setFixedWidth(72)
        self._new_session_btn.clicked.connect(self._new_session)
        top_layout.addWidget(self._new_session_btn)

        root.addWidget(top_group)

        # ── Middle: chat scroll area ───────────────────────────────────────────
        self._chat_scroll = QScrollArea()
        self._chat_scroll.setObjectName("chat_scroll")
        self._chat_scroll.setWidgetResizable(True)
        self._chat_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._chat_scroll.viewport().setStyleSheet("background: transparent;")
        self._chat_content = QWidget()
        self._chat_layout = QVBoxLayout(self._chat_content)
        self._chat_layout.setSpacing(4)
        self._chat_layout.addStretch()

        self._chat_scroll.setWidget(self._chat_content)
        root.addWidget(self._chat_scroll, 1)

        # ── Bottom: input area ─────────────────────────────────────────────────
        input_group = QGroupBox()
        input_row = QHBoxLayout(input_group)
        input_row.setContentsMargins(4, 4, 4, 4)
        input_row.setSpacing(8)

        self._input = QTextEdit()
        self._input.setMaximumHeight(40)
        self._input.setPlaceholderText("输入问题…")
        input_row.addWidget(self._input, 1)

        self._send_btn = QPushButton("发送")
        self._send_btn.setStyleSheet(Theme.btn_primary())
        self._send_btn.setFixedWidth(72)
        self._send_btn.clicked.connect(self.send_question)
        input_row.addWidget(self._send_btn)

        root.addWidget(input_group)

        self._on_scope_changed(self._scope_combo.currentText())

    # ── Theme ──────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"background-color:{Theme.BG_CARD};")
        self._new_session_btn.setStyleSheet(Theme.btn_primary())
        self._send_btn.setStyleSheet(Theme.btn_primary())

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
        """Reset session: new thread_id + clear chat bubbles."""
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
        cit_lbl.setStyleSheet(Theme.caption() + "margin-left:4px;")
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
        bubble.setStyleSheet(_user_bubble_style() if is_user else _asst_bubble_style())
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
