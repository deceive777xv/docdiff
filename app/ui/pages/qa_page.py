"""QA page — chat-style retrieval-augmented question answering with streaming."""
from __future__ import annotations

import asyncio
import logging
import uuid

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.types import RetrievalScope
from app.db import document_repo, qa_repo
from app.ui.app_context import AppContext
from app.ui.theme import Theme

logger = logging.getLogger(__name__)

_SCOPE_MAP: dict[str, RetrievalScope] = {
    "当前文档": RetrievalScope.CURRENT_DOC,
    "对比文档": RetrievalScope.COMPARE,
    "文档库": RetrievalScope.STANDARD_LIB,
    "全部": RetrievalScope.ALL,
}
_SCOPE_TEXT_BY_VALUE = {scope.value: text for text, scope in _SCOPE_MAP.items()}

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


class CustomTextEdit(QTextEdit):
    def __init__(self, button=None, parent=None):
        super().__init__(parent)
        self.button = button

    def keyPressEvent(self, event):
        # 检查是否按下 Enter 键
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # 检查是否同时按下了 Shift
            if event.modifiers() == Qt.ShiftModifier:
                # Shift+Enter: 正常换行
                super().keyPressEvent(event)
            else:
                # 单独的 Enter: 触发按钮点击
                if self.button and self.button.isEnabled():
                    self.button.click()
        else:
            # 其他按键正常处理
            super().keyPressEvent(event)


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
        compare_task_id: str | None = None,
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
        self._compare_task_id = compare_task_id

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
                "compare_task_id": self._compare_task_id,
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

    sessions_changed = Signal()

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._thread: QThread | None = None
        self._threads: set[QThread] = set()
        self._thread_id: str = str(uuid.uuid4())
        self._session_persisted = False
        self._loading_session = False
        self._accumulated: str = ""
        self._current_bubble: QLabel | None = None
        self._build_ui()
        self.refresh_documents()
        self.refresh_compare_tasks()
        self.refresh_sessions()
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

        tmp_label = QLabel("会话：")
        tmp_label.setStyleSheet(Theme.form_label_large())
        top_layout.addWidget(tmp_label)
        self._session_combo = QComboBox()
        self._session_combo.setMinimumWidth(220)
        self._session_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._session_combo.currentIndexChanged.connect(self._on_session_selected)
        top_layout.addWidget(self._session_combo)

        self._new_session_btn = QPushButton("新会话")
        self._new_session_btn.setStyleSheet(Theme.btn_primary())
        self._new_session_btn.setFixedWidth(72)
        self._new_session_btn.clicked.connect(self._new_session)
        top_layout.addWidget(self._new_session_btn)

        self._delete_session_btn = QPushButton("删除")
        self._delete_session_btn.setStyleSheet(Theme.btn_danger())
        self._delete_session_btn.setFixedWidth(64)
        self._delete_session_btn.clicked.connect(self._delete_session)
        top_layout.addWidget(self._delete_session_btn)

        tmp_label = QLabel("检索范围：")
        tmp_label.setStyleSheet(Theme.form_label_large())
        top_layout.addWidget(tmp_label)
        self._scope_combo = QComboBox()
        self._scope_combo.addItems(list(_SCOPE_MAP.keys()))
        self._scope_combo.currentTextChanged.connect(self._on_scope_changed)
        top_layout.addWidget(self._scope_combo)

        self._doc_label = QLabel("文档：")
        self._doc_label.setStyleSheet(Theme.form_label_large())
        top_layout.addWidget(self._doc_label)
        self._doc_combo = QComboBox()
        self._doc_combo.setMinimumWidth(200)
        self._doc_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._doc_combo.currentIndexChanged.connect(self._on_doc_changed)
        top_layout.addWidget(self._doc_combo)

        self._compare_task_label = QLabel("对比任务：")
        self._compare_task_label.setStyleSheet(Theme.form_label_large())
        top_layout.addWidget(self._compare_task_label)
        self._compare_task_combo = QComboBox()
        self._compare_task_combo.setMinimumWidth(280)
        self._compare_task_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._compare_task_combo.currentIndexChanged.connect(self._on_compare_task_changed)
        top_layout.addWidget(self._compare_task_combo)

        top_layout.addStretch()

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

        self._send_btn = QPushButton("发送")
        self._send_btn.setStyleSheet(Theme.btn_primary())
        self._send_btn.setFixedWidth(72)
        self._send_btn.clicked.connect(self.send_question)

        self._input = CustomTextEdit(button=self._send_btn)
        self._input.setMaximumHeight(40)
        self._input.setPlaceholderText("输入问题…")
        input_row.addWidget(self._input, 1)
        
        input_row.addWidget(self._send_btn)

        root.addWidget(input_group)

        self._on_scope_changed(self._scope_combo.currentText())

    # ── Theme ──────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"background-color:{Theme.BG_CARD};")
        self._new_session_btn.setStyleSheet(Theme.btn_primary())
        self._delete_session_btn.setStyleSheet(Theme.btn_danger())
        self._send_btn.setStyleSheet(Theme.btn_primary())
        self._doc_label.setStyleSheet(Theme.form_label_large())
        self._compare_task_label.setStyleSheet(Theme.form_label_large())
        self._restyle_chat_bubbles()

    # ── Public API ─────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        self.refresh_documents()
        self.refresh_compare_tasks()
        self.refresh_sessions(
            select_session_id=self._thread_id if self._session_persisted else None,
            load_selected=False,
        )
        
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
                       ct.id AS task_id,
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
                    {
                        "task_id": row["task_id"],
                        "version_ids": (
                            row["baseline_version_id"],
                            row["target_version_id"],
                        ),
                    },
                )
        except Exception as exc:
            logger.warning("refresh_compare_tasks failed: %s", exc)
        finally:
            self._compare_task_combo.blockSignals(False)

    def refresh_sessions(
        self,
        *,
        select_session_id: str | None = None,
        load_selected: bool = True,
    ) -> None:
        self._session_combo.blockSignals(True)
        selected_id: str | None = None
        try:
            self._session_combo.clear()
            rows = qa_repo.list_sessions(self.ctx.conn)
            if not rows:
                self._session_combo.addItem("新会话", None)
                self._delete_session_btn.setEnabled(False)
                selected_id = None
            else:
                for row in rows:
                    self._session_combo.addItem(row["title"] or "未命名会话", row["id"])
                wanted = select_session_id or (self._thread_id if self._session_persisted else None)
                index = self._session_combo.findData(wanted) if wanted else 0
                if index < 0:
                    index = 0
                self._session_combo.setCurrentIndex(index)
                selected_id = self._session_combo.itemData(index)
                self._delete_session_btn.setEnabled(True)
        except Exception as exc:
            logger.warning("refresh_sessions failed: %s", exc)
        finally:
            self._session_combo.blockSignals(False)

        if selected_id and load_selected:
            self._load_session(selected_id)
        elif selected_id is None:
            self._session_persisted = False
            if not self._thread_id:
                self._thread_id = str(uuid.uuid4())

    def _show_new_session_placeholder(self) -> None:
        self._session_combo.blockSignals(True)
        try:
            placeholder_index = self._session_combo.findData(None)
            if placeholder_index < 0:
                self._session_combo.insertItem(0, "新会话", None)
                placeholder_index = 0
            self._session_combo.setCurrentIndex(placeholder_index)
            self._delete_session_btn.setEnabled(False)
        finally:
            self._session_combo.blockSignals(False)

    def _load_session(self, session_id: str) -> None:
        row = qa_repo.get_session(self.ctx.conn, session_id)
        if row is None:
            self._new_session()
            return

        self._loading_session = True
        try:
            self._thread_id = session_id
            self._session_persisted = True
            self._delete_session_btn.setEnabled(True)
            scope_text = _SCOPE_TEXT_BY_VALUE.get(row["scope"], "全部")
            scope_index = self._scope_combo.findText(scope_text)
            if scope_index >= 0:
                self._scope_combo.setCurrentIndex(scope_index)

            version_ids = qa_repo.decode_version_ids(row["current_version_ids_json"])
            if row["scope"] == RetrievalScope.CURRENT_DOC.value and version_ids:
                doc_index = self._doc_combo.findData(version_ids[0])
                if doc_index >= 0:
                    self._doc_combo.setCurrentIndex(doc_index)
            elif row["scope"] == RetrievalScope.COMPARE.value:
                task_id = row["compare_task_id"]
                for index in range(self._compare_task_combo.count()):
                    data = self._compare_task_combo.itemData(index)
                    if isinstance(data, dict) and data.get("task_id") == task_id:
                        self._compare_task_combo.setCurrentIndex(index)
                        break

            self._clear_chat()
            for message in qa_repo.list_messages(self.ctx.conn, session_id):
                self._add_message(message["role"], message["content"])
        finally:
            self._loading_session = False

    def send_question(self) -> None:
        question = self._input.toPlainText().strip()
        if not question:
            return

        if self.ctx.embedder is None or self.ctx.lc_model is None:
            self._add_message("assistant", "请先在设置页面配置模型")
            return

        scope, current_version_ids, compare_task_id = self._selected_scope_context()
        self._ensure_current_session(question, scope, current_version_ids, compare_task_id)
        qa_repo.add_message(self.ctx.conn, self._thread_id, "user", question)

        self._add_message("user", question)
        self._input.clear()
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
            compare_task_id=compare_task_id,
        )
        self._thread = thread
        self.worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.token_received.connect(self._on_token)
        worker.citations_ready.connect(self._on_citations)
        worker.error.connect(self._on_error)
        worker.done.connect(self._on_done)
        worker.done.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda th=thread: self._threads.discard(th))
        self._threads.add(thread)
        thread.start()

    def _selected_scope_context(self) -> tuple[RetrievalScope, list[str], str | None]:
        scope_text = self._scope_combo.currentText()
        scope = _SCOPE_MAP.get(scope_text, RetrievalScope.ALL)

        current_version_ids: list[str] = []
        compare_task_id: str | None = None
        if scope == RetrievalScope.CURRENT_DOC:
            vid = self._doc_combo.currentData()
            if vid:
                current_version_ids = [vid]
        elif scope == RetrievalScope.COMPARE:
            task_data = self._compare_task_combo.currentData()
            if isinstance(task_data, dict):
                compare_task_id = task_data.get("task_id")
                current_version_ids = list(task_data.get("version_ids") or [])
            elif task_data:
                values = list(task_data)
                if len(values) >= 3:
                    compare_task_id = values[0]
                    current_version_ids = values[1:3]
                else:
                    current_version_ids = values
        return scope, current_version_ids, compare_task_id

    def _ensure_current_session(
        self,
        question: str,
        scope: RetrievalScope,
        current_version_ids: list[str],
        compare_task_id: str | None,
    ) -> None:
        if not self._session_persisted or qa_repo.get_session(self.ctx.conn, self._thread_id) is None:
            self._thread_id = qa_repo.create_session(
                self.ctx.conn,
                session_id=self._thread_id,
                title=qa_repo.title_from_question(question),
                scope=scope.value,
                current_version_ids=current_version_ids,
                compare_task_id=compare_task_id,
            )
            self._session_persisted = True
        else:
            row = qa_repo.get_session(self.ctx.conn, self._thread_id)
            title = row["title"] if row is not None else qa_repo.title_from_question(question)
            qa_repo.update_session(
                self.ctx.conn,
                self._thread_id,
                title=title,
                scope=scope.value,
                current_version_ids=current_version_ids,
                compare_task_id=compare_task_id,
            )
        self.refresh_sessions(select_session_id=self._thread_id, load_selected=False)

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_scope_changed(self, text: str) -> None:
        is_current_doc = text == "当前文档"
        is_compare = text == "对比文档"
        self._doc_label.setVisible(is_current_doc)
        self._doc_combo.setVisible(is_current_doc)
        self._compare_task_label.setVisible(is_compare)
        self._compare_task_combo.setVisible(is_compare)
        if text == "对比文档" and self._compare_task_combo.count() == 0:
            self.refresh_compare_tasks()
        if not self._loading_session:
            self._start_new_ephemeral_session(clear_chat=True)

    def _on_doc_changed(self) -> None:
        if not self._loading_session and self._scope_combo.currentText() == "当前文档":
            self._start_new_ephemeral_session(clear_chat=True)

    def _on_compare_task_changed(self) -> None:
        if not self._loading_session and self._scope_combo.currentText() == "对比文档":
            self._start_new_ephemeral_session(clear_chat=True)

    def _on_session_selected(self) -> None:
        if self._loading_session:
            return
        session_id = self._session_combo.currentData()
        if session_id:
            self._load_session(session_id)
        else:
            self._start_new_ephemeral_session(clear_chat=True)

    def _new_session(self) -> None:
        """Reset session: new thread_id + clear chat bubbles."""
        self._start_new_ephemeral_session(clear_chat=True)

    def _start_new_ephemeral_session(self, *, clear_chat: bool) -> None:
        self._thread_id = str(uuid.uuid4())
        self._session_persisted = False
        self._current_bubble = None
        self._accumulated = ""
        if clear_chat and hasattr(self, "_chat_layout"):
            self._clear_chat()
        if hasattr(self, "_session_combo"):
            self._show_new_session_placeholder()

    def _delete_session(self) -> None:
        session_id = self._session_combo.currentData()
        if not session_id:
            return
        answer = QMessageBox.question(
            self,
            "确认删除",
            "删除该问答会话将移除历史消息和本地记忆。\n\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            qa_repo.delete_session(self.ctx.conn, session_id)
            if session_id == self._thread_id:
                self._start_new_ephemeral_session(clear_chat=True)
            self.refresh_sessions(load_selected=True)
            self.sessions_changed.emit()
        except Exception as exc:
            logger.exception("delete QA session failed")
            QMessageBox.critical(self, "删除失败", f"删除问答会话失败：{exc}")

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
        if self._session_persisted:
            qa_repo.add_message(self.ctx.conn, self._thread_id, "assistant", f"错误：{msg}")
            self.sessions_changed.emit()

    def _on_done(self) -> None:
        if self._session_persisted and self._accumulated.strip():
            qa_repo.add_message(self.ctx.conn, self._thread_id, "assistant", self._accumulated)
            self.refresh_sessions(select_session_id=self._thread_id, load_selected=False)
            self.sessions_changed.emit()
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
        bubble.setProperty("qa_role", role)
        bubble.setWordWrap(True)
        self._apply_bubble_style(bubble)
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

    def _apply_bubble_style(self, bubble: QLabel) -> None:
        role = bubble.property("qa_role")
        bubble.setStyleSheet(_user_bubble_style() if role == "user" else _asst_bubble_style())

    def _restyle_chat_bubbles(self) -> None:
        if not hasattr(self, "_chat_content"):
            return
        for bubble in self._chat_content.findChildren(QLabel):
            if bubble.property("qa_role") in ("user", "assistant"):
                self._apply_bubble_style(bubble)
