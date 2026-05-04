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
        self.setStyleSheet(f"background-color:{Theme.BG_CARD};")

        # ── Top: scope and document selectors ─────────────────────────────────
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
        top_layout.addWidget(self._doc_combo)

        self._compare_task_label = QLabel("对比任务：")
        top_layout.addWidget(self._compare_task_label)
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
        self._compare_task_label.setVisible(text == "对比文档")
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
