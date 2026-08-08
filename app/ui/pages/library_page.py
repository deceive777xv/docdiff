"""Document library page."""
from __future__ import annotations
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui.app_context import AppContext
from app.core.normalization import NormalizationDepth
from app.ui.theme import Theme
from app.db import document_repo

logger = logging.getLogger(__name__)


DOCUMENT_ID_ROLE = Qt.ItemDataRole.UserRole
VERSION_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 1


DOCUMENT_FILE_FILTER = (
    "支持的文档 (*.pdf *.doc *.docx *.docm *.ppt *.pps *.pot *.pptx *.pptm *.ppsx *.ppsm "
    "*.xls *.xlsx *.xlsm *.xlsb *.odt *.ods *.odp *.rtf *.epub *.csv *.html *.htm "
    "*.json *.xml *.txt *.md *.markdown)"
    ";;PDF (*.pdf)"
    ";;Word (*.doc *.docx *.docm)"
    ";;PowerPoint (*.ppt *.pps *.pot *.pptx *.pptm *.ppsx *.ppsm)"
    ";;Excel (*.xls *.xlsx *.xlsm *.xlsb)"
    ";;OpenDocument (*.odt *.ods *.odp)"
    ";;网页 (*.html *.htm)"
    ";;Markdown (*.md *.markdown)"
    ";;其他 (*.rtf *.epub *.csv *.json *.xml *.txt)"
)


_INGEST_NODE_PROGRESS = {
    "file_check": ("file_checked", 10, "正在解析文档"),
    "parse_doc": ("parsed", 35, "正在规范化段落与跨页表格"),
    "save_document": ("saved", 85, "正在构建检索索引"),
    "build_embeddings": ("completed", 100, "导入完成"),
}


def _progress_from_graph_update(
    update: object,
) -> tuple[int, str] | None:
    if not isinstance(update, dict):
        return None
    for node_name, (expected_status, percent, stage) in _INGEST_NODE_PROGRESS.items():
        node_update = update.get(node_name)
        if (
            isinstance(node_update, dict)
            and node_update.get("status") == expected_status
            and not node_update.get("error")
        ):
            return percent, stage
    return None


class _ImportProgressDialog(QDialog):
    """Non-blocking aggregate progress for one import batch."""

    def __init__(self, paths: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("正在导入文档")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setMinimumWidth(520)
        self._paths = list(paths)
        self._file_progress = {path: 0 for path in self._paths}
        self._file_states = {path: "running" for path in self._paths}
        self._items: dict[str, QListWidgetItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self._title = QLabel("正在导入并规范化文档")
        layout.addWidget(self._title)

        self._summary = QLabel(f"已完成 0/{len(self._paths)}")
        layout.addWidget(self._summary)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        layout.addWidget(self._progress)

        self._list = QListWidget()
        self._list.setMinimumHeight(110)
        self._list.setMaximumHeight(220)
        for path in self._paths:
            item = QListWidgetItem(f"{Path(path).name} — 准备导入")
            self._list.addItem(item)
            self._items[path] = item
        layout.addWidget(self._list)

        self._hint = QLabel("关闭此窗口不会停止后台导入。")
        layout.addWidget(self._hint)
        self._apply_theme()
        from app.ui.theme_manager import ThemeManager
        ThemeManager.instance().theme_changed.connect(self._apply_theme)

    def _apply_theme(self) -> None:
        self._title.setStyleSheet(Theme.page_title())
        self._summary.setStyleSheet(Theme.label_secondary())
        self._hint.setStyleSheet(Theme.label_secondary())
        self.setStyleSheet(
            f"QDialog{{background:{Theme.BG_PAGE};}}"
            f"QListWidget{{background:{Theme.BG_CARD};color:{Theme.TEXT_PRIMARY};"
            f"border:1px solid {Theme.BORDER};border-radius:6px;}}"
            f"QProgressBar{{background:{Theme.BG_CARD};color:{Theme.TEXT_PRIMARY};"
            f"border:1px solid {Theme.BORDER};border-radius:5px;text-align:center;}}"
            f"QProgressBar::chunk{{background:{Theme.COLOR_PRIMARY};border-radius:4px;}}"
        )

    def update_file(
        self,
        file_path: str,
        percent: int,
        stage: str,
        *,
        state: str = "running",
    ) -> None:
        if file_path not in self._file_progress:
            return
        self._file_progress[file_path] = max(
            self._file_progress[file_path],
            max(0, min(100, int(percent))),
        )
        self._file_states[file_path] = state
        state_prefix = {
            "success": "完成",
            "failed": "失败",
        }.get(state, stage)
        self._items[file_path].setText(
            f"{Path(file_path).name} — {state_prefix}"
        )
        overall = int(
            sum(self._file_progress.values()) / max(len(self._file_progress), 1)
        )
        completed = sum(
            state in {"success", "failed"}
            for state in self._file_states.values()
        )
        self._progress.setValue(overall)
        self._summary.setText(f"已完成 {completed}/{len(self._paths)}")

    def finish(self) -> None:
        self.hide()
        self.deleteLater()

    def closeEvent(self, event) -> None:
        self.hide()
        event.ignore()


class _IngestWorker(QObject):
    """Run ingest in a background thread."""
    finished = Signal()   
    error = Signal(str, str)
    refresh_needed = Signal(str)
    progress = Signal(str, int, str)

    def __init__(
        self,
        ctx: AppContext,
        file_path: str,
        document_id: str | None = None,
        normalization_depth: str = NormalizationDepth.OFF.value,
    ):
        super().__init__()
        self.ctx = ctx
        self.file_path = file_path
        self.document_id = document_id
        self.normalization_depth = NormalizationDepth(normalization_depth).value

    def run(self) -> None:
        try:
            from app.agent.ingest_graph import ingest_graph
            from app.db.schema import open_db

            conn = open_db(self.ctx.data_dir)
            try:
                state = {
                    "file_path": self.file_path,
                    "data_dir": self.ctx.data_dir,
                    "source_type": "standard",
                    "document_id": self.document_id,
                    "embedder": self.ctx.embedder,
                    "provider": self.ctx.provider,
                    "normalization_depth": self.normalization_depth,
                    "conn": conn,
                    "llm_client": self.ctx.openai_client,
                    "llm_model": self.ctx.openai_model,
                }
                result = dict(state)
                self.progress.emit(self.file_path, 0, "准备导入")
                for update in ingest_graph.stream(state, stream_mode="updates"):
                    if isinstance(update, dict):
                        for node_update in update.values():
                            if isinstance(node_update, dict):
                                result.update(node_update)
                    progress = _progress_from_graph_update(update)
                    if progress is not None:
                        self.progress.emit(self.file_path, *progress)
            finally:
                conn.close()

            if result.get("error"):
                self.error.emit(self.file_path, str(result["error"]))
            else:
                self.refresh_needed.emit(self.file_path)
                self.finished.emit()
                
        except Exception as e:
            logger.exception("Ingest worker failed")
            self.error.emit(self.file_path, f"导入失败：{e}")


class LibraryPage(QWidget):
    """Document library management page."""

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._thread: QThread | None = None
        self._threads: set[QThread] = set()
        self._workers: set[QObject] = set()
        self._worker_by_thread: dict[QThread, QObject] = {}
        self._import_batch: dict[str, object] | None = None
        self._progress_dialog: _ImportProgressDialog | None = None
        self._build_ui()
        self._apply_theme()
        from app.ui.theme_manager import ThemeManager
        ThemeManager.instance().theme_changed.connect(self._apply_theme)
        self.refresh()

    def _build_ui(self) -> None:
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Theme.PAGE_MARGIN, Theme.PAGE_MARGIN, Theme.PAGE_MARGIN, Theme.PAGE_MARGIN)

        # Header
        header = QHBoxLayout()
        title = QLabel("文档库")
        title.setStyleSheet(Theme.page_title())
        self._title = title
        header.addWidget(title)
        header.addStretch()

        depth_label = QLabel("思考深度")
        depth_label.setStyleSheet(Theme.form_label())
        self._normalization_depth_label = depth_label
        header.addWidget(depth_label)
        depth_combo = QComboBox()
        depth_combo.addItem("低", NormalizationDepth.OFF.value)
        depth_combo.addItem("中", NormalizationDepth.STANDARD.value)
        depth_combo.addItem("高", NormalizationDepth.REVIEW.value)
        depth_combo.setCurrentIndex(0)
        depth_combo.setToolTip(
            "低：不执行规范化，速度最快；中：每个候选判断一次；高：对变更再复核一次。"
        )
        self._normalization_depth = depth_combo
        header.addWidget(depth_combo)

        import_btn = QPushButton("导入文档")
        import_btn.setStyleSheet(Theme.btn_primary())
        import_btn.clicked.connect(self._import_document)
        self._import_btn = import_btn
        header.addWidget(import_btn)

        self._add_version_btn = QPushButton("新增版本")
        self._add_version_btn.setStyleSheet(Theme.btn_success())
        self._add_version_btn.setEnabled(False)
        self._add_version_btn.clicked.connect(self._add_version)
        header.addWidget(self._add_version_btn)

        self._delete_btn = QPushButton("删除")
        self._delete_btn.setStyleSheet(Theme.btn_danger())
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._delete_selected)
        header.addWidget(self._delete_btn)

        layout.addLayout(header)
        layout.addSpacing(12)

        # Table — one document per row, with latest version surfaced.
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["文档名称", "类型", "版本", "导入时间"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        
        layout.addWidget(self._table, 1)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

        # Status bar
        self._status = QLabel("")
        self._status.setStyleSheet(Theme.label_secondary())
        layout.addWidget(self._status)

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
        if hasattr(self, '_title'):
            self._title.setStyleSheet(Theme.page_title())
        if hasattr(self, '_import_btn'):
            self._import_btn.setStyleSheet(Theme.btn_primary())
        if hasattr(self, '_add_version_btn'):
            self._add_version_btn.setStyleSheet(Theme.btn_success())
        if hasattr(self, '_delete_btn'):
            self._delete_btn.setStyleSheet(Theme.btn_danger())
        
        if hasattr(self, '_status'):
            self._status.setStyleSheet(Theme.label_secondary())
        if hasattr(self, '_normalization_depth_label'):
            self._normalization_depth_label.setStyleSheet(Theme.form_label())

    def refresh(self) -> None:
        """Reload documents from DB."""
        try:
            entries = document_repo.list_library_entries(self.ctx.conn)
            self._table.setRowCount(len(entries))
            document_ids: set[str] = set()
            version_total = 0
            for row, entry in enumerate(entries):
                document_id = str(entry["document_id"])
                document_ids.add(document_id)
                version_id = entry["version_id"]
                if version_id is not None:
                    version_total += 1
                item0 = QTableWidgetItem(entry["doc_name"])
                item0.setData(DOCUMENT_ID_ROLE, document_id)
                item0.setData(VERSION_ID_ROLE, version_id)
                self._table.setItem(row, 0, item0)
                self._table.setItem(
                    row,
                    1,
                    QTableWidgetItem(entry["doc_type"].upper()),
                )
                self._table.setItem(
                    row,
                    2,
                    QTableWidgetItem(self._format_version(entry)),
                )
                created_at = (
                    entry["version_created_at"]
                    if version_id is not None
                    else entry["document_created_at"]
                )
                created = str(created_at)[:10]
                self._table.setItem(row, 3, QTableWidgetItem(created))
            self._status.setText(
                f"共 {len(document_ids)} 份文档，{version_total} 个版本"
            )
        except Exception as e:
            logger.exception("Failed to refresh library")
            self._status.setText(f"加载失败：{e}")

    def _format_version(self, entry) -> str:
        if entry["version_id"] is None:
            return "暂无版本"
        version = f"v{entry['version_no']}"
        label = entry["version_label"] or ""
        return f"{version}（{label}）" if label else version

    def _import_document(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择文档", "",
            DOCUMENT_FILE_FILTER,
        )
        if not paths:
            return
        self._start_ingest_batch(paths)

    def _start_ingest_batch(
        self,
        paths: list[str],
        document_id: str | None = None,
    ) -> None:
        unique_paths = list(dict.fromkeys(paths))
        if not unique_paths:
            return
        if self._import_batch is not None:
            if self._progress_dialog is not None:
                self._progress_dialog.show()
                self._progress_dialog.raise_()
                self._progress_dialog.activateWindow()
            return
        self._import_batch = {
            "pending": set(unique_paths),
            "successes": set(),
            "failures": {},
            "normalization_depth": self._normalization_depth.currentData(),
        }
        self._progress_dialog = _ImportProgressDialog(unique_paths, self)
        self._import_btn.setEnabled(False)
        self._add_version_btn.setEnabled(False)
        self._delete_btn.setEnabled(False)
        self._normalization_depth.setEnabled(False)
        self._progress_dialog.show()
        for path in unique_paths:
            self._run_ingest(path, document_id=document_id)

    def _run_ingest(self, file_path: str, document_id: str | None = None) -> None:
        if self._import_batch is None:
            self._start_ingest_batch([file_path], document_id=document_id)
            return
        thread = QThread()
        depth = (
            self._import_batch.get("normalization_depth", NormalizationDepth.OFF.value)
            if self._import_batch is not None
            else NormalizationDepth.OFF.value
        )
        worker = _IngestWorker(
            self.ctx,
            file_path,
            document_id=document_id,
            normalization_depth=str(depth),
        )
        self._thread = thread
        self.worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.refresh_needed.connect(self._on_ingest_done)
        worker.progress.connect(self._on_ingest_progress)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_ingest_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_ingest_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._threads.add(thread)
        self._workers.add(worker)
        self._worker_by_thread[thread] = worker
        thread.start()

    @Slot()
    def _on_ingest_thread_finished(self, thread: QThread | None = None) -> None:
        finished_thread = thread or self.sender()
        if finished_thread not in self._threads:
            return
        self._threads.discard(finished_thread)
        worker = self._worker_by_thread.pop(finished_thread, None)
        if worker is not None:
            self._workers.discard(worker)
        self._finish_import_batch_if_ready()

    @Slot(str, int, str)
    def _on_ingest_progress(
        self,
        file_path: str,
        percent: int,
        stage: str,
    ) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.update_file(file_path, percent, stage)

    @Slot(str)
    def _on_ingest_done(self, file_path: str) -> None:
        if self._import_batch is None:
            return
        pending = self._import_batch["pending"]
        successes = self._import_batch["successes"]
        if isinstance(pending, set) and file_path in pending:
            pending.discard(file_path)
            if isinstance(successes, set):
                successes.add(file_path)
            if self._progress_dialog is not None:
                self._progress_dialog.update_file(
                    file_path,
                    100,
                    "导入完成",
                    state="success",
                )
        self._finish_import_batch_if_ready()

    @Slot(str, str)
    def _on_ingest_error(self, file_path: str, msg: str) -> None:
        if self._import_batch is None:
            return
        pending = self._import_batch["pending"]
        failures = self._import_batch["failures"]
        if isinstance(pending, set) and file_path in pending:
            pending.discard(file_path)
            if isinstance(failures, dict):
                failures[file_path] = msg
            if self._progress_dialog is not None:
                self._progress_dialog.update_file(
                    file_path,
                    100,
                    msg,
                    state="failed",
                )
        self._finish_import_batch_if_ready()

    def _finish_import_batch_if_ready(self) -> None:
        batch = self._import_batch
        if batch is None:
            return
        pending = batch["pending"]
        if not isinstance(pending, set) or pending or self._threads:
            return
        successes = batch["successes"]
        failures = batch["failures"]
        success_count = len(successes) if isinstance(successes, set) else 0
        failure_count = len(failures) if isinstance(failures, dict) else 0
        if self._progress_dialog is not None:
            self._progress_dialog.finish()
        self._import_batch = None
        self._progress_dialog = None
        self._import_btn.setEnabled(True)
        self._normalization_depth.setEnabled(True)
        self._on_selection_changed()
        self.refresh()
        if failure_count:
            failure_lines = [
                f"- {Path(path).name}: {message}"
                for path, message in failures.items()
            ] if isinstance(failures, dict) else []
            QMessageBox.warning(
                self,
                "导入完成（部分失败）",
                "\n".join(
                    [
                        f"成功 {success_count} 个，失败 {failure_count} 个。",
                        *failure_lines,
                    ]
                ),
            )
        else:
            QMessageBox.information(
                self,
                "导入完成",
                f"成功导入 {success_count} 个文档。",
            )

    def _on_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        has_selection = bool(rows)
        if has_selection:
            row = rows[0].row()
            version_id = self._table.item(row, 0).data(VERSION_ID_ROLE)
            self._delete_btn.setText(
                "删除版本" if version_id is not None else "删除文档"
            )
        else:
            self._delete_btn.setText("删除")
        self._add_version_btn.setEnabled(
            has_selection and self._import_batch is None
        )
        self._delete_btn.setEnabled(
            has_selection and self._import_batch is None
        )

    def _add_version(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        doc_id = self._table.item(row, 0).data(DOCUMENT_ID_ROLE)
        doc_name = self._table.item(row, 0).text()
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"为《{doc_name}》选择新版本文件",
            "",
            DOCUMENT_FILE_FILTER,
        )
        self._start_ingest_batch(paths, document_id=doc_id)

    def _delete_selected(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        item = self._table.item(row, 0)
        doc_id = item.data(DOCUMENT_ID_ROLE)
        version_id = item.data(VERSION_ID_ROLE)
        doc_name = item.text()

        if version_id is None:
            message = f"删除空文档《{doc_name}》的记录，且不可恢复。\n\n是否继续？"
        else:
            version_text = self._table.item(row, 2).text()
            versions = document_repo.list_versions(self.ctx.conn, doc_id)
            last_version_note = (
                "\n这是该文档的最后一个版本，文档记录也会一并删除。"
                if len(versions) == 1
                else ""
            )
            message = (
                f"删除《{doc_name}》{version_text} 将同时删除该版本的规格化产物、"
                f"向量索引和关联对比记录，且不可恢复。{last_version_note}"
                "\n\n是否继续？"
            )

        confirm = QMessageBox.question(
            self,
            "确认删除",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            if version_id is None:
                cleanup_failures = document_repo.delete_document(
                    self.ctx.conn,
                    doc_id,
                    data_dir=self.ctx.data_dir,
                )
            else:
                result = document_repo.delete_version(
                    self.ctx.conn,
                    version_id,
                    data_dir=self.ctx.data_dir,
                )
                cleanup_failures = result.cleanup_failures
            self.refresh()
            if cleanup_failures:
                paths = "\n".join(f"- {path}" for path in cleanup_failures)
                QMessageBox.warning(
                    self,
                    "删除完成（部分文件未清理）",
                    f"版本记录已删除，但以下本地文件未能清理：\n{paths}",
                )
        except Exception as e:
            logger.exception("Failed to delete library selection")
            QMessageBox.critical(self, "删除失败", str(e))
