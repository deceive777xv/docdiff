"""Document library page."""
from __future__ import annotations
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui.app_context import AppContext
from app.ui.theme import Theme
from app.db import document_repo

logger = logging.getLogger(__name__)


class _IngestWorker(QObject):
    """Run ingest in a background thread."""
    finished = Signal()   
    error = Signal(str)
    refresh_needed = Signal(str)

    def __init__(self, ctx: AppContext, file_path: str, document_id: str | None = None):
        super().__init__()
        self.ctx = ctx
        self.file_path = file_path
        self.document_id = document_id

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
                    "llm_client": self.ctx.openai_client,
                    "llm_model": self.ctx.openai_model,
                })
            finally:
                conn.close()

            if result.get("error"):
                self.error.emit(result["error"])
            else:
                self.refresh_needed.emit(self.file_path)
                self.finished.emit()
                
        except Exception as e:
            logger.exception("Ingest worker failed")
            self.error.emit(f"导入失败：{e}")


class LibraryPage(QWidget):
    """Document library management page."""

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._thread: QThread | None = None
        self._threads: set[QThread] = set()
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
        self._delete_btn.clicked.connect(self._delete_document)
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

    def refresh(self) -> None:
        """Reload documents from DB."""
        try:
            docs = self.ctx.conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
            self._table.setRowCount(len(docs))
            version_total = 0
            for row, doc in enumerate(docs):
                versions = document_repo.list_versions(self.ctx.conn, doc["id"])
                version_total += len(versions)
                item0 = QTableWidgetItem(doc["doc_name"])
                item0.setData(Qt.UserRole, doc["id"])
                self._table.setItem(row, 0, item0)
                self._table.setItem(row, 1, QTableWidgetItem(doc["doc_type"].upper()))
                self._table.setItem(row, 2, QTableWidgetItem(self._format_version_summary(versions)))
                created = str(doc["created_at"])[:10]
                self._table.setItem(row, 3, QTableWidgetItem(created))
            self._status.setText(f"共 {len(docs)} 份文档，{version_total} 个版本")
        except Exception as e:
            logger.exception("Failed to refresh library")
            self._status.setText(f"加载失败：{e}")

    def _format_version_summary(self, versions) -> str:
        if not versions:
            return "暂无版本"
        latest = versions[0]
        version = f"v{latest['version_no']}"
        label = latest["version_label"] or ""
        latest_text = f"{version}（{label}）" if label else version
        return f"{latest_text} · 共 {len(versions)} 版"

    def _import_document(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择文档", "",
            "支持的文档 (*.pdf *.docx *.pptx *.xlsx *.xls *.html *.htm *.csv *.json *.xml *.epub *.txt *.md *.markdown)"
            ";;PDF (*.pdf);;Word (*.docx);;PowerPoint (*.pptx)"
            ";;Excel (*.xlsx *.xls);;网页 (*.html *.htm);;Markdown (*.md *.markdown)"
            ";;其他 (*.csv *.json *.xml *.epub *.txt)"
        )
        if not paths:
            return
        for path in paths:
            self._run_ingest(path)

    def _run_ingest(self, file_path: str, document_id: str | None = None) -> None:
        thread = QThread()
        worker = _IngestWorker(self.ctx, file_path, document_id=document_id)
        self._thread = thread
        self.worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.refresh_needed.connect(self._on_ingest_done)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_ingest_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda th=thread: self._threads.discard(th))
        self._threads.add(thread)
        thread.start()

    @Slot(str)
    def _on_ingest_done(self, file_path: str) -> None:
        name = Path(file_path).name
        QMessageBox.information(self, "导入成功", f"《{name}》已成功导入文档库。")
        self.refresh()

    @Slot(str)
    def _on_ingest_error(self, msg: str) -> None:
        QMessageBox.warning(self, "导入错误", msg)

    def _on_selection_changed(self) -> None:
        has_selection = len(self._table.selectedItems()) > 0
        self._add_version_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)

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
            "支持的文档 (*.pdf *.docx *.pptx *.xlsx *.xls *.html *.htm *.csv *.json *.xml *.epub *.txt *.md *.markdown)"
            ";;PDF (*.pdf);;Word (*.docx);;PowerPoint (*.pptx)"
            ";;Excel (*.xlsx *.xls);;网页 (*.html *.htm);;Markdown (*.md *.markdown)"
            ";;其他 (*.csv *.json *.xml *.epub *.txt)",
        )
        for path in paths:
            self._run_ingest(path, document_id=doc_id)

    def _delete_document(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        doc_id = self._table.item(row, 0).data(Qt.UserRole)
        doc_name = self._table.item(row, 0).text()

        confirm = QMessageBox.question(
            self,
            "确认删除",
            f"删除《{doc_name}》将同时删除该文档的所有版本、向量索引和关联的对比记录，且不可恢复。\n\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            from app.db.document_repo import delete_document
            delete_document(self.ctx.conn, doc_id, data_dir=self.ctx.data_dir)
            self.refresh()
        except Exception as e:
            logger.exception("Failed to delete document")
            QMessageBox.critical(self, "删除失败", str(e))
