"""Settings dialog — configure providers, embedding, backup/restore, and updates."""
from __future__ import annotations
import logging

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from app.config import settings as settings_mod
from app.config.settings import LocalEmbeddingConfig, ProviderConfig
from app.ui.app_context import AppContext
from app.ui.theme import Theme

logger = logging.getLogger(__name__)


class _UpdateCheckThread(QThread):
    done = Signal(object)  # emits str (new version) or None

    def run(self) -> None:
        from app.services.update_checker import check_for_update
        self.done.emit(check_for_update())


class SettingsDialog(QDialog):
    """Modal settings dialog for provider and embedding configuration."""

    provider_changed = Signal()

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("设置")
        self.setMinimumWidth(600)
        self.setModal(True)
        self._build_ui()
        self._apply_theme()
        from app.ui.theme_manager import ThemeManager
        ThemeManager.instance().theme_changed.connect(self._apply_theme)
        self._load_values()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(16)

        # Provider group
        provider_group = QGroupBox("模型 API 配置")
        provider_form = QFormLayout(provider_group)
        provider_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        provider_form.setSpacing(10)

        self._provider_name = QLineEdit()
        self._provider_name.setPlaceholderText("配置名称（如：DeepSeek）")
        provider_form.addRow("名称：", self._provider_name)

        self._base_url = QLineEdit()
        self._base_url.setPlaceholderText("https://api.deepseek.com/v1")
        provider_form.addRow("Base URL：", self._base_url)

        self._api_key = QLineEdit()
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setPlaceholderText("sk-…")
        provider_form.addRow("API Key：", self._api_key)

        self._chat_model = QLineEdit()
        self._chat_model.setPlaceholderText("deepseek-chat")
        provider_form.addRow("Chat 模型：", self._chat_model)

        self._embed_model = QLineEdit()
        self._embed_model.setPlaceholderText("text-embedding-ada-002")
        provider_form.addRow("Embed 模型：", self._embed_model)

        outer.addWidget(provider_group)

        # Local embedding group
        embed_group = QGroupBox()
        embed_layout = QVBoxLayout(embed_group)

        self._local_embed_enabled = QCheckBox("启用本地 Embedding 模型")
        embed_layout.addWidget(self._local_embed_enabled)

        path_row = QHBoxLayout()
        self._local_model_path = QLineEdit()
        self._local_model_path.setPlaceholderText("模型目录路径…")
        path_row.addWidget(self._local_model_path)
        browse_btn = QPushButton("浏览…")
        browse_btn.setFixedWidth(64)
        browse_btn.clicked.connect(self._browse_model_path)
        path_row.addWidget(browse_btn)
        embed_layout.addLayout(path_row)

        outer.addWidget(embed_group)

        # Data management group (backup / restore)
        data_group = QGroupBox()
        data_layout = QVBoxLayout(data_group)

        backup_row = QHBoxLayout()
        self._backup_label = QLabel("备份当前数据库、解析缓存和配置文件到 zip")
        self._backup_label.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:14px;")
        backup_row.addWidget(self._backup_label)
        backup_row.addStretch()
        backup_btn = QPushButton("创建备份…")
        backup_btn.setFixedWidth(110)
        backup_btn.clicked.connect(self._do_backup)
        backup_row.addWidget(backup_btn)
        data_layout.addLayout(backup_row)

        restore_row = QHBoxLayout()
        self._restore_label = QLabel("从备份 zip 恢复数据（将覆盖当前数据，请谨慎操作）")
        self._restore_label.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:14px;")
        restore_row.addWidget(self._restore_label)
        restore_row.addStretch()
        restore_btn = QPushButton("从备份恢复…")
        restore_btn.setFixedWidth(110)
        restore_btn.clicked.connect(self._do_restore)
        restore_row.addWidget(restore_btn)
        data_layout.addLayout(restore_row)

        outer.addWidget(data_group)

        # Update check group
        update_group = QGroupBox()
        update_layout = QHBoxLayout(update_group)
        from app.services.update_checker import APP_VERSION
        self._update_status = QLabel(f"当前版本：{APP_VERSION}")
        self._update_status.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:14px;")
        update_layout.addWidget(self._update_status)
        update_layout.addStretch()
        check_btn = QPushButton("检查更新")
        check_btn.setFixedWidth(90)
        check_btn.clicked.connect(self._check_update)
        update_layout.addWidget(check_btn)
        outer.addWidget(update_group)

        outer.addStretch()

        # Bottom button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(
            f"background-color:transparent;color:{Theme.TEXT_PRIMARY};"
            f"border:1px solid {Theme.BORDER};padding:8px 20px;"
            "border-radius:6px;font-size:13px;"
        )
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        self._cancel_btn = cancel_btn
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("保存")
        save_btn.setStyleSheet(
            f"background-color:{Theme.TEXT_PRIMARY};color:{Theme.NAV_ACTIVE_TEXT};"
            "border:none;padding:8px 20px;border-radius:6px;font-size:13px;"
        )
        save_btn.setFixedWidth(80)
        save_btn.clicked.connect(self._save)
        self._save_btn = save_btn
        btn_row.addWidget(save_btn)

        outer.addLayout(btn_row)

    # ── Logic ──────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"background-color:{Theme.BG_PAGE};")
        self._backup_label.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:14px;")
        self._restore_label.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:14px;")
        self._update_status.setStyleSheet(f"color:{Theme.TEXT_SECONDARY};font-size:14px;")
        self._cancel_btn.setStyleSheet(
            f"background-color:transparent;color:{Theme.TEXT_PRIMARY};"
            f"border:1px solid {Theme.BORDER};padding:8px 20px;"
            "border-radius:6px;font-size:13px;"
        )
        self._save_btn.setStyleSheet(
            f"background-color:{Theme.TEXT_PRIMARY};color:{Theme.NAV_ACTIVE_TEXT};"
            "border:none;padding:8px 20px;border-radius:6px;font-size:13px;"
        )

    def _load_values(self) -> None:
        s = self.ctx.settings

        if s.providers:
            p = s.providers[0]
            self._provider_name.setText(p.name)
            self._base_url.setText(p.base_url)
            self._api_key.setText(p.api_key)
            self._chat_model.setText(p.chat_model)
            self._embed_model.setText(p.embed_model)

        le = s.local_embedding
        self._local_embed_enabled.setChecked(le.enabled)
        self._local_model_path.setText(le.model_path)

    def _browse_model_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择本地模型目录")
        if path:
            self._local_model_path.setText(path)

    def _do_backup(self) -> None:
        dest = QFileDialog.getExistingDirectory(self, "选择备份保存目录")
        if not dest:
            return
        try:
            from app.services.backup_service import backup
            zip_path = backup(self.ctx.data_dir, dest)
            QMessageBox.information(self, "备份完成", f"备份已保存至：\n{zip_path}")
        except Exception as e:
            logger.exception("Backup failed")
            QMessageBox.critical(self, "备份失败", str(e))

    def _do_restore(self) -> None:
        zip_path, _ = QFileDialog.getOpenFileName(
            self, "选择备份文件", "", "Zip 文件 (*.zip)"
        )
        if not zip_path:
            return
        confirm = QMessageBox.question(
            self,
            "确认恢复",
            "恢复将覆盖当前数据和配置，操作不可撤销。\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            from app.services.backup_service import restore
            restore(zip_path, self.ctx.data_dir)
            QMessageBox.information(
                self, "恢复完成", "数据已恢复，请重启应用以使更改生效。"
            )
        except Exception as e:
            logger.exception("Restore failed")
            QMessageBox.critical(self, "恢复失败", str(e))

    def _check_update(self) -> None:
        self._update_status.setText("正在检查更新…")
        self._thread = _UpdateCheckThread()
        self._thread.done.connect(self._on_update_result)
        self._thread.start()

    def _on_update_result(self, new_version) -> None:
        from app.services.update_checker import APP_VERSION
        if new_version:
            self._update_status.setText(f"发现新版本：{new_version}（当前：{APP_VERSION}）")
            QMessageBox.information(
                self,
                "发现新版本",
                f"新版本 {new_version} 已发布，请前往项目主页下载更新。",
            )
        else:
            self._update_status.setText(f"当前版本：{APP_VERSION}（已是最新）")

    def _save(self) -> None:
        name = self._provider_name.text().strip()
        if not name:
            QMessageBox.warning(self, "验证错误", "配置名称不能为空。")
            return

        provider = ProviderConfig(
            name=name,
            type="openai_compatible",
            api_key=self._api_key.text().strip(),
            base_url=self._base_url.text().strip(),
            chat_model=self._chat_model.text().strip() or "deepseek-chat",
            embed_model=self._embed_model.text().strip() or "text-embedding-ada-002",
        )
        self.ctx.settings.providers = [provider]
        self.ctx.settings.active_provider = name
        self.ctx.settings.local_embedding = LocalEmbeddingConfig(
            enabled=self._local_embed_enabled.isChecked(),
            model_path=self._local_model_path.text().strip(),
        )

        try:
            settings_mod.save(self.ctx.settings)
            from app.core.model.factory import build_provider, get_embedder
            self.ctx.provider = build_provider(provider)
            self.ctx.embedder = get_embedder(self.ctx.settings)
            self.provider_changed.emit()
            self.accept()
        except Exception as e:
            logger.exception("Failed to save settings")
            QMessageBox.critical(self, "保存失败", str(e))
