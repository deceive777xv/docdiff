"""Settings dialog — configure providers and embedding."""
from __future__ import annotations
import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
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
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("保存")
        save_btn.setStyleSheet(
            f"background-color:{Theme.TEXT_PRIMARY};color:white;"
            "border:none;padding:8px 20px;border-radius:6px;font-size:13px;"
        )
        save_btn.setFixedWidth(80)
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)

        outer.addLayout(btn_row)

    # ── Logic ──────────────────────────────────────────────────────────────────

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
