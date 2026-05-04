# Phase 2 设计文档：UI 优化 + LangGraph 架构迁移

**日期**：2026-05-02  
**状态**：已批准  
**范围**：UI 全面优化、LangGraph 编排架构迁移、I5 补丁（LibraryPage 新增版本入口）

---

## 1. 目标

1. 将现有散落的硬编码样式统一到集中主题模块，采用低饱和度配色，各功能区有清晰视觉区分。
2. 将现有直接函数调用的三个工作流（入库/对比/问答）迁移到 LangGraph StateGraph，为 Phase 2 新功能（OCR 节点、多模式问答节点等）打下基础。
3. 修复 I5：LibraryPage 缺少"新增版本"UI 入口。

---

## 2. UI 主题设计

### 2.1 主题模块

新建 `app/ui/theme.py`，所有页面从此处导入颜色常量，不再散落硬编码值。

```python
class Theme:
    # 布局
    SIDEBAR_WIDTH = 140

    # 侧边栏
    BG_SIDEBAR       = "#1e2736"
    NAV_ACTIVE_BG    = "#3b5080"
    NAV_ACTIVE_TEXT  = "#ffffff"
    NAV_TEXT         = "#c4cad8"
    LOGO_TEXT_COLOR  = "#a8b8d8"

    # 页面区域
    BG_PAGE          = "#f4f5f7"   # 页面底色
    BG_CARD          = "#ffffff"   # 卡片/面板
    BG_HEADER        = "#edf0f5"   # 页面顶部 header 区

    # 文字
    TEXT_PRIMARY     = "#2c3a52"
    TEXT_SECONDARY   = "#6b7a99"
    TEXT_PLACEHOLDER = "#9ca3af"

    # 边框
    BORDER           = "#dde1ea"

    # 操作色
    COLOR_PRIMARY    = "#3d5fa0"   # 主操作按钮（低饱和靛蓝）
    COLOR_SUCCESS    = "#3e7d6a"   # 导入/成功
    COLOR_DANGER     = "#a04040"   # 删除/错误
    COLOR_WARNING    = "#b09830"   # 警告

    # 差异高亮（同步更新 diff_template.html）
    DIFF_ADDED       = "#4a9e72"
    DIFF_DELETED     = "#c05050"
    DIFF_MINOR       = "#b09830"
    DIFF_MAJOR       = "#c07840"
    DIFF_REWRITE     = "#7a58c0"
    DIFF_FORMAT      = "#9ca3af"
```

### 2.2 图标集成

- **窗口图标**：`main.py` 中 `QApplication.setWindowIcon(QIcon("assets/icons/docdiff.ico"))`
- **侧边栏 Logo**：`QLabel` + `QPixmap("assets/icons/docdiff.png").scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)`
- **移除 emoji**：所有按钮标签去掉 `＋`、`🔍`、`💬` 等 emoji，改为纯文字

### 2.3 功能区视觉区分

| 区域 | 背景色 | 说明 |
|------|--------|------|
| 侧边栏 | `#1e2736` | 深色，与内容区强对比 |
| 页面 header 区 | `#edf0f5` | 含标题、工具栏按钮 |
| 内容主体 | `#f4f5f7` | 浅灰底 |
| 卡片/表格/面板 | `#ffffff` | 白色卡片，含 border-radius 和 box-shadow |

### 2.4 需修改的文件

| 文件 | 改动 |
|------|------|
| `app/ui/theme.py` | 新建，定义 Theme 类 |
| `app/ui/main_window.py` | 导入 Theme，更新 SideBar 和 NavButton 样式，添加 Logo 图片，设置窗口图标 |
| `app/ui/pages/home_page.py` | 导入 Theme，更新所有 setStyleSheet，移除 emoji |
| `app/ui/pages/library_page.py` | 同上，+ I5 补丁 |
| `app/ui/pages/compare_page.py` | 同上 |
| `app/ui/pages/qa_page.py` | 同上 |
| `app/ui/pages/settings_page.py` | 同上 |
| `assets/diff_template.html` | 更新差异高亮色值 |
| `main.py` | 设置 QApplication 窗口图标 |

---

## 3. LangGraph 架构迁移

### 3.1 新目录结构

```
app/agent/
├── __init__.py
├── states.py          # 三个 TypedDict 状态定义
├── ingest_graph.py    # 入库 StateGraph
├── compare_graph.py   # 对比 StateGraph
└── qa_graph.py        # 问答 StateGraph
```

现有 `app/services/` 保留不删，内部逻辑函数供 graph 节点调用。

### 3.2 State 定义（`app/agent/states.py`）

```python
from typing import TypedDict, Any, Optional

class IngestState(TypedDict, total=False):
    # 输入
    file_path: str
    data_dir: str
    source_type: str          # "standard" | "uploaded"
    document_id: str          # 新增版本时传入，入库新文档时为空
    embedder: Any
    # 节点间传递
    doc_id: str
    version_id: str
    # 状态
    error: Optional[str]
    status: str               # pending/running/completed/failed

class CompareState(TypedDict, total=False):
    # 输入
    data_dir: str
    baseline_version_id: str
    target_version_id: str
    provider: Any
    embedder: Any
    # 节点间传递
    task_id: str
    diff_items: list
    # 状态
    error: Optional[str]
    status: str

class QAState(TypedDict, total=False):
    # 输入
    data_dir: str
    question: str
    scope: str                # "current_doc" | "standard_lib" | "all"
    version_id: Optional[str]
    provider: Any
    embedder: Any
    # 节点间传递
    chunks: list
    # 输出
    answer: str
    citations: list
    # 状态
    error: Optional[str]
    status: str
```

### 3.3 Ingest Graph（`app/agent/ingest_graph.py`）

**节点**：`file_check` → `parse_document` → `save_document` → `build_embeddings`

| 节点 | 职责 | 对应原逻辑 |
|------|------|-----------|
| `file_check` | 哈希计算、重复检测、格式校验 | `ingest_service` 前段 |
| `parse_document` | 调用 parser router 生成 DocumentIR | `ingest_service` 中段 |
| `save_document` | 写 documents/versions/chunks 表 | `ingest_service` 中段 |
| `build_embeddings` | 向量化并写 FAISS 索引 | `ingest_service` 后段 |

**错误路由**：每节点出错设 `state["error"]`，条件边检测后跳到 `END`。

```python
def _route(state) -> str:
    return END if state.get("error") else "next"
```

**编译产物**：模块级 `ingest_graph = build_ingest_graph()` 供 worker 导入。

### 3.4 Compare Graph（`app/agent/compare_graph.py`）

**节点**：`create_task` → `ensure_parsed` → `align_structure` → `semantic_compare` → `classify_diffs` → `persist_result`

| 节点 | 职责 |
|------|------|
| `create_task` | 写 compare_tasks 记录，返回 task_id |
| `ensure_parsed` | 检查两版本是否已有 parsed_json，否则触发解析 |
| `align_structure` | StructureAligner 章节对齐 |
| `semantic_compare` | SemanticMatcher 段落对齐 |
| `classify_diffs` | DiffClassifier LLM 分类 |
| `persist_result` | 写 diff_items，更新 compare_tasks.status |

### 3.5 QA Graph（`app/agent/qa_graph.py`）

**节点**：`resolve_scope` → `retrieve_chunks` → `generate_answer` → `attach_citations`

| 节点 | 职责 |
|------|------|
| `resolve_scope` | 根据 scope 确定检索范围（version_id 列表）|
| `retrieve_chunks` | FAISS 向量检索，返回候选 chunks |
| `generate_answer` | 调用 LLM 生成答案 |
| `attach_citations` | 格式化引用来源，写入 state.citations |

### 3.6 UI Worker 改动

三个 worker 的 `run()` 方法均改为调用对应 graph：

```python
# _IngestWorker.run()
from app.agent.ingest_graph import ingest_graph
result = ingest_graph.invoke({
    "file_path": self.file_path,
    "data_dir": self.ctx.data_dir,
    "source_type": "standard",
    "embedder": self.ctx.embedder,
})
if result.get("error"):
    self.error.emit(result["error"])
else:
    self.finished.emit(result["doc_id"], result["version_id"])

# _CompareWorker.run()
from app.agent.compare_graph import compare_graph
result = compare_graph.invoke({
    "data_dir": self.ctx.data_dir,
    "baseline_version_id": self.baseline_version_id,
    "target_version_id": self.target_version_id,
    "provider": self.ctx.provider,
    "embedder": self.ctx.embedder,
})

# _QaWorker.run()
from app.agent.qa_graph import qa_graph
result = qa_graph.invoke({
    "data_dir": self.ctx.data_dir,
    "question": self.question,
    "scope": self.scope,
    "version_id": self.version_id,
    "provider": self.ctx.provider,
    "embedder": self.ctx.embedder,
})
```

`.invoke()` 是同步阻塞调用，天然适合在 QThread worker 内运行，不影响 Qt 事件循环。

### 3.7 测试兼容性

- 现有 105 个测试针对 services 层函数，服务层函数签名不变，测试继续有效。
- 新增 `tests/test_agent/` 目录，针对各 graph 的集成测试。

---

## 4. I5 补丁：LibraryPage 新增版本入口

在 LibraryPage 工具栏 `import_btn` 旁新增 `add_version_btn`：

- 默认禁用，表格选中行后激活
- 点击弹出文件选择器
- 调用 `ingest_graph.invoke({..., "document_id": selected_doc_id, "source_type": "standard"})`
- 成功后刷新列表

```python
self._add_version_btn = QPushButton("新增版本")
self._add_version_btn.setEnabled(False)
self._table.itemSelectionChanged.connect(self._on_selection_changed)

def _on_selection_changed(self):
    self._add_version_btn.setEnabled(len(self._table.selectedItems()) > 0)
```

---

## 5. 依赖新增

```toml
# pyproject.toml dependencies 新增
"langgraph>=0.2.0"
```

---

## 6. 不在本次范围内

- OCR 增强节点（Phase 2 功能，下次迭代）
- 实质修改/重写分类增强
- 多模式问答（对比/标准库/混合）
- 差异报告导出
- PyInstaller 打包
