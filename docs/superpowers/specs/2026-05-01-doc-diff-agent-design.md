# Doc-Diff-Agent 设计文档

**日期**：2026-05-01（最后更新：2026-05-07）  
**状态**：第三阶段已完成（2026-05-07）  
**技术路线**：方案 A（轻量直接模式，MVP 不引入 LangGraph）

---

## 1. 项目目标

面向 Windows 桌面的智能文档对比与问答应用，核心能力：

1. 多格式文档解析（PDF、DOCX，OCR 接口预留）
2. 本地标准文档库（SQLite + FAISS，支持版本管理）
3. 语义级文档对比（结构对齐 → 向量匹配 → LLM 分类）
4. 差异分类标注与可视化（新增/删减/微调/实质修改/重写/格式变化）
5. 检索增强问答（RAG，支持单文档/对比/标准库/混合四种模式）
6. 多模型 API 兼容（OpenAI 兼容接口、本地 embedding）
7. Windows 独立桌面应用打包，不依赖 Docker/虚拟机

---

## 2. 技术栈

| 层次 | 技术选型 |
|------|----------|
| 桌面 UI | PySide6 |
| Diff 视图 | QWebEngineView + QWebChannel |
| 文档解析 | Docling（主）+ PyMuPDF + python-docx（回退）|
| OCR | 接口预留，Tesseract 二期实现 |
| 向量索引 | FAISS-cpu（本地文件持久化）|
| 本地数据库 | SQLite（via SQLAlchemy Core）|
| Embedding | 混合：优先本地 sentence-transformers，fallback API |
| Agent 编排 | MVP 直接函数调用；二期引入 LangGraph |
| 打包 | PyInstaller（onedir 模式）+ Inno Setup |
| 加密 | cryptography 库 + 机器唯一标识派生密钥 |

---

## 3. 项目结构

```
doc_diff_agent/
├── main.py
├── app/
│   ├── ui/
│   │   ├── main_window.py
│   │   ├── pages/
│   │   │   ├── home_page.py
│   │   │   ├── compare_page.py
│   │   │   ├── library_page.py
│   │   │   ├── qa_page.py
│   │   │   └── settings_page.py
│   │   └── components/
│   ├── services/
│   │   ├── ingest_service.py
│   │   ├── compare_service.py
│   │   └── qa_service.py
│   ├── core/
│   │   ├── parser/
│   │   │   ├── router.py          # 解析路由
│   │   │   ├── docling_adapter.py
│   │   │   ├── pymupdf_extractor.py
│   │   │   ├── docx_extractor.py
│   │   │   ├── ocr_interface.py   # OCR 预留接口（MVP 不实现）
│   │   │   └── ir_builder.py      # → DocumentIR
│   │   ├── diff/
│   │   │   ├── structure_aligner.py
│   │   │   ├── semantic_matcher.py
│   │   │   └── diff_classifier.py
│   │   ├── retrieval/
│   │   │   ├── indexer.py
│   │   │   └── searcher.py
│   │   └── model/
│   │       ├── base_provider.py
│   │       ├── openai_compatible.py
│   │       ├── azure_provider.py
│   │       └── local_embedding.py
│   ├── db/
│   │   ├── schema.py              # 建表 DDL
│   │   ├── document_repo.py
│   │   ├── chunk_repo.py
│   │   ├── compare_repo.py
│   │   └── faiss_store.py
│   └── config/
│       ├── settings.py            # 配置读写
│       └── crypto.py              # API Key 加解密
├── assets/
│   ├── diff_template.html         # WebEngine diff 视图模板
│   └── icons/
├── tests/
└── build/
    ├── doc_diff_agent.spec        # PyInstaller spec
    └── installer.iss              # Inno Setup 脚本
```

---

## 4. 核心数据结构

### 4.1 DocumentIR（文档中间表示）

```python
@dataclass
class Sentence:
    text: str

@dataclass
class Paragraph:
    paragraph_id: str
    page_no: int
    text: str
    sentences: list[Sentence]

@dataclass
class Section:
    section_id: str
    title: str
    level: int                  # 1/2/3
    paragraphs: list[Paragraph]

@dataclass
class DocumentIR:
    doc_id: str
    title: str
    file_hash: str
    sections: list[Section]
    plain_text: str             # 全文纯文本，用于快速检索
```

### 4.2 DiffItem（差异项）

```python
@dataclass
class DiffItem:
    diff_id: str
    section_path: str           # 如 "第一章/第2条"
    diff_type: Literal["新增", "删减", "微调", "实质修改", "重写", "格式变化"]
    risk_level: Literal["high", "medium", "low"]
    baseline_text: str
    target_text: str
    similarity_score: float
    explanation: str
    baseline_page: int
    target_page: int
```

### 4.3 配置文件（%APPDATA%/DocDiffAgent/config.json）

```json
{
  "providers": [
    {
      "name": "默认",
      "type": "openai_compatible",
      "api_key": "<encrypted>",
      "base_url": "https://api.deepseek.com/v1",
      "chat_model": "deepseek-chat",
      "embed_model": "text-embedding-ada-002"
    }
  ],
  "local_embedding": {
    "enabled": false,
    "model_path": ""
  },
  "active_provider": "默认",
  "data_dir": ""
}
```

---

## 5. 数据库表设计

```sql
-- 文档主表
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    doc_name TEXT NOT NULL,
    doc_type TEXT NOT NULL,         -- pdf / docx
    file_path TEXT NOT NULL,        -- docs/ 目录下按 hash 命名的文件
    file_hash TEXT UNIQUE NOT NULL,
    source_type TEXT NOT NULL,      -- standard / uploaded
    business_category TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- 文档版本
CREATE TABLE document_versions (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    version_no INTEGER NOT NULL,
    version_label TEXT,
    status TEXT NOT NULL,           -- active / archived / needs_review
    parsed_json_path TEXT,          -- parsed/ 目录下的 IR JSON
    summary TEXT,
    created_at TEXT NOT NULL
);

-- 文本块（用于检索和向量化）
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL REFERENCES document_versions(id),
    chunk_no INTEGER NOT NULL,
    section_path TEXT,
    page_no INTEGER,
    text TEXT NOT NULL,
    faiss_index_id INTEGER           -- FAISS 内部 ID
);

-- 对比任务
CREATE TABLE compare_tasks (
    id TEXT PRIMARY KEY,
    baseline_version_id TEXT NOT NULL,
    target_version_id TEXT NOT NULL,
    status TEXT NOT NULL,           -- pending / running / completed / failed
    result_json_path TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);

-- 差异明细
CREATE TABLE diff_items (
    id TEXT PRIMARY KEY,
    compare_task_id TEXT NOT NULL REFERENCES compare_tasks(id),
    section_path TEXT,
    diff_type TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    baseline_text TEXT,
    target_text TEXT,
    similarity_score REAL,
    explanation TEXT,
    baseline_page INTEGER,
    target_page INTEGER
);
```

**MVP 简化**：`qa_sessions` / `qa_messages` 不落库，问答历史只在内存维护。

---

## 6. 核心引擎接口

### 6.1 Parser

```python
def parse_document(file_path: str, mode: Literal["standard", "fast"] = "standard") -> DocumentIR: ...
def evaluate_quality(ir: DocumentIR) -> ParseQualityReport: ...
```

**解析路由**：
- DOCX → python-docx 主解，Docling 补结构
- PDF（原生文本）→ PyMuPDF 提文本，Docling 提结构
- PDF（扫描件）→ 返回 `needs_ocr=True`，MVP 阶段提示用户

### 6.2 Diff 引擎（三阶段）

```python
@dataclass
class ComparePolicy:
    similarity_threshold: float = 0.75   # 低于此值判定为新增/删减
    use_llm_classify: bool = True         # 是否调用 LLM 做语义分类
    rule_strengthen: bool = True          # 是否启用规则补强

def compare(baseline: DocumentIR, target: DocumentIR, policy: ComparePolicy) -> DiffResult: ...
```

1. `StructureAligner`：标题相似度匹配，建立章节/段落对齐表
2. `SemanticMatcher`：embedding 余弦相似度，低于 `policy.similarity_threshold`（默认 0.75）→ 新增/删减
3. `DiffClassifier`：对已对齐段落调用 LLM 判定类别 + 规则补强（数字/否定词/责任动词）

### 6.3 Model Provider

```python
class BaseProvider(ABC):
    def chat(self, messages: list[dict], **kwargs) -> str: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def health_check(self) -> bool: ...
```

实现：`OpenAICompatibleProvider`、`AzureOpenAIProvider`、`LocalEmbeddingProvider`

**Embedding 路由逻辑**：
```python
def get_embedder() -> BaseProvider:
    if settings.local_embedding.enabled and Path(settings.local_embedding.model_path).exists():
        return LocalEmbeddingProvider(settings.local_embedding.model_path)
    return get_active_provider()
```

### 6.4 Retrieval

```python
def build_index(version_id: str, chunks: list[Chunk]) -> None: ...
def search(query: str, scope: RetrievalScope, top_k: int = 5) -> list[ChunkHit]: ...
```

`RetrievalScope`：`CURRENT_DOC / BASELINE / TARGET / STANDARD_LIB / ALL`

---

## 7. UI 设计

### 主窗口

左侧导航栏（图标+文字）+ 右侧 QStackedWidget 主内容区。

### 对比页布局

```
┌──────────────┬───────────────────────────┬────────────┐
│ 章节导航树   │  左：基准文档             │ 差异详情   │
│ （带差异数   │  右：目标文档             │ 类别标签   │
│  角标）      │  QWebEngineView           │ 风险等级   │
│              │  行内高亮 HTML            │ AI 解释    │
└──────────────┴───────────────────────────┴────────────┘
  顶部：差异概览栏（总数 / 类型分布 / 筛选按钮）
```

QWebEngine ↔ Python 通信：`QWebChannel`（点击差异项 → Python 更新右侧详情面板）

### 差异高亮颜色规范

| 类型 | 颜色 |
|------|------|
| 新增 | 绿色 `#22c55e` |
| 删减 | 红色 `#ef4444` |
| 微调 | 黄色 `#eab308` |
| 实质修改 | 橙色 `#f97316` |
| 重写 | 紫色 `#a855f7` |
| 格式变化 | 灰色 `#9ca3af` |

---

## 8. 打包方案

```
PyInstaller --onedir
预估安装包：300~400MB（不含本地 embedding 模型）
```

- `build/doc_diff_agent.spec`：PyInstaller 配置，含 WebEngine 资源收集
- `build/installer.iss`：Inno Setup 脚本，生成 `DocDiffAgent-v1.0-setup.exe`
- 支持：安装目录自定义、桌面+开始菜单快捷方式、首次运行初始化数据目录

---

## 9. 开发阶段（MVP 优先）

### 第一阶段 MVP（目标：跑通核心闭环）✅ 已完成（2026-05-01）

- [x] 项目骨架搭建
- [x] 文档解析（PDF + DOCX，原生文本）
- [x] SQLite + FAISS 数据层
- [x] 标准文档入库与版本管理
- [x] 基础语义对比（新增/删减/微调三类）
- [x] 对比结果 WebEngine 可视化
- [x] 单文档问答（RAG）
- [x] Provider Adapter（OpenAI 兼容 + 本地 embedding）
- [x] Windows 本地运行

**已知遗留问题（待二期处理）**：
- I5：LibraryPage 缺少"新增版本"入口，`ingest_new_version()` 已实现但 UI 未接入

### 第二阶段

- [x] LangGraph 编排层接入（2026-05-03 完成：ingest/compare/QA 三图迁移，UI 全局 Theme，I5 新增版本入口）
- [ ] OCR 增强（Tesseract）← 跳过，按需增加
- [x] 实质修改/重写分类（相似度感知，规则+LLM，2026-05-03 完成）
- [x] 对比/标准库/混合问答模式（2026-05-03 完成）
- [x] 差异报告导出（Word/HTML，2026-05-03 完成）

### 第三阶段

- [x] PyInstaller 打包配置（build/doc_diff_agent.spec）+ Inno Setup 安装包脚本（build/installer.iss）（2026-05-07 完成）
- [x] 数据备份恢复（app/services/backup_service.py，设置页集成，2026-05-07 完成）
- [x] 应用内更新检查（app/services/update_checker.py，设置页集成，2026-05-07 完成）

---

## 10. 安全设计

- API Key 用 `cryptography.fernet` + 机器唯一标识派生密钥加密，存 `config.json`
- 日志不打印完整 Key（只打印前 8 位）
- 文档文件按 SHA256 hash 命名，防覆盖
- 导出报告支持脱敏选项（二期）

---

## 11. 非目标（本期不做）

- 多人协作编辑
- 云端集中式文档管理
- Linux / macOS 支持
- LangGraph 编排（MVP 阶段）
- OCR 扫描件识别（MVP 阶段）
- 问答历史持久化（MVP 阶段）
