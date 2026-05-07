# Doc Diff Agent

Doc Diff Agent 是一个面向 Windows 桌面场景的文档比对与问答工具，聚焦于法规、制度、合同、规范类文档的版本管理、语义差异识别和检索式问答。

项目以 PySide6 桌面应用的形式运行，核心能力覆盖文档导入、版本比对、流式 RAG 问答与差异报告导出。

## 核心能力

- **文档入库与版本管理**：支持导入 PDF、DOCX 文档，自动计算文件哈希避免重复导入，支持同一文档下新增版本。
- **语义级文档比对**：章节对齐 → 段落级语义匹配 → LLM 分类，差异类型涵盖新增、删减、微调、实质修改、重写、格式变化。
- **混合检索问答（RAG）**：BM25 词法检索与 FAISS 向量检索并行，通过 Reciprocal Rank Fusion（RRF）融合排序，支持当前文档、对比任务、标准文档库、全部四种检索范围。
- **流式问答与会话记忆**：基于 LangGraph `astream_events` 实现逐 Token 流式输出，`MemorySaver` 保持单次会话的上下文记忆，「新会话」按钮清空历史。
- **差异报告导出**：支持导出 HTML 与 DOCX 对比报告，包含差异统计、风险等级、相似度和详细内容。
- **模型接入**：OpenAI 兼容接口（聊天 + Embedding），可选本地 sentence-transformers Embedding；Azure Provider 接口已预留。
- **数据备份恢复**：一键备份数据库、向量索引和配置文件为 ZIP；支持从备份还原。
- **应用内更新检查**：从 GitHub Release 获取最新版本号，在设置页提示可用更新。

## 技术栈

| 层次 | 技术 |
|------|------|
| 桌面界面 | PySide6 |
| 文档解析 | Docling、PyMuPDF、python-docx |
| Agent 编排 | LangGraph（ingest / compare / QA 三图） |
| 流式生成 | LangChain `ChatOpenAI`（streaming=True）|
| 向量检索 | FAISS-cpu |
| 词法检索 | rank-bm25（字符级中文分词） |
| 数据持久化 | SQLite |
| 模型适配 | OpenAI Compatible API、sentence-transformers |
| 打包 | PyInstaller（onedir）+ Inno Setup |

## 运行要求

- Windows 10 x64 及以上
- Python 3.11+

建议使用虚拟环境（uv 或 venv）。

## 快速开始

### 1. 安装依赖

```powershell
# 使用 uv（推荐）
uv sync

# 或使用 pip
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. 启动应用

```powershell
python main.py
```

启动后在设置页配置模型提供方、API Key、本地 Embedding 路径和数据目录。

## 配置说明

- 配置文件：`%APPDATA%\DocDiffAgent\config.json`（API Key 经 Fernet 加密存储）
- 默认数据目录：`%LOCALAPPDATA%\DocDiffAgent\data`

数据目录结构：

```
data/
  app.db          SQLite 数据库
  faiss/          每个文档版本的 FAISS 索引
  docs/           导入后的原始文档副本
  parsed/         解析后的 DocumentIR JSON
  exports/        对比结果导出文件
```

## 问答范围说明

| 检索范围 | 说明 |
|----------|------|
| 当前文档 | 仅在选定的单个文档版本中检索 |
| 对比文档 | 在对比任务的基准版与目标版中检索 |
| 标准文档库 | 仅在已入库的标准文档中检索 |
| 全部 | 当前文档 + 标准文档库 |

## 文档处理说明

- **PDF**：standard 模式优先使用 Docling，失败后回退到 PyMuPDF；fast 模式直接使用 PyMuPDF。
- **DOCX**：使用 python-docx 提取文本与结构。
- **扫描件 PDF**：当前版本返回 `needs_ocr` 提示，OCR 接口已预留。

## 测试

```powershell
uv run pytest
# 或
pytest
```

## 项目结构

```
app/
  agent/      LangGraph 工作流（ingest_graph / compare_graph / qa_graph）
  config/     配置读写与 API Key 加解密
  core/
    diff/     结构对齐、语义匹配、差异分类
    model/    BaseProvider、OpenAI 适配、本地 Embedding、LangChain 工厂
    parser/   文档解析路由（Docling / PyMuPDF / python-docx）
    retrieval/ BM25 + FAISS 混合检索
    types.py  核心数据结构
  db/         SQLite 仓储层（documents / chunks / compare_tasks）
  services/   导入、比对、问答、报告、备份、更新检查
  ui/         桌面界面（主窗口 + 5 个页面）
assets/       模板、字体、图标
build/        PyInstaller spec + Inno Setup 脚本
tests/        自动化测试（166 个用例）
```

## 打包

```powershell
# 生成 onedir 包
pyinstaller build/doc_diff_agent.spec

# 生成 Windows 安装程序（需要 Inno Setup 6）
iscc build/installer.iss
```

## 当前边界

- 仅支持 Windows，不支持 Linux / macOS。
- Azure OpenAI Provider 接口已预留，当前版本不建议启用。
- OCR 扫描件识别接口已预留，当前版本不实现。
- 问答会话历史仅在内存中保持，重启后清空。

## 开发建议

- 修改解析链路后，优先运行 `tests/test_parser/` 与 `tests/test_services/`。
- 修改比对逻辑后，优先运行 `tests/test_diff/` 与 `tests/test_agent/`。
- 修改检索逻辑后，优先运行 `tests/test_retrieval/`。
- 接入本地 Embedding 时，确保模型目录可被 sentence-transformers 正确加载。
