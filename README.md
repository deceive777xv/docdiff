# Doc Diff Agent

Doc Diff Agent 是一个面向 Windows 桌面场景的文档比对与问答工具，聚焦于法规、制度、合同、规范类文档的版本管理、语义差异识别和检索式问答。

项目以 PySide6 桌面应用的形式运行，核心能力覆盖文档导入、版本比对、流式 RAG 问答、本地会话记忆与差异报告导出。

## 核心能力

- **文档入库与版本管理**：支持导入 PDF、DOCX、PPTX、XLSX、HTML、CSV、EPUB 等多种格式，自动计算文件哈希避免重复导入，支持同一文档下新增版本，并在文档库展示最新版本与总版本数。
- **语义级文档比对**：章节对齐 → 段落/句子/表格行语义匹配 → LLM 分类，差异类型涵盖新增、删减、微调、实质修改、重写、格式变化。
- **差异定位与风险判断**：对比页双栏保留完整文档，可从章节或差异卡片定位到对应行与变化词；风险等级支持高/中/低/无风险，并结合 LLM 的语义一致性判断。
- **混合检索问答（RAG）**：BM25 词法检索与 FAISS 向量检索并行，通过 Reciprocal Rank Fusion（RRF）融合排序，支持当前文档、对比任务、文档库、全部四种检索范围；引用优先显示页码，无页码时回退到段落位置。
- **流式问答与本地会话记忆**：基于 LangGraph `astream_events` 实现逐 Token 流式输出，使用 SQLite-backed checkpointer 持久化上下文记忆，支持历史会话加载、切换与删除，并在回答生成中避免重复提交。
- **首页任务管理**：最近对比任务可直接打开报告、恢复意外中断的任务、查看差异统计结果，并支持删除任务记录。
- **差异报告导出**：支持导出 HTML 与 DOCX 对比报告，包含差异统计、风险等级、相似度和详细内容。
- **模型接入**：OpenAI 兼容接口（聊天 + Embedding），可选本地 sentence-transformers Embedding；Azure Provider 接口已预留。
- **数据备份恢复**：一键备份数据库、原始文档副本、解析缓存、向量索引和配置文件为 ZIP；支持从备份还原。
- **应用内更新检查**：从 GitHub Release 获取最新版本号，在设置页提示可用更新。

## 技术栈

| 层次 | 技术 |
|------|------|
| 桌面界面 | PySide6 |
| 文档解析 | pymupdf4llm（PDF）+ AnyDoc（Office / OpenDocument / RTF / EPUB / CSV）+ MarkItDown 兼容解析 |
| Agent 编排 | LangGraph（ingest / compare / QA 三图） |
| 流式生成 | LangChain `ChatOpenAI`（streaming=True）|
| 向量检索 | FAISS-cpu |
| 词法检索 | rank-bm25（字符级中文分词） |
| 数据持久化 | SQLite（文档、对比任务、问答会话、LangGraph checkpoints） |
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
  app.db          SQLite 数据库（文档、对比任务、问答会话与本地记忆）
  faiss/          每个文档版本的 FAISS 索引
  docs/           导入后的原始文档副本
  parsed/         解析后的 DocumentIR JSON
  exports/        对比结果导出文件
```

## 问答范围说明

| 检索范围 | 说明 |
|----------|------|
| 当前文档 | 仅在选定的单个文档版本中检索 |
| 对比文档 | 在对比任务的基准版与目标版中检索，并附加该任务的差异结果上下文 |
| 文档库 | 仅在已入库的文档中检索 |
| 全部 | 当前文档 + 文档库 |

检索引用会优先使用“章节 + 页码”；当解析结果没有页码或页码不可用时，自动显示“章节 + 段落序号”，避免出现无效的第 0 页引用。

## 文档处理说明

文档解析采用显式三级路由：PDF 由 [pymupdf4llm](https://github.com/pymupdf/RAG) 处理；Word、PowerPoint、Excel、OpenDocument、RTF、EPUB 和 CSV 由本地 [AnyDoc](https://github.com/firecrawl/anydoc) 转换为干净的 GitHub-Flavored Markdown；AnyDoc 不支持的 HTML、JSON、XML、TXT 和 Markdown 继续由 [MarkItDown](https://github.com/microsoft/markitdown) 兼容解析。各路径随后转换为内部 DocumentIR，供导入规范化、检索、比对和报告生成复用。

支持格式：`.pdf`；`.doc`、`.docx`、`.docm`；`.ppt`、`.pps`、`.pot`、`.pptx`、`.pptm`、`.ppsx`、`.ppsm`；`.xls`、`.xlsx`、`.xlsm`、`.xlsb`；`.odt`、`.ods`、`.odp`；`.rtf`、`.epub`、`.csv`；`.html`、`.htm`、`.json`、`.xml`、`.txt`、`.md`、`.markdown`。

**OCR 支持**：AnyDoc 是纯本地结构解析器，不执行 OCR。MarkItDown 兼容路径在设置 OpenAI 兼容 API 后仍可启用 markitdown-ocr 插件。PDF 当前使用 pymupdf4llm 文本抽取；若解析质量不足，质量报告会提示可能需要 OCR 或人工检查。

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
    parser/   文档解析路由（pymupdf4llm / AnyDoc / MarkItDown + 共享 Markdown IR）
    retrieval/ BM25 + FAISS 混合检索
    types.py  核心数据结构
  db/         SQLite 仓储层（documents / chunks / compare_tasks / qa_sessions / checkpoints）
  services/   导入、比对、问答、报告、备份、更新检查
  ui/         桌面界面（主窗口 + 5 个页面）
assets/       模板、字体、图标
build/        PyInstaller spec + Inno Setup 脚本
tests/        自动化测试（254 个用例）
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
- 问答会话与本地记忆保存在 SQLite 中，当前不支持多设备云端同步。

## 开发建议

- 修改解析链路后，优先运行 `tests/test_parser/` 与 `tests/test_services/`。
- 修改比对逻辑后，优先运行 `tests/test_diff/` 与 `tests/test_agent/`。
- 修改检索逻辑后，优先运行 `tests/test_retrieval/`、`tests/test_agent/test_qa_graph.py` 与 `tests/test_services/test_qa_service.py`。
- 接入本地 Embedding 时，确保模型目录可被 sentence-transformers 正确加载。
