# Doc Diff Agent 技术说明

本文档面向开发、维护、部署和二次扩展人员，说明 Doc Diff Agent 的整体架构、核心数据结构、业务链路、存储模型、模型接入、测试与打包方式。

更新日期：2026-06-02

## 1. 项目定位

Doc Diff Agent 是一个面向 Windows 桌面的文档版本管理、语义比对和检索问答工具，主要服务于法规、制度、合同、规范类文档。应用以 PySide6 桌面程序运行，核心能力包括：

- 多格式文档导入、解析、版本管理和去重。
- 文档结构化为统一的 `DocumentIR`，并构建检索 chunk 与 FAISS 索引。
- 两个文档版本之间的章节对齐、段落/句子/表格行级语义匹配、LLM 差异分类。
- 差异结果持久化、风险等级判断、HTML/DOCX 报告导出。
- 基于 BM25 + FAISS 的混合检索问答，支持本地 SQLite 会话记忆，并兼容无页码文档的段落引用。
- 首页任务管理、任务恢复、删除任务、QA 会话管理、包含原始文档副本的备份恢复和更新检查。

## 2. 技术栈

| 层次 | 主要技术 | 说明 |
| --- | --- | --- |
| 桌面 UI | PySide6 / Qt WebEngine | 主窗口、页面导航、表格、对比双栏 Web 视图、后台线程 |
| 工作流编排 | LangGraph | 导入、比对、问答三个工作流 |
| 文档解析 | pymupdf4llm + firecrawl-anydoc + MarkItDown | PDF、AnyDoc 支持格式、兼容文本格式显式三级路由 |
| 数据库 | SQLite | 文档、版本、chunk、任务、差异、QA 会话和 checkpoint |
| 向量索引 | FAISS-cpu | 每个文档版本独立索引 |
| 关键词检索 | rank-bm25 | 中文使用字符级 token 化 |
| 模型接入 | OpenAI Compatible API / sentence-transformers | 远程 chat + embedding，本地 embedding 可选 |
| 流式问答 | LangChain ChatOpenAI streaming | QA 页面逐 token 输出 |
| 报告导出 | python-docx / HTML | 导出 DOCX 和独立 HTML |
| 打包 | PyInstaller onedir + Inno Setup | 生成 Windows 离线安装器 |
| 测试 | pytest / pytest-qt | 当前全量测试 632 个通过 |

## 3. 目录结构

```text
app/
  agent/       LangGraph 工作流和 SQLite checkpointer
  config/      配置读写、API Key 加解密
  core/
    diff/      章节对齐、语义匹配、差异分类
    model/     模型 provider、OpenAI 兼容适配、本地 embedding
    parser/    文档解析路由、Markdown 转 IR、chunk 构建
    retrieval/ BM25、FAISS、RRF 混合检索
    types.py   核心 dataclass 和枚举
  db/          SQLite schema、仓储层、FAISS 文件存储
  services/    导入、比对、问答、报告、备份、更新检查服务
  ui/          PySide6 主窗口、页面、主题系统
assets/        图标、字体、HTML 模板
build/         PyInstaller spec 与 Inno Setup 脚本
tests/         单元测试、集成测试、UI 测试
main.py        应用入口
```

## 4. 分层架构

项目按“UI -> 工作流/服务 -> core -> db”的方向组织：

```text
PySide6 UI
  -> AppContext 共享运行时对象
  -> QThread worker 调用 agent graph 或 service
  -> core/parser, core/diff, core/retrieval, core/model
  -> db repository + SQLite + FAISS files
```

各层职责如下：

- `ui/`：负责用户交互、主题、页面状态、后台线程启动、结果展示。
- `agent/`：定义导入、比对、QA 的 LangGraph 工作流；QA 使用 SQLite checkpointer 保存上下文。
- `services/`：提供可直接调用的业务服务，如报告导出、备份恢复、更新检查。
- `core/`：放置领域逻辑，包括解析、比对、检索、模型适配和核心类型。
- `db/`：数据库 schema、CRUD 仓储，以及 FAISS 索引文件的构建/加载。

UI 层通过 `AppContext` 共享 `settings`、SQLite connection、data directory、provider、embedder、LangChain model 和正在运行的对比任务集合。

## 5. 启动流程

入口文件是 `main.py`。

启动流程：

1. 创建 `QApplication`，设置应用名、图标和字体。
2. 读取 `%APPDATA%\DocDiffAgent\config.json`。
3. 初始化主题系统 `ThemeManager`。
4. 确定数据目录并初始化 SQLite 数据库。
5. 根据设置构建：
   - `provider`：OpenAI Compatible chat + embedding provider。
   - `embedder`：本地 sentence-transformers 或远程 embedding provider。
   - `lc_model`：LangChain `ChatOpenAI` 流式模型。
   - `openai_client`：供 MarkItDown OCR 插件使用。
6. 创建 `MainWindow` 和四个主页面：首页、对比、文档库、问答。
7. 连接页面间信号，如首页打开/恢复任务、设置变更刷新 provider、QA 会话变更刷新首页。
8. 刷新首页统计、文档库、对比版本列表和 QA 文档/任务列表。

数据目录的实际来源是 `AppSettings.data_dir`。默认设置中 `data_dir` 来自 `%APPDATA%\DocDiffAgent`；如果配置项为空，`main.py` 会退回到 `%LOCALAPPDATA%\DocDiffAgent\data`。

## 6. 核心数据结构

核心类型定义在 `app/core/types.py`。

### 6.1 DocumentIR

所有解析后的文档都会转成统一结构：

```text
DocumentIR
  doc_id
  title
  file_hash
  sections[]
    section_id
    title
    level
    paragraphs[]
      paragraph_id
      text
      sentences[]
        text
  plain_text
```

设计目的：

- 屏蔽 PDF、Word、Excel、HTML 等格式差异。
- 让导入、检索、比对、QA、报告导出都使用同一份结构化表示。
- 支持大段落、表格和句子的细粒度拆分。

### 6.2 Chunk

`Chunk` 是检索单元：

- `version_id` 指向一个文档版本。
- `chunk_no` 保持原始顺序。
- `section_path` 用于引用章节。
- `text` 是检索文本。
- `page_no` 是可选页码，缺省或不可用时为 `0`。
- `faiss_index_id` 映射到 FAISS 索引中的向量行号。

### 6.3 DiffResult / DiffItem

对比结果由 `DiffResult` 和多个 `DiffItem` 组成。差异类型包括：

- `新增`
- `删减`
- `微调`
- `实质修改`
- `重写`
- `格式变化`

风险等级包括：

- `high`：高风险
- `medium`：中风险
- `low`：低风险
- `none`：无风险

风险判断优先参考 LLM 对语义影响的判断；规则逻辑用于 fallback 或增强关键数值、否定词、义务词等明确触发项。新增、删减这类单侧文本也会根据硬触发词提升风险，避免重要义务或禁止条款被低估。

## 7. 数据库设计

数据库初始化位于 `app/db/schema.py`。SQLite 启用：

- `PRAGMA journal_mode=WAL`
- `PRAGMA foreign_keys=ON`

主要表：

| 表 | 用途 |
| --- | --- |
| `documents` | 文档元信息，按文件 hash 去重 |
| `document_versions` | 同一文档的多个版本 |
| `chunks` | 文档版本对应的检索 chunk |
| `compare_tasks` | 对比任务状态、版本关系、结果路径 |
| `diff_items` | 对比任务下的结构化差异详情 |
| `qa_sessions` | QA 会话元信息 |
| `qa_messages` | QA 用户/助手消息历史 |
| `qa_checkpoints` | LangGraph checkpoint 主表 |
| `qa_checkpoint_writes` | checkpoint pending writes |
| `qa_checkpoint_blobs` | checkpoint channel blob |

`diff_items.risk_level` 的 CHECK 约束支持 `high / medium / low / none`。代码包含旧库迁移逻辑，会重建缺少 `none` 的 legacy `diff_items` 表。

为了降低首页统计、文档库列表、检索、任务恢复和会话列表的查询成本，schema 初始化时会确保以下索引存在：

| 索引 | 主要用途 |
| --- | --- |
| `idx_documents_source_created` | 按来源类型和创建时间列出文档 |
| `idx_document_versions_document_version` | 查询同一文档的最新版本 |
| `idx_chunks_version_chunk_no` | 按原始顺序加载版本 chunk |
| `idx_chunks_version_faiss_id` | 将 FAISS 行号批量映射回 chunk |
| `idx_compare_tasks_created` | 首页最近任务排序 |
| `idx_compare_tasks_status_created` | 按状态恢复或统计任务 |
| `idx_diff_items_task_section` | 加载对比任务差异和章节定位 |
| `idx_qa_sessions_updated` | QA 历史会话按更新时间排序 |
| `idx_qa_messages_session_rowid` | 按写入顺序加载会话消息 |

## 8. 文件存储结构

应用数据目录下保存：

```text
data_dir/
  app.db
  docs/
    <file_hash>.<ext>
  parsed/
    <doc_id>.json
  faiss/
    <version_id>/
      index.faiss
  exports/
    <compare_task_id>.json
```

说明：

- `docs/` 保存导入文件副本。
- `parsed/` 保存 `DocumentIR` JSON，供比对页完整文档双栏和后续任务复用。
- `faiss/` 按版本保存索引，避免不同版本向量混在同一索引中。
- `exports/` 保存对比任务 JSON 结果，HTML/DOCX 报告由用户选择路径导出。

## 9. 文档导入链路

导入工作流定义在 `app/agent/ingest_graph.py`。

```text
file_check
  -> parse_doc
  -> save_document
  -> build_embeddings
```

### 9.1 file_check

职责：

- 检查文件是否存在。
- 计算文件 hash。
- 如果不是“新增版本”模式，则检查 `documents.file_hash` 是否已存在。
- 对重复导入给出明确提示：如需新增版本，应在文档库选择已有文档后点击“新增版本”。

### 9.2 parse_doc

调用 `app/core/parser/router.py`：

- `.pdf`：使用 `pymupdf4llm_adapter.extract()`。
- Word、PowerPoint、Excel、OpenDocument、RTF、EPUB、CSV：使用 `anydoc_adapter.extract()`。
- HTML、JSON、XML、TXT、Markdown：使用 `markitdown_adapter.extract()`。

支持格式：

```text
.pdf,
.doc, .docx, .docm,
.ppt, .pps, .pot, .pptx, .pptm, .ppsx, .ppsm,
.xls, .xlsx, .xlsm, .xlsb,
.odt, .ods, .odp, .rtf, .epub, .csv,
.html, .htm, .json, .xml, .txt, .md, .markdown
```

解析后会进行质量评估：

- 无章节结构或无段落内容：质量分低，提示可能是扫描件或解析失败。
- 平均段落过短：降低质量分。
- 过短段落比例过高：提示检查解析结果。
- `quality_score < 0.4` 时 `needs_ocr=True`。

### 9.3 save_document

职责：

- 将原始文件复制到 `docs/`。
- 将 `DocumentIR` 写入 `parsed/<doc_id>.json`。
- 插入 `documents` 和 `document_versions`，或给已有文档插入新版本。
- 调用 `build_chunks()` 生成 chunk 并插入 `chunks` 表。

### 9.4 build_embeddings

如果当前配置中有 embedder，则调用 `app/core/retrieval/indexer.py` 构建 FAISS 索引，并回写 chunk 的 `faiss_index_id`。

## 10. 文档解析实现

### 10.1 PDF 解析

PDF 当前优先使用 `pymupdf4llm.to_markdown()` 转 Markdown，再走统一 Markdown -> `DocumentIR` 解析逻辑。

优势：

- 对 PDF 文本抽取和版面结构更友好。
- 输出 Markdown，方便后续统一处理表格、标题和段落。

### 10.2 AnyDoc 非 PDF 解析

AnyDoc 支持的 Word、PowerPoint、Excel、OpenDocument、RTF、EPUB 和 CSV 由 `anydoc.to_markdown()` 在本地转换为 GitHub-Flavored Markdown。AnyDoc 的公开文档模型不携带页码，因此这些格式生成的 paragraph 保持 `page_no=None`。

### 10.3 MarkItDown 兼容解析

HTML、JSON、XML、TXT、Markdown 等 AnyDoc 不支持的原有格式继续由 `MarkItDown` 转换。

当设置页配置了 OpenAI 兼容 API 后，`MarkItDown` 初始化时会启用插件链，供 `markitdown-ocr` 辅助处理可识别内容。

### 10.4 Markdown 到 DocumentIR

解析器会：

- 用 `#`、`##`、`###` 识别章节层级。
- 用空行分割普通段落。
- 识别 Markdown 表格行，连续表格行保存为一个 paragraph。
- 对普通文本按中英文句末标点拆分 sentence。
- 对表格将每一行作为 sentence，方便后续表格行级比对。

`markdown_cleanup.py` 只删除 Markdown 表格单元格中的 `=DISPIMG(...)` 解析噪声。`NaN`、`None`、`NA`、`N/A` 等字面值作为原始内容保留。AnyDoc 与 MarkItDown 共用 `markdown_ir.py` 转换逻辑。

## 11. Chunk 与索引

`build_chunks(ir, version_id, max_chars=2000)` 的策略：

- 普通短段落直接作为一个 chunk。
- 超过 `2000` 字符的段落按 `sentences` 拆成句子级 chunk。
- 表格行在解析阶段已保存为 sentence，因此大表格也能拆成更细的检索单元。

索引流程：

1. embedder 对所有 chunk 文本批量生成 embedding。
2. `faiss_store.build_and_save()` 保存 `index.faiss`。
3. `chunk_repo.update_faiss_ids()` 将 FAISS 行号回写到 `chunks.faiss_index_id`。

当前 chunk 的 `page_no` 字段允许为空或为 `0`。QA 引用不会假设所有解析器都能产出页码：有有效页码时显示页码；没有页码时使用 `chunk_no + 1` 显示段落序号。

## 12. 文档比对链路

比对工作流定义在 `app/agent/compare_graph.py`。

```text
parsed DocumentIR pair
  -> align_sections
  -> reconstruct_table_pairs
  -> match_paragraphs (fresh embeddings of reconstructed in-memory text)
  -> classify
  -> persist diff JSON + reconstruction sidecar
```

### 12.1 create_task

如果是新任务：

- 插入 `compare_tasks`，状态从 `pending` 改为 `running`。

如果是恢复任务：

- 校验任务存在。
- 删除旧 `diff_items`。
- 将任务重置为 `running`。

### 12.2 ensure_parsed

根据 `baseline_version_id` 和 `target_version_id` 从 `document_versions.parsed_json_path` 加载两个 `DocumentIR`。

如果解析 JSON 不存在，任务会标记为 `failed`。

### 12.3 do_align

`structure_aligner.align_sections()` 用章节标题相似度对齐两个文档结构。未匹配章节会保留为一侧为空的 section pair，用于生成新增或删减。

### 12.4 do_reconstruct_tables

`table_reconstruction_pipeline.reconstruct_table_pairs()` 在章节对齐之后、语义匹配之前联合分析两个版本的跨页表格片段。它识别逻辑列、重复页眉/页脚边界和跨页续行；高置信度规则直接作出合并或保留决定，中置信度候选在 provider 可用时才交给 LLM 裁决。provider 不可用、裁决失败或返回无效结果时，中置信度候选保守地保持分离，不会猜测或补写原文。

重建只在 `DocumentIR` 的深拷贝上执行，并重新对齐重建后的章节。原始解析 JSON 和传入的 `DocumentIR` 不会被修改。后续 `match_paragraphs()` 针对重建后的内存文本重新计算 embedding，不复用导入阶段保存的 chunk 向量。

每个候选决定和实际变换都会记录在版本化、可重放的 sidecar：`exports/<task_id>.reconstruction.json`。sidecar 包含 schema/algorithm 版本、两侧 `doc_id` 与 `file_hash`、候选 ID、规则证据和冲突、可选 LLM 裁决，以及按顺序执行的列投影、边界删除、行/片段合并操作。差异 JSON 与 sidecar 先分别写入同目录临时文件，再发布正式文件；任一发布失败都会令对比任务标记为 `failed`，不会标记为完成。

对比页先加载原始 `DocumentIR`，校验 sidecar 的版本和两侧文档来源，再重放操作用于完整文档双栏。sidecar 缺失、不可读、JSON 无效、版本/来源不匹配或操作无效时，对比页记录带类别的 warning，并整体回退到两侧原始 IR；不会部分应用重建，也不会在展示时调用模型或检索服务。

表格重建与检索基础设施明确隔离：`app/core/retrieval/searcher.py`、`Chunk.faiss_index_id` 和磁盘中保存的 FAISS 索引仅供 retrieval/QA 使用。comparison reconstruction 不读取这些对象，不加载或重建已保存的 FAISS 索引，也不依赖检索 chunk。

### 12.5 do_semantic_compare

`semantic_matcher.match_paragraphs()` 在已对齐章节内匹配段落。

关键策略：

- 对短段落直接按 paragraph 比对。
- 对超过 `500` 字符或看起来像表格的 paragraph 拆成 sentence/table row 单元。
- 表格行使用第一列作为 `match_key`，相同 key 的行优先匹配。
- embedding cosine similarity 作为主评分。
- 规则惩罚会关注：
  - 数字
  - 否定词
  - 义务词，如“应、须、必须、不得、禁止”

### 12.6 do_classify

`diff_classifier.classify()` 将段落/句子/表格行 pair 转成结构化差异。

处理规则：

- baseline 为空、target 存在：`新增`。
- target 为空、baseline 存在：`删减`。
- 两侧都有文本：调用 LLM 或规则进行分类。
- 对拆分单元，如果忽略空白后相同，直接跳过，避免表格未变行进入结果。

LLM prompt 要求只输出 JSON：

```json
{
  "diff_type": "微调|实质修改|重写|格式变化",
  "risk_level": "high|medium|low|none",
  "explanation": "简短的差异说明"
}
```

如果 LLM 调用失败或未配置 provider，会使用规则 fallback。

### 12.7 persist_result

持久化内容：

- `diff_items` 表保存结构化差异。
- `exports/<task_id>.json` 保存差异 JSON。
- `exports/<task_id>.reconstruction.json` 保存可校验、可重放的表格重建 sidecar。
- `compare_tasks` 状态更新为 `completed`，并保存结果路径。

失败时任务状态更新为 `failed`。

## 13. 对比页面实现要点

`app/ui/pages/compare_page.py` 负责对比任务创建、恢复、结果展示和报告导出。

实现要点：

- 后台使用 `_CompareWorker` + `QThread` 运行比对，避免阻塞 UI。
- `AppContext.active_compare_task_ids` 记录正在运行的任务，首页可显示“进行中”并避免重复恢复。
- 中间双栏使用 WebEngine 展示完整文档。
- 双栏从 `parsed_json_path` 加载完整 `DocumentIR`，而不只展示差异片段。
- 差异卡片与双栏内容通过 `diff_id` 和 WebBridge 联动。
- 点击差异卡片可滚动定位到对应内容位置。
- 点击双栏中的差异片段会同步选择差异卡片和筛选状态。
- Markdown 渲染会处理表格、行内格式和 HTML escape。
- 差异卡片在暗色模式下使用主题色而非半透明高饱和彩色背景。

## 14. 检索问答链路

QA 工作流定义在 `app/agent/qa_graph.py`。

```text
resolve_scope
  -> retrieve_chunks
  -> generate_answer
  -> attach_citations
```

### 14.1 检索范围

支持范围：

| 范围 | 说明 |
| --- | --- |
| 当前文档 | 使用用户选择的一个版本 |
| 对比文档 | 使用对比任务的基准版和目标版，并附加差异结果上下文 |
| 文档库 | 使用所有标准文档的最新版本 |
| 全部 | 当前选择版本 + 标准文档库最新版本 |

### 14.2 混合检索

`app/core/retrieval/searcher.py` 同时执行：

- FAISS 向量检索。
- BM25 关键词检索。

再用 Reciprocal Rank Fusion 合并排序：

```text
score = 1 / (RRF_K + faiss_rank) + 1 / (RRF_K + bm25_rank)
```

默认 `RRF_K = 60`，默认返回 `top_k=5`。

性能相关实现：

- FAISS 命中会通过 `chunk_repo.get_chunks_by_faiss_ids()` 批量映射回 chunk，避免逐条查询。
- 文档库范围通过 `document_repo.list_latest_versions()` 一次取出各文档最新版本，避免重复扫描版本表。
- `app/db/faiss_store.py` 维护最多 2 个 FAISS 索引的 LRU 缓存，并用 `index.faiss` 的 `mtime_ns` 自动失效；重新构建索引时会清理旧缓存。
- `app/core/retrieval/bm25_searcher.py` 维护最多 8 个 BM25 语料缓存；当单个版本 chunk 数超过 2000 时跳过缓存，避免在低内存机器上长期占用大量内存。

这些缓存都保持较小上限，目标是减少重复加载和重复建模，同时兼容性能较差或内存较小的运行环境。

### 14.3 对比任务 QA 上下文

当范围是“对比文档”且存在 `compare_task_id` 时：

- 先对两个文档版本执行 chunk 检索。
- 再从 `diff_items` 重建 `DiffResult`。
- 将差异统计和最多 20 条差异摘要加入 system prompt。
- 差异摘要优先按风险等级排序：高风险、中风险、低风险、无风险。

这样用户问“两者有什么差异”时，即使 chunk 检索没有命中，也能基于已持久化的对比结果回答。

### 14.4 QA 上下文预算

为避免模型上下文过长，QA 生成前会进行字符预算控制：

- 检索 + 对比上下文默认预算：`12000` 字符。
- 历史消息总预算：`4000` 字符。
- 单条历史消息上限：`1200` 字符。
- 历史最多发送最近 `6` 条消息。
- 超长内容会追加“已截断，已优先保留最相关内容”提示。

这些预算可以通过 LangGraph config 的 configurable 字段覆盖：

```text
qa_context_char_budget
qa_history_char_budget
qa_history_message_char_limit
```

当前实现使用字符数近似 token 预算，优点是无需额外 tokenizer 依赖；缺点是不能精确匹配不同模型的 token 规则。

### 14.5 检索引用位置

QA prompt 中的检索片段由 `app/core/retrieval/context_format.py` 统一格式化：

- 有章节时显示 `章节：...`。
- `page_no > 0` 时显示 `第 N 页`。
- 没有页码、页码为 `0` 或解析器未提供页码时，显示 `段落：第 N 段`。

这避免了无页码文档在回答中出现“第 0 页”或空引用。LangGraph QA 工作流和 `qa_service.answer()` 使用同一套格式化函数，保证 UI 流式问答和 service 问答行为一致。

## 15. QA 会话记忆

QA 会话由两层持久化组成：

1. `qa_sessions` / `qa_messages`
   - 供 UI 显示会话列表、加载历史、删除会话。
   - 保存用户消息和助手回复。

2. `SQLiteCheckpointSaver`
   - 供 LangGraph 恢复状态。
   - 写入 `qa_checkpoints`、`qa_checkpoint_writes`、`qa_checkpoint_blobs`。
   - `thread_id` 使用 QA session id。

删除会话时会同时删除：

- `qa_messages`
- `qa_sessions`
- 对应 `thread_id` 的 LangGraph checkpoint 三张表记录

首页“已完成问答”数量直接查询 `qa_sessions`，所以删除 QA 会话后会同步变化。

## 16. 模型接入

### 16.1 OpenAI Compatible Provider

`app/core/model/openai_compatible.py` 封装 chat 和 embedding：

- `chat(messages)` 调用兼容 OpenAI Chat Completions 的接口。
- `embed(texts)` 调用 embedding API。

由 `factory.build_provider()` 根据 `ProviderConfig` 创建。

### 16.2 本地 Embedding

如果设置启用了 `local_embedding`，且模型目录存在，则 `get_embedder()` 使用 `LocalEmbeddingProvider`。否则使用远程 provider 的 embedding 能力。

本地 embedding 的主要价值：

- 文档入库和检索不依赖远程 embedding API。
- 降低检索成本。
- 对离线或内网环境更友好。

### 16.3 LangChain 流式模型

QA 使用 `app/core/model/lc_factory.py` 创建 `ChatOpenAI`：

- `model` 来自 active provider 的 `chat_model`。
- `api_key` 和 `base_url` 来自 provider 配置。
- `streaming=True`。

如果未配置 `lc_model`，QA 会返回“请先在设置页面配置模型”。

### 16.4 API Key 加密

配置文件位于：

```text
%APPDATA%\DocDiffAgent\config.json
```

API Key 使用 `cryptography.fernet.Fernet` 加密。密钥由机器标识派生：

- Windows 优先读取 `HKLM\SOFTWARE\Microsoft\Cryptography\MachineGuid`。
- 失败时使用主机名、机器架构和固定盐组合。

这意味着配置文件不适合直接跨机器迁移。备份恢复到另一台机器后，API Key 可能无法解密，需要重新配置。

## 17. UI 页面说明

### 17.1 MainWindow

`MainWindow` 提供：

- 左侧导航栏。
- 首页、对比、文档库、问答四个页面容器。
- 设置入口。
- 主题切换按钮。
- HarmonyOS Sans 和 Font Awesome 字体加载。

### 17.2 HomePage

首页展示：

- 文档数量。
- 对比任务数量。
- 已完成问答数量。
- 最近对比任务表格。

任务表格支持：

- 打开已完成任务。
- 查看进行中任务。
- 恢复意外中断的任务。
- 删除任务。
- 展示差异统计和风险数量。

### 17.3 LibraryPage

文档库支持：

- 导入新文档。
- 为已有文档新增版本。
- 展示文档名称、类型、最新版本和总版本数。
- 后台线程导入，避免阻塞 UI。

### 17.4 ComparePage

对比页支持：

- 选择基准版本和目标版本。
- 创建或恢复任务。
- 查看差异概览、筛选差异类型和风险等级。
- 双栏完整文档对照。
- 差异卡片与文档定位联动。
- 导出 HTML / DOCX 报告。

### 17.5 QaPage

问答页支持：

- 当前文档、对比文档、文档库、全部四种范围。
- 根据范围显示不同的文档或对比任务选择器。
- 流式输出。
- 引用检索片段。
- 回答生成中禁用发送按钮，避免重复提交同一问题。
- 历史会话列表、加载、删除、新建会话。

### 17.6 SettingsDialog

设置页支持：

- 配置 provider 名称、base URL、API Key、chat model、embedding model。
- 配置本地 embedding 模型路径。
- 配置数据目录。
- 切换主题。
- 备份与恢复。
- 检查更新。

## 18. 线程模型

耗时任务均放入后台线程：

- 文档导入：`LibraryPage._IngestWorker`
- 文档比对：`ComparePage._CompareWorker`
- QA 流式回答：`QaPage._QaWorker`
- 更新检查：`SettingsDialog._UpdateCheckThread`

线程间通过 Qt Signal 通信，避免在后台线程直接操作 UI。

数据库 connection 当前由 `AppContext` 共享，并在 SQLite 初始化时设置 `check_same_thread=False`。这简化了页面和 worker 的协作，但也要求长任务尽量在仓储函数内保持短事务，避免 UI 与 worker 间长时间占用数据库写锁。

## 19. 报告导出

`app/services/report_service.py` 支持：

- `export_docx(result, output_path)`
- `export_html(result, output_path)`

DOCX 报告包含：

- 标题和生成时间。
- 差异总数。
- 差异类型统计表。
- 每条差异的类型、章节、风险、相似度、基准文本、目标文本、说明。

HTML 报告是独立文件，包含内联 CSS 和 escaped 文本，避免用户文档中的 HTML 片段影响报告结构。

## 20. 备份与恢复

`app/services/backup_service.py` 将以下内容打包为 ZIP：

- `config.json`
- `data/app.db`
- `data/docs/`
- `data/faiss/`
- `data/parsed/`

恢复时会覆盖对应文件。恢复逻辑会校验 ZIP 内路径，只允许写入配置文件和数据目录下的预期内容，避免带有 `..` 的路径穿越条目写出目标目录。

注意：

- 如果备份迁移到另一台机器，API Key 可能因机器密钥不同而无法解密。

## 21. 更新检查

`app/services/update_checker.py` 从远端版本文件读取最新版本号，并与本地 `APP_VERSION` 比较。设置页会提示发现的新版本。

该机制只负责提示，不负责自动下载安装包。

## 22. 打包与离线安装器

打包文件：

- `build/doc_diff_agent.spec`
- `build/installer.iss`

标准流程：

```powershell
pyinstaller build/doc_diff_agent.spec
iscc build/installer.iss
```

PyInstaller 产物：

```text
dist/DocDiffAgent/
  DocDiffAgent.exe
  _internal/
```

Inno Setup 输出：

```text
dist/DocDiffAgent-v1.0.1-setup.exe
```

当前安装器脚本语言使用 Inno Setup 内置英文语言文件。应用本体 UI 文案仍由项目代码控制，主要为中文。

## 23. 测试体系

测试目录按模块组织：

| 目录 | 覆盖内容 |
| --- | --- |
| `tests/test_agent/` | LangGraph 工作流、QA 流式、SQLite checkpointer |
| `tests/test_db/` | SQLite schema、仓储层、FAISS 文件存储 |
| `tests/test_diff/` | 章节对齐、语义匹配、差异分类、表格行比对 |
| `tests/test_model/` | OpenAI provider、本地 embedding、LangChain factory |
| `tests/test_parser/` | AnyDoc、MarkItDown、PDF、共享 Markdown IR、清理、OCR 接口与解析路由 |
| `tests/test_retrieval/` | BM25、FAISS indexer、混合检索 |
| `tests/test_services/` | 导入、比对、QA service、报告、备份、更新 |
| `tests/ui/` | 首页、文档库、对比页、QA 页交互与样式 |
| `tests/test_ui/` | 主窗口和主题基础导入测试 |

推荐命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

如果当前 Windows 临时目录权限异常，可显式设置临时目录：

```powershell
$tmp = Join-Path (Get-Location) ".tmp\pytest"
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
$env:TEMP = $tmp
$env:TMP = $tmp
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

最近一次验证结果：

```text
632 passed, 3 warnings
```

## 24. 主要扩展点

### 24.1 新增模型 Provider

实现步骤：

1. 在 `app/core/model/` 新增 provider 类，实现 `BaseProvider`。
2. 在 `factory.build_provider()` 中增加类型分支。
3. 在设置页增加对应配置字段。
4. 增加 provider 单元测试。

### 24.2 新增解析格式

实现步骤：

1. 在 `SUPPORTED_EXTENSIONS` 中加入扩展名。
2. 在 `router.parse_document()` 中增加路由。
3. 输出必须转换为 `DocumentIR`。
4. 增加 parser 测试和导入服务测试。

### 24.3 优化风险判断

可以扩展：

- LLM prompt 中的风险规则。
- `_rule_classify()` 中的数值、日期、主体、义务词识别。
- diff item 的结构化字段，例如影响对象、条款编号、变更摘要。

注意保持 `risk_level` 仍落在 `high / medium / low / none`，否则数据库 CHECK 会失败。

### 24.4 精确 token 预算

当前 QA 使用字符预算控制上下文长度。若要更精确，可引入 tokenizer：

- 按模型类型选择 tokenizer。
- 对 system context、history、question 分配 token budget。
- 保留高风险差异和高分检索命中。
- 在 UI 上提示“部分上下文已裁剪”。

## 25. 当前边界与注意事项

- 当前主要面向 Windows 桌面环境。
- Azure provider 分支已预留，但当前会抛出 `NotImplementedError`。
- SQLite connection 使用 `check_same_thread=False`，应避免长事务。
- QA 本地记忆不支持多设备同步。
- 配置文件中的 API Key 依赖机器派生密钥，不适合直接跨机器复制。
- QA 引用页码依赖解析器输出；无页码或页码不可用时会显示段落位置。
- BM25 与 FAISS 缓存上限偏保守，优先保证低性能环境的内存稳定性。
- 对比风险等级依赖 LLM 输出质量；规则 fallback 只覆盖数字、否定词、义务词等明确模式。
- QA 上下文预算当前按字符裁剪，不是严格 token 裁剪。

## 26. 开发建议

常见改动对应测试：

| 改动类型 | 建议优先测试 |
| --- | --- |
| 文档解析 | `tests/test_parser/`, `tests/test_services/test_ingest_service.py` |
| 文档比对 | `tests/test_diff/`, `tests/test_agent/test_compare_graph.py` |
| 检索问答 | `tests/test_retrieval/`, `tests/test_db/test_faiss_store.py`, `tests/test_agent/test_qa_graph.py`, `tests/test_services/test_qa_service.py`, `tests/test_agent/test_qa_stream.py` |
| QA 会话 | `tests/test_db/test_qa_repo.py`, `tests/test_agent/test_sqlite_checkpointer.py`, `tests/ui/test_qa_page.py` |
| UI 样式和主题 | `tests/ui/`, `tests/test_ui/test_theme.py` |
| 打包 | 手动执行 PyInstaller + Inno Setup，并检查 `dist/` 产物 |

开发时建议遵守：

- 领域逻辑优先放在 `core/` 或 `services/`，UI 只负责展示和用户交互。
- 新增数据库字段时同步修改 schema、仓储层和迁移测试。
- 新增 workflow 节点时确保失败路径会写入明确 `status` 和 `error`。
- 对用户可见行为增加 UI 或 service 测试。
- 对大文件、安装包、缓存产物保持 `.gitignore` 覆盖，避免误提交。
