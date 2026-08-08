# 非 PDF 通用解析器迁移至 AnyDoc 设计

日期：2026-08-08

## 背景

当前解析路由使用 `pymupdf4llm` 处理 PDF，使用 MarkItDown 处理其余支持格式。MarkItDown 的 Office 与表格输出包含较多解析噪声，项目随后通过 `clean_md_table_cells()` 把表格单元格中的 `NaN`、`nan`、`None`、`NA`、`N/A` 清空，并删除 `=DISPIMG(...)` 片段。

AnyDoc 为 Word、PowerPoint、Excel、OpenDocument、RTF、EPUB、CSV 和 PDF 提供本地 Python 绑定，并把各格式统一转换为 GitHub-Flavored Markdown。它的 Office 格式覆盖面和输出整洁度更适合作为新的通用文档解析器。本次只替换非 PDF 路径；AnyDoc 不支持但项目已经支持的格式继续由 MarkItDown 解析。

## 已确认决策

- `.pdf` 继续使用 `pymupdf4llm_adapter`，不切换至 AnyDoc。
- AnyDoc 明确支持的非 PDF 格式使用新的 `anydoc_adapter`。
- AnyDoc 不支持但项目已经支持的 `.html`、`.htm`、`.json`、`.xml`、`.txt`、`.md`、`.markdown` 继续使用 MarkItDown。
- 不在 AnyDoc 失败时隐式回退到 MarkItDown；解析器由受支持扩展名显式决定。
- 取消 `nan_patterns` 处理，表格中的 `NaN`、`nan`、`None`、`NA`、`N/A` 作为真实文档内容原样保留。
- 保留 `=DISPIMG(...)` 噪声清理，因为 AnyDoc 也可能输出该片段。清理同时适用于 AnyDoc 与 MarkItDown 输出。
- Markdown 到 `DocumentIR` 的转换逻辑由两个适配器共享，不复制实现，也不让 AnyDoc 适配器依赖 MarkItDown 适配器。
- 已导入文档和已经持久化的原始、归一化 JSON 不迁移、不重算；新行为只影响后续导入和新增版本。
- 比较阶段保持只读，只消费导入阶段持久化的最终 JSON。

## 目标

1. 用 AnyDoc 提升 Office、OpenDocument、RTF、EPUB 和 CSV 的解析质量与格式覆盖面。
2. 保持 PDF 解析、导入规范化、比较和持久化边界不变。
3. 保留项目当前支持但 AnyDoc 不支持的格式。
4. 停止对 NaN 类字面值进行有损清空，只删除已经确认的 `DISPIMG` 噪声。
5. 让解析器路由、UI 文件选择器、依赖清单、打包配置、文档和测试使用同一套显式格式集合。

## 非目标

- 不改用 AnyDoc 的 PDF 解析能力。
- 不为图片型或扫描文档新增 OCR 流程。
- 不改变 `DocumentIR`、`ParseQualityReport`、导入 artifact 或数据库 schema。
- 不重新设计 Markdown 的标题、段落、句子或表格切分规则。
- 不迁移或重新解析已经导入的版本。
- 不根据 AnyDoc 未来版本的动态能力自动扩大可导入格式。
- 不顺带优化或删除 MarkItDown 及 `markitdown-ocr` 依赖。

## 方案选择

### 方案 A：显式三级路由和独立适配器

建立 PDF、AnyDoc、MarkItDown 三组扩展名；每个解析器由独立适配器负责；共享 Markdown 到 `DocumentIR` 的转换模块。

这是选定方案。它使能力边界、失败语义、UI 白名单和测试矩阵保持显式一致。

### 方案 B：在现有 MarkItDown 适配器内切换解析引擎

改动量较小，但单个适配器会同时承担两种解析引擎，模块命名、异常边界和测试职责含混。

### 方案 C：运行时使用 `anydoc.format_from_path()` 动态选择

可以随 AnyDoc 升级自动获得新格式，但后端能力可能与 UI、文档和打包验证漂移，且不利于稳定复现解析行为。

## 总体数据流

```text
输入文件
  -> router.parse_document
     -> .pdf                                  -> pymupdf4llm_adapter
     -> AnyDoc 支持的非 PDF 扩展名            -> anydoc_adapter
     -> 其余保留扩展名                        -> markitdown_adapter
  -> Markdown
  -> 删除表格单元格中的 DISPIMG 噪声
  -> 共享 Markdown -> DocumentIR 转换
  -> ParseQualityReport
  -> prepare_import_ir
  -> 原始 JSON、规范化 JSON、trace、chunks / index
```

比较阶段不选择解析器、不重新解析源文件，也不修改导入 artifact。

## 格式路由

### PDF

| 类别 | 扩展名 | 解析器 |
| --- | --- | --- |
| PDF | `.pdf` | `pymupdf4llm_adapter` |

### AnyDoc

| 类别 | 扩展名 |
| --- | --- |
| Word | `.doc`, `.docx`, `.docm` |
| PowerPoint | `.ppt`, `.pps`, `.pot`, `.pptx`, `.pptm`, `.ppsx`, `.ppsm` |
| Excel | `.xls`, `.xlsx`, `.xlsm`, `.xlsb` |
| OpenDocument | `.odt`, `.ods`, `.odp` |
| Rich Text Format | `.rtf` |
| EPUB | `.epub` |
| CSV | `.csv` |

### MarkItDown 兼容路径

| 类别 | 扩展名 |
| --- | --- |
| HTML | `.html`, `.htm` |
| 结构化文本 | `.json`, `.xml` |
| 纯文本 | `.txt` |
| Markdown | `.md`, `.markdown` |

`SUPPORTED_EXTENSIONS` 是以上三组扩展名的并集。路由仍先校验扩展名；不允许任意扩展名仅凭 AnyDoc 内容检测进入导入流程。AnyDoc 在已允许范围内仍可使用自身的内容检测识别误标文件。

## 模块职责

### `app/core/parser/router.py`

- 定义 `PDF_EXTENSIONS`、`ANYDOC_EXTENSIONS`、`MARKITDOWN_EXTENSIONS`。
- 用三组集合的并集生成 `SUPPORTED_EXTENSIONS`。
- 保持 `parse_document(file_path, llm_client=None, llm_model="")` 公共签名不变。
- PDF 分派至 `pymupdf4llm_adapter.extract()`。
- AnyDoc 格式分派至 `anydoc_adapter.extract()`。
- MarkItDown 格式分派至 `markitdown_adapter.extract()`，只在该分支传递 `llm_client` 与 `llm_model`。
- 对生成的 `DocumentIR` 继续执行现有 `evaluate_quality()`。

### `app/core/parser/anydoc_adapter.py`

- 提供 `is_available() -> bool`。
- 计算标题和文件 SHA-256。
- 调用 `anydoc.to_markdown(file_path)` 获取 Markdown。
- 调用共享的 `DISPIMG` 清理和 Markdown 到 `DocumentIR` 转换。
- 不接收或使用 LLM/OCR 参数。
- AnyDoc 未安装时抛出明确的 `RuntimeError`；转换异常不做隐式降级。

### `app/core/parser/markitdown_adapter.py`

- 只承担 MarkItDown 兼容格式。
- 保持现有 `llm_client`、`llm_model` 参数和插件启用语义。
- `.md`、`.markdown` 可以继续直接读取 UTF-8 文本，不必调用 MarkItDown。
- 调用与 AnyDoc 相同的 `DISPIMG` 清理和共享 Markdown 到 `DocumentIR` 转换。
- 不再包含通用 Markdown IR 构建实现。

### 共享 Markdown IR 模块

新模块负责当前 `_parse_markdown()`、`_split_sentences()`、标题识别、表格行识别和 `DocumentIR` 构建。迁移只改变所有权，不改变现有解析语义：

- 无标题内容仍创建“正文”默认章节；
- 只识别一级至三级 Markdown 标题；
- 连续 Markdown 表格行仍作为同一个 paragraph；
- 表格每行仍作为一个 sentence；
- 非 PDF paragraph 的 `page_no` 仍为 `None`；
- 文档标题、哈希、ID、plain text 的生成规则保持不变。

具体文件名在实现时采用能清楚表达职责的 `markdown_ir.py`。

## `DISPIMG` 清理与 NaN 保真

现有 `clean_md_table_cells()` 同时承担两类行为：删除已知噪声和清空疑似缺失值。新设计只保留前者。

清理函数仅处理完整 Markdown 表格数据行中的单元格内容，并删除匹配 `=DISPIMG(...)` 的片段：

- 保持表格分隔行不变；
- 保持单元格原有空格风格；
- 不修改非表格行；
- 不把任何完整单元格值视为缺失值；
- `NaN`、`nan`、`None`、`NA`、`N/A` 和相邻空格原样保留；
- 除 `DISPIMG` 模式外不增加新的内容清理规则。

为避免名称继续暗示广泛清理，函数和模块应改为只表达 `DISPIMG` artifact 清理的名称。两个非 PDF 适配器都在 IR 构建前调用它。

## 依赖与打包

- 在 `pyproject.toml` 和 `requirements.txt` 增加 `firecrawl-anydoc>=0.1.3`。
- 保留 `markitdown[all]`、`markitdown-ocr` 和 `pymupdf4llm`。
- PyInstaller 配置显式包含 AnyDoc Python 包及 Windows 原生 `_anydoc.pyd` 扩展，避免动态导入漏收集。
- 不使用对整个 site-packages 的宽泛递归收集；只声明 AnyDoc 运行所需模块和原生二进制。
- 构建规格测试应展开并验证原生扩展被包含，而不只检查源码中出现包名。

## UI 与文档

文档库普通导入和新增版本使用相同的扩展名集合。文件选择器增加：

- Word 旧格式与宏格式；
- PowerPoint 旧格式、模板、放映与宏格式；
- Excel 宏和二进制格式；
- OpenDocument；
- RTF。

已有 PDF、HTML、结构化文本、纯文本、Markdown、EPUB、CSV 入口继续保留。README 与 TECHNICAL 文档同步说明三级路由、AnyDoc 本地转换、格式集合和 OCR 边界；不得继续描述“所有非 PDF 都由 MarkItDown 解析”。

## 错误处理

- 扩展名不在并集内：保持 `ValueError("Unsupported format: ...")`。
- AnyDoc 未安装：抛出包含依赖包名的明确运行时错误。
- AnyDoc 加密、损坏、不支持、资源限制、缺少必要部件或文件读取异常：保留原始异常类型和信息，由现有 ingest service / graph 失败路径记录并展示。
- AnyDoc 已选中但转换失败：不回退到 MarkItDown，避免相同扩展名因失败原因不同产生不可预测的解析权威。
- MarkItDown 插件或 OCR 失败：保持现有兼容路径行为。
- 单个文件失败不改变批次内其他文件的解析器选择。

## 测试设计

### 路由

- 三组扩展名互不重叠，其并集等于 `SUPPORTED_EXTENSIONS`。
- 每个 AnyDoc 扩展名都分派至 `anydoc_adapter`。
- `.pdf` 只分派至 `pymupdf4llm_adapter`。
- `.html/.htm/.json/.xml/.txt/.md/.markdown` 只分派至 `markitdown_adapter`。
- AnyDoc 异常不会触发 MarkItDown。
- 不支持扩展名继续失败关闭。

### AnyDoc 适配器

- `is_available()` 返回布尔值。
- `to_markdown()` 的 Markdown 被转换为现有 `DocumentIR` 结构。
- 标题、哈希和 ID 正确生成。
- 非 PDF paragraph 的 `page_no` 为 `None`。
- 缺少依赖时错误信息明确。
- AnyDoc 转换异常原样传播。

### 共享 Markdown 转换

- 迁移现有无标题、单标题、多级标题、普通段落、句子和表格测试。
- MarkItDown 与 AnyDoc 对相同 Markdown 使用同一转换函数并产生等价 IR 结构。
- 共享模块不依赖任一解析引擎。

### 清理

- AnyDoc 表格输出中的 `=DISPIMG(...)` 被删除。
- MarkItDown 表格输出中的 `=DISPIMG(...)` 被删除。
- `NaN`、`nan`、`None`、`NA`、`N/A` 均原样保留。
- 非表格正文中的同名文本不被修改。
- 表格分隔行、空格风格和其他单元格内容不变。

### UI、依赖与打包

- 两个导入文件选择器包含完整新增格式，且共用同一格式定义或经过一致性断言。
- `pyproject.toml` 与 `requirements.txt` 都声明 `firecrawl-anydoc`。
- 构建规格显式收集 `_anydoc.pyd`。
- 干净 PyInstaller 构建产物中包含 AnyDoc 原生扩展。

### 真实样本与回归

- 使用真实 CSV、DOCX、XLSX 小样执行 AnyDoc 转换并进入 `prepare_import_ir()`。
- 使用 HTML、TXT 或 Markdown 小样确认 MarkItDown 兼容路径仍可用。
- 使用 PDF 小样确认原有 `pymupdf4llm` 路径未改变。
- 运行解析、导入、规范化、比较、UI 和构建规格相关测试。
- 运行完整测试套件。
- 执行一次干净 PyInstaller 构建，检查产物并启动应用完成基本导入入口检查。

## 完成标准

1. PDF、AnyDoc 和 MarkItDown 三条路由与已确认扩展名完全一致。
2. AnyDoc 支持的全部非 PDF 格式可由 UI 选择并进入 AnyDoc。
3. AnyDoc 不支持的原有格式继续由 MarkItDown 解析。
4. 两条非 PDF 路径共享相同 Markdown 到 `DocumentIR` 转换逻辑。
5. `DISPIMG` 噪声继续删除，所有 NaN 类字面值保持原样。
6. AnyDoc 失败不会隐式改变解析器；错误原因可由现有导入失败流程观察。
7. 依赖、文档、UI 与 PyInstaller 产物均包含新的解析器能力。
8. 已有导入 artifact 不迁移，compare 仍只读取导入阶段的最终 JSON。
9. 目标测试、相关回归、完整测试和干净打包验证均通过。
