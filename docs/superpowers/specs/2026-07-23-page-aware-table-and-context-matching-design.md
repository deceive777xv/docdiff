# 分页感知的跨页表格重建与上下文段落匹配设计

## 背景

当前 PDF 解析链路同时使用 PyMuPDF4LLM 和一个可选的 pdfplumber 表格修复路径。PyMuPDF4LLM 先生成整篇 Markdown，pdfplumber 再扫描同一 PDF，并尝试用行文本签名替换或追加表格。该实现存在三个相互关联的问题：

1. PyMuPDF4LLM 默认返回整篇字符串，`DocumentIR` 不保存页码，后续只能猜测表格片段是否位于分页交界。
2. pdfplumber 的结果在进入合并逻辑前丢失页码和坐标；当 PyMuPDF4LLM 已包含部分表格行时，现有 70% 行签名规则可能把完整 pdfplumber 表追加到文档末尾，产生重复和错序。
3. 跨页表格重建主要依赖复杂结构规则。现有 LLM 只处理规则标为中置信度的固定三行候选，高置信度规则会绕过 LLM。列数异常、两行表格式页眉、正文表格与页眉表格宽度不同等情况仍可能造成错误合并或不安全列投影。

用户提供的两份外部 JSON 进一步暴露了短段落匹配问题：

- 两份文档分别出现 80 次和 77 次重复短段落命中；
- 当前算法在整节内进行全量两两评分，再按分数贪心占用目标段落；
- 相同短文本没有邻域或顺序约束，能够稳定复现“第一组内容已经删除，但重复标题仍错误匹配第一组”的问题。

当前 HEAD 已通过候选级投影预检避免部分结构异常导致整次对比失败，但该修复只负责安全降级，不解决语义误判。

## 目标

1. 让 PyMuPDF4LLM 成为 PDF 的唯一解析来源，消除重复解析和全文级表格合并。
2. 保存 PDF 物理页码，使新解析文档能准确定位相邻页交界。
3. 在分页交界处向 LLM 提供前后各最多 6 个逻辑项，让 LLM 主要判断页眉、页脚、续行和新表。
4. 保留确定性结构校验，禁止 LLM 生成文本、猜测列映射或绕过非空单元格保护。
5. 使用邻域和顺序为重复短段落消歧，同时保留表格业务行换序匹配。
6. 保持旧 JSON、历史 trace、数据库 schema、差异结果格式和对比页面布局兼容。

## 非目标

- 不恢复或推断页面上印刷的逻辑页码；只保存 PDF 文件内部从 1 开始的物理页序号。
- 不修改或覆盖已经保存的原始解析 JSON。
- 不引入新的 PDF 表格解析器或 OCR 管线。
- 不让 LLM 生成、改写、概括或补全任何正文或单元格内容。
- 不修改 `DiffItem`、数据库 schema、检索索引结构或现有页面交互。
- 不在本阶段增加设置页开关、人工确认界面或用户可调阈值。

## 方案比较

### 方案 A：单一 PyMuPDF4LLM 解析源（采用）

使用 `page_chunks=True` 获取分页 Markdown 和物理页码，移除全部 pdfplumber 提取与合并逻辑。

优点：

- 不会因两个解析器的部分重叠产生重复表格；
- 页码、文本和表格均来自同一输出，来源一致；
- 只扫描一次 PDF，链路和测试显著简化；
- 与后续分页交界上下文构建自然衔接。

代价是个别 PyMuPDF4LLM 未识别的表格可能退化为普通文本。该取舍已确认：宁可极少数表格退化，也不能产生重复或错序。

### 方案 B：页级条件式 pdfplumber 兜底（不采用）

只对疑似漏表页运行 pdfplumber。该方案仍需要判断何时漏表，并可能在部分表格场景重复解析，语义和测试复杂度较高。

### 方案 C：按页码和坐标区域级双解析仲裁（不采用）

改用 `pdfplumber.Page.find_tables()` 获取 `Table.bbox`，再与 PyMuPDF4LLM 的布局框对齐，每个区域选择一个结果。该方案精度潜力最高，但需要坐标系转换、区域匹配和冲突策略，超出当前问题的必要范围。

## 总体数据流

```text
PDF
  -> PyMuPDF4LLM(page_chunks=True)
  -> 分页 Markdown -> DocumentIR(page_no)
  -> align_sections
  -> 构造分页交界上下文
  -> LLM 分类页眉/页脚/续行/新表
  -> 确定性列投影与安全预检
  -> 生成并重放 ReconstructionTrace v2
  -> 普通段落上下文顺序匹配 / 表格业务行匹配
  -> 差异分类
  -> 持久化差异结果和 trace
```

graph、同步比较入口和页面重放必须使用同一个规范化模块，不能各自复制判断规则。

## 模块与接口

### PDF 解析模块

外部接口保持：

```python
extract(file_path: str) -> DocumentIR
```

调用方不需要知道 PyMuPDF4LLM 的 page chunk 结构。模块内部负责：

1. 调用 `pymupdf4llm.to_markdown(file_path, page_chunks=True)`；
2. 验证返回值是按页排列的字典列表；
3. 按页解析 Markdown，同时保持章节状态跨页连续；
4. 为本页产生的每个 `Paragraph` 写入物理 `page_no`；
5. 生成与现有格式一致的 `plain_text`。

完全移除：

- `_extract_pdf_tables()`；
- `_merge_pdf_tables_into_markdown()`；
- pdfplumber 表格归一化、签名、替换和末尾追加辅助函数；
- 只为上述逻辑存在的测试。

PyMuPDF4LLM 的 `metadata.page_number` 不依赖页面中是否印刷页码。它表示 PDF 内部物理顺序，当前版本返回从 1 开始的页号，适合判断分页相邻关系。

### DocumentIR 页码与编解码

`Paragraph` 增加向后兼容字段：

```python
page_no: int | None = None
```

表格 Markdown 在每页独立解析为 `Paragraph`，其行级 `Sentence` 通过所属段落获得页码，不重复保存 `Sentence.page_no`。

新增统一的 `DocumentIR` 编解码模块，提供小型接口：

```python
document_ir_from_dict(data: Mapping[str, object]) -> DocumentIR
document_ir_to_dict(document: DocumentIR) -> dict[str, object]
load_document_ir(path: str | Path) -> DocumentIR
```

该模块集中处理可选字段和嵌套类型。旧 JSON 缺少 `page_no` 时读取为 `None`。graph、同步比较入口和页面不得再手写不同版本的反序列化逻辑。

### 分页交界上下文模块

对具有真实页码的新文档，只检查物理相邻页 `N` 与 `N+1`。当交界两侧存在表格行或表格片段时，构造一个 `BoundaryContext`：

- 前页末尾最多 6 个逻辑项；
- 后页开头最多 6 个逻辑项；
- 普通段落、表格行、分隔行、表头候选、页眉和页脚均保留；
- 每项包含稳定 `item_id`、side、页码、章节、原始顺序、类型提示、来源引用和原始文本或单元格；
- 附带另一个版本中的有限对应行作为参考，但不要求两个版本在同一业务行分页。

旧 JSON 的所有 `page_no` 均为 `None` 时，以现有疑似表格片段交界作为退化定位方式，仍使用相同的 `BoundaryContext` 和 LLM 输出契约。

规则层只负责发现有限交界和生成稳定来源引用，不再通过多项语义证据自动合并高置信度候选。

## LLM 分页交界裁决

### 调用原则

- 每个疑似交界最多调用一次；
- 明显没有表格的交界不调用；
- 以前由规则自动接受的高置信度候选也必须经过 LLM；
- 请求最多包含 12 个逻辑项；
- 每项文本最多 800 个字符，请求总文本最多 12,000 个字符；
- 前一页末行优先保留单元格尾部，后一页首行优先保留单元格头部，以保存跨页续接线索。

### 严格输出

```json
{
  "boundary_id": "stable-boundary-id",
  "confidence": 0.93,
  "items": [
    {"item_id": "x1", "role": "body_row"},
    {"item_id": "x2", "role": "page_footer"},
    {"item_id": "x3", "role": "page_header"},
    {"item_id": "x4", "role": "continuation_row"}
  ],
  "row_action": {
    "decision": "merge",
    "previous_row_id": "x1",
    "continuation_row_id": "x4"
  },
  "table_action": "merge_fragments",
  "reason": "页眉页脚位于两个正文表格片段之间，续行语义承接上一行"
}
```

允许的角色固定为：

- `body_row`
- `continuation_row`
- `table_header`
- `page_header`
- `page_footer`
- `ordinary_text`
- `new_table`

`row_action.decision` 只能是 `merge` 或 `keep_separate`，`table_action` 只能是 `merge_fragments` 或 `keep_separate`。只有整体 `confidence >= 0.75` 时，分类和操作才可进入确定性预检。

响应必须满足：

- `boundary_id` 与请求一致；
- `item_id` 全部属于当前窗口且不能重复；
- 每个输入项恰好分类一次；
- `previous_row_id` 和 `continuation_row_id` 在 `merge` 时必须存在；
- `reason` 为非空字符串；
- 不含额外字段、替换文本、列映射或窗口外操作。

### 确定性安全预检

LLM 只能做语义判断。实际操作必须满足：

1. 两个来源行位于相邻物理页，或同一旧 JSON 推断交界；
2. previous 在 continuation 之前；
3. 来源没有被其他不兼容操作占用；
4. 列映射保持单调并覆盖所有保留的非空单元格；
5. 两个同时非空且冲突的关键列不得合并；
6. 被标记为页眉或页脚的项目只从规范化副本移除；
7. operations 可以完整构建、重放并保持幂等；
8. 原始 `DocumentIR` 不可变。

任一预检失败只将当前交界降级为 `keep_separate`，并记录稳定冲突码，不得使整个对比任务失败。`key column conflict at logical column 0` 等候选级结构冲突必须在生成最终 operations 之前被捕获。

### 失败策略

以下情况均保留原内容和顺序：

- 没有 Provider；
- Provider 超时或异常；
- JSON 无效；
- ID 越界、重复或缺失；
- 置信度不足；
- 确定性安全预检失败。

单个候选失败不影响其他交界。最终 trace 持久化或整份 trace 重放失败仍属于任务级错误。

## ReconstructionTrace v2

v2 在现有来源引用、判断和 operations 基础上增加：

- `boundary_id`
- 两侧物理页码或 `None`
- 有序上下文来源引用
- LLM 对各 item 的角色分类
- LLM 整体置信度和理由
- 最终行级与片段级操作
- 安全降级冲突码

读取端同时支持 v1 和 v2：

- v1 继续按既有 operations 重放；
- v2 使用已经保存的 operations，不重新调用 LLM；
- 不迁移历史 sidecar；
- 页面加载时仍校验 schema、文档 ID 和文件哈希。

主差异结果与 trace 继续采用临时文件加原子替换方式写入。页面不得根据角色分类临时重新推导操作。

## 上下文感知的普通段落匹配

公开接口保持：

```python
match_paragraphs(...) -> list[ParagraphPair]
```

内部把普通段落和表格行分开处理。

### 唯一锚点

首先匹配双方均只出现一次、归一化文本足够长且综合相似度至少为 0.90 的普通段落。只接受不产生顺序交叉的锚点。锚点把章节切分成更小的待匹配区间。

### 歧义段落

满足任一条件即进入上下文消歧：

- 归一化长度不超过 24 个字符；
- 相同归一化文本在任一侧出现多次；
- 第一、第二候选基础分数差不超过 0.05；
- 文本本身缺少完整语义。

为每个歧义段落构造临时匹配特征：

```text
前两个有效普通段落
+ 当前段落
+ 后两个有效普通段落
+ 章节路径
+ 物理页码（存在时）
```

页眉、页脚、表格分隔行和已过滤的结构噪声不进入普通段落上下文。

综合分数由以下部分组成：

- 当前段落文本相似度：55%
- 邻域上下文相似度：30%
- 锚点区间内相对位置：10%
- 段落结构角色一致性：5%

普通段落在每个锚点区间内使用一对一、顺序单调的动态规划对齐。插入和删除通过 gap 操作表达，不能通过交叉匹配重复短文本来掩盖。

上下文仅用于选对匹配对象，不会把多个原始 `Paragraph` 永久合并，也不会改变逐段落差异输出。

### 表格行

表格行不使用普通段落的严格单调约束：

- 继续按业务内容、关键列和归一化单元格匹配；
- 唯一业务行允许跨位置匹配；
- 行顺序变化继续归类为格式变化；
- 多个近似表格行组成冲突簇时，保留受限 LLM 一对一重排裁决。

这样可以修复普通短段落误配，而不破坏现有表格行换序测试。

### 残余歧义的 LLM 兜底

上下文评分后仍有多个近似候选的普通段落形成有限冲突簇。LLM 只看到：

- 基准上下文块；
- 目标上下文块；
- 分数矩阵；
- 固定的局部索引。

输出只能选择一对一匹配或不匹配，不能改写内容。无效响应退回确定性动态规划结果。

## 错误处理与不变量

- PDF 解析整体失败：导入任务失败并保留原异常，不再静默切换 pdfplumber。
- 单个 LLM 交界或匹配簇失败：保留确定性结果并继续。
- 候选投影失败：局部降级，保留所有非空内容。
- trace 写入、哈希校验或最终重放失败：对比任务失败，避免结果与页面结构不一致。
- 所有解析、重建和匹配函数不得修改调用者传入的原始文档。
- graph、同步入口和页面重放必须共享同一编解码和重建接口。

## 兼容性

- 新 PDF 保存 `Paragraph.page_no`；
- 旧 JSON 缺少该字段时默认 `None`；
- 旧 JSON 继续通过推断交界工作；
- 历史 trace v1 原样可重放；
- 新 trace v2 记录分页上下文；
- `DiffItem`、数据库 schema 和页面布局不变；
- 已保存的原始 JSON、Chunk 和 FAISS 数据不迁移；
- 只有重新导入的 PDF 获得真实物理页码。

## 修改范围

预计修改：

- `app/core/types.py`
- `app/core/parser/pymupdf4llm_adapter.py`
- 新增 `app/core/document_ir_codec.py`，统一编解码 `DocumentIR`
- 新增 `app/core/diff/table_boundary_context.py`，构造分页交界上下文
- 新增 `app/core/diff/contextual_matcher.py`，完成普通段落顺序匹配
- `app/core/diff/__init__.py`
- `app/core/diff/table_reconstruction.py`
- `app/core/diff/table_reconstruction_pipeline.py`
- `app/core/diff/table_reconstruction_llm.py`
- `app/core/diff/reconstruction_trace.py`
- `app/core/diff/semantic_matcher.py`
- `app/agent/compare_graph.py`
- `app/services/compare_service.py`
- `app/ui/pages/compare_page.py`
- 对应 parser、diff、graph、service 和 UI 测试

明确删除：

- pdfplumber 解析和全文合并代码；
- 只验证 pdfplumber 平铺文本修复的测试。

明确不修改：

- 数据库 schema；
- 检索索引结构；
- `DiffItem` 格式；
- 现有页面布局和交互；
- 历史原始 JSON。

## 测试策略

### 解析与编解码

- 动态生成无印刷页码的两页 PDF，确认 `page_no` 为 1、2；
- 章节跨页时保持同一章节状态；
- 每页表格形成独立 `Paragraph`；
- 新 JSON 往返保留 `page_no`；
- 旧 JSON 缺少 `page_no` 时正常加载；
- 解析路径不导入或调用 pdfplumber；
- 部分表格不再因第二解析器被重复追加。

### 分页交界与 LLM

- 两行表格式页眉；
- 页眉表格比正文表格宽；
- 跨页表格和跨页表格行；
- 页脚夹在两个表格片段之间；
- 前后页面实际开始新表；
- 两个版本在不同业务行分页；
- 无 Provider、超时、无效 JSON、低置信度和越界 ID；
- LLM 不能返回新文本或列映射；
- 不安全列投影局部降级；
- `key column conflict` 不终止整个任务；
- v1/v2 trace 均可校验和重放；
- 重放幂等且原始 IR 不变。

### 普通段落与表格行匹配

- 删除第一组内容后，重复短标题匹配到具有相同邻域的第二组；
- 多个“电流？”等相同短文本按上下文区分；
- 普通段落匹配不产生顺序交叉；
- 真正新增或删除的短段落保持 unmatched；
- 段落拆分差异不制造大量新增、删减；
- 表格业务行换序仍匹配同一业务内容并归为格式变化；
- 表格近似行冲突簇的 LLM 失败时安全回退。

### 集成与页面

- graph 与同步比较入口产生一致的规范化结果和 trace；
- 差异匹配文本与页面重放文本一致；
- v2 派生 ID 仍可用于差异聚焦和同步滚动；
- trace 缺失或无效时按既有防崩策略展示原始文档；
- sidecar 写入失败时任务不能标记为完成。

### 外部样本

用户提供的两份 JSON 继续用于本地端到端验收，不作为仓库路径依赖。仓库提交脱敏、最小化的结构样本。由于没有原始 PDF：

- 旧 JSON 验证退化交界定位、LLM 裁决和上下文匹配；
- 动态生成的多页 PDF 验证真实页码与分页解析；
- 不根据样本中的固定页眉文案、业务列名、列号或页码编写特殊分支。

## 完成标准

1. PDF 解析链路中不存在 pdfplumber 调用或表格追加逻辑。
2. 新解析 PDF 的每个段落都具有正确物理页码。
3. 两行宽页眉、跨页表格行和不同列数片段不会造成任务级异常或内容丢失。
4. LLM 是非显然跨页语义决策的主要来源，确定性代码只负责有限候选发现和结构安全。
5. 重复短段落通过邻域和顺序匹配，不再依赖全节贪心抢占。
6. 表格业务行换序能力无回归。
7. 旧 JSON、trace v1、数据库、差异结果和页面布局保持兼容。
8. 定向测试、完整 pytest 回归和两份外部 JSON 本地验收全部通过。
