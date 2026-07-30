# 导入期统一规范化、思考深度与上下文匹配设计

日期：2026-07-30

## 背景

当前规范化架构已经把部分跨页表格处理前移到导入阶段，但 compare 阶段仍通过 `normalize_pair()` 发现或复核导入时延后的候选，并在任务副本上应用表格操作。表格实现仍主要位于 `app/core/diff`，同时混合了三类不同职责：单文档结构分析、跨版本增强证据和 compare 任务级重放。

这种边界导致以下问题：

1. 同一文档可能在导入和每次 compare 时重复执行表格分析或 LLM 判断。
2. 规范化后的 JSON 已经移除了部分页边界噪声，但旧表格逻辑仍重新推断 header/body 区域。下一页第一个保留正文行可能被推断为 header 或 unknown 并被候选生成跳过，随后第二行因含有新关键值触发 `new_key_value` 与 `conflicting_key_cells`，整张跨页表合并失败。
3. 当前逻辑把“两个 fragment 是否属于同一张表”与“下一页首行是否续接上一页末行”绑定在一个决策中。新关键值本应只否决行续接，却会连带阻止 fragment 合并。
4. 页边界噪声在可能删除内容时执行初判和 review。固定双调用对全部导入来说成本过高，又缺少用户可见的质量与速度选择。
5. 所有文档格式都会进入当前结构修复，但产品没有清晰表达不同质量档位的含义。
6. 普通段落合并会用 `Sentence(merged_text)` 把整个输出 paragraph 压成一句，丢失原有句子边界。相同正文因解析断句不同可能在 compare 中出现伪差异。
7. 普通段落候选使用“长度不超过 20 且不以终止标点结尾”作为小标题硬否决。短的真实续段会在调用 LLM 前被错误拦截。

## 已确认决策

- 所有支持的文档格式都可以执行规范化，不再限定 PDF。
- 导入界面提供低、中、高三档“思考深度”。
- 低档完全跳过规范化；中档对每个语义候选执行一次 LLM 初判；高档对所有可能改变内容或归属的初判接受结果执行独立 review。
- 高档只有初判和 review 一致同意且确定性安全校验通过时才修改内容。
- 所有表格合并只在导入阶段完成；compare 阶段不得发现、复核或应用表格合并。
- 旧导入数据不迁移，也不保留 compare 期兼容兜底；用户需要重新导入才能使用新规范化行为。
- compare 的 paragraph 上下文必须包含文档标题和完整 section 标题路径。
- 不在导入阶段重新切分复杂句子；不同 sentence 边界由 compare 的 paragraph 上下文匹配处理。

## 目标

1. 建立单一的导入期规范化所有权，彻底移除 compare 期表格规范化。
2. 用清晰的三档思考深度控制速度、LLM 调用量和变更确认强度。
3. 将通用结构安全能力迁入 `app/core/normalization`，使 `app/core/diff` 只负责比较。
4. 基于结构规范化后的单文档 JSON 生成表格候选，不复用不适合该阶段的跨版本语义规则。
5. 分离 fragment 连续性和 row 连续性，使新业务行不会阻止同表 fragment 合并。
6. 保留普通 paragraph 内部的原始 sentence 结构，只处理被确认的跨段连接边界。
7. 让 compare 在完整 paragraph、标题路径和邻近正文上下文中处理不同断句，而不是把 sentence 当作不可组合的事实单元。
8. 保持原始 IR 只读，并保留无损投影、内容守恒、来源唯一性和幂等重放作为最终安全权威。

## 非目标

- 不自动迁移或重新规范化已经导入的版本。
- 不让 compare 回写任何导入 artifact。
- 不让 LLM 生成、改写或补全文本。
- 不做全文级 Markdown 列表标记清洗。
- 不用固定业务文案、列名、列号或样例文本识别表格。
- 不在导入阶段实现通用自然语言分句器。
- 不改变最终差异分类的业务类型、风险等级或严格 `should_report` 布尔契约。

## 方案选择

### 方案 A：保留 diff 表格模块，仅停止 compare 调用

优点是改动最小。缺点是表格规范化仍由 `app/core/diff` 拥有，双文档规则、deferred 候选和不适合规范化后 JSON 的 veto 继续存在，后续容易再次产生职责回流。

### 方案 B：提取结构安全能力并重建导入期候选流水线

将 Markdown 行解析、列映射、显式操作、无损投影和重放迁入 `app/core/normalization`；删除跨版本候选和 compare 复核；重新定义文档级 fragment、row 和 paragraph 候选。

这是选定方案。它保留已经验证的安全能力，同时移除问题来源，而不承担全量重写的回归风险。

### 方案 C：从头重写规范化

模块边界最干净，但容易丢失极端列宽映射、来源追踪、无损投影和幂等保护，风险与本次目标不匹配。

## 总体数据流

### 导入

```text
文件
  -> parse_document
  -> 原始 DocumentIR
  -> 按思考深度执行或跳过 DocumentNormalizer
  -> 最终 DocumentIR + 文档级 normalization trace
  -> 持久化最终 JSON
  -> 构建 chunks / FAISS index
```

### 比较

```text
读取两个版本已经持久化的最终 JSON
  -> 章节对齐
  -> 带标题路径的 paragraph / sentence-window 匹配
  -> 差异分类
  -> 保存 compare 结果
```

compare 不调用 `normalize_pair()`，不加载 deferred table candidates，不生成表格操作，也不重放任务级 reconstruction trace。对比页面直接渲染版本记录指向的最终 JSON。

## 思考深度

定义稳定枚举 `NormalizationDepth`，持久化值不使用 UI 文案：

| UI | 持久化值 | 规范化 | LLM 行为 |
| --- | --- | --- | --- |
| 低（最快） | `off` | 完全跳过 | 不调用规范化 LLM |
| 中 | `standard` | 执行 | 每个语义候选只初判一次 |
| 高 | `review` | 执行 | 初判建议变更时执行独立 review |

低档仍执行解析、持久化、chunk 和索引构建。其最终 JSON 与解析后的 raw JSON 内容相同，normalization trace 记录 `status: skipped`。

中档在初判合法、置信度达到阈值且确定性校验通过时应用操作。高档只 review 初判建议改变内容或归属的候选；初判为 keep 的候选不产生第二次调用。两轮必须绑定同一候选、固定来源和固定结构方案。

任一轮异常、输出非法、低置信度、候选 ID 不匹配或两轮动作分歧，都使该候选最终 keep。失败不得阻止其他候选继续处理。

## 导入界面与批次语义

文档库顶部在导入操作附近增加“思考深度”选择框，默认低档。选择同时适用于普通导入和“新增版本”。

开始一个导入批次时，UI 将当前深度复制为不可变批次参数，再复制到每个 `_IngestWorker` 和 `IngestState`。导入运行期间修改选择框只影响下一个批次，不改变已经启动的文件。

同步 `ingest_document()`、`ingest_new_version()`、LangGraph ingest 路径和测试辅助入口使用同一个深度枚举与默认值。

## 模块边界

`app/core/normalization` 成为唯一规范化模块：

- `pipeline.py`：编排深度、候选阶段、操作应用和最终校验。
- `candidates.py`：定义页边界噪声、段落合并、fragment 连接、row 续接和章节归属候选。
- `resolver.py`：统一初判、review、严格输出校验和 fail-closed 行为。
- `paragraphs.py`：普通段落发现、边界拼接和增量重新排队。
- `tables.py`：单文档表格分析、列映射及 fragment/row 两级决策。
- `operations.py`：显式操作、无损预检、重放、来源唯一性和内容守恒。
- `trace.py`：文档级 trace 序列化与加载。

具体文件可以在实施计划中按现有代码规模合并，以上名称表达职责边界，不要求为每项创建空壳文件。

现有 `app/core/structure_repair` 的有效能力迁入或收编为 normalization 的私有实现，不再保留一条可以绕过思考深度、resolver 和统一 trace 的公开规范化流水线。

`app/core/diff` 只保留章节对齐、上下文匹配、差异分类和 compare 结果持久化。现有表格代码中的以下纯能力迁入 normalization：

- Markdown 表格行的原样/分析单元格解析；
- 有效列和单调列映射；
- 显式 operation 构建；
- 逐列投影、fragment/row 应用；
- 来源追踪、内容守恒、无损预检和幂等重放。

以下能力删除，不迁入新流水线：

- baseline/target 联合分析；
- cross-version support；
- ordinal peer fragment 回填；
- deferred pair review；
- compare 任务级表格修正副本；
- 将新关键值作为 fragment 合并硬否决的旧决策模型。

## 表格候选模型

当前一个候选同时表达 fragment 和 row 合并。新设计拆为三个固定候选类型。

### BoundaryNoiseCandidate

判断页眉、页脚、重复表头或其他边界噪声是否可以删除。删除只由该候选产生，并受思考深度控制。表格连接操作不得顺带删除未确认内容。

### FragmentLinkCandidate

判断相邻物理页上的两个固定 table fragments 是否属于同一张逻辑表：

- `merge_fragments`
- `keep_separate`

新关键值和关键单元格差异是语义证据，不是该候选的硬否决。只要结构可以无损投影，下一页以新业务行开始仍可合并 fragments。

### RowContinuationCandidate

仅在 fragment link 被接受后，判断固定的前页末行与后页首个保留内容行是否属于同一业务行：

- `merge_rows`
- `keep_rows`

`new_key_value` 和 `conflicting_key_cells` 可以否决或强烈反对 row merge，但不能回退已经确认的 fragment link。row keep 表示将后页首行作为同表的新业务行保留。

### 首行选择

表格候选消费完成边界噪声处理后的当前 DocumentIR，而不是原始 IR 或 compare 双文档副本。

- 后页第一个保留的非空、非 separator 行必须进入候选分析。
- 不能仅因区域推断将其标为 `header` 或 `unknown` 而跳过。
- 角色、关键值、占用模式和类型分布作为 LLM 与规则证据。
- 真正的硬否决仅包括：物理页不相邻、明确跨 section、来源重叠、没有任何无损列映射、非空业务单元格无法唯一投影或操作无法重放。
- 每个接受操作仍需增量通过无损投影预检；失败只降级当前候选。

因此，即使后页第二行具有新关键值，算法也不能静默忽略 `sentence_index: 0` 后直接用第二行代表边界。fragment 与 row 决策分别记录在 trace 中。

## 普通段落候选与增量处理

删除“长度不超过 20 且无终止标点即视为标题”的全局硬否决。只有已有 section 结构、明确编号标题模式或已确认的标题操作可以构成标题 veto。未知的短文本交给受约束 LLM 判断。

候选处理基于当前 section 状态：

1. 为相邻普通 paragraphs 建立固定来源候选。
2. 接受合并后，用生成 paragraph 替换两个来源。
3. 使所有引用已消费来源的旧候选失效。
4. 将生成 paragraph 左右两侧的新边界重新入队。
5. keep 或失败只前进当前边界，不阻止后续候选。

每次成功合并都会减少 paragraph 数量，候选处理必然终止。重叠来源不能被两个已接受操作重复消费。

## 普通段落边界拼接与 sentence 保真

`paragraph.text` 是正文事实来源。导入规范化不尝试根据标点重新切分完整 paragraph，也不修改未参与跨段边界的 sentence。

当两个 paragraphs 被确认属于同一句跨段碎片时：

- 只拼接前段最后一个 sentence 与后段第一个 sentence；
- 前段此前的 sentences 和后段其余 sentences 原样保留；
- 输出 paragraph 的 `text` 对两个来源 text 的固定连接边界执行同一 splice；边界 sentences 同步执行等价 splice，不能通过重新拼接全部 sentences 改变原有换行或其他非边界文本；
- 不把整个输出压成单个 `Sentence(merged_text)`；
- 复杂括号内部的 `？`、`！` 等原始断句不做额外修复。

如果续段第一句在连接位置以单个 Markdown 无序列表标记开头，且 LLM 已确认这是同一句续接，join policy 可以仅移除该连接位置的标记。输出开头原有的列表标记保持不变。该移除写入 operation trace，并进入内容守恒允许减少集合；禁止全文搜索替换列表标记。

join policy 确定性处理连接处空白：中文字符之间不额外插入空格；连续 ASCII 字母或数字在需要时保留一个空格。LLM 只决定是否合并，不返回合并文本。

操作校验必须证明 paragraph text splice 与 sentence 边界 splice 消费相同的两个来源边界，且除显式记录的连接标记和连接空白外，字符序列保持守恒。

## 标题与 paragraph 上下文

每个普通 paragraph 合并候选和 compare 匹配窗口都包含只读上下文：

```json
{
  "document_title": "文档标题",
  "section_path": ["一级标题", "二级标题", "当前三级标题"],
  "current_section_title": "当前标题",
  "before": ["前文摘要"],
  "candidate": ["固定候选正文"],
  "after": ["后文摘要"]
}
```

`section_path` 根据文档中的 section 顺序与 level 用确定性栈构建。标题不并入正文匹配文本，LLM 不能修改标题、选择其他 section 或创建新路径。

标题路径用于约束候选：

- 同一标题路径内优先匹配；
- `1:N` 或 `N:1` 窗口不能跨越不兼容标题路径；
- 相同短句位于不同标题语义下时不能仅凭正文相同跨章节匹配；
- 完整正文确定性相等仍是强证据，标题差异不能把相同正文改写为不同文本，但 section 归属变化可以作为独立结构变化处理。

## compare 的不同断句处理

compare 不要求两侧 `sentences` 数量相同，也不在加载时重写它们。

1. 先在标题路径和相邻 paragraph 上下文中对齐完整 paragraph。
2. 如果两侧完整 `paragraph.text` 去除空白和展示性段首列表标记后相等，视为正文相同，不因 sentence 数量或边界不同报告差异。
3. 全文不完全相等时，在同一 paragraph 来源和兼容标题路径内执行单调的 `1:N` / `N:1` sentence window 匹配。
4. paragraph 内窗口不使用固定“三句”上限，而以总字符数和来源范围限制；窗口不得跨 paragraph 或标题路径。
5. 多个窗口具有相同确定性证据时不强行选择，使用完整标题和邻近正文上下文进行 LLM rerank；LLM 只能选择固定窗口或无匹配。
6. 差异分类接收匹配窗口拼接后的正文，而不是被任一侧原始断句截断的片段。

这项 compare 改进适用于低、中、高三档，因为它不修改导入 artifact。

## Trace 与持久化

每份导入文档保存：

```text
parsed/raw/<doc_id>.json
parsed/<doc_id>.json
parsed/traces/<doc_id>.normalization.json
```

normalization trace 至少记录：

- schema 与 algorithm version；
- `normalization_depth`；
- `status: skipped | unchanged | normalized | fallback`；
- 文档引用和哈希；
- 候选类型、固定来源及标题上下文引用；
- 规则证据和硬否决；
- 初判 judgment；
- 高档 review judgment；
- 最终动作和失败码；
- 显式 operations；
- 校验结果；
- LLM 初判、review、成功、失败和跳过次数。

compare 只持久化 diff 结果及 compare 自身必要的审计信息，不再写入或依赖任务级 table reconstruction sidecar。对比页面直接加载版本的最终 JSON；sidecar 缺失不再是警告或 fallback 条件。

## 错误处理与安全性

- 原始 DocumentIR 和 raw JSON 始终只读。
- provider 不可用、LLM 异常、非法 JSON、字段集合错误、候选绑定错误、低置信度或 review 分歧均 keep 当前候选。
- 单个候选失败不终止同一文档的后续候选。
- operation 应用前执行来源存在性、唯一消费、列映射和无损投影预检。
- operation 应用后执行内容守恒、唯一 ID、来源覆盖和幂等重放校验。
- 单候选结构预检失败只降级自身。
- 全文内容守恒、唯一 ID 或最终重放校验失败时，整份规范化结果回退为解析后的原始 IR，并持久化 fallback trace。

## 删除与迁移清单

实施完成后删除：

- `normalize_pair()` 公共入口；
- compare graph 的 `do_reconstruct_tables` 节点；
- compare service 对 deferred candidates 和 pair normalization 的调用；
- 导入 trace 中的 `deferred_table_candidates`；
- pair-only 候选、cross-version 回填和任务级表格副本；
- compare page 的 reconstruction trace 加载与 replay；
- 相应 legacy/fallback 测试。

旧版本不做 schema 迁移。缺少新 normalization trace 的版本不会在 compare 时惰性修复；用户重新导入后获得新行为。

## 测试设计

### 思考深度与导入入口

- 低档不调用 normalization provider，最终 JSON 与 raw 相同，trace 为 skipped。
- 中档每个候选最多一次有效判断。
- 高档只对初判建议变更的候选 review；两轮一致才改变内容。
- 初判 keep 不触发 review。
- 所有支持格式、普通导入、新增版本、同步 service 和 graph 使用相同深度。
- 批次启动后修改 UI 不改变已启动 worker 的深度。

### 表格

- 后页 `sentence_index: 0` 被推断为 header/unknown 时仍进入固定候选。
- 新关键值只阻止 row merge，不阻止已确认的 fragment merge。
- 同表以完整新业务行开页时只合并 fragments，不合并 rows。
- 同表以残缺续行开页时先合并 rows，再连接 fragments。
- 重复表头删除只来自 BoundaryNoiseCandidate。
- 极端物理列差仍可生成有界安全映射。
- 未映射非空业务单元格导致当前候选降级。
- 原始 IR 不变，操作可重放且幂等。

### 普通段落

- 合并后只改变前段最后一句和后段第一句，其他 sentences 原样保留。
- 括号内多个问号的原始 sentence 列表不被重新切分。
- 续段连接位置的单个 `- ` 可移除，输出开头及其他真实列表标记保留。
- `- 时间内的霍尔变化）` 不因短文本规则在 LLM 前被拦截。
- 连续两次以上合并会重新检查生成 paragraph 的左右邻居。
- keep、LLM 失败和 review 分歧不会阻止后续独立候选。
- 来源 paragraph 不能被重复消费。

### compare 上下文

- 完整 paragraph 正文相同但一侧一句、另一侧多句时不产生差异。
- 复杂括号断句不同但完整正文相同时不产生差异。
- `1:N` 和 `N:1` 窗口保持单调且不跨 paragraph。
- 窗口包含文档标题和完整 section path。
- 相同正文位于不同标题语义时不错误跨章节匹配。
- 标题相同、空白或展示性段首列表标记不同的正文可确定性匹配。
- 不唯一窗口只能由绑定候选的 LLM rerank 选择。

### 全链路

- compare graph 和 service 均不调用 normalization、deferred loader 或 reconstruction replay。
- 对比页面直接展示最终 JSON。
- 导入 graph 与同步 ingest service 产生等价 artifact。
- chunks 和索引基于最终 JSON 构建。
- 原始 JSON 保持不变。
- normalization trace 严格往返序列化。
- 目标测试、相关回归测试和完整测试套件全部通过。

## 完成标准

1. 表格规范化只发生在导入期，compare 无任何表格合并入口或兼容兜底。
2. 所有格式共享低、中、高三档行为，UI、graph、service 和 trace 一致。
3. fragment link、row continuation 和 boundary noise 是独立候选与独立操作。
4. 后页首个保留内容行不会因旧区域角色推断被跳过。
5. 段落合并保留内部 sentence 结构，并可增量处理连续碎片。
6. compare 以完整 paragraph、文档标题、完整 section path 和邻近正文处理不同断句。
7. 原始数据只读，任何内容变更都具有固定来源、严格 LLM 绑定、可审计 trace 和确定性安全校验。
8. 旧数据不迁移；重新导入后可获得新规范化结果。
