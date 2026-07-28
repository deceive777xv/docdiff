# 统一文档规范化与跨页重建架构设计

日期：2026-07-28

## 背景

当前项目存在两套规范化流程：

1. 导入阶段的 `app/core/structure_repair` 对单份 `DocumentIR` 执行标题、章节、普通段落、页眉页脚和简单表格修复，并持久化规范化 IR 与 structure trace。
2. 比较阶段的 `app/core/diff/table_reconstruction_pipeline.py` 同时分析 baseline 与 target，重新识别页边界和表格结构，判断跨页表格续行，生成任务级 reconstruction trace。

两套流程都理解页边界、都可能调用 LLM，但复用关系很弱。由此产生以下问题：

- 同一文档的页边界和表格结构会在不同 compare task 中重复分析。
- 跨页表格重建按版本对重复执行，无法复用导入时已经具备的单文档证据。
- 导入阶段普通段落 LLM 候选存在固定 24 次上限，后续候选被静默跳过，规范化结果受候选出现顺序影响。
- table 页眉与正文位于同一 paragraph 时，导入流程缺少行级区域分析；宽页眉可能污染正文有效列。
- 极端列差下，当前列映射覆盖率使用两侧有效列数的较大值作为分母，导致安全的局部映射在候选生成前被拒绝，LLM 无法介入。
- 规则、LLM、缓存、重试、trace 和性能统计分别由两套流程实现，容易产生行为分叉。

跨页表格重建所依赖的主要判据，包括物理页邻接、区域划分、有效列、单调列映射、关键列、续行文本、页眉页脚和无损投影，都可以在单文档中获得。另一版本提供的是稀疏画像回填、映射加分、完整行佐证和页眉确认等增强证据，而非领域上的硬依赖。

## 目标

1. 建立一个共享规范化核心，对外保留单文档导入和双文档比较两个入口。
2. 将跨页表格重建主体前移到导入阶段，使每个版本只执行一次主要规范化。
3. compare 阶段只复核导入阶段明确保留的歧义候选，不再完整重跑表格重建。
4. 取消普通段落 LLM 候选数量上限，保证每个合格候选都得到明确结果。
5. 完整保留现有规则证据、硬否决、冲突检测、候选绑定、无损预检和幂等重放能力。
6. 凡是会改变业务文本归属且规则无法证明唯一结论的操作，都必须由 LLM 判断。
7. 支持同一 table fragment 中的多行宽页眉、极端物理列差和多个安全映射候选。
8. 统一 LLM 调度、严格输出校验、缓存、有限并发、重试、调用统计和失败语义。
9. 原始 JSON/IR 始终只读；所有转换只应用于规范化副本，并具有可重放 trace。

## 非目标

- 本次不清洗正文行首的 `- `，不新增“列表还是 PDF 折行噪声”的 LLM 判断。
- 本次不改变最终差异分类的业务类型、风险等级或 `should_report` 严格布尔契约。
- 本次不把多个相似候选放进同一个 LLM prompt 批量自由选择。
- 本次不允许 LLM 生成正文、修改单元格文本、创造行引用或创造列映射。
- 本次不硬编码页眉文案、业务列名、固定列号、页码或外部样例内容。
- 本次不生成独立 implementation plan；实施以本设计和后续任务拆分为依据。

## 总体架构

统一规范化核心提供两个外部接口：

```python
normalize_document(
    raw_document: DocumentIR,
    provider: BaseProvider | None,
    model: str,
) -> DocumentNormalizationResult

normalize_pair(
    baseline: NormalizedDocumentArtifacts,
    target: NormalizedDocumentArtifacts,
    provider: BaseProvider | None,
) -> PairNormalizationResult
```

调用关系如下：

```text
原始 DocumentIR
  -> BoundaryAnalyzer
  -> DocumentBoundaryProfile
  -> DocumentNormalizer
  -> 规范化 IR + 文档候选 + 文档 Trace
  -> 持久化，供检索、QA 和后续比较复用

两个已规范化版本
  -> PairNormalizer
  -> 只读取 deferred_pair_review 候选
  -> 使用另一版本补充证据
  -> 任务级可选修正副本 + Pair Trace
  -> 段落匹配与差异分类
```

`normalize_document` 与 `normalize_pair` 是统一规范化 module 的外部 seam。页边界分析、候选发现、LLM 调度、冲突处理和操作重放是 module 内部 seam，不向导入图或比较图暴露实现细节。

## 共享文档边界画像

`BoundaryAnalyzer` 是纯分析实现，不修改输入，输出可序列化的 `DocumentBoundaryProfile`。画像至少包含：

- 文档哈希、算法版本和画像 schema 版本；
- section、paragraph、sentence/table-row 的稳定源引用；
- paragraph 的物理页码、页内顺序及页首/页尾关系；
- 普通 paragraph、table paragraph 和 table row 的类别；
- table fragment、连续 region 及 `header/body/boundary/unknown` 角色；
- 每行原始单元格、分析单元格、宽度、占用模式和值类型；
- 正文有效列、列画像、关键列画像和局部单调映射候选；
- 同文档重复位置、重复内容、分隔行、页码和页眉页脚证据；
- 规则置信度、冲突、硬否决和仍需语义判断的原因。

画像只保存分析事实和候选，不直接删除或合并内容。table 页眉即使暂时无法确认，也必须保留其 region 和行引用，供导入 LLM 或 compare 增强复核使用。

## 导入阶段规范化

`DocumentNormalizer` 消费原始 IR 与 `DocumentBoundaryProfile`，负责所有只依赖单文档的规范化：

- 标题 Markdown 强调清理与章节层级修复；
- 普通页码、稳定重复页眉页脚、严格图片占位符和确定性结构噪声；
- 普通段落跨页续接；
- 无编号章节归属；
- table 内部 header/body/boundary 区域识别；
- table 页眉、页脚和分隔区域处理；
- 不同物理宽度之间的正文逻辑列映射；
- 跨页表格续行与 fragment 合并；
- 逐列投影、逐列文本拼接、内容守恒和幂等重放。

导入结果包括：

```text
normalized_document
boundary_profile
document_trace
candidate_records
normalization_metrics
```

候选不得因数量、文档长度或候选出现顺序被静默跳过。原有 `_MAX_LLM_PARAGRAPH_CANDIDATES = 24` 行为被取消。

## compare 阶段增强复核

`PairNormalizer` 不重新发现和处理全部表格，只读取两份导入 trace 中状态为 `deferred_pair_review` 的候选。以下情况可以进入增强复核：

- 导入时没有可用 provider；
- LLM 超时、异常或输出未通过严格校验；
- LLM 置信度低于接受阈值；
- 单文档存在多个安全映射或多个续行候选，置信度差不足；
- 单文档证据无法确认某一 region 是页眉、续行还是新表；
- 稀疏片段无法仅凭同文档恢复关键列画像，但仍存在安全的有界映射候选。

另一版本只补充以下证据：

- 对应 ordinal fragment 的完整列画像；
- 对应完整业务行或已合并业务行；
- 对应页眉、页脚和重复边界角色；
- 候选映射的跨版本类型相似度。

增强复核只应用于当前 compare task 的深拷贝，不回写导入规范化 IR。导入阶段已经确定并通过无损校验的操作不重新调用 LLM，也不由 compare 推翻。

## 规则与 LLM 决策契约

规则性判据必须完整保留，并承担候选边界和最终安全权威。决策矩阵如下：

| 情况 | 最终处理 |
| --- | --- |
| 明确页码、严格图片占位符、完全重复且位置稳定的页眉页脚、table 分隔行 | 规则直接处理 |
| 新关键值、关键列冲突、跨真实正文行、新章节或新表、无法无损投影、没有任何安全映射 | 规则硬否决，保留原文，不调用 LLM |
| 普通段落是否是上一段续接 | 必须调用 LLM |
| table 行是否是上一行续行 | 必须调用 LLM |
| header/body/continuation/new-table 角色无法由确定性规则唯一证明 | 必须调用 LLM |
| 存在一个或多个安全映射，但业务语义不能唯一确定 | 每个固定候选分别调用 LLM |
| LLM 同意 merge | 仍必须通过规则无损预检和增量重放验证 |
| LLM 失败、低置信度或输出无效 | 导入时保留原文并标记 `deferred_pair_review`；compare 仍失败则最终保留 |

凡是会改变业务文本归属的普通段落合并和表格续行合并，即使规则证据达到当前 `high`，仍然调用 LLM。规则高置信度只影响候选排序和上下文说明，不替代语义判断。

当前 `low` 规则证据但没有硬否决的候选不得直接 `keep_separate`。只要候选具有明确的源槽位和至少一个安全结构方案，就必须进入 LLM；如果连安全结构方案都不存在，则由规则确定性保留。

## 候选发现、冲突组与完整处理

候选发现必须基于不可变快照，先完成全量发现，再进入判断和应用。统一候选外壳包含：

```text
candidate_id
kind
source_refs
fixed_slots
structural_options
rule_evidence
conflicts
vetoes
conflict_group
context
cache_key
```

规则如下：

- `candidate_id` 由算法版本、候选类型、稳定源引用和固定结构方案生成，不使用可变规范化文本作为唯一身份。
- 普通段落没有行 ID 时只使用 paragraph ID 和固定的 previous/continuation 槽位，不虚构 item ID。
- table row 内部可继续使用 paragraph ID 与 sentence index 形成 `SourceRowRef`，但 LLM 输出只回显 `candidate_id`，不选择任意源行。
- 共享 paragraph 或 source row 的候选进入同一 `conflict_group`。
- 冲突组内按文档顺序处理；不同章节或互不重叠的冲突组可以有限并发。
- 普通段落成功合并后，只重新检查新合并 paragraph 的直接邻居，支持一个原始段落跨三页以上，同时保证每次成功合并都会减少 paragraph 数量，流程自然终止。
- 每个候选必须记录 `rule_resolved`、`llm_resolved`、`cache_hit`、`deferred_pair_review`、`failed_keep` 或 `structural_override` 中的一个状态。

## 极端列差与有界映射救援

正常单调映射保持现有算法和阈值。只有正常映射失败时才进入救援路径，不能全局降低 `minimum_mapping_coverage` 或 `mapping_compatibility_threshold`。

救援触发条件：

- 两个 fragment 位于同一 section 的相邻物理页或经过已确认边界噪声后的相邻位置；
- 正常映射因物理列数、有效列数、占用模式或画像差异返回 `None`；
- 前一 fragment 存在可用正文逻辑列；
- 后一 fragment 至少有一条可能续行或下一条完整正文行；
- 可以生成至少一个顺序单调、逻辑列唯一且不丢失候选业务内容的局部映射。

救援不得要求“已经明显识别出前置页眉”，避免页眉识别失败再次阻断 LLM。页眉特征只作为候选证据和 LLM 上下文。

每个页边界最多保留三个安全映射候选。映射覆盖率只针对需要保留的业务行和业务单元格计算，不把已确认 header/boundary region 的装饰列计入分母。候选中的全部非空业务单元格必须映射到唯一逻辑列；未映射的 header/boundary 内容只有在其角色已经确认并产生显式 drop operation 时才允许排除。

每个安全映射形成独立 `candidate_id`，分别调用 LLM。多个候选都返回 merge 时：

1. 按严格校验后的 LLM confidence 降序排列；
2. 第一名必须达到 merge 阈值；
3. 第一名与第二名必须达到既定最小差值；
4. 差值不足则全部保留并记录歧义；
5. 选中候选仍必须通过现有 `build_reconstruction_operations` 无损预检。

## LLM 调度与严格协议

`CandidateResolver` 统一处理导入和 compare 的 LLM 调用，但每种候选保留自己的详细 system prompt、payload 构造器和响应校验器。

统一要求：

- 一个请求只判断一个固定候选，不批量混入相似候选。
- system prompt 必须详细说明任务边界、固定槽位、允许动作、全部字段、字段类型、置信度范围和禁止行为。
- 输入只包含判断所需的原始/分析文本、结构证据、候选映射及邻近上下文，不传递无用的 `kind`、内部定位元数据或可由程序推导的字段。
- 邻近上下文继续保留前页末尾最多六项和后页开头最多六项，以覆盖多行页眉页脚。
- 输出必须是单个 JSON object，不允许 Markdown fence 或额外说明。
- 输出字段集合必须完全匹配该候选协议；`candidate_id` 必须与输入完全一致。
- action 必须来自允许枚举；confidence 必须是 0.0 到 1.0 的数值；reason 必须是非空字符串。
- LLM 不能更换 previous/continuation 槽位，不能选择输入之外的行，不能创建 mapping，不能返回独立且可能矛盾的 row/table 动作。
- 首次输出无效时只重试一次，并在重试 system prompt 中明确列出失败原因和完整格式要求。
- 网络异常、第二次输出无效或低置信度均按 fail-closed 处理。

## 缓存、并发与性能统计

有效 LLM judgment 的缓存 key 包含：

```text
candidate payload hash
candidate kind
normalization algorithm version
prompt schema version
provider/model identity
```

只缓存通过严格格式校验的 judgment，不缓存网络异常、超时或无效响应。文档级缓存随导入 artifact 持久化；版本对增强 judgment 使用 baseline hash、target hash 和上述字段形成 pair cache。

独立 conflict group 使用可配置有限并发，初始默认并发度为 2。单个 conflict group 内保持顺序，避免重叠 paragraph 或 row 的判断基于已失效上下文。provider 限流或线程安全约束由 LLM adapter 隐藏，不暴露给规范化调用方。

每次运行记录：

- 各阶段 `duration_ms`；
- 发现、规则解决、LLM 解决、缓存命中、延后、失败和结构覆盖的候选数量；
- LLM 首次调用数、重试数、成功数和失败数；
- 文档级与 pair 级缓存命中数；
- 映射救援触发数、安全映射数和最终接受数。

## Trace 与持久化

统一 trace 外壳至少包含：

```text
scope: document | pair
schema_version
algorithm_version
source_document_refs
candidate_records
decisions
operations
validation_results
metrics
warnings
```

每个 decision 记录：

```text
candidate_id
kind
decision_source: rule | llm | cache
rule_evidence
conflicts
vetoes
action
confidence
llm_judgment
validation_result
failure_code
```

文档级 artifacts：

```text
parsed/raw/<doc_id>.json
parsed/<doc_id>.json
parsed/profiles/<doc_id>.boundary.json
parsed/traces/<doc_id>.normalization.json
```

pair trace 继续与 compare task 一起持久化。相同版本对可复用 pair cache，但每个 task 仍保存自己实际采用的 trace 快照，保证 UI 重放不依赖可变缓存。

原有 structure trace 与 reconstruction trace loader 在迁移期继续可读。已有导入版本缺少 boundary profile 时，首次使用时执行一次惰性画像回填并持久化；不得在每个 compare task 中重复回填。

## 错误处理与无数据丢失

- 任意分析或 LLM 异常只影响对应候选，不能阻止后续候选处理。
- 硬否决、LLM keep、LLM 失败和结构预检失败都保留原文。
- 导入阶段 unresolved 候选以 `deferred_pair_review` 持久化，不假装已经完成 merge。
- compare 增强复核仍不能确定时最终保留，并记录稳定 failure code。
- operation builder 必须逐候选增量预检；单个不安全候选只降级自身，不能弱化全局无损约束。
- 规范化后进行内容守恒、唯一 ID、source ref 覆盖和幂等重放校验。
- 任何任务级 fallback 都返回输入副本，原始 IR、原始 JSON 和已持久化导入 artifact 不被部分覆盖。

## 能力迁移与兼容性

迁移采用能力清单和特征测试保护，不直接重写后删除旧入口。

1. 冻结现有行为：为全部现有正向 evidence、veto、conflict、映射、boundary drop、candidate binding、confidence arbitration、无损预检和 replay 建立特征测试。
2. 提取共享分析：将现有纯规则函数迁移到统一规范化内部 seam，保持输入输出等价。
3. 持久化画像：导入阶段生成 `DocumentBoundaryProfile`，compare 仍暂时使用旧重建入口进行结果对照。
4. 前移表格重建：导入阶段生成文档级 table operations 和 deferred candidates，比较新旧规范化文本、source coverage 和 trace decision。
5. 切换 compare：compare 只处理 deferred candidates；在 fixture 和真实脱敏样例上验证与旧流程能力相同或更强。
6. 移除完整重跑：只有在结果、trace、幂等和 UI replay 验收通过后，才移除 compare 阶段完整表格重建入口。
7. 启用缓存与有限并发：先记录串行基线，再开启缓存和并发，验证输出顺序与结果确定性不变。

迁移过程中必须保留：

- JSON-only、源数据只读；
- 不同物理宽度和空列位置的表格映射；
- 数字关键列与文本关键列；
- 重复宽页眉与真实重复业务行区分；
- 普通 paragraph 页眉页脚与 table 页眉页脚；
- 多行页眉页脚上下文；
- 高、中、低规则证据及全部 evidence code；
- 所有 hard veto 和 conflict code；
- LLM 严格候选绑定与详细格式校验；
- 多候选置信度选择和最小差值；
- unsafe projection 只降级单个候选；
- 原始 IR 不变、trace 可重放、结果幂等；
- graph 与同步 service 路径结果一致；
- compare page 继续从任务 trace 重放规范化全文。

## 测试设计

### 文本与普通段落

- 候选数量超过 24 时，所有候选都有明确状态且后部候选可以被合并。
- LLM 失败不阻止后续候选。
- 三页及以上段落碎片通过邻居重新入队完整合并。
- 重叠候选不会重复消费同一 paragraph。
- 行首 `- ` 保持现有行为，不新增导入清洗或 LLM 请求。

### table 区域与映射

- 多行宽页眉与续行位于同一 paragraph/fragment。
- 独立且确认的 table 页眉 paragraph 从规范化副本剔除。
- 前页正文三列、后页物理八列或更多时仍能生成安全救援映射。
- 有效列占用模式差异很大时，候选仍能进入 LLM。
- 未映射 header 装饰列不影响业务覆盖率。
- 任意未映射非空业务单元格导致候选降级。
- 没有安全映射时不调用 LLM。
- 最多三个安全映射分别绑定 candidate ID。
- 多个 merge judgment 按 confidence 和最小差值选择；不足时全部保留。

### 单文档与双文档职责

- 没有 target 文档时，导入阶段仍可完成主要跨页表格重建。
- 跨版本证据不会成为单文档候选生成的硬条件。
- 导入成功确定的候选在 compare 时不重复调用 LLM。
- deferred candidate 在 compare 中获得 peer 支持后可安全合并到任务副本。
- compare 增强操作不回写任一导入 IR。
- 相同版本对重复 compare 命中缓存并产生相同 trace 快照。

### LLM、缓存和性能

- 一个请求只包含一个固定候选。
- system prompt 完整列出字段、类型、允许枚举和禁止行为。
- 错误 candidate ID、额外字段、缺失字段、字符串 confidence、无效 action 和空 reason 均被拒绝。
- 无效输出只重试一次。
- 缓存 key 随 prompt、算法或模型版本变化而失效。
- 不缓存失败响应。
- 有限并发下决策和 operation 顺序保持确定。
- metrics 精确反映调用、重试、缓存和失败数量。

### 全链路回归

- 导入 graph 与同步 ingest service 产生相同 artifacts。
- compare graph 与同步 compare service 产生相同规范化文本和 trace。
- 检索和 QA 使用导入规范化 IR。
- diff 匹配与分类使用 pair 增强后的任务副本。
- 现有 reconstruction trace 仍可由 compare page 重放。
- 全量测试、脱敏 JSON 验收和内容守恒检查全部通过。

## 验收标准

1. 导入规范化不再存在固定 24 次候选截断，所有合格候选均有 trace 状态。
2. 跨页表格主体在单文档导入阶段完成，compare 不再完整重跑已确定候选。
3. 另一版本只增强 deferred candidates，不是表格重建的硬前提。
4. 所有现有规则证据、硬否决、冲突和无损预检能力均通过特征测试。
5. 所有改变业务文本归属的非确定性操作均经过严格绑定的 LLM 判断。
6. 同 fragment 多行宽页眉和极端列差能够生成有界安全候选并进入 LLM。
7. LLM 不能选择候选之外的行或映射；任何格式不一致均 fail closed。
8. 原始 IR 不变，导入和 pair 规范化结果均可通过 trace 幂等重放。
9. 相同文档和版本对能够复用画像与有效 judgment，不重复产生相同 LLM 请求。
10. 本次不改变正文行首 `- ` 的存储、清洗或 LLM 行为。
