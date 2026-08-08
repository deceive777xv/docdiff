# Compare 逻辑章节作用域对齐设计

## 背景

当前 compare 先调用 `structure_aligner.align_sections()`，按 section title 的字符 Jaccard 相似度生成一对一 `SectionPair`，再由 `semantic_matcher.match_paragraphs()` 在对齐后的章节作用域内匹配正文。

这个结构存在两类相关缺口：

1. 真实章节同时更换标题和位置时，如果新旧标题相似度低于当前门槛，章节无法对齐。完全相同的正文有时能被后置的跨路径精确调和抵消，但正文只要发生修改，通常仍会形成错误的“新增/删减”。
2. 解析或导入规格化可能把普通正文误判成 section title，导致一侧比另一侧多出 fake section。该标题以下的正文因此被错误的 section scope 隔开，原有 `1:n`、`n:1` 相邻窗口不能跨越该边界。

现有 paragraph/title 角色调和只会抵消标题文本本身，不会改变正文匹配前已经建立的章节作用域，也不会开放 `_exact_adjacent_window_matches()` 当前的同 `section_path` 限制。

## 目标

- 为 compare 构建只读的逻辑章节作用域对齐计划。
- 支持有充分正文证据的真实章节改名和换位。
- 支持由 paragraph/title 角色错位证明的 fake section，并只开放被证明为 fake 的边界。
- 让逻辑组内正文继续复用现有 `1:1`、`1:n`、`n:1` 匹配和分类链路。
- 匹配失败时保留普通新增、删减或修改结果，不因结构推断丢失内容。
- 保持导入规范化、表格重建、报告 schema 和 UI 行为不变。

## 非目标

- 不修改或重新持久化导入后的 `DocumentIR`。
- 不在 compare 阶段合并、重建、投影或删除表格。
- 不新增“章节标题修改”或“章节移动”差异类型。
- 不报告 section title 本身的新增、删减、改名或移动。
- 不用正文匹配是否成功反向证明或撤销 fake section 判定。
- 不使用 LLM、模糊包含或跨单元格拼接判断 fake title。
- 不允许所有 paragraph 脱离章节结构做全文自由匹配。

## 模块 seam

现有 `app/core/diff/structure_aligner.py` 的 `align_sections()` 和 `SectionPair` 继续保留。导入阶段 `app/core/normalization/table_pipeline.py` 依赖这套保守的一对一标题对齐来寻找跨版本表格证据；compare 的推断不能反向扩大导入阶段表格候选。

新增 compare 专用的深 module：

```python
align_compare_scopes(
    baseline: DocumentIR,
    target: DocumentIR,
    embedder: BaseProvider,
) -> SectionAlignmentPlan
```

它隐藏标题候选、正文锚点、换位判定、fake-title 索引、冲突选择和失败关闭规则。compare 的调用者只需要知道输入两份只读 IR 和 embedder，输出可供 paragraph matcher 消费的计划。

`app/core/diff/__init__.py`、`app/services/compare_service.py` 和 `app/agent/compare_graph.py` 改为调用 `align_compare_scopes()`。导入和表格规范化调用者继续调用原 `align_sections()`。

## 数据模型

建议使用不可变的请求内模型：

```python
@dataclass(frozen=True)
class AlignmentContentRef:
    side: Literal["baseline", "target"]
    paragraph_id: str
    sentence_index: int | None = None
    cell_index: int | None = None


@dataclass(frozen=True)
class SectionAlignmentEvidence:
    kind: Literal[
        "title",
        "body_exact",
        "body_semantic",
        "fake_paragraph",
        "fake_table_boundary_cell",
    ]
    score: float
    baseline_section_id: str | None
    target_section_id: str | None
    content_ref: AlignmentContentRef | None = None


@dataclass(frozen=True)
class SectionScopeGroup:
    group_id: str
    baseline_sections: tuple[Section, ...]
    target_sections: tuple[Section, ...]
    evidence: tuple[SectionAlignmentEvidence, ...]
    baseline_crossable_boundaries: frozenset[tuple[str, str]]
    target_crossable_boundaries: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class SectionAlignmentPlan:
    groups: tuple[SectionScopeGroup, ...]
    baseline_section_order: tuple[str, ...]
    target_section_order: tuple[str, ...]
```

`group_id` 只在当前 compare 请求内使用，不持久化，也不冒充跨版本稳定 ID。每个组至少一侧非空；单侧组直接表达未匹配章节，避免同时维护 group 与 unmatched 两套状态。每个源 section 在计划中恰好属于一个逻辑组。

两侧 `section_order` 保留输入 IR 的完整 section ID 顺序，使换位后的逻辑组仍能重建各自原始层级路径和 paragraph 文档顺序，不能从逻辑组排列反推源文档顺序。

`crossable_boundaries` 使用源 section ID 的有序相邻对表达。逻辑组相同不代表组内所有边界都可用于 `1:n` 窗口；只有显式列出的 fake 边界可以跨越。

`content_ref` 在 fake paragraph 证据中精确指向被 section title 覆盖的 paragraph；在 fake table 证据中精确指向提供锚点的 table paragraph、sentence 和 cell。它只用于请求内调和和完整性校验，不持久化。

`score` 对精确标题、精确正文和 fake 证据固定为 `1.0`；弱标题证据保存字符相似度；semantic 正文证据保存本阶段定义的字符加权平均 paragraph 相似度。它用于稳定排序和诊断，不绕过各阶段的准入条件。

## 第一阶段：初始真实章节锚点

compare 对齐不直接复用当前贪心 `SectionPair` 结果作为不可撤销事实，而是生成候选后统一选择：

1. 两侧规格化标题完全相同且各自唯一时，建立强 `title` 候选。
2. 其他标题继续计算现有字符相似度。只有一侧 section 在另一侧恰好有一个达到现有 `0.3` 门槛的候选、反向也恰好只有当前 section 达到门槛且层级兼容时，才建立弱 `title` 候选。
3. 强标题候选立即固定；弱标题候选只记录为延后候选。正文精确和 semantic 身份证据优先消费未对齐 section，最后才用双方仍未消费的弱标题候选补齐，避免偶然字符重叠抢占正文身份明确的章节。
4. 标题候选不要求文档位置相同，因此唯一且可靠的同名章节可以换位。
5. 只靠弱标题候选形成的映射保持锚点区间内的单调顺序，不能用低强度标题相似度制造交叉匹配。

标题规格化沿用现有标题匹配语义，不使用本设计后述的“去标点 fake-title 规格化”。fake-title 规格化只用于证明标题/正文角色错位，不能放宽真实章节标题身份。

## 第二阶段：真实章节改名和换位

对初始阶段仍未对齐的 section，建立只读正文特征。表格行不参与真实章节正文身份评分，避免 compare 的 section 身份推断取得表格重建职责；表格仍可在 fake-title 阶段提供边界单元格证据。

正文特征包含：

- 规格化后非空的普通 paragraph 文本；
- 两侧全文唯一、长度至少为 24 个规格化字符的精确 paragraph 锚点；
- unresolved 普通 paragraph 的批量 embedding；
- section level、父级路径和文档顺序。

候选 section pair 按下列互斥证据层级选择，前一层已经消费的 section 不进入后一层：

1. **精确正文层**：一个 section 中的全部全文唯一长段锚点必须指向同一个对侧 section，对侧锚点也必须反向指回当前 section。双方至少有两个锚点，或者锚点规格化字符总数在双方普通正文中占比都不低于 `0.5`。任一锚点指向冲突 section 或覆盖率不足时，当前 section 在本层不匹配并继续进入 semantic 层。
2. **多段 semantic 层**：至少存在两个顺序一致、互为最佳的 semantic paragraph 候选，双方被候选覆盖的普通正文字符比例都不低于 `0.5`。候选质量取这些 paragraph 相似度按规格化字符数加权的平均值；两侧都必须以该质量互选为最佳 section，且与各自次优质量的差都不低于 `0.15`。
3. **单段 semantic 层**：双方都只有一个普通 paragraph，使用与后续 paragraph matcher 一致的“embedding/词法相似度取高值后扣除数值、否定词和义务词差异惩罚”评分；该 paragraph 必须互为最佳、达到 compare policy 的 paragraph similarity threshold，且与各自次优 section 候选的相似度差都不低于 `0.15`。

多段 semantic 候选沿用 compare policy 的 paragraph similarity threshold。最终真实章节映射还必须：

- 不重复消费任一 section；
- 不与已经固定的强标题锚点冲突。

同 level 的真实章节在强正文证据下允许跨位置形成非单调映射，从而处理章节换位。跨父级移动只有在 level 相同且满足上述强正文证据时才允许；证据不足时保留 unmatched。正文 embedding 只作为候选收窄证据，不产生章节标题差异。

若 embedder 在本阶段不可用或返回非法结果，不使用 semantic 证据扩大作用域；精确标题和精确正文锚点仍可生成计划。后续 paragraph matcher 对 provider 失败继续沿用现有 compare 错误处理。

## 第三阶段：保留现有层级归属

真实章节映射完成后，尚未一对一对齐的后代 section 暂时归入本侧最近的已对齐祖先组，保持当前 `_section_match_scopes()` 的候选范围行为。这个归属只表示 paragraph 候选 scope，不代表该 section 已经找到对侧身份，也不会开放任何 section 边界。

没有已对齐祖先的 section 形成单侧逻辑组。后续 fake 判定仍会检查所有尚未取得对侧身份的 section，包括已经暂归祖先组的后代。若 fake 证据指向其他逻辑组，可将该 section 从临时归属移动到证据所属组；若证据就在当前组，只增加明确的 fake 边界许可。

临时层级归属不能成为 fake section 的新锚点，也不能作为真实章节正文身份的成功证据。

## 第四阶段：fake-title 证据索引

### 专用规格化

fake-title 使用独立的 `_normalize_fake_title_evidence()`，不能全局替换现有 `_normalize_match_text()`：

1. 使用 Unicode NFKC 统一全角和兼容字符；
2. 表格单元格先去除已有 Markdown/HTML 展示标记；
3. 删除所有 Unicode 空白；
4. 删除 Unicode category 以 `P` 开头的标点字符；
5. 转为小写；
6. 保留汉字、字母、数字以及 `%`、`+`、`≥`、货币符号等非标点业务符号。

规格化后为空的文本不成为候选。标题长度、是否编号、是否有句末标点和“标题形态”都不是 fake 判定条件。

### 普通 paragraph 候选

对候选区间内的完整普通 paragraph 建立索引。title 必须与整个 paragraph 规格化后相等；不接受子串包含、多个 paragraph 拼接或 sentence 子单元命中。

### 表格边界单元格候选

Markdown table paragraph 已按 sentence 保存原始行。每个表格只检查：

- 首个非空、非分隔符业务行；
- 末个非空、非分隔符业务行；
- 首尾为同一行时只计一次。

在这些行中逐个检查完整非空单元格。title 必须与一个完整单元格规格化后相等。不检查中间业务行，不拼接多个单元格，也不因单元格命中而消费、修改或抑制整行表格。

## 第五阶段：局部锚点区间和 fake 判定

只检查真实章节对齐后仍未取得对侧身份的 section。它可能位于单侧组，也可能因层级关系暂归一个已对齐祖先组。fake 候选和真实章节候选相互排斥：已经被真实标题或正文证据对齐的 section 不再重新解释为 fake。

对每个 unmatched section：

1. 在同一侧查找文档顺序上最近的前、后已对齐真实章节锚点。
2. 将锚点投影到另一侧。前后锚点顺序一致时，候选区间包含两个相邻锚点所属 scope 及其之间的 section；这样既能覆盖从前一 section 尾部误切出的标题，也能覆盖从后一 section 首部误切出的标题。
3. 只有一个锚点时，只检查该锚点对应的同一父级 scope 及其已归属子 section。
4. 没有可靠锚点、两侧锚点顺序冲突或无法确定父级 scope 时，不判定 fake。
5. 在候选区间内合并统计普通 paragraph 和表格首尾业务行单元格候选。
6. title 规格化文本在合并索引中恰好命中一次时，判定为 fake；零个或多个候选都保持 unmatched。

局部唯一性只在锚点区间内计算，区间外重复不影响当前候选。一个文本同时命中一个普通 paragraph 和一个表格边界单元格时计为两个候选，因此不开放边界。

多个 fake section 可以独立附着到同一个逻辑组，但所有判断都基于真实章节锚点和原始候选区间；fake 判定结果不能级联成为其他 fake section 的新锚点。

## 第六阶段：构建逻辑组和开放边界

fake title 命中普通 paragraph 或表格边界单元格后，将 fake section 附着到匹配对象所属 `SectionScopeGroup` 的对应侧。

如果该组在 fake 所在侧已经包含相邻真实 section，则将 fake 与该真实 section 之间的源文档相邻边界加入 `crossable_boundaries`。如果该侧组中只有 fake section，则无需声明额外边界；其正文直接在该逻辑组内参与候选匹配。

fake 判定的职责到此结束：

- 不检查 fake section 正文是否能够匹配；
- 不用后续匹配成功率确认或撤销 fake；
- 不强制其正文与任何 paragraph 配对；
- 不改变正文 similarity threshold；
- 不抑制后续正常新增或删减。

命中普通 paragraph 的 title 证据通过计划中精确的 `content_ref` 交给 paragraph/title 角色等价规则抵消，不再因锚点区间外的重复文本推翻已经证明的局部唯一性，也不生成标题差异。命中表格单元格时只使用该单元格定位逻辑组，表格行仍完整进入原有表格行比较。

## Paragraph matcher 集成

`match_paragraphs()` 改为消费 `SectionAlignmentPlan`。现有测试可在测试 helper 中从简单一对一 section 构造最小 plan，不增加生产代码的双重输入接口。

主要调整：

1. `_section_match_scopes()` 由计划中的 `groups` 取代，不能再次根据 `SectionPair` 猜测 scope。
2. `_ParagraphUnit` 保留原始 `section_path` 和 `section_level`，并增加来源 section ID。
3. 普通 `1:1` paragraph 候选在同一逻辑组内生成，继续沿用现有 ordinary/table 类型隔离、上下文评分、LLM rerank 和单调消费规则。
4. `_exact_adjacent_window_matches()` 不再要求窗口内所有 unit 的 `section_path` 字符串相同。窗口跨越的每一条源 section 边界都必须出现在该侧组的 `crossable_boundaries`；没有显式许可的真实 section 边界仍不可跨越。
5. 现有 `_reconcile_unique_exact_unmatched()`、paragraph/title 角色调和和短段覆盖使用同一 `SectionAlignmentPlan` 解析 scope，避免不同阶段重新推导不一致的父级关系。
6. 最终 `ParagraphPair.section_path` 继续使用两侧来源中较具体且兼容的原始路径。alignment plan 不改写报告展示路径。
7. 单侧逻辑组中的 paragraph 直接生成现有新增或删减 `ParagraphPair`，不需要另一套 unmatched 输入。

真实章节换位后的 paragraph 匹配在各逻辑组内独立执行，不要求不同逻辑组之间保持全局单调。单个逻辑组内部仍保持现有普通 paragraph 单调规则。

## 失败关闭规则

以下情况均不扩大正文候选范围：

- 真实章节标题或正文候选不是互为最佳；
- 与次优候选差距不足；
- 重复标题或正文锚点无法唯一消歧；
- semantic embedding 为空、非数值、非有限值或维度不一致；
- 真实章节映射重复消费 section；
- 弱证据产生顺序交叉或层级不兼容；
- fake section 没有可靠局部锚点；
- 局部锚点在另一侧顺序冲突；
- fake title 规格化后为空；
- fake title 在局部普通 paragraph 和表格边界单元格合并索引中命中零次或多次；
- 仅能通过子串、sentence、跨 paragraph、跨单元格或表格中间行命中；
- 表格行为空、仅为 Markdown 分隔符或无法稳定拆分单元格；
- 计划完整性校验发现 section 丢失、重复归属或开放了非相邻边界。

失败时 section 保持 unmatched，后续正文按现有新增、删减路径处理。正文在已经开放的 fake 逻辑组内仍可能匹配失败；这是正常内容结果，不是结构错误，也不会触发回滚或特殊抑制。

## 计划完整性校验

`align_compare_scopes()` 返回前执行确定性校验：

1. 两侧每个 section 恰好出现一次；
2. 每个 group 至少一侧包含 section；
3. 每个开放边界对应同一侧源文档中物理相邻的两个 section；
4. 开放边界两端都属于同一逻辑组；
5. 每个开放边界恰有一端是已确认 fake section，另一端是由标题或正文证据对齐的原始真实 section；
6. evidence 引用的 section 和 `content_ref` 候选内容都存在于输入 IR，table sentence/cell 索引没有越界，且同一内容候选不能被多条 fake evidence 复用；
7. 输出 group 和 section 顺序确定，不依赖集合迭代或 provider 返回顺序。

任何校验失败都不得返回部分计划。compare 记录错误并沿用现有任务失败处理，不静默丢弃 section。

## 测试策略

### Compare section alignment

- 标题相同、位置相同维持现有一对一结果。
- 标题相同、章节换位仍能分别对齐且正文无变化时没有差异。
- 标题彻底变化且存在至少两个一致的唯一长段锚点，或唯一锚点在双方正文中覆盖率均达到 `0.5` 时建立真实章节映射。
- 标题彻底变化且正文有局部修改时，使用多个 mutual-best 正文候选识别同一章节，只报告实际修改。
- 单 paragraph 章节在高相似度和明确 margin 下可以改名换位。
- 重复正文、双方次优差距不足、层级冲突和弱证据交叉时保持 unmatched。
- 一侧 section 不得被多个逻辑组消费。

### Fake title 普通正文证据

- 短、长、编号型 title 均可与完整普通 paragraph 匹配。
- 中英文标点、全半角和空白差异可以命中。
- `%`、`+`、`≥` 等业务符号不同不能因去标点而相等。
- title 只被 paragraph 包含、只匹配一个 sentence 或需要拼接 paragraph 时不命中。
- 区间外重复不影响局部唯一；区间内重复必须拒绝。
- 缺少锚点、单侧锚点父 scope 不兼容或双锚点顺序冲突时拒绝。
- baseline fake 和 target fake 两个方向行为对称。

### Fake title 表格证据

- title 与首个业务行完整单元格相等时命中。
- title 与末个业务行完整单元格相等时命中。
- 表格只有一个业务行时不重复计数。
- 空行、分隔行和中间业务行不作为候选。
- 单元格子串、多个单元格拼接和多个边界单元格重复命中时拒绝。
- 一个普通 paragraph 和一个表格单元格同时命中时拒绝。
- 命中只开放 scope，不消费整行，也不隐藏其他单元格差异。

### 跨 fake 边界正文匹配

- fake section 正文与相邻真实 section 正文可以共同形成 `2:1` 和 `1:2` 精确窗口。
- `n:1` 和 `1:n` 两个方向对称。
- 只允许跨已记录的 fake 边界；同一逻辑组中的其他真实边界仍不可跨越。
- fake 判定成功但正文无候选时，保留正常新增或删减。
- fake 判定成功但正文相似度不足时，不强行匹配、不撤销 fake，也不抑制差异。
- paragraph/title 普通证据继续静默抵消；表格证据行继续走表格比较。

### 回归和调用链

- `structure_aligner.align_sections()` 现有测试保持通过。
- import-only `table_pipeline` 仍使用原 `SectionPair`，表格重建结果和 trace 不变化。
- semantic matcher 的普通 paragraph、表格行、短段覆盖和 paragraph/title 调和回归测试保持通过。
- `app/core/diff.compare()`、`compare_service` 和 `compare_graph` 使用 plan 的集成测试通过。
- compare graph/test doubles 更新新的 state 字段默认值。
- 完整测试套件、Python compile 和 `git diff --check` 通过；若存在已知基线失败，必须分别复现并证明本改动未新增失败。

## 验收标准

1. 完全改名和换位的真实章节在强、唯一正文证据下仍能比较正文修改。
2. 标题、正文和表格边界证据均不能重复消费 section 或内容候选。
3. fake title 不受长度、编号或弱标题形态限制。
4. fake 判定只依赖局部唯一的完整 paragraph 或表格首尾完整单元格证据。
5. fake 判定后只开放明确记录的边界；不使用正文匹配结果反证、撤销或扩大 fake 判定。
6. 正文在开放后的逻辑组内匹配失败时仍正常报告新增或删减。
7. 表格证据不消费表格行、不改变单元格、不隐藏表格差异。
8. 原始 `DocumentIR`、导入表格规范化、数据库 schema、报告和 UI 均不变化。
9. 所有歧义、重复、层级冲突、非完整文本命中和计划完整性错误都 fail closed。
