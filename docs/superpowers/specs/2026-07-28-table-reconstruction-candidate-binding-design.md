# 表格重建候选绑定与精简 LLM 协议设计

## 目标

修复同一边界存在多个续行候选时，LLM 在候选 A 的请求中实际选择候选 B、但代码仍把 `action=merge` 归给候选 A 的问题。同时压缩 LLM 输入输出，避免让模型处理不能影响最终决定的数据。

本次只改变候选裁决协议和多候选冲突消解，不改变候选生成、列映射、源 `DocumentIR`、重放 Trace 结构或无数据丢失预检。

## 已否决方案

### 使用源文档行 ID

源数据只有稳定的 paragraph 标识，没有可依赖的业务行 ID。由 `sentence_index` 拼出的值只能表示当前解析结果中的位置，不应成为 LLM 协议依赖。

### 继续让 LLM 为全部上下文分配 roles

这会扩大输出、重复传递上下文身份信息，并允许 LLM 把另一个相邻行标为 `continuation_row`。即使增加校验，协议仍然让模型执行不必要的逐行分类。

### 只按置信度选择候选

这会把角色错位当成合法候选参与比较，不能修复动作归属错误。置信度只能用于比较已经通过候选绑定校验的判断。

## 推荐协议

每次调用只裁决一个固定候选。`previous`、`continuation` 和 `next` 是请求内的语义槽位，不是源文档 ID。模型不能从邻近上下文中改选另一个 continuation。

### 输入

```json
{
  "candidate_id": "stable-candidate-id",
  "candidate": {
    "previous": ["12", "drive", "prefix"],
    "continuation": ["", "", "suffix"],
    "next": ["13", "stop", "complete"]
  },
  "nearby_context": {
    "before": ["..."],
    "after": ["..."]
  },
  "peer_rows": [
    ["12", "drive", "prefix suffix"]
  ]
}
```

约束：

- `previous`、`continuation` 和 `next` 使用已确定的逻辑列投影；LLM 不选择或返回列映射。
- `nearby_context` 排除本次 `previous` 和 `continuation` 行，避免重复。
- `nearby_context` 沿用现有 `TableBoundaryContext` 的选取范围：前页最后最多六项、后页最前最多六项，以保留多行页眉、页脚和连续边界伪行。
- `before` 和 `after` 只保留按原顺序排列的完整文本；删除 `kind`、ID、section、paragraph、sentence index 和页码。上下文文本不截断。
- `peer_rows` 最多两行；没有跨版本参考时省略该字段。
- `next` 不存在时省略，不传 `null`。
- 空的 `nearby_context` 分组和空的可选字段直接省略。

删除以下现有输入：

- `boundary_id`、`side`
- `physical_mapping`、`mapping_candidates`、`logical_column_roles`
- `rule_evidence`、`rule_conflicts`
- `context_items` 中的 `kind`、来源 ID、页码和其他定位元数据；上下文范围、顺序及完整文本继续保留
- 超过两行的 `cross_version_rows`

这些数据要么已由程序确定、LLM 无权改变，要么与候选槽位内容重复。

### 输出

```json
{
  "candidate_id": "stable-candidate-id",
  "continuation_role": "continuation_row",
  "action": "merge",
  "confidence": 0.92,
  "reason": "continuation completes the previous row"
}
```

输出必须严格且只包含五个字段：

- `candidate_id` 必须与请求一致。
- `continuation_role` 必须是 `continuation_row`、`table_header`、`page_header`、`page_footer`、`ordinary_text` 或 `new_table` 之一。
- `action` 只能是 `merge` 或 `keep`。
- `action=merge` 时，`continuation_role` 必须是 `continuation_row`。
- `confidence` 必须是 `0.0..1.0` 的非布尔数字。
- `reason` 必须是长度不超过 200 字符的非空字符串。

不再输出 `roles`、`boundary_id`、`mapping_id`、`row_action` 或 `table_action`。内部 `LLMJudgment` 仍可由程序从 `action` 派生兼容字段，避免扩大 Trace 变更范围。

## System prompt 规范

精简的是每次请求携带的数据，不是 system prompt 的约束强度。System prompt 必须完整说明任务语义和可由程序逐项验证的格式，至少包含以下内容。

### 任务边界

- 当前请求只判断 `candidate.previous` 与 `candidate.continuation` 是否属于同一逻辑表格行。
- `candidate.previous` 和 `candidate.continuation` 是固定槽位；不得从 `nearby_context` 或 `peer_rows` 中改选其他行。
- `candidate.next`、`nearby_context` 和 `peer_rows` 只提供判断背景，不能成为本次 `action` 的目标。
- LLM 不得改写单元格、创建文本、选择列映射或返回重构操作。

### 输出格式

System prompt 必须逐字段声明以下严格 schema：

```text
Return exactly one JSON object with exactly these five members:
candidate_id, continuation_role, action, confidence, reason.

candidate_id:
- JSON string.
- Copy the supplied candidate_id exactly.

continuation_role:
- JSON string.
- One of: continuation_row, table_header, page_header,
  page_footer, ordinary_text, new_table.
- Classifies only candidate.continuation.

action:
- JSON string.
- Exactly merge or keep.
- merge is valid only when continuation_role is continuation_row.

confidence:
- JSON number from 0.0 through 1.0 inclusive.
- Do not return a quoted number, boolean, null, NaN, or infinity.

reason:
- Non-empty JSON string of at most 200 characters.
- Explain the relationship between candidate.previous and
  candidate.continuation only.
```

### JSON 严格性

System prompt 还必须明确：

- 只输出一个 JSON object，前后不得包含解释文字。
- 不得使用 Markdown code fence。
- 不得缺少字段、增加字段或重复字段。
- 所有键名和枚举值区分大小写，必须与 schema 完全一致。
- 不得发明或修改 `candidate_id`。
- `action=merge` 与非 `continuation_row` 的组合属于无效响应，而不是保守判断。
- 如果认为候选可能是续行但证据不足，可以返回 `continuation_role=continuation_row` 与 `action=keep`；此组合是合法的保守判断。

解析器必须执行与 system prompt 相同的全部检查，不能把提示词约束当作可信输入。首次响应无效时，重试 system prompt 应指出响应未通过严格校验，但不得回显或猜测具体错误字段，仍要求模型依据同一输入重新生成完整对象。

System prompt 可包含一个合法输出示例，但不包含多个冗长示例；字段规范和无效格式清单是主要约束来源。

## 响应失败处理

首次响应不满足严格 schema 或候选绑定规则时，沿用一次重试。重试提示只说明当前候选槽位不可替换，并附上同一份精简输入。第二次仍无效时，该候选保留为 `keep_separate`，且不影响其他候选。

## 多候选冲突消解

候选首先独立通过以下准入条件：

1. 响应通过严格解析和候选绑定校验；
2. `action=merge`；
3. `confidence >= 0.75`；
4. 候选没有规则 veto。

若同一个 previous row 只有一个准入候选，继续执行既有结构预检。

若有多个准入候选：

- 按 LLM 置信度降序排列；
- 仅当最高分唯一且比第二名至少高 `0.05` 时选择最高分候选；
- 其他候选降级为 `keep_separate`，追加 `lower_confidence_continuation_choice`；
- 最高分并列或差值小于 `0.05` 时，全部保持 `keep_separate`，沿用 `ambiguous_continuation_choices`。

选出的候选仍必须通过 `build_reconstruction_operations` 的增量预检。预检失败时沿用 `unsafe_fragment_projection`，不得回退选择较低置信度候选，也不得削弱无数据丢失保护。

## Trace 与兼容性

- 保留每个候选原始 LLM 判断，即使后续因置信度冲突或结构预检而降级。
- 不修改已持久化 Trace 的 schema version。
- 新协议不兼容旧 provider 响应；同步迁移仓库内测试 provider 和响应样例。
- 不修改候选 ID 生成、原始 JSON、chunks 或 FAISS 索引。

## 测试范围

### LLM 协议

- 精简输入只包含允许字段；邻近上下文仍覆盖前后页各最多六项，peer rows 最多两行。
- 候选行不会在 `nearby_context` 中重复出现。
- 多行页眉或页脚在去除 `kind` 和定位元数据后仍完整保留顺序和文本。
- `merge + continuation_row` 被接受。
- `merge + 其他 role` 被拒绝并重试一次。
- 错误 candidate ID、额外或重复字段、缺失字段、非法置信度、空 reason 和超长 reason 被拒绝。
- `keep` 允许所有合法的非 continuation role，也允许模型保守地把 continuation row 保持分离。

### 流水线

- 一个合法 merge 候选正常进入结构预检。
- 两个合法候选分数差至少 `0.05` 时只选择最高分。
- 分数并列或差值不足 `0.05` 时全部安全降级。
- 被角色校验拒绝的高置信度响应不参与择优。
- 获胜候选结构预检失败时保持分离，不尝试次优候选。
- provider 异常和单个候选无效不会中断其他候选。

## 验收标准

- 原问题中的“LLM 在候选 A 请求中实际选择候选 B”不能产生候选 A 的有效 merge 判断。
- 存在唯一、明显更高置信度的合法候选时只生成该候选的 `merge_rows`。
- 所有不确定或结构不安全情况继续无数据丢失地保持分离。
- 目标 LLM 与流水线测试通过，全量测试无新增失败。
