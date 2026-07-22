# 跨页表格安全投影降级修复设计

## 背景与复现结论

跨页表格候选经过规则或 LLM 判定为 `merge` 后，`build_reconstruction_operations` 会为候选涉及的全部保留行生成列投影。当前实现已经能够为“物理槽位发生整体位移、但逻辑列数完整且锚点稳定”的行生成行级投影；当保留行包含无法映射的非空单元格时，它会抛出 `ValueError("fragment projection contains unmapped retained cells")`，防止静默丢列。

问题不在这项防丢校验，而在调用层：单个候选的投影失败会终止整个重建阶段，并进一步使整次文档对比失败。

使用 `E:\Project\test` 中的七份解析 JSON 进行本地验证时，纯规则模式的 21 种组合全部成功；使用不联网的本地假 Provider 模拟中置信度候选被 LLM 接受后，有六种组合触发同一异常。`升降器设计校核表-20251212-v1.json` 与 `升降器设计校核表-20251213-v1.json` 可以稳定复现。测试数据不得复制进仓库，测试过程不得调用外部 Provider。

## 方案比较

### 方案 A：候选级增量预检并安全降级（采用）

按稳定顺序逐个处理已解析的候选。对于每个 `merge` 候选，将它临时加入此前已验证的候选集合，并调用现有 operation builder 完成真实的行级投影和冲突校验：

- 成功：保留 `merge`；
- 失败：只将当前候选改为 `keep_separate`，并记录结构冲突码；
- 最终使用全部已验证决策重新生成一次 operations。

该方案复用已有安全投影规则，也能发现两个候选之间的投影冲突；无需新增猜列启发式。由于候选数量只覆盖页边界附近的表格片段，增量预检的计算开销可接受。

### 方案 B：先把片段拆成物理布局一致的子片段

该方案可能保留更多合并机会，但会改变候选生成、边界识别、来源追踪和片段合并语义，改动面较大。当前复现数据中的歧义行可能同时属于业务行与宽表边界，直接拆分容易制造错误合并，不适合作为本次缺陷的最小修复。

### 方案 C：捕获 operation builder 异常并取消所有合并

实现最小，但一个坏候选会使同一文档内其他安全候选全部失效，也无法在 trace 中说明具体拒绝对象。该方案不采用。

## 详细设计

在 `table_reconstruction_pipeline.py` 中新增一个内部验证步骤，输入为已完成规则/LLM决策的 `CandidateAssessment` 列表以及边界行、边界段落集合，输出为经过投影安全校验的 assessment 列表。

处理顺序沿用候选现有的 `(side, candidate_id)` 稳定排序：

1. 先用空候选集合构建边界 operations，确认边界数据本身有效；该步骤失败时继续抛出，不得伪装成候选错误。
2. `keep_separate` 候选原样保留。
3. 对每个 `merge` 候选，用“此前已验证 assessments + 当前候选”试建 operations。
4. 试建成功时接受当前候选。
5. 试建抛出结构性 `ValueError` 时，仅将当前候选降级为 `keep_separate`，并在候选 `conflicts` 中追加稳定冲突码 `unsafe_fragment_projection`。
6. 全部候选验证完毕后，再调用一次 operation builder 生成最终 operations。最终调用仍保留原有异常保护，避免未知错误被吞掉。

LLM judgment 原样保留在 trace 中，因此可以看到“LLM建议合并，但结构完整性校验拒绝”的完整决策链。最终 `final_action` 必须反映安全降级结果。

## 数据流与边界

```text
规则评估
  -> 中置信度 LLM 判断（可选）
  -> 候选级增量投影预检
       -> 安全：merge
       -> 歧义：keep_separate + unsafe_fragment_projection
  -> 最终 operations
  -> replay
  -> 后续语义匹配
```

本修复不修改：

- 原始或解析后的 PDF/JSON；
- `DocumentIR` 持久化结构；
- FAISS、chunk 或 embedding 索引；
- Provider 请求格式和阈值；
- trace schema 版本。

## 错误处理

- 已知的候选投影不安全属于局部拒绝条件，不再导致整次对比失败。
- 边界数据自身无效、最终 operation 构建失败、trace 校验失败或 replay 失败仍然是全局错误，必须继续失败并记录堆栈。
- 不根据异常文本推测新的列位置，不删除未映射单元格，也不让 LLM覆盖结构完整性约束。

## 测试与验收

自动测试必须覆盖：

1. 一个由 LLM 接受但含未映射保留单元格的中置信度候选被降级，pipeline 正常返回。
2. trace 保留 LLM judgment，`final_action` 为 `keep_separate`，`rule_conflicts` 包含 `unsafe_fragment_projection`。
3. 同一批候选中的其他安全候选仍能生成 `merge_rows` 和相关 operations。
4. 已支持的完整移位行继续生成正确的行级投影。
5. operation builder 的最终防丢校验继续存在，直接传入不安全候选时仍然抛错。
6. 使用 `E:\Project\test` 中自然配对 JSON 和本地假 Provider 复现时不再抛异常，且不调用网络。
7. 运行表格重建定向测试以及完整 `pytest` 回归。

验收标准是：局部歧义候选不影响整次对比，安全候选不受牵连，任何非空单元格都不会因为修复而被猜测、覆盖或静默丢弃。
