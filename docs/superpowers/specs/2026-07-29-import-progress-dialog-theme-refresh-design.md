# 导入进度弹窗动态主题刷新设计

## 问题

导入进度弹窗在创建时把当前 `Theme` 值写入局部 QSS。应用在弹窗保持打开时切换 light/dark 主题，`ThemeManager` 会更新全局 QSS 和 `Theme` 属性，但弹窗没有监听 `theme_changed`，其局部 QSS 继续覆盖全局样式，因此部分背景和文字保留旧主题颜色。

## 目标

- 打开的导入进度弹窗在 light/dark 主题切换后立即刷新全部局部样式。
- 保留现有颜色、布局、进度计算、关闭行为和导入流程。
- 弹窗自行维护主题刷新，不让 `LibraryPage` 感知其内部控件。

## 设计

`_ImportProgressDialog` 在构造完成后连接 `ThemeManager.instance().theme_changed` 到自身 `_apply_theme()`，与设置弹窗等现有独立窗口保持一致。

当前只以局部变量保存的标题和底部提示标签改为实例属性。`_apply_theme()` 每次执行时：

1. 使用现有 `Theme.BG_PAGE`、`Theme.BG_CARD`、`Theme.BORDER`、`Theme.TEXT_PRIMARY` 和 `Theme.COLOR_PRIMARY` 重建弹窗、列表和进度条的原有 QSS；
2. 重新应用标题、汇总文字和底部提示文字的现有主题样式；
3. 不改变任何颜色来源、字号、边距、圆角或控件层级。

Qt 会在接收对象销毁时自动移除 signal 连接，因此不增加手动断开或生命周期状态。

## 测试

通过 `_ImportProgressDialog` 的公开 Qt 行为验证：

- 在 light 主题下创建弹窗，确认其局部样式包含 light palette；
- 弹窗保持存活时切换到 dark 主题，确认弹窗背景以及标题、汇总、进度条、列表和提示标签都更新为 dark palette；
- 再切回 light，确认样式可以双向刷新；
- 现有进度聚合、关闭仅隐藏和批量导入测试继续通过。

## 完成标准

1. 打开的导入进度弹窗不会在主题切换后残留旧主题背景或文字样式。
2. 修复不改变现有配色和布局。
3. 主题切换和导入进度相关测试通过。
