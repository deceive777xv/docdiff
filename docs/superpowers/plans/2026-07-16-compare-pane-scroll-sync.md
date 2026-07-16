# Compare Pane Scroll Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stable bidirectional proportional scrolling to the document comparison panes while keeping explicit diff selections centered and the initial result view at the document top.

**Architecture:** Keep all high-frequency scrolling logic inside `assets/diff_template.html` so PySide6 does not receive scroll events. The template exposes small global functions for scroll reset and diff focus; `ComparePage` only injects content and invokes those functions through the existing `QWebEnginePage.runJavaScript(scriptSource)` call, which remains valid in the current Qt for Python API without requiring a callback.

**Tech Stack:** Python 3.11+, PySide6 6.7+, Qt WebEngine, HTML/CSS/vanilla JavaScript, pytest, pytest-qt.

## Global Constraints

- Initial comparison display must remain at the top of both documents and must not auto-select the first diff.
- Manual scrolling must synchronize in both directions by relative scroll progress, not copied pixel offsets.
- Explicit diff selection must highlight every matching `data-diff-id` element and center the matching element within each pane.
- A diff that exists on only one side must not force the other pane to jump.
- Scroll synchronization must prevent recursive feedback and visible jitter.
- Do not add dependencies or move scroll calculations into Python.

---

### Task 1: Bidirectional proportional pane scrolling and top reset

**Files:**
- Modify: `assets/diff_template.html:399-472`
- Modify: `app/ui/pages/compare_page.py:917-983`
- Test: `tests/ui/test_compare_page.py:648-655`

**Interfaces:**
- Consumes: DOM panes `#baseline-pane` and `#target-pane`; existing Python HTML injection in `ComparePage._render_diff(result)`.
- Produces: JavaScript functions `maximumScrollTop(pane) -> number`, `syncPaneScroll(sourcePane, targetPane) -> void`, `attachPaneScrollSync() -> void`, and `resetDiffPaneScroll() -> void`; global `window.resetDiffPaneScroll` callable from Python-injected JavaScript.

- [ ] **Step 1: Write the failing template-contract test**

Add the following test beside `test_diff_template_exposes_focus_diff_function`:

```python
def test_diff_template_syncs_panes_by_relative_scroll_progress():
    """Either document pane should drive proportional scrolling in the other pane."""
    from pathlib import Path

    template = Path("assets/diff_template.html").read_text(encoding="utf-8")

    assert "function maximumScrollTop(pane)" in template
    assert "function syncPaneScroll(sourcePane, targetPane)" in template
    assert "sourcePane.scrollTop / sourceMaximum" in template
    assert "progress * targetMaximum" in template
    assert "baselinePane.addEventListener('scroll'" in template
    assert "targetPane.addEventListener('scroll'" in template
    assert "scrollSyncInProgress" in template
```

- [ ] **Step 2: Write the failing initial-top test**

Add this focused Python integration test:

```python
def test_render_diff_resets_both_panes_without_auto_focusing(compare_page):
    """Injecting a result should show document tops without selecting a diff."""
    from app.core.types import DiffItem, DiffResult

    result = DiffResult(
        task_id="task-top",
        baseline_version_id="baseline-top",
        target_version_id="target-top",
        items=[
            DiffItem(
                diff_id="diff-top",
                section_path="第一章",
                diff_type="实质修改",
                risk_level="medium",
                baseline_text="旧内容",
                target_text="新内容",
                similarity_score=0.5,
                explanation="",
            )
        ],
    )

    compare_page._render_diff(result)

    script = compare_page._web_view.page().runJavaScript.call_args.args[0]
    assert "resetDiffPaneScroll();" in script
    assert "focusDiff(" not in script
```

- [ ] **Step 3: Run the two tests and verify RED**

Run:

```powershell
pytest tests/ui/test_compare_page.py::test_diff_template_syncs_panes_by_relative_scroll_progress tests/ui/test_compare_page.py::test_render_diff_resets_both_panes_without_auto_focusing -v
```

Expected: both tests fail because the template has no synchronization/reset functions and `_render_diff` does not call `resetDiffPaneScroll()`.

- [ ] **Step 4: Add proportional scrolling with re-entry protection**

Inside the template IIFE, before `matchingDiffElements`, add:

```javascript
      var scrollSyncInProgress = false;

      function maximumScrollTop(pane) {
        return Math.max(0, pane.scrollHeight - pane.clientHeight);
      }

      function syncPaneScroll(sourcePane, targetPane) {
        if (scrollSyncInProgress) {
          return;
        }

        var sourceMaximum = maximumScrollTop(sourcePane);
        var targetMaximum = maximumScrollTop(targetPane);
        var progress = sourceMaximum > 0
          ? sourcePane.scrollTop / sourceMaximum
          : 0;

        scrollSyncInProgress = true;
        targetPane.scrollTop = progress * targetMaximum;
        window.requestAnimationFrame(function () {
          scrollSyncInProgress = false;
        });
      }

      function attachPaneScrollSync() {
        var baselinePane = document.getElementById('baseline-pane');
        var targetPane = document.getElementById('target-pane');

        baselinePane.addEventListener('scroll', function () {
          syncPaneScroll(baselinePane, targetPane);
        });
        targetPane.addEventListener('scroll', function () {
          syncPaneScroll(targetPane, baselinePane);
        });
      }

      function resetDiffPaneScroll() {
        document.getElementById('baseline-pane').scrollTop = 0;
        document.getElementById('target-pane').scrollTop = 0;
      }
```

At template initialization, call `attachPaneScrollSync();` once. Expose the reset helper with:

```javascript
      window.resetDiffPaneScroll = resetDiffPaneScroll;
```

- [ ] **Step 5: Reset both panes after each result injection**

In both `_render_diff` JavaScript payload branches, change the final injected statement from:

```python
"attachDiffHandlers();"
```

to:

```python
"resetDiffPaneScroll();\n"
"attachDiffHandlers();"
```

Do not call `focusDiff` from result rendering.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run:

```powershell
pytest tests/ui/test_compare_page.py::test_diff_template_syncs_panes_by_relative_scroll_progress tests/ui/test_compare_page.py::test_render_diff_resets_both_panes_without_auto_focusing -v
```

Expected: `2 passed`.

- [ ] **Step 7: Commit Task 1**

```powershell
git add assets/diff_template.html app/ui/pages/compare_page.py tests/ui/test_compare_page.py
git commit -m "feat: 同步滚动文档对比双栏"
```

---

### Task 2: Center matching diffs from either pane without scroll feedback

**Files:**
- Modify: `assets/diff_template.html:425-460`
- Test: `tests/ui/test_compare_page.py:648-660`

**Interfaces:**
- Consumes: `focusDiff(diffId)` and `selectDiff(diffId)` from the template; QWebChannel bridge object `window.bridge`.
- Produces: `pausePaneScrollSync() -> void`, used only during smooth programmatic centering; document diff clicks invoke `focusDiff(diffId)` instead of selection-only behavior.

- [ ] **Step 1: Write the failing diff-click focus test**

Add this test beside the other template tests:

```python
def test_diff_template_click_centers_matching_items_and_pauses_sync_feedback():
    """Clicking a document diff should center both matches without scroll fighting."""
    from pathlib import Path

    template = Path("assets/diff_template.html").read_text(encoding="utf-8")

    assert "function pausePaneScrollSync()" in template
    assert "pausePaneScrollSync();" in template
    assert "target.scrollIntoView({ behavior: 'smooth', block: 'center' });" in template
    assert "focusDiff(diffId);" in template
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
pytest tests/ui/test_compare_page.py::test_diff_template_click_centers_matching_items_and_pauses_sync_feedback -v
```

Expected: FAIL because there is no focus-time scroll synchronization pause and document diff clicks still call `selectDiff(diffId)`.

- [ ] **Step 3: Pause proportional synchronization during smooth centering**

Extend the template synchronization state:

```javascript
      var scrollSyncInProgress = false;
      var scrollSyncPaused = false;
      var scrollSyncPauseTimer = null;

      function pausePaneScrollSync() {
        scrollSyncPaused = true;
        if (scrollSyncPauseTimer !== null) {
          window.clearTimeout(scrollSyncPauseTimer);
        }
        scrollSyncPauseTimer = window.setTimeout(function () {
          scrollSyncPaused = false;
          scrollSyncPauseTimer = null;
        }, 450);
      }
```

Update the guard at the start of `syncPaneScroll`:

```javascript
        if (scrollSyncInProgress || scrollSyncPaused) {
          return;
        }
```

At the start of `focusDiff`, after selection, call:

```javascript
        pausePaneScrollSync();
```

This allows both panes to run their own smooth centering animation without treating either animation as a manual source scroll.

- [ ] **Step 4: Route document diff clicks through the shared focus behavior**

In `_diffClickHandler`, retain the QWebChannel notification and replace:

```javascript
            selectDiff(diffId);
```

with:

```javascript
            focusDiff(diffId);
```

If a diff exists only in one pane, `focusDiff` finds no target in the other pane and leaves that pane unchanged.

- [ ] **Step 5: Run the focused template tests and verify GREEN**

Run:

```powershell
pytest tests/ui/test_compare_page.py::test_diff_template_exposes_focus_diff_function tests/ui/test_compare_page.py::test_diff_template_click_centers_matching_items_and_pauses_sync_feedback -v
```

Expected: `2 passed`.

- [ ] **Step 6: Run the full compare-page regression suite**

Run:

```powershell
pytest tests/ui/test_compare_page.py -v
```

Expected: all tests pass with no new warnings or errors.

- [ ] **Step 7: Verify the interaction in a browser**

Open `assets/diff_template.html` with Playwright, inject long unequal-height content and matching diff elements into both panes, then verify:

```text
1. Initial pane scrollTop values are both 0.
2. Scrolling either pane to 50% places the other pane near 50% of its own range.
3. Scrolling either pane to its bottom places the other pane at its bottom.
4. Clicking a two-sided diff highlights and centers both matching elements.
5. Clicking a one-sided diff centers only the pane containing it.
6. No visible bounce occurs after either action.
```

- [ ] **Step 8: Commit Task 2**

```powershell
git add assets/diff_template.html tests/ui/test_compare_page.py
git commit -m "feat: 聚焦双栏匹配差异"
```

---

### Task 3: Final verification

**Files:**
- Verify only: `assets/diff_template.html`
- Verify only: `app/ui/pages/compare_page.py`
- Verify only: `tests/ui/test_compare_page.py`

**Interfaces:**
- Consumes: all behavior implemented by Tasks 1 and 2.
- Produces: a verified feature ready for normal branch integration.

- [ ] **Step 1: Run all UI tests**

```powershell
pytest tests/ui tests/test_ui -v
```

Expected: all tests pass.

- [ ] **Step 2: Run the complete test suite**

```powershell
pytest -q
```

Expected: all tests pass; environment-only Windows sandbox warnings, if any, are reported separately and are not hidden by product-code changes.

- [ ] **Step 3: Check the final diff and worktree state**

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` has no output; status contains no unintended files.

