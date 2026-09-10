# CADE Multiclass Compatibility Spike + Monitor-Fit Label Bug (#94) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Issue #94 asks for an investigation into whether `src/firce/drift_monitor/cade_monitor.py` (which wraps the external `cade-firce` package) needs changes to support multiclass. The investigation is complete: it doesn't. But it surfaced a real, separate bug in firce's own `bootstrap.py` that breaks monitor fitting (both CE and CADE) for every live multiclass simulation. Per explicit user direction, that bug is fixed here as part of closing out #94, rather than filed separately.

**Architecture:** `CadeRuntimeDetector.fit()`/`detect()` (in `~/dfair/CADE_FIRCE/src/cade/runtime.py`, the source for the `cade-firce` PyPI package `cade_monitor.py` depends on) already computes per-class centroids over `np.unique(y_train)` with no binary-specific assumptions, and explicitly requires `len(classes) >= 2` — it was written with multiclass (K>2) support as a first-class case, not bolted on. No changes are needed in the external repo. `firce/drift_monitor/cade_monitor.py` is a thin, already-generic pass-through wrapper — also no changes needed.

The actual bug: `firce/runtime/bootstrap.py::build_runtime_monitor` (line 411) does `train_df['Label']` for `model_type != BINARY`, but `'Label'` is the *binary* raw-label column (see `ce_model_training.py:254/273`, where it's mapped into `BinLabel`) — not `MC_Label`, the canonical multiclass label column established across #85-#92. Real multiclass CSVs produced via `mc_labeling.py` (see the `_make_multiclass_csv` fixture in `tests/test_config_validation.py`) have no `'Label'` column at all, so this isn't silently-wrong data — it's a `KeyError` crash the moment any multiclass simulation with `monitor_type != NONE` tries to fit its drift monitor (CE or CADE, both call `build_runtime_monitor`). `_label_column(config.model_type)` (in `firce/runtime/constants.py`, already imported into `bootstrap.py` and already used correctly one function up at line 243) is the fix — this line was simply missed during the #90 label-column threading work.

**Tech Stack:** Python 3.11+, `pandas`, `pytest` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- Reuse the existing `_make_config`/`_make_multiclass_csv` fixtures in `tests/test_config_validation.py` rather than duplicating them — the new test belongs in that file, alongside the existing `test_initialize_simulation_runtime_multiclass_all_variants`.
- Reproduce the bug through the real integration path (`initialize_simulation_runtime`), not by calling `build_runtime_monitor` directly with a hand-built `train_df` — the whole point is that the real CSV schema (`MC_Label` only, no `'Label'`) is what triggers the crash.

---

## File Structure

- Modify: `tests/test_config_validation.py` — add `test_initialize_simulation_runtime_multiclass_fits_ce_monitor`, proving the monitor-fit path works end-to-end for multiclass with a real CE monitor enabled.
- Modify: `src/firce/runtime/bootstrap.py:411` — fix the label column lookup.

---

### Task 1: Reproduce and fix the monitor-fit label bug

**Files:**
- Modify: `tests/test_config_validation.py`
- Modify: `src/firce/runtime/bootstrap.py`

**Interfaces:**
- Consumes: `_label_column(model_type: ModelType) -> str` from `firce.runtime.constants` (already imported in `bootstrap.py`).
- Consumes: `_make_config`, `_make_multiclass_csv` from `tests/test_config_validation.py` (already defined in that file, shown above).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config_validation.py` (after `test_initialize_simulation_runtime_multiclass_all_variants`):

```python
def test_initialize_simulation_runtime_multiclass_fits_ce_monitor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(
        tmp_path,
        model_variant=ModelVariant.DT,
        aggregated_path=csv_path,
        flows_path=csv_path,
        monitor_type=MonitorType.CE,
        ce_type=CEType.ICE,
    )

    runtime = initialize_simulation_runtime(config)

    assert runtime.monitor is not None
    thresholds = runtime.monitor._evaluator.thresholds
    assert set(thresholds.keys()) == {'Benign', 'PortScan', 'XMasAttack'}
```

- [ ] **Step 2: Run it to confirm it fails with the real bug**

Run: `uv run pytest tests/test_config_validation.py::test_initialize_simulation_runtime_multiclass_fits_ce_monitor -v`
Expected: FAIL with `KeyError: 'Label'` (raised from `build_runtime_monitor` at `bootstrap.py:411`), confirming the real crash path — the multiclass fixture CSV genuinely has no `'Label'` column.

- [ ] **Step 3: Fix `bootstrap.py`**

In `src/firce/runtime/bootstrap.py`, change line 411 from:

```python
    y_train = train_df['BinLabel'] if config.model_type == ModelType.BINARY else train_df['Label']
```

to:

```python
    y_train = train_df[_label_column(config.model_type)]
```

- [ ] **Step 4: Run the test again to confirm it passes**

Run: `uv run pytest tests/test_config_validation.py::test_initialize_simulation_runtime_multiclass_fits_ce_monitor -v`
Expected: PASS

- [ ] **Step 5: Run the full test file to confirm no regressions in binary or other multiclass tests**

Run: `uv run pytest tests/test_config_validation.py -v`
Expected: all pass, including the existing `test_initialize_simulation_runtime_multiclass_all_variants` (all 5 variants) and any binary-path tests in the same file.

- [ ] **Step 6: Commit**

```bash
git add tests/test_config_validation.py src/firce/runtime/bootstrap.py
git commit -m "fix: use correct multiclass label column when fitting the runtime drift monitor

build_runtime_monitor read train_df['Label'] (the binary raw-label
column) for multiclass runs instead of MC_Label via _label_column,
crashing with KeyError the moment any multiclass simulation with a
CE or CADE monitor enabled tried to fit it. Found while scoping #94
(CADE multiclass compatibility) - CADE's own detector already
handles multiclass correctly, this bug was in firce's own wiring."
```

---

### Task 2: Lint, format, and final regression

**Files:**
- `tests/test_config_validation.py`, `src/firce/runtime/bootstrap.py` (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any issues**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
```

If issues appear, run `uv run ruff check --fix .` and `uv run ruff format .`, then re-run the full suite.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures beyond the pre-existing/expected skips (xgboost not installed in lean group, unrelated MC rolling-logger skips)

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format monitor-fit label fix"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** #94 asks to investigate/scope CADE multiclass compatibility and file follow-up issues in the external repo if needed. Investigation is complete and documented in the plan's Architecture section and will be recorded in the issue-closing comment: no external-repo changes needed, `CadeRuntimeDetector` was already multiclass-correct. The one real gap found (`bootstrap.py`'s monitor-fit label column) is firce-side, not CADE-side, and per explicit user choice is fixed here rather than filed separately.

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code.

**Explicitly out of scope:** No changes to the external `cade-firce` repo (none needed). No changes to `CadeMonitorConfig`'s `dims` handling (user-configured architecture parameter, not class-count-derived — no bug found there). General multiclass test coverage beyond this specific fix (#95) and the broader firce test-coverage gap (#104) are separate.
