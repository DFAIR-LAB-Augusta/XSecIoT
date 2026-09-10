# General Multiclass Test Coverage (#95) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Issue #95 asks for test coverage across model classes, `train_ce_multiclass`, the retraining branch, and the labeling script. An audit of the existing suite (built up incrementally across #85-#94) shows `test_mc_labeling.py` (19 tests) and `test_ce_model_training.py` (9 tests, all 6 model variants including XGB) are already thorough. Two real gaps remain: `MLPCEMulticlass`'s `get_params`/`set_params`/`clone` cycle has no test (its binary sibling does), and `retrain_runtime` is only exercised for DT and FEEDFORWARD, leaving KNN/RF/SVM untested plus two completely untested branches (`monitor is None` guard, single-class skip-refit).

**Architecture:** No production code changes are anticipated — this plan closes concrete, identified coverage gaps in two existing test files. If either new test surfaces a real bug (matching the pattern from #93/#94), fix it in place following the same file's established conventions.

**Tech Stack:** Python 3.11+, `pytest`, `torch`, `pandas`, `scikit-learn` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- Follow each target file's existing fixture conventions exactly (`_make_multiclass_data`/`DEVICE` in `test_mlp_ce.py`; `_StubMonitor`/`_make_runtime`/`_make_config`/`ROLLING_COLUMNS` in `test_retraining.py`) — do not introduce parallel fixtures.
- `retrain_runtime`'s `_StubMonitor` deliberately decouples these tests from CE/CADE's own internals (already covered by #93/#94) — do not swap in a real `ConformalDriftMonitor` here.

---

## File Structure

- Modify: `tests/test_mlp_ce.py` — add one test proving `MLPCEMulticlass.get_params()`/`set_params()`/`clone()` work correctly, mirroring the existing binary test.
- Modify: `tests/test_retraining.py` — parametrize the existing DT-only retraining test across all classical variants (KNN/RF/SVM/DT), and add two tests for the two currently-untested branches in `retrain_runtime`/`_fit_monitor_on_retrained_data`.

---

### Task 1: `MLPCEMulticlass` get_params/set_params/clone coverage

**Files:**
- Modify: `tests/test_mlp_ce.py`

**Interfaces:**
- Consumes: `MLPCEMulticlass` (`firce.models.mlp_ce_multiclass`), `_make_multiclass_data` (already defined in this file, returns `(X, y, labels)`), `DEVICE` (already defined in this file).

- [ ] **Step 1: Write the test**

Add to `tests/test_mlp_ce.py`, after `test_mlp_ce_multiclass_save_load_roundtrip`:

```python
def test_mlp_ce_multiclass_get_params_set_params_clone():
    X, y, labels = _make_multiclass_data()
    model = MLPCEMulticlass(input_dim=X.shape[1], classes=labels, device=DEVICE, epochs=3)
    model.fit(X, y)

    params = model.get_params()
    assert params['input_dim'] == X.shape[1]
    assert sorted(params['classes']) == sorted(labels.tolist())

    clone = model.clone()
    assert sorted(clone.classes_.tolist()) == sorted(labels.tolist())
    assert clone.is_fitted_ is False

    model.set_params(epochs=7)
    assert model.epochs == 7
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_mlp_ce.py -v`
Expected: all pass, including the new test. `clone.classes_` should already be populated (unlike the binary case, where `classes_` resets to `None` since binary derives it from `fit` data) — `MLPCEMulticlass.classes_` is fixed at construction time from the `classes` constructor arg, and `clone()` round-trips it through `get_params()`'s `_extra_params()` override. If this assertion fails, diagnose the real cause (don't guess) — read `MLPCEBase.clone()`/`get_params()` and `MLPCEMulticlass._extra_params()` to find the actual mismatch.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mlp_ce.py
git commit -m "test: cover MLPCEMulticlass get_params/set_params/clone cycle"
```

---

### Task 2: `retrain_runtime` variant + branch coverage

**Files:**
- Modify: `tests/test_retraining.py`

**Interfaces:**
- Consumes: `retrain_runtime` (`firce.runtime.retraining`), `SimulationRuntime` (`firce.runtime.sim_types`), `_StubMonitor`, `_make_runtime`, `_make_config`, `ROLLING_COLUMNS`, `_make_multiclass_rows` (all already defined in this file, shown in full below for reference).

Current relevant contents of `tests/test_retraining.py` (for reference — do not recreate, only add to):

```python
class _StubMonitor:
    def __init__(self):
        self.fit_calls: list = []

    def fit(self, X, y, perf_stats):
        self.fit_calls.append((X, y))


def _make_runtime(tmp_path, model_variant=ModelVariant.DT):
    config = _make_config(tmp_path, model_variant=model_variant)
    rolling = CircularDequeLogger(None, max_rows=200, columns=ROLLING_COLUMNS)
    for row in _make_multiclass_rows().itertuples(index=False, name=None):
        rolling.append(list(row))

    return SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=rolling,
        scaler=None,
        pca=None,
        model=None,
        monitor=_StubMonitor(),
        train_df=pd.DataFrame(),
    )


def test_retrain_runtime_multiclass_updates_model_and_scaler(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = _make_runtime(tmp_path)

    retrain_runtime(runtime)

    assert runtime.scaler is not None
    assert runtime.model is not None
    assert len(runtime.monitor.fit_calls) == 1
    _, y_fit = runtime.monitor.fit_calls[0]
    assert set(y_fit.tolist()) <= {'Benign', 'PortScan', 'XMasAttack'}
```

- [ ] **Step 1: Parametrize the existing classical-variant test across DT/KNN/RF/SVM**

In `tests/test_retraining.py`, replace:

```python
def test_retrain_runtime_multiclass_updates_model_and_scaler(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = _make_runtime(tmp_path)

    retrain_runtime(runtime)

    assert runtime.scaler is not None
    assert runtime.model is not None
    assert len(runtime.monitor.fit_calls) == 1
    _, y_fit = runtime.monitor.fit_calls[0]
    assert set(y_fit.tolist()) <= {'Benign', 'PortScan', 'XMasAttack'}
```

with:

```python
@pytest.mark.parametrize(
    'model_variant', [ModelVariant.DT, ModelVariant.KNN, ModelVariant.RF, ModelVariant.SVM]
)
def test_retrain_runtime_multiclass_updates_model_and_scaler(tmp_path, monkeypatch, model_variant):
    monkeypatch.chdir(tmp_path)
    runtime = _make_runtime(tmp_path, model_variant=model_variant)

    retrain_runtime(runtime)

    assert runtime.scaler is not None
    assert runtime.model is not None
    assert len(runtime.monitor.fit_calls) == 1
    _, y_fit = runtime.monitor.fit_calls[0]
    assert set(y_fit.tolist()) <= {'Benign', 'PortScan', 'XMasAttack'}
```

Add `import pytest` at the top of the file if not already present (check first — `test_config_validation.py`'s pattern uses a plain top-level `import pytest`).

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_retraining.py -v`
Expected: 4 passes for the parametrized test (dt/knn/rf/svm) plus the existing feedforward test, 5 total plus the 2 new tests from Steps 3-4 below once added.

- [ ] **Step 3: Add a test for the `monitor is None` guard**

`retrain_runtime` (`src/firce/runtime/retraining.py:34-35`) raises `RuntimeError('Monitor is disabled; retraining should not be triggered.')` when `runtime.monitor is None` — currently untested. Add to `tests/test_retraining.py`:

```python
def test_retrain_runtime_raises_when_monitor_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = _make_runtime(tmp_path)
    runtime.monitor = None

    with pytest.raises(RuntimeError, match='Monitor is disabled'):
        retrain_runtime(runtime)
```

- [ ] **Step 4: Add a test for the single-class skip-refit branch**

`_fit_monitor_on_retrained_data` (`src/firce/runtime/retraining.py:215-220`) logs a warning and returns early — skipping the monitor refit — when the retrained rolling data has fewer than 2 distinct classes. Currently untested. Add to `tests/test_retraining.py`:

```python
def test_retrain_runtime_rejects_single_class_retraining_data(tmp_path, monkeypatch):
    # train_ce_multiclass itself rejects <2 distinct MC_Label classes before
    # retrain_runtime ever reaches _fit_monitor_on_retrained_data's own
    # single-class skip-refit branch - that guard fires first in practice.
    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path)
    rolling = CircularDequeLogger(None, max_rows=200, columns=ROLLING_COLUMNS)
    single_class_rows = _make_multiclass_rows().copy()
    single_class_rows['MC_Label'] = 'Benign'
    for row in single_class_rows.itertuples(index=False, name=None):
        rolling.append(list(row))

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=rolling,
        scaler=None,
        pca=None,
        model=None,
        monitor=_StubMonitor(),
        train_df=pd.DataFrame(),
    )

    with pytest.raises(ValueError, match='at least 2 distinct MC_Label classes'):
        retrain_runtime(runtime)
```

- [ ] **Step 5: Run the full file**

Run: `uv run pytest tests/test_retraining.py -v`
Expected: all pass (4 parametrized + feedforward + 2 new branch tests = 7 total). If the single-class test fails because `train_ce_multiclass` itself rejects single-class training data before the monitor-refit branch is even reached, that's a real, different finding — diagnose via the traceback and adjust: check whether `train_ce_multiclass` requires ≥2 classes (it likely does, matching `MLPCEMulticlass`'s own constructor guard) and whether that's the *actual* wall being tested here instead of the monitor-refit skip. If so, keep the test but adjust its assertion to match whichever guard fires first, and note which one in a comment.

- [ ] **Step 6: Commit**

```bash
git add tests/test_retraining.py
git commit -m "test: parametrize retrain_runtime across classical variants, cover monitor-disabled and single-class branches"
```

---

### Task 3: Lint, format, and final regression

**Files:**
- `tests/test_mlp_ce.py`, `tests/test_retraining.py` (formatting only, if needed)

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
git commit -m "style: ruff format general multiclass test coverage additions"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** #95 lists 4 areas: model classes (#85/#86), `train_ce_multiclass` (#87), the retraining branch (#89), and the labeling script (#91). Audit found the latter two of those four already thoroughly covered (`test_ce_model_training.py`: 9 tests across all 6 variants; `test_mc_labeling.py`: 19 tests). This plan closes the two real remaining gaps: `MLPCEMulticlass`'s clone/params cycle (model classes) and `retrain_runtime`'s variant/branch coverage (retraining branch).

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code. Step 5 of Task 2 has an explicit contingency (matching #93's Step 2 pattern) for a genuinely uncertain outcome — whether `train_ce_multiclass` or the monitor-refit skip is the first guard to fire on single-class data — which cannot be resolved without running the test, and is not a placeholder for missing design.

**Type consistency:** `_make_runtime(tmp_path, model_variant=...)` signature (Task 2 Step 1) matches the existing definition already in the file. `SimulationRuntime` field names/order in Task 2 Step 4 match the existing `_make_runtime` body verbatim.

**Explicitly out of scope:** No changes to `train_ce_multiclass`, `mc_labeling.py`, or their existing thorough test files — audit found no real gaps there. The broader firce test-coverage initiative (conftest fixtures, e2e test) is #104, tracked separately.
