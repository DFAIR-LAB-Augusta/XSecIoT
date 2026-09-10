# Fix Shared-Model CE-Calibration Mutation + FIRCE Multiclass E2E Test (#125) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver #125 (full end-to-end test for the FIRCE multiclass live simulation pipeline). Building the e2e test surfaced a severe, previously-undiscovered bug: for any classical model variant (DT/KNN/RF/XGB) combined with `monitor_type=CE`, the CE monitor's calibration step silently corrupts the *production* model used for live predictions. This plan fixes that bug (the real blocker for #125), a smaller related logging bug found in the same investigation, and delivers the e2e test itself plus a dedicated regression test locking in the fix.

**Architecture:**

*The shared-model bug.* `firce/runtime/bootstrap.py::_build_monitor_model` returns the classifier used internally by the CE monitor. For `use_svm`/`use_mlp` it correctly builds a **fresh** model; for every other variant (the default/fallthrough case — DT, KNN, RF, XGB) it returned `model` directly: the *exact same object* already assigned to `runtime.model`. `ICE.calibrate()` (`firce/conformalEval/ice.py:98`) does `self.model.fit(X_train, y_train)` — an **in-place refit** as part of its own calibration split — and since `self.model` and `runtime.model` are the same object, this silently retrains the "production" model on the monitor's calibration data (and, for multiclass, on *raw string* `MC_Label` values rather than the `LabelEncoder`-integer-encoded labels `train_ce_multiclass` originally fit it on — `build_runtime_monitor`'s `y_train = train_df[_label_column(config.model_type)]` is the raw string column). The result: `runtime.model.classes_` becomes string-valued (`['Benign', 'TCPAttack', 'XMasAttack']`) instead of integer (`[0, 1, 2]`), and `predict_row`'s generic branch (`out = int(np.asarray(pred).reshape(-1)[0])`) crashes with `ValueError: invalid literal for int() with base 10: 'Benign'` on the very first live prediction.

This is not multiclass-specific in its root cause (the shared-reference-plus-in-place-refit pattern), but multiclass is where it's *observable* as a hard crash — `BinLabel` is already `0`/`1` int, so a binary refit-on-calibration-data silently produces a *different but still-int* model without an obvious crash (a separate, quieter correctness concern: the "production" model ends up being whatever the CE calibration's train-split refit produced, not the artifact `train_ce_binary`/`train_ce_multiclass` actually saved to disk — noted but not separately chased here, since the fix below eliminates it for both model types at once).

**Fix:** `_build_monitor_model`'s default case returns `clone_model(model)` (the existing, already-tested cloning utility from `firce/conformalEval/utils.py`, used identically by `CrossConformalEvaluator`/`ApproxCrossConformalEvaluator` for per-fold cloning) instead of `model` directly — the monitor gets its own independent, unfitted copy to calibrate, and `runtime.model` is never touched.

*The logging bug.* `firce/runtime/inference.py::_process_chunk_rows` had `if prediction not in [0, 1]: logger.error(...)` — a leftover binary-only assumption. For any multiclass run with more than 2 classes, every prediction with class index ≥2 gets spuriously logged at `ERROR` level (confirmed during this plan's design: fifty consecutive correct predictions all logged as errors). Fixed to check against the actual valid range for the run's model type.

*The e2e test.* Mirrors `test_run_simulation_pipeline_end_to_end_binary` (#104) but for `model_type=MULTI`, using a small (150/170-row), 3-class (`Benign`/`XMasAttack`/`TCPAttack`) sample of the real `combined_data_mc.csv` dataset (same reasoning as #104's binary fixture: the real dataset already matches `FINAL_LOG_COLUMNS`'s schema exactly, sidestepping all column-engineering risk). Manually validated during this plan's design: after the two fixes above, the full pipeline runs to completion with **100% training and streaming accuracy** (a small, cleanly-separable fixture, so perfect accuracy is expected and further confirms nothing is silently broken).

**Tech Stack:** Python 3.11+, `pandas`, `pytest`, `scikit-learn` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files/fixtures go in `tests/`/`tests/fixtures/`, run via `pytest -q` (see `pytest.ini`).
- Do not reference `/home/claude/dfair/repos/combined_data_mc.csv` (outside this git repo) from test code — only the committed `tests/fixtures/*.csv` samples.
- This plan closes #125.

---

## File Structure

- Create: `tests/fixtures/ce_flows_e2e_mc_train.csv` (150 rows, 3 classes) and `tests/fixtures/ce_flows_e2e_mc_stream.csv` (170 rows, 3 classes) — already generated and validated during this plan's design.
- Modify: `src/firce/runtime/bootstrap.py` — `_build_monitor_model`'s default case, plus a `clone_model` import.
- Modify: `src/firce/runtime/inference.py` — `_process_chunk_rows`'s prediction-range check.
- Modify: `tests/test_config_validation.py` — strengthen `test_initialize_simulation_runtime_multiclass_fits_ce_monitor` with the regression assertion that would have caught this bug.
- Modify: `tests/test_e2e_simulation.py` — add the multiclass e2e test and a multiclass drift→retrain wiring test.

---

### Task 1: Commit the multiclass e2e fixture CSVs

**Files:**
- Create: `tests/fixtures/ce_flows_e2e_mc_train.csv`
- Create: `tests/fixtures/ce_flows_e2e_mc_stream.csv`

Already generated during this plan's design (sampled from the real `combined_data_mc.csv`: 150 rows — 70 `Benign` + 50 `XMasAttack` + 30 `TCPAttack` — for training; 170 rows — 100/50/20 of the same three classes — for streaming).

- [ ] **Step 1: Verify the fixture files**

Run:
```bash
uv run python -c "
import pandas as pd
train = pd.read_csv('tests/fixtures/ce_flows_e2e_mc_train.csv')
stream = pd.read_csv('tests/fixtures/ce_flows_e2e_mc_stream.csv')
assert train.shape == (150, 86), train.shape
assert stream.shape == (170, 86), stream.shape
assert set(train['MC_Label'].unique()) == {'Benign', 'XMasAttack', 'TCPAttack'}
assert set(stream['MC_Label'].unique()) == {'Benign', 'XMasAttack', 'TCPAttack'}
assert not train.select_dtypes('number').isna().any().any()
assert not stream.select_dtypes('number').isna().any().any()
print('fixtures OK')
"
```
Expected: `fixtures OK`

- [ ] **Step 2: Commit**

```bash
git add tests/fixtures/ce_flows_e2e_mc_train.csv tests/fixtures/ce_flows_e2e_mc_stream.csv
git commit -m "test: add multiclass e2e fixture CSVs sampled from real combined_data_mc data"
```

---

### Task 2: Fix the shared-model CE-calibration mutation bug

**Files:**
- Modify: `src/firce/runtime/bootstrap.py`
- Modify: `tests/test_config_validation.py`

**Interfaces:**
- Consumes: `clone_model` (`firce.conformalEval.utils`) — already exists, used by `CrossConformalEvaluator`/`ApproxCrossConformalEvaluator`.

- [ ] **Step 1: Write a failing regression test**

In `tests/test_config_validation.py`, find `test_initialize_simulation_runtime_multiclass_fits_ce_monitor` and add an assertion to it (do not create a new test — this strengthens the existing one, which already exercises the exact buggy code path: `monitor_type=CE`, `ce_type=CEType.ICE`, `model_variant=ModelVariant.DT`):

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
    # Regression guard: CE calibration must not mutate the production model.
    # ICE.calibrate() does an in-place self.model.fit(...) as part of its own
    # calibration split - if _build_monitor_model shares runtime.model's exact
    # object reference instead of cloning it, this refit corrupts runtime.model
    # (fit on raw string MC_Label instead of the LabelEncoder-integer-encoded
    # labels train_ce_multiclass originally used), breaking every live prediction.
    assert all(isinstance(c, (int, np.integer)) for c in runtime.model.classes_)
```

- [ ] **Step 2: Run it to confirm it fails against the current (buggy) code**

Run: `uv run pytest tests/test_config_validation.py::test_initialize_simulation_runtime_multiclass_fits_ce_monitor -v`
Expected: FAIL — `runtime.model.classes_` are numpy string values (`'Benign'`, etc.), not integers.

- [ ] **Step 3: Fix `_build_monitor_model`**

In `src/firce/runtime/bootstrap.py`, add the import:

```python
from firce.conformalEval.utils import clone_model
```

(placed alphabetically among the existing `from firce...` imports, right after `from firce.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController`).

Change the final line of `_build_monitor_model` from:

```python
    return model
```

to:

```python
    return clone_model(model)
```

- [ ] **Step 4: Run the test again to confirm it passes**

Run: `uv run pytest tests/test_config_validation.py::test_initialize_simulation_runtime_multiclass_fits_ce_monitor -v`
Expected: PASS

- [ ] **Step 5: Run the full test file to confirm no regressions**

Run: `uv run pytest tests/test_config_validation.py -v`
Expected: all pass, including the binary equivalent (`test_initialize_simulation_runtime_binary_fits_ce_monitor`) — binary was never crashing (BinLabel is already int), but this confirms the clone doesn't break the binary path either.

- [ ] **Step 6: Commit**

```bash
git add src/firce/runtime/bootstrap.py tests/test_config_validation.py
git commit -m "fix: clone the model before handing it to the CE monitor

_build_monitor_model returned the exact same model object already
assigned to runtime.model for every classical variant (DT/KNN/RF/
XGB) - only use_svm/use_mlp got a fresh model. ICE.calibrate() (and
likely CCE/Approx-TCE/Approx-CCE similarly) does an in-place
self.model.fit(...) refit as part of its own calibration split,
silently corrupting the shared production model. For multiclass
this is a hard crash (fit on raw string MC_Label instead of the
LabelEncoder-integer-encoded labels train_ce_multiclass used,
breaking predict_row's int() cast on the very first live
prediction); for binary it would have silently swapped the
carefully-trained-and-saved model for whatever the monitor's
calibration split produced. Fixed by cloning via the existing
clone_model utility, matching what CCE/Approx-CCE already do for
per-fold models."
```

---

### Task 3: Fix the multiclass prediction-range logging bug

**Files:**
- Modify: `src/firce/runtime/inference.py`

- [ ] **Step 1: Fix `_process_chunk_rows`**

In `src/firce/runtime/inference.py`, change:

```python
        if prediction not in [0, 1]:
            logger.error('Row %d prediction: %r', row_index, prediction)
```

to:

```python
        if runtime.config.model_type == ModelType.BINARY:
            if prediction not in (0, 1):
                logger.error('Row %d prediction: %r', row_index, prediction)
        elif runtime.label_encoder is not None and prediction not in range(len(runtime.label_encoder.classes_)):
            logger.error('Row %d prediction: %r', row_index, prediction)
```

- [ ] **Step 2: Commit**

```bash
git add src/firce/runtime/inference.py
git commit -m "fix: don't log every multiclass prediction as an error

_process_chunk_rows checked 'prediction not in [0, 1]' regardless of
model_type - for any multiclass run with more than 2 classes, every
prediction with class index >= 2 was spuriously logged at ERROR
level. Confirmed during #125's e2e testing: fifty consecutive
correct predictions all logged as errors. Now checks against the
actual valid class-index range for the run's model type."
```

(No dedicated test for this one — it's a pure logging-level cosmetic fix with no behavioral effect; verified manually during design that the noisy `ERROR` lines disappear after the fix while predictions remain correct. Covered implicitly by Task 4's e2e test running cleanly.)

---

### Task 4: Full multiclass e2e test + drift→retrain wiring test

**Files:**
- Modify: `tests/test_e2e_simulation.py`

**Interfaces:**
- Consumes: everything already imported in `tests/test_e2e_simulation.py` (`run_simulation_pipeline`, `process_chunk`, `SimulationRuntime`, `DriftDetectionResult`, `CircularDequeLogger`, `get_rolling_columns`, `train_ce_binary`, config classes) plus `train_ce_multiclass` (`firce.ce_model_training`).

- [ ] **Step 1: Add the full pipeline multiclass e2e test**

Add to `tests/test_e2e_simulation.py`, after `test_run_simulation_pipeline_end_to_end_binary`:

```python
def test_run_simulation_pipeline_end_to_end_multiclass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'CETrain_mc_e2e'
    ds_dir.mkdir()
    train_csv = ds_dir / 'train.csv'
    stream_csv = ds_dir / 'stream.csv'
    shutil.copy(FIXTURES / 'ce_flows_e2e_mc_train.csv', train_csv)
    shutil.copy(FIXTURES / 'ce_flows_e2e_mc_stream.csv', stream_csv)

    config = SimulationConfig(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        ce_type=CEType.ICE,
        aggregated_path=train_csv,
        flows_path=stream_csv,
        is_unsw=False,
        seed=0,
        device=DEVICE,
        monitor_type=MonitorType.CE,
        chunk_size=50,
        use_circular_logger=True,
        use_pca=False,
    )

    run_simulation_pipeline(config)

    model_dir = tmp_path / 'multi_class_models' / 'CETrain_mc_e2e'
    assert (model_dir / 'dt_model_multi.pkl').exists()
    assert (model_dir / 'scaler_multi.pkl').exists()
    assert (model_dir / 'label_encoder_multi.pkl').exists()

    plot_dir = tmp_path / 'logging' / 'chunk_size_50' / 'DFAIR'
    assert (plot_dir / 'dt_ice_multi_0_0_accuracy_plot.png').exists()
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_e2e_simulation.py::test_run_simulation_pipeline_end_to_end_multiclass -v`
Expected: PASS — manually validated during this plan's design (after Tasks 2-3's fixes) to complete with 100% training and streaming accuracy and produce exactly these artifacts.

- [ ] **Step 3: Add the multiclass drift→retrain wiring test**

Add to the same file, after `test_process_chunk_triggers_retrain_on_detected_drift`:

```python
def test_process_chunk_triggers_retrain_on_detected_drift_multiclass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    train_csv = ds_dir / 'train.csv'
    shutil.copy(FIXTURES / 'ce_flows_e2e_mc_train.csv', train_csv)

    config = SimulationConfig(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=train_csv,
        flows_path=train_csv,
        is_unsw=False,
        seed=0,
        device=DEVICE,
    )

    model_dir = train_ce_multiclass(config, str(train_csv), variant=ModelVariant.DT, use_pca=False)
    scaler = joblib.load(model_dir / 'scaler_multi.pkl')
    model = joblib.load(model_dir / 'dt_model_multi.pkl')
    label_encoder = joblib.load(model_dir / 'label_encoder_multi.pkl')

    retrain_calls = []
    monkeypatch.setattr(
        'firce.runtime.inference.retrain_runtime',
        lambda runtime: retrain_calls.append(runtime),
    )

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=500, columns=get_rolling_columns(config)),
        scaler=scaler,
        pca=None,
        model=model,
        label_encoder=label_encoder,
        monitor=_AlwaysDriftMonitor(),
        train_df=pd.DataFrame(),
    )

    chunk = pd.read_csv(FIXTURES / 'ce_flows_e2e_mc_stream.csv').head(20)
    process_chunk(runtime, chunk, chunk_num=0)

    assert len(retrain_calls) == 1
    assert retrain_calls[0] is runtime
    assert runtime.perf_stats.drift_detected_indices == [0]
```

- [ ] **Step 4: Add the `train_ce_multiclass` import**

At the top of `tests/test_e2e_simulation.py`, change:

```python
from firce.ce_model_training import train_ce_binary
```

to:

```python
from firce.ce_model_training import train_ce_binary, train_ce_multiclass
```

- [ ] **Step 5: Run the full file**

Run: `uv run pytest tests/test_e2e_simulation.py -v`
Expected: all 4 tests pass (2 binary from #104, 2 new multiclass).

- [ ] **Step 6: Commit**

```bash
git add tests/test_e2e_simulation.py
git commit -m "test: add full multiclass e2e test + drift-to-retrain wiring test"
```

---

### Task 5: Lint, format, and final regression

**Files:**
- All files modified in Tasks 1-4 (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any issues**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
```

If issues appear, run `uv run ruff check --fix .` and `uv run ruff format .`, then re-run the full suite.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures beyond the pre-existing/expected skips (xgboost/tensorflow/cade not installed in lean group, unrelated MC rolling-logger skips). Pay particular attention to any other test that builds a `SimulationConfig` with `monitor_type=MonitorType.CE` and a classical model variant — the `clone_model` fix changes real behavior (the monitor's internal model is now genuinely independent from `runtime.model`), so any test that assumed the old shared-reference behavior (e.g., checking identity between a monitor-internal model and `runtime.model`) would need updating, not working around.

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format multiclass e2e test and bug fixes"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** #125 asks for a full end-to-end test for the FIRCE multiclass live simulation pipeline, mirroring #104's binary one and covering completion, artifacts/logs, and a drift→retrain path. Task 4 delivers exactly that. Building it surfaced a severe, blocking bug (Task 2) and a smaller related one (Task 3) — both fixed as part of closing #125, since the e2e test cannot pass without them, matching this session's established pattern of fixing real blockers found while writing real tests rather than working around them.

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code, all validated by real manual runs during this plan's design (the bug was reproduced, root-caused via direct object inspection — not guessed — and the fix confirmed via a 100%-accuracy full pipeline run).

**Type consistency:** `clone_model(model)` matches the exact call signature already used by `firce/conformalEval/cce.py`/`approx_cce.py`. The regression test in Task 2 asserts on `runtime.model.classes_` directly, the same attribute the original crash traceback pointed at.

**Explicitly out of scope:** The quieter binary-side consequence of the same root bug (before this fix, a binary CE-monitored classical model's "production" model was silently whatever the monitor's calibration-split refit produced, not the artifact saved to disk) is fixed by the same `clone_model` change but not separately investigated/tested here — the fix is unconditional (applies to both model types), so no separate binary-specific test is needed to validate it beyond Task 2 Step 5's regression check that the binary e2e test still passes.
