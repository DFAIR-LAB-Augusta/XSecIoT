# General Test Coverage Pt.2: Binary-Path Component Tests (#104) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue #104 (Part 2 of several — see `~/dfair/repos/xseciot/docs/superpowers/plans/2026-09-08-general-test-coverage-pt1-conftest-binary-training.md` for Part 1). Every existing test in `test_bootstrap.py`, `test_retraining.py`, `test_inference.py`, and `test_config_validation.py` exercises only the multiclass path (added incrementally across #85-#95) — the binary path through the exact same functions has zero coverage, despite being the original, still-primary code path. `firce/runtime/monitoring.py` (`filter_ce_kwargs`) has no test file at all.

**Architecture:** Every target function already has an established multiclass test in the corresponding file, using a `_make_config`/`_make_*_rows`/`_make_*_csv` helper pattern. This plan adds binary-path siblings using the exact same helper pattern (swap `MC_Label` string column for `BinLabel` int column, `ModelType.MULTI` for `ModelType.BINARY`, `train_ce_multiclass` for `train_ce_binary`), so each new test is a close mirror of an existing, already-verified-correct one — low risk of design mistakes, high confidence from precedent. `monitoring.py::filter_ce_kwargs` is small and self-contained (24 lines) and gets a new dedicated `test_monitoring.py`. `test_config_validation.py` gets one new binary integration test through `initialize_simulation_runtime` with a real CE monitor enabled — the same shape as the multiclass test added in #94's PR (`mc-cade-monitor-fit-label`), which re-validates that #94's `bootstrap.py:411` fix (`_label_column`) didn't regress the binary path it already handled correctly.

**Tech Stack:** Python 3.11+, `pytest`, `torch`, `pandas`, `scikit-learn` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- Mirror each existing multiclass test's exact structure/naming convention in its own file — do not introduce a new fixture style. `tests/conftest.py` (Part 1) fixtures are available but not required; existing files' local helpers may be extended in place, matching how each file already works.
- If a new binary test surfaces a real bug (matching the #93/#94/#114 pattern), do not fix it inline — file it as a GitHub issue and document current behavior in the test (same approach as #114), to avoid changing runtime behavior as a side effect of a test-coverage PR.

---

## File Structure

- Modify: `tests/test_bootstrap.py` — binary-path tests for `ensure_model_artifacts`.
- Modify: `tests/test_retraining.py` — binary-path test for `retrain_runtime`.
- Modify: `tests/test_inference.py` — binary-path tests for `_prepare_chunk`, `predict_row`, `_record_prediction_outcome`.
- Modify: `tests/test_config_validation.py` — binary-path integration test through `initialize_simulation_runtime` with a real CE monitor.
- Create: `tests/test_monitoring.py` — unit tests for `filter_ce_kwargs`.

---

### Task 1: `monitoring.py::filter_ce_kwargs` unit tests

**Files:**
- Create: `tests/test_monitoring.py`

**Interfaces:**
- Consumes: `filter_ce_kwargs` (`firce.runtime.monitoring`), `SimulationConfig`/`CEType`/`ModelType`/`ModelVariant` (`firce.utils.config`).

- [ ] **Step 1: Write the tests**

```python
import pytest

from firce.runtime.monitoring import filter_ce_kwargs
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig


def _make_config(tmp_path, **overrides):
    dummy = tmp_path / 'dummy.csv'
    dummy.write_text('a\n1\n')
    defaults = dict(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.ICE,
        aggregated_path=dummy,
        flows_path=dummy,
        is_unsw=False,
        seed=0,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_filter_ce_kwargs_raises_when_ce_disabled(tmp_path):
    config = _make_config(tmp_path, ce_type=CEType.NONE)

    with pytest.raises(RuntimeError, match='CE is disabled'):
        filter_ce_kwargs(config)


def test_filter_ce_kwargs_keeps_only_constructor_params_for_ice(tmp_path):
    config = _make_config(tmp_path, ce_type=CEType.ICE, ce_kwargs={'calibration_split': 0.3, 'not_a_real_param': 123})

    result = filter_ce_kwargs(config)

    assert result == {'calibration_split': 0.3}


def test_filter_ce_kwargs_keeps_only_constructor_params_for_cce(tmp_path):
    config = _make_config(tmp_path, ce_type=CEType.CCE, ce_kwargs={'folds': 3, 'not_a_real_param': 123})

    result = filter_ce_kwargs(config)

    assert result == {'folds': 3}


def test_filter_ce_kwargs_empty_kwargs_returns_empty_dict(tmp_path):
    config = _make_config(tmp_path, ce_type=CEType.APPROX_TCE, ce_kwargs={})

    result = filter_ce_kwargs(config)

    assert result == {}
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_monitoring.py -v`
Expected: all 4 pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_monitoring.py
git commit -m "test: add filter_ce_kwargs unit tests"
```

---

### Task 2: `bootstrap.py::ensure_model_artifacts` binary-path tests

**Files:**
- Modify: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: `ensure_model_artifacts` (`firce.runtime.bootstrap`), the existing `_make_config` helper in this file (add `model_type` as an overridable param — check it already accepts `**overrides`, which it does).

Current relevant contents of `tests/test_bootstrap.py` (for reference — do not recreate):

```python
def _make_config(csv_path, **overrides):
    defaults = dict(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=csv_path,
        flows_path=csv_path,
        is_unsw=False,
        seed=0,
        device=DEVICE,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)
```

- [ ] **Step 1: Add a `_make_binary_csv` helper and two binary tests**

Add to `tests/test_bootstrap.py`, after `_make_multiclass_csv`:

```python
def _make_binary_csv(tmp_path, n=60, seed=0, dirname='DS'):
    rng = np.random.default_rng(seed)
    ds_dir = tmp_path / dirname
    ds_dir.mkdir(exist_ok=True)
    csv_path = ds_dir / 'flows.csv'
    labels = np.array(['Benign', 'Attack'])
    idx = rng.integers(0, 2, size=n)
    df = pd.DataFrame({
        'device_id': range(n),
        'session_id': range(n),
        'src_ip': ['192.168.1.1'] * n,
        'dst_ip': ['192.168.1.2'] * n,
        'src_port': rng.integers(1024, 65535, size=n),
        'dst_port': rng.integers(1, 1024, size=n),
        'protocol': rng.integers(0, 2, size=n),
        'timestamp': ['01-01-2020 00:00'] * n,
        'flow_duration': rng.random(n) * 100,
        'tot_fwd_pkt': rng.integers(1, 50, size=n),
        'tot_bwd_pkts': rng.integers(0, 50, size=n),
        'totlen_fwd_pkts': rng.random(n) * 1000,
        'totlen_bwd_pkts': rng.random(n) * 1000,
        'Label': labels[idx],
    })
    df.to_csv(csv_path, index=False)
    return csv_path
```

Add after `test_ensure_model_artifacts_multiclass_propagates_real_errors`:

```python
def test_ensure_model_artifacts_binary_trains(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_binary_csv(tmp_path)
    config = _make_config(csv_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)

    ensure_model_artifacts(config, PerformanceStats())

    assert (tmp_path / 'binary_models' / 'DS' / 'dt_model_binary.pkl').exists()


def test_ensure_model_artifacts_binary_propagates_real_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    pd.DataFrame({'flow_duration': [1, 2, 3], 'tot_fwd_pkt': [1, 2, 3]}).to_csv(csv_path, index=False)
    config = _make_config(csv_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)

    with pytest.raises(ValueError, match="must contain either 'Label' or 'BinLabel'"):
        ensure_model_artifacts(config, PerformanceStats())
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: 5 pass (3 existing multiclass + 2 new binary).

- [ ] **Step 3: Commit**

```bash
git add tests/test_bootstrap.py
git commit -m "test: add ensure_model_artifacts binary-path tests"
```

---

### Task 3: `retraining.py::retrain_runtime` binary-path test

**Files:**
- Modify: `tests/test_retraining.py`

**Interfaces:**
- Consumes: `retrain_runtime`, `SimulationRuntime`, `_StubMonitor`, `_make_config`, `CircularDequeLogger`, `PerformanceStats` (all already defined/imported in this file).

- [ ] **Step 1: Add a binary rolling-columns constant, `_make_binary_rows` helper, and the binary test**

Add to `tests/test_retraining.py`, near `ROLLING_COLUMNS`:

```python
BINARY_ROLLING_COLUMNS = [
    'device_id',
    'session_id',
    'src_ip',
    'dst_ip',
    'src_port',
    'dst_port',
    'protocol',
    'timestamp',
    'flow_duration',
    'tot_fwd_pkt',
    'tot_bwd_pkts',
    'totlen_fwd_pkts',
    'totlen_bwd_pkts',
    'BinLabel',
]


def _make_binary_rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 2, size=n)
    return pd.DataFrame({
        'device_id': range(n),
        'session_id': range(n),
        'src_ip': ['192.168.1.1'] * n,
        'dst_ip': ['192.168.1.2'] * n,
        'src_port': rng.integers(1024, 65535, size=n),
        'dst_port': rng.integers(1, 1024, size=n),
        'protocol': rng.integers(0, 2, size=n),
        'timestamp': ['01-01-2020 00:00'] * n,
        'flow_duration': rng.random(n) * 100,
        'tot_fwd_pkt': rng.integers(1, 50, size=n),
        'tot_bwd_pkts': rng.integers(0, 50, size=n),
        'totlen_fwd_pkts': rng.random(n) * 1000,
        'totlen_bwd_pkts': rng.random(n) * 1000,
        'BinLabel': idx,
    })
```

Add after `test_retrain_runtime_rejects_single_class_retraining_data`:

```python
def test_retrain_runtime_binary_updates_model_and_scaler(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)
    rolling = CircularDequeLogger(None, max_rows=200, columns=BINARY_ROLLING_COLUMNS)
    for row in _make_binary_rows().itertuples(index=False, name=None):
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

    retrain_runtime(runtime)

    assert runtime.scaler is not None
    assert runtime.model is not None
    assert len(runtime.monitor.fit_calls) == 1
    _, y_fit = runtime.monitor.fit_calls[0]
    assert set(y_fit.tolist()) <= {0, 1}
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_retraining.py -v`
Expected: 8 pass (7 existing + 1 new binary).

- [ ] **Step 3: Commit**

```bash
git add tests/test_retraining.py
git commit -m "test: add retrain_runtime binary-path test"
```

---

### Task 4: `inference.py` binary-path tests

**Files:**
- Modify: `tests/test_inference.py`

**Interfaces:**
- Consumes: `_prepare_chunk`, `predict_row`, `_record_prediction_outcome` (`firce.runtime.inference`), `train_ce_binary` (`firce.ce_model_training`), `SimulationRuntime`, `CircularDequeLogger`, `DROP_COLS`/`PRED_THRESHOLD` (all already imported in this file, plus one new import: `train_ce_binary`).

- [ ] **Step 1: Add `train_ce_binary` import, a `BINARY_FLOW_COLUMNS` constant, `_make_binary_rows`, and `_train_and_build_binary_runtime` helper**

Change the import line:

```python
from firce.ce_model_training import train_ce_multiclass
```

to:

```python
from firce.ce_model_training import train_ce_binary, train_ce_multiclass
```

Add after `FLOW_COLUMNS`:

```python
BINARY_FLOW_COLUMNS = [
    'device_id',
    'session_id',
    'src_ip',
    'dst_ip',
    'src_port',
    'dst_port',
    'protocol',
    'timestamp',
    'flow_duration',
    'tot_fwd_pkt',
    'tot_bwd_pkts',
    'totlen_fwd_pkts',
    'totlen_bwd_pkts',
    'BinLabel',
]


def _make_binary_rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 2, size=n)
    return pd.DataFrame({
        'device_id': range(n),
        'session_id': range(n),
        'src_ip': ['192.168.1.1'] * n,
        'dst_ip': ['192.168.1.2'] * n,
        'src_port': rng.integers(1024, 65535, size=n),
        'dst_port': rng.integers(1, 1024, size=n),
        'protocol': rng.integers(0, 2, size=n),
        'timestamp': ['01-01-2020 00:00'] * n,
        'flow_duration': rng.random(n) * 100,
        'tot_fwd_pkt': rng.integers(1, 50, size=n),
        'tot_bwd_pkts': rng.integers(0, 50, size=n),
        'totlen_fwd_pkts': rng.random(n) * 1000,
        'totlen_bwd_pkts': rng.random(n) * 1000,
        'BinLabel': idx,
    })
```

Add after `_train_and_build_runtime`:

```python
def _train_and_build_binary_runtime(tmp_path, monkeypatch, model_variant=ModelVariant.DT):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    _make_binary_rows().to_csv(csv_path, index=False)

    config = _make_config(tmp_path, model_type=ModelType.BINARY, model_variant=model_variant)

    model_dir = train_ce_binary(config, str(csv_path), PerformanceStats())
    scaler = joblib.load(model_dir / 'scaler_binary.pkl')

    if model_variant == ModelVariant.FEEDFORWARD:
        from firce.models.feedforward_binary import FeedForwardBinary

        ckpt = torch.load(model_dir / 'feedforward_model_binary.pt', map_location='cpu')
        model = FeedForwardBinary(input_dim=int(ckpt['input_dim']))
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model.eval()
    else:
        model = joblib.load(model_dir / f'{model_variant.value}_model_binary.pkl')

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=200, columns=BINARY_FLOW_COLUMNS),
        scaler=scaler,
        pca=None,
        model=model,
        label_encoder=None,
        monitor=None,
        train_df=pd.DataFrame(),
    )
    return runtime, csv_path
```

- [ ] **Step 2: Add the four binary-path tests**

Add at the end of the file:

```python
def test_prepare_chunk_extracts_binlabel_ground_truth(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_binary_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)

    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)

    assert ground_truth is not None
    assert set(ground_truth.tolist()) <= {0, 1}
    assert 'BinLabel' not in clean_chunk.columns


@pytest.mark.parametrize('model_variant', [ModelVariant.DT, ModelVariant.FEEDFORWARD])
def test_predict_row_binary_returns_valid_class(tmp_path, monkeypatch, model_variant):
    runtime, csv_path = _train_and_build_binary_runtime(tmp_path, monkeypatch, model_variant=model_variant)
    chunk = pd.read_csv(csv_path)
    clean_chunk, _ = _prepare_chunk(runtime, chunk)
    row = clean_chunk.iloc[[0]]

    prediction = predict_row(row, DROP_COLS, runtime.scaler, runtime.pca, runtime.config, runtime.model, PRED_THRESHOLD)

    assert prediction in (0, 1)


def test_record_prediction_outcome_binary_tracks_correctness(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_binary_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)
    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)
    raw_row = clean_chunk.iloc[0]
    row_to_log = clean_chunk.iloc[[0]].copy()

    true_value = int(ground_truth.iloc[0])

    _record_prediction_outcome(
        runtime=runtime,
        row_index=0,
        raw_row=raw_row,
        row_to_log=row_to_log,
        prediction=true_value,
        ground_truth=ground_truth,
    )

    assert row_to_log['BinLabel'].iloc[0] == true_value
    assert runtime.perf_stats.correct_log == [True]


def test_record_prediction_outcome_binary_tracks_incorrect(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_binary_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)
    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)
    raw_row = clean_chunk.iloc[0]
    row_to_log = clean_chunk.iloc[[0]].copy()

    true_value = int(ground_truth.iloc[0])
    wrong_value = 1 - true_value

    _record_prediction_outcome(
        runtime=runtime,
        row_index=0,
        raw_row=raw_row,
        row_to_log=row_to_log,
        prediction=wrong_value,
        ground_truth=ground_truth,
    )

    assert row_to_log['BinLabel'].iloc[0] == wrong_value
    assert runtime.perf_stats.correct_log == [False]
```

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/test_inference.py -v`
Expected: 9 pass (5 existing + 4 new binary; the parametrized `test_predict_row_binary_returns_valid_class` counts as 2). If `test_prepare_chunk_extracts_binlabel_ground_truth` fails because `clean_data` doesn't produce a numeric `BinLabel` dtype the way expected, diagnose via the traceback — don't guess — and adjust the fixture (not the production code) to match `clean_data`'s actual non-UNSW passthrough behavior (confirmed via direct code reading: for non-UNSW input, `clean_data` only does `Unnamed: 0` cleanup, inf→NaN replacement, and index sorting — it does not rename or synthesize label columns, so the CSV must already contain a `BinLabel` column exactly as `_make_binary_rows` provides).

- [ ] **Step 4: Commit**

```bash
git add tests/test_inference.py
git commit -m "test: add inference.py binary-path tests (_prepare_chunk, predict_row, _record_prediction_outcome)"
```

---

### Task 5: `initialize_simulation_runtime` binary-path integration test with a real CE monitor

**Files:**
- Modify: `tests/test_config_validation.py`

**Interfaces:**
- Consumes: `initialize_simulation_runtime` (`firce.runtime.bootstrap`), the existing `_make_config` helper in this file (already accepts `model_type` via `**overrides`).

- [ ] **Step 1: Add a `_make_binary_csv` helper and the integration test**

Add to `tests/test_config_validation.py`, after `_make_multiclass_csv`:

```python
def _make_binary_csv(tmp_path, n=60, seed=0, dirname='DS'):
    rng = np.random.default_rng(seed)
    ds_dir = tmp_path / dirname
    ds_dir.mkdir(exist_ok=True)
    csv_path = ds_dir / 'flows.csv'
    labels = np.array(['Benign', 'Attack'])
    idx = rng.integers(0, 2, size=n)
    df = pd.DataFrame({
        'device_id': range(n),
        'session_id': range(n),
        'src_ip': ['192.168.1.1'] * n,
        'dst_ip': ['192.168.1.2'] * n,
        'src_port': rng.integers(1024, 65535, size=n),
        'dst_port': rng.integers(1, 1024, size=n),
        'protocol': rng.integers(0, 2, size=n),
        'timestamp': ['01-01-2020 00:00'] * n,
        'flow_duration': rng.random(n) * 100,
        'tot_fwd_pkt': rng.integers(1, 50, size=n),
        'tot_bwd_pkts': rng.integers(0, 50, size=n),
        'totlen_fwd_pkts': rng.random(n) * 1000,
        'totlen_bwd_pkts': rng.random(n) * 1000,
        'Label': labels[idx],
    })
    df.to_csv(csv_path, index=False)
    return csv_path
```

Add after `test_initialize_simulation_runtime_multiclass_fits_ce_monitor`:

```python
def test_initialize_simulation_runtime_binary_fits_ce_monitor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_binary_csv(tmp_path)
    config = _make_config(
        tmp_path,
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        aggregated_path=csv_path,
        flows_path=csv_path,
        monitor_type=MonitorType.CE,
        ce_type=CEType.ICE,
    )

    runtime = initialize_simulation_runtime(config)

    assert runtime.monitor is not None
    thresholds = runtime.monitor._evaluator.thresholds
    assert set(thresholds.keys()) == {0, 1}
    assert runtime.label_encoder is None
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_config_validation.py -v`
Expected: 10 pass (9 existing + 1 new binary). This re-validates that #94's `bootstrap.py:411` fix (`_label_column`) works correctly for binary too, not just multiclass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_config_validation.py
git commit -m "test: add initialize_simulation_runtime binary-path CE-monitor integration test"
```

---

### Task 6: Lint, format, and final regression

**Files:**
- All files modified in Tasks 1-5 (formatting only, if needed)

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
git commit -m "style: ruff format binary component test additions"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** This is Part 2 of #104. Covers the issue's explicit examples almost verbatim: `bootstrap.py::ensure_model_artifacts` tested against a tmp_path artifact directory (Task 2), `retraining.py`'s retrain path tested against a fake rolling-log dataframe verifying correct label column/training function selection (Task 3), `inference.py::_record_prediction_outcome` tested against a stub-ish `SimulationRuntime` (Task 4). Adds `monitoring.py` coverage (previously zero, Task 1) and closes the loop on the full bootstrap integration path for binary (Task 5).

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code. Task 4 Step 3 has an explicit, reasoned contingency (matching #93/#95's established pattern) rather than a placeholder.

**Type consistency:** `_train_and_build_binary_runtime(tmp_path, monkeypatch, model_variant=ModelVariant.DT)` mirrors `_train_and_build_runtime`'s exact signature. `BINARY_ROLLING_COLUMNS`/`BINARY_FLOW_COLUMNS` mirror `ROLLING_COLUMNS`/`FLOW_COLUMNS` with `MC_Label` swapped for `BinLabel`.

**Explicitly out of scope for this PR (remaining #104 work):** binary-path `conformalEval`/`drift_monitor` tests, the full e2e test, dead test file triage (`test_simulations.py`/`test_main.py`/`test_models.py`), `is_unsw=True` path for `train_ce_binary` (already deferred from Part 1). Do not close #104 after this PR merges — more parts remain.
