# Fix UNSW Multiclass Rolling Schema (#108) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix #108 — the UNSW rolling-log schema (`ROLLING_COLS`) has no `MC_Label` slot, so UNSW + multiclass live simulation has been hard-blocked since #92 with an explicit `ValueError`. This plan removes that block and makes the whole chain (training-frame loading, rolling-log schema, row appending) correctly multiclass-aware for UNSW runs, matching the pattern already used for the non-UNSW path.

**Architecture:** Three functions needed fixing, found by tracing the full chain and validating with a manual smoke test against a realistic raw NetFlow-v3-format fixture (confirmed working during this plan's design):

1. `firce/runtime/constants.py` — new `get_unsw_rolling_columns(model_type) -> list[str]` helper: `ROLLING_COLS[:-1] + [_label_column(model_type)]` (the label slot is always the last element of `ROLLING_COLS`, already relied upon elsewhere via `ROLLING_COLS[:-1]`).
2. `firce/runtime/bootstrap.py::get_rolling_columns` — delegates to the new helper for the `is_unsw` branch instead of unconditionally returning `ROLLING_COLS.copy()`.
3. `firce/runtime/bootstrap.py::load_training_frame` — this function had a **second, deeper gap** beyond what #108's title suggested: it only ever derived `BinLabel` for UNSW (from `Label`), never `MC_Label` (from `Attack`) — so a UNSW+multiclass dataset would trip its `extra_features` check (which compares against `FINAL_LOG_COLUMNS`, itself `BinLabel`-only) before training even started. Fixed by adding the same `Attack`→`MC_Label` derivation `train_ce_multiclass` already does internally, and making the `extra_features` comparison set swap `BinLabel`→`MC_Label` for multiclass (mirroring `get_rolling_columns`'s existing swap pattern).
4. `firce/runtime/inference.py::_append_unsw_row` — used the fixed `ROLLING_COLS` constant as its `allowed` reindex target and unconditionally coerced `pruned['BinLabel']` via `_coerce_binary_label`. Fixed to use `get_unsw_rolling_columns(runtime.config.model_type)` and only coerce for `ModelType.BINARY` (multiclass `MC_Label` values are already valid strings from `_record_prediction_outcome`, no coercion needed).
5. `firce/runtime/bootstrap.py::initialize_simulation_runtime` — the `#92`-added guard blocking `is_unsw and model_type == MULTI` is removed entirely, since the underlying gap it guarded against is now fixed.

**A separate, pre-existing, non-multiclass-specific bug was found and filed separately (#121), not fixed here**: `inference.py::_prepare_chunk` hardcodes `clean_data(chunk, False)` regardless of `runtime.config.is_unsw`, so live streaming chunks never get UNSW column renaming applied (only the one-time `aggregated_path` load via `load_training_frame` does). This affects UNSW binary streaming equally and is unrelated to the rolling-log schema gap #108 is about — tests in this plan validate #108's actual scope (`initialize_simulation_runtime`'s training/seeding path, and `_append_unsw_row` in isolation) without depending on `_prepare_chunk`'s streaming-chunk cleaning being correct.

**Tech Stack:** Python 3.11+, `pandas`, `pytest` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- UNSW test fixtures must use **raw NetFlow-v3-style column names** (`IPV4_SRC_ADDR`, `L4_SRC_PORT`, `IN_PKTS`, etc., all-caps) for `aggregated_path` — confirmed during this plan's design that `clean_data(df, is_unsw=True)` only performs its renaming/derived-feature computation when these exact raw names are present; anything else causes a silent early-return passthrough (which is why `load_training_frame`'s `extra_features` check then correctly rejects the un-renamed columns as "unexpected"). This differs from `train_ce_multiclass`/`train_ce_binary`'s own internal UNSW handling in `ce_model_training.py`, which does NOT call `clean_data` a second time and expects already-short-named input — the two functions have different input-format expectations for `aggregated_path`, both validated during design.
- Do not attempt to fix #121 (`_prepare_chunk`'s hardcoded `clean_data(chunk, False)`) as part of this plan.

---

## File Structure

- Modify: `src/firce/runtime/constants.py` — add `get_unsw_rolling_columns`.
- Modify: `src/firce/runtime/bootstrap.py` — `get_rolling_columns`, `load_training_frame`, remove the guard in `initialize_simulation_runtime`.
- Modify: `src/firce/runtime/inference.py` — `_append_unsw_row`.
- Modify: `tests/test_config_validation.py` — replace the now-obsolete rejection test with tests proving UNSW+multiclass works; add `get_rolling_columns`/`get_unsw_rolling_columns` UNSW coverage; add a binary UNSW regression test.
- Modify: `tests/test_inference.py` — add direct `_append_unsw_row` tests for both model types.

---

### Task 1: `get_unsw_rolling_columns` + wire into `get_rolling_columns`

**Files:**
- Modify: `src/firce/runtime/constants.py`
- Modify: `src/firce/runtime/bootstrap.py`

- [ ] **Step 1: Add the helper**

In `src/firce/runtime/constants.py`, immediately after the `ROLLING_COLS` list definition, add:

```python
def get_unsw_rolling_columns(model_type: ModelType) -> list[str]:
    """Return the UNSW rolling-log schema for the given model type."""
    return ROLLING_COLS[:-1] + [_label_column(model_type)]
```

- [ ] **Step 2: Wire it into `get_rolling_columns`**

In `src/firce/runtime/bootstrap.py`:
- Change the import line `from firce.runtime.constants import FINAL_LOG_COLUMNS, FULL_DROP_COLS, ROLLING_COLS, _label_column` to `from firce.runtime.constants import FINAL_LOG_COLUMNS, FULL_DROP_COLS, _label_column, get_unsw_rolling_columns` (drop the now-unused `ROLLING_COLS`, add `get_unsw_rolling_columns`).
- In `get_rolling_columns`, change `return ROLLING_COLS.copy()` to `return get_unsw_rolling_columns(config.model_type)`.

- [ ] **Step 3: Write tests for both**

Add to `tests/test_config_validation.py`, after `test_get_rolling_columns_binary_still_includes_bin_label`:

```python
def test_get_rolling_columns_unsw_multiclass_includes_mc_label(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.MULTI, is_unsw=True)
    columns = get_rolling_columns(config)

    assert 'MC_Label' in columns
    assert 'BinLabel' not in columns
    assert len(columns) == 21


def test_get_rolling_columns_unsw_binary_still_includes_bin_label(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT, is_unsw=True)
    columns = get_rolling_columns(config)

    assert 'BinLabel' in columns
    assert 'MC_Label' not in columns
    assert len(columns) == 21
```

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_config_validation.py -v -k rolling_columns`
Expected: all 4 pass (2 existing non-UNSW + 2 new UNSW).

- [ ] **Step 5: Commit**

```bash
git add src/firce/runtime/constants.py src/firce/runtime/bootstrap.py tests/test_config_validation.py
git commit -m "feat: add get_unsw_rolling_columns, branch UNSW rolling schema on model_type"
```

---

### Task 2: Fix `load_training_frame`'s UNSW multiclass label derivation

**Files:**
- Modify: `src/firce/runtime/bootstrap.py`

- [ ] **Step 1: Add `MC_Label` derivation and fix the `extra_features` check**

In `src/firce/runtime/bootstrap.py::load_training_frame`, change:

```python
    if config.model_type == ModelType.BINARY and 'BinLabel' not in df_train.columns and 'Label' in df_train.columns:
        if config.is_unsw:
            df_train['BinLabel'] = df_train['Label']
        else:
            df_train['BinLabel'] = df_train['Label'].map({'Benign': 0}).fillna(1).astype(int)

    df_train = df_train.drop(columns='Label', errors='ignore')
    df_train = df_train.drop(columns='Unnamed: 0', errors='ignore')

    if config.is_unsw:
        df_train = _unsw_clean(clean_data(df_train, config.is_unsw))
        extra_features = set(df_train.columns) - set(FINAL_LOG_COLUMNS)
        logger.debug('UNSW extra features beyond mandatory set: %s', extra_features)
        if extra_features:
            raise RuntimeError('Unexpected UNSW features found. Diagnose before retraining.')

    return df_train
```

to:

```python
    if config.model_type == ModelType.BINARY and 'BinLabel' not in df_train.columns and 'Label' in df_train.columns:
        if config.is_unsw:
            df_train['BinLabel'] = df_train['Label']
        else:
            df_train['BinLabel'] = df_train['Label'].map({'Benign': 0}).fillna(1).astype(int)

    if (
        config.model_type == ModelType.MULTI
        and config.is_unsw
        and 'MC_Label' not in df_train.columns
        and 'Attack' in df_train.columns
    ):
        df_train['MC_Label'] = df_train['Attack']

    df_train = df_train.drop(columns='Label', errors='ignore')
    df_train = df_train.drop(columns='Unnamed: 0', errors='ignore')

    if config.is_unsw:
        df_train = _unsw_clean(clean_data(df_train, config.is_unsw))
        label_col = _label_column(config.model_type)
        expected_columns = {label_col if col == 'BinLabel' else col for col in FINAL_LOG_COLUMNS}
        extra_features = set(df_train.columns) - expected_columns
        logger.debug('UNSW extra features beyond mandatory set: %s', extra_features)
        if extra_features:
            raise RuntimeError('Unexpected UNSW features found. Diagnose before retraining.')

    return df_train
```

(`_unsw_clean` drops `Attack` and `Label` via its `UNSW_DROP_COLS` list — confirmed during design — so no separate drop step is needed for `Attack` after deriving `MC_Label` from it.)

- [ ] **Step 2: Remove the `initialize_simulation_runtime` guard**

In the same file, change:

```python
def initialize_simulation_runtime(config: SimulationConfig) -> SimulationRuntime:
    """
    Build and return a fully initialized simulation runtime.

    Args:
        config: Simulation configuration.

    Returns:
        Fully initialized runtime state.

    Raises:
        ValueError: If `is_unsw` and `model_type=multi` are combined — the UNSW
            rolling-log schema has no MC_Label slot yet (see xseciot issue #108).
    """
    if config.is_unsw and config.model_type == ModelType.MULTI:
        raise ValueError(
            'UNSW + multiclass live simulation is not yet supported: the UNSW rolling-log '
            'schema has no MC_Label slot (see xseciot issue #108). Training via '
            'train_ce_multiclass works standalone; use model_type=binary with is_unsw=True, '
            'or model_type=multi with is_unsw=False, for live simulation until #108 lands.'
        )

    sig_controller = create_sig_controller(config)
```

to:

```python
def initialize_simulation_runtime(config: SimulationConfig) -> SimulationRuntime:
    """
    Build and return a fully initialized simulation runtime.

    Args:
        config: Simulation configuration.

    Returns:
        Fully initialized runtime state.
    """
    sig_controller = create_sig_controller(config)
```

- [ ] **Step 3: Commit**

```bash
git add src/firce/runtime/bootstrap.py
git commit -m "fix: derive MC_Label from Attack in load_training_frame for UNSW multiclass; remove #92's UNSW+multi block"
```

(Verification for this task happens via Task 3's integration test — `load_training_frame` doesn't have direct unit tests today; it's exercised through `initialize_simulation_runtime`, matching the existing test file's convention.)

---

### Task 3: `_append_unsw_row` model-type-aware coercion + integration tests

**Files:**
- Modify: `src/firce/runtime/inference.py`
- Modify: `tests/test_config_validation.py`
- Modify: `tests/test_inference.py`

- [ ] **Step 1: Fix `_append_unsw_row`**

In `src/firce/runtime/inference.py`:
- Change the import: `from firce.runtime.constants import (DROP_COLS, FULL_DROP_COLS, PRED_THRESHOLD, ROLLING_COLS, _label_column)` to add `get_unsw_rolling_columns` (keep `ROLLING_COLS` — still used by `_prepare_chunk`'s `ROLLING_COLS[:-1]` feature-only slice, which is model-type-agnostic and doesn't need changing).
- In `_append_unsw_row`, change:

```python
    allowed = ROLLING_COLS
    logger_obj = runtime.rolling
```

to:

```python
    allowed = get_unsw_rolling_columns(runtime.config.model_type)
    label_col = _label_column(runtime.config.model_type)
    logger_obj = runtime.rolling
```

- Change:

```python
    pruned = series.reindex(index=allowed)
    pruned['BinLabel'] = _coerce_binary_label(pruned['BinLabel'])

    assert len(pruned) == len(allowed), f'[rolling] row width mismatch: {len(pruned)} vs expected {len(allowed)}'
```

to:

```python
    pruned = series.reindex(index=allowed)
    if runtime.config.model_type == ModelType.BINARY:
        pruned[label_col] = _coerce_binary_label(pruned[label_col])

    assert len(pruned) == len(allowed), f'[rolling] row width mismatch: {len(pruned)} vs expected {len(allowed)}'
```

- Update the docstring's `Raises:` line from `ValueError: If BinLabel is invalid.` to `ValueError: If BinLabel is invalid (binary runs only).`

- [ ] **Step 2: Write direct `_append_unsw_row` tests**

Add to `tests/test_inference.py`. First check whether `_append_unsw_row` and `ModelType` are already imported — if not, add them to the existing import block. Then add:

```python
def test_append_unsw_row_multiclass_writes_mc_label(tmp_path, monkeypatch):
    from firce.runtime.inference import _append_unsw_row
    from firce.runtime.constants import get_unsw_rolling_columns

    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path, model_type=ModelType.MULTI, is_unsw=True)
    columns = get_unsw_rolling_columns(ModelType.MULTI)
    rolling = CircularDequeLogger(None, max_rows=200, columns=columns)
    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=rolling,
        scaler=None,
        pca=None,
        model=None,
        label_encoder=None,
        monitor=None,
        train_df=pd.DataFrame(),
    )

    row_values = {col: 1.0 for col in columns if col != 'MC_Label'}
    row_values['MC_Label'] = 'DoS'
    row_to_log = pd.DataFrame([row_values])

    _append_unsw_row(runtime, row_to_log)

    appended = rolling.to_dataframe()
    assert len(appended) == 1
    assert appended['MC_Label'].iloc[0] == 'DoS'


def test_append_unsw_row_binary_still_coerces_label(tmp_path, monkeypatch):
    from firce.runtime.inference import _append_unsw_row
    from firce.runtime.constants import get_unsw_rolling_columns

    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path, model_type=ModelType.BINARY, is_unsw=True)
    columns = get_unsw_rolling_columns(ModelType.BINARY)
    rolling = CircularDequeLogger(None, max_rows=200, columns=columns)
    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=rolling,
        scaler=None,
        pca=None,
        model=None,
        label_encoder=None,
        monitor=None,
        train_df=pd.DataFrame(),
    )

    row_values = {col: 1.0 for col in columns if col != 'BinLabel'}
    row_values['BinLabel'] = 'Attack'
    row_to_log = pd.DataFrame([row_values])

    _append_unsw_row(runtime, row_to_log)

    appended = rolling.to_dataframe()
    assert len(appended) == 1
    assert appended['BinLabel'].iloc[0] == 1
```

Check `tests/test_inference.py`'s existing `_make_config` helper accepts `is_unsw` as an override (it should, matching the file's established `**overrides` pattern) before assuming this works verbatim.

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/test_inference.py -v -k append_unsw_row`
Expected: both pass.

- [ ] **Step 4: Write the `initialize_simulation_runtime` integration test**

Add to `tests/test_config_validation.py`, replacing `test_initialize_simulation_runtime_rejects_unsw_multiclass`:

```python
def _make_unsw_raw_csv(tmp_path, n=60, seed=0, dirname='UNSW_DS'):
    rng = np.random.default_rng(seed)
    ds_dir = tmp_path / dirname
    ds_dir.mkdir(exist_ok=True)
    csv_path = ds_dir / 'flows.csv'
    labels = np.array(['Benign', 'DoS', 'Reconnaissance'])
    idx = rng.integers(0, 3, size=n)
    base_ms = 1_600_000_000_000
    df = pd.DataFrame({
        'IPV4_SRC_ADDR': ['10.0.0.1'] * n,
        'IPV4_DST_ADDR': ['10.0.0.2'] * n,
        'L4_SRC_PORT': rng.integers(1024, 65535, size=n),
        'L4_DST_PORT': rng.integers(1, 1024, size=n),
        'PROTOCOL': rng.integers(0, 2, size=n),
        'FLOW_START_MILLISECONDS': base_ms + np.arange(n) * 1000,
        'FLOW_END_MILLISECONDS': base_ms + np.arange(n) * 1000 + 500,
        'FLOW_DURATION_MILLISECONDS': rng.random(n) * 100,
        'IN_PKTS': rng.integers(1, 50, size=n),
        'OUT_PKTS': rng.integers(1, 50, size=n),
        'IN_BYTES': rng.random(n) * 1000,
        'OUT_BYTES': rng.random(n) * 1000,
        'SRC_TO_DST_IAT_MIN': rng.random(n) * 10,
        'SRC_TO_DST_IAT_MAX': rng.random(n) * 10,
        'SRC_TO_DST_IAT_AVG': rng.random(n) * 10,
        'SRC_TO_DST_IAT_STDDEV': rng.random(n) * 10,
        'DST_TO_SRC_IAT_MIN': rng.random(n) * 10,
        'DST_TO_SRC_IAT_MAX': rng.random(n) * 10,
        'DST_TO_SRC_IAT_AVG': rng.random(n) * 10,
        'DST_TO_SRC_IAT_STDDEV': rng.random(n) * 10,
        'Label': (idx != 0).astype(int),
        'Attack': labels[idx],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def test_initialize_simulation_runtime_unsw_multiclass_seeds_mc_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_unsw_raw_csv(tmp_path)
    config = _make_config(
        tmp_path,
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        aggregated_path=csv_path,
        flows_path=csv_path,
        is_unsw=True,
        monitor_type=MonitorType.NONE,
        use_circular_logger=True,
    )

    runtime = initialize_simulation_runtime(config)

    assert 'MC_Label' in runtime.rolling.columns
    assert 'BinLabel' not in runtime.rolling.columns
    seeded = runtime.rolling.to_dataframe()
    assert len(seeded) == 60
    assert set(seeded['MC_Label'].unique()) == {'Benign', 'DoS', 'Reconnaissance'}
    assert runtime.label_encoder is not None
    assert sorted(runtime.label_encoder.classes_.tolist()) == ['Benign', 'DoS', 'Reconnaissance']


def test_initialize_simulation_runtime_unsw_binary_still_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_unsw_raw_csv(tmp_path)
    config = _make_config(
        tmp_path,
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        aggregated_path=csv_path,
        flows_path=csv_path,
        is_unsw=True,
        monitor_type=MonitorType.NONE,
        use_circular_logger=True,
    )

    runtime = initialize_simulation_runtime(config)

    assert 'BinLabel' in runtime.rolling.columns
    assert 'MC_Label' not in runtime.rolling.columns
    seeded = runtime.rolling.to_dataframe()
    assert len(seeded) == 60
    assert set(seeded['BinLabel'].unique()) <= {0, 1}
```

- [ ] **Step 5: Run it**

Run: `uv run pytest tests/test_config_validation.py -v`
Expected: all pass — this was manually smoke-tested during this plan's design with this exact fixture shape and confirmed to train, seed 60 rows with correct `MC_Label`/`BinLabel` values, and load the label encoder correctly for multiclass.

- [ ] **Step 6: Commit**

```bash
git add src/firce/runtime/inference.py tests/test_inference.py tests/test_config_validation.py
git commit -m "fix: make _append_unsw_row model-type-aware; add UNSW multiclass integration tests"
```

---

### Task 4: Lint, format, and final regression

**Files:**
- All files modified in Tasks 1-3 (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any issues**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
```

If issues appear, run `uv run ruff check --fix .` and `uv run ruff format .`, then re-run the full suite.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures beyond the pre-existing/expected skips (xgboost/tensorflow/cade not installed in lean group, unrelated MC rolling-logger skips).

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format UNSW multiclass rolling schema fix"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** #108 asks to either add a parallel multiclass rolling schema and branch on `model_type`, or confirm the combination isn't needed and document it as unsupported. This plan takes the first option, following the exact pattern the rest of the multiclass roadmap already established (`_label_column`, swap-in-place list comprehensions). All three functions in the actual data-flow chain (`get_rolling_columns`, `load_training_frame`, `_append_unsw_row`) are fixed, not just the one named in the issue title — `load_training_frame`'s gap was found during this plan's design via a real smoke test, not assumed from the issue text alone.

**Placeholder scan:** No TBD/TODO markers; every step has complete code, validated by a real manual run during design (`initialize_simulation_runtime` for both UNSW+multiclass and UNSW+binary, plus a direct `process_chunk` streaming append) before being written into the plan.

**Type consistency:** `get_unsw_rolling_columns(model_type: ModelType) -> list[str]` matches `_label_column`'s existing signature convention. Test fixtures (`_make_unsw_raw_csv`) use the validated raw NetFlow-v3 column set exactly as confirmed working during design.

**Explicitly out of scope:** #121 (`_prepare_chunk`'s hardcoded `clean_data(chunk, False)`) — a separate, pre-existing, binary-equally-affecting bug found during this plan's design, filed independently rather than fixed here to keep this PR focused on #108's actual scope.
