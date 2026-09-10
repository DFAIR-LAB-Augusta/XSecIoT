# Multiclass CLI/Config Validation + Rolling Schema Fix (#92) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Issue #92 asks to verify `--modelType multi` works for every variant (or fails loudly if not) and add validation for invalid combinations. Its stated premise — "feedforward+multi silently trains nothing" — is already fixed by #87/#88/#90 and covered by their tests. Scoping this properly surfaced two real, previously-undiscovered gaps that "verify it works for every variant" actually requires fixing first.

**What's actually broken (discovered while scoping):**

1. **The default (non-UNSW) rolling-log schema has no `MC_Label` slot at all.** `firce/runtime/constants.py::FINAL_LOG_COLUMNS` (83 columns, drives `bootstrap.py::get_rolling_columns` for every non-UNSW run) includes `BinLabel` but not `MC_Label`. A real multiclass bootstrap run would seed the rolling logger via `reindex(columns=rolling_cols)`, which silently drops the `MC_Label` column entirely — the first live retrain would then `KeyError: 'MC_Label'` in `_load_retraining_frame`. This was masked in #89/#90's tests because they built `CircularDequeLogger` directly with a hand-crafted `columns=[...]` list that happened to include `MC_Label`, bypassing the real `get_rolling_columns` path entirely. Confirmed via direct inspection: `'MC_Label' in FINAL_LOG_COLUMNS` is `False`. This is a different, broader gap than #108 (which is UNSW-specific) — it affects the *primary* DFAIR/cicflowmeter path this whole roadmap targets. Per explicit user direction, fixed here rather than deferred.
2. **UNSW + multiclass + live simulation** would still hit #108's separate, UNSW-specific rolling-schema gap (`ROLLING_COLS` has no `MC_Label` slot either, and that one isn't being fixed in this plan — it's tracked in #108). Since `SimulationConfig` is legitimately constructed with `is_unsw=True, model_type=MULTI` by #87's own tests (to test `train_ce_multiclass`'s UNSW label-mapping logic, independent of live simulation), a blanket `SimulationConfig`-level validator would break that. The guard belongs at the actual live-simulation entry point (`bootstrap.py::initialize_simulation_runtime`), not at config construction.

**Architecture:** `_label_column(model_type) -> str` currently lives in `retraining.py`, but `bootstrap.py` needs it too — and `retraining.py` already imports `SimulationRuntime` from `bootstrap.py`, so importing the other direction would be circular. Move `_label_column` to `firce/runtime/constants.py` (a dependency-free constants module both files already import from), and update `retraining.py`/`inference.py` to import it from there.

**Tech Stack:** Python 3.11+, `torch`, `pandas`, `pytest` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- `firce.ce_model_training` also defines a `FINAL_LOG_COLUMNS` constant — same name, different file, different purpose (UNSW extra-feature validation in `train_ce_binary`, not rolling-log schema). Do not touch it; this plan only modifies `firce/runtime/constants.py::FINAL_LOG_COLUMNS` usage via `get_rolling_columns`, not the constant's contents (to avoid affecting the UNSW extra-feature check).
- UNSW + multiclass live simulation remains explicitly unsupported (guarded with a clear error citing #108) — not fixed here.

---

## File Structure

- Modify: `src/firce/runtime/constants.py` — add `_label_column(model_type) -> str` (moved from `retraining.py`).
- Modify: `src/firce/runtime/retraining.py` — remove local `_label_column` definition, import from `constants.py` instead.
- Modify: `src/firce/runtime/inference.py` — import `_label_column` from `constants.py` instead of `retraining.py`.
- Modify: `src/firce/runtime/bootstrap.py` — `get_rolling_columns` swaps in the correct label column for multiclass; `initialize_simulation_runtime` guards against UNSW+multiclass.
- Create: `tests/test_config_validation.py` — tests for the UNSW+multiclass guard and full end-to-end `initialize_simulation_runtime` coverage across all variants.

---

### Task 1: Move `_label_column` to a shared, import-cycle-free location

**Files:**
- Modify: `src/firce/runtime/constants.py`
- Modify: `src/firce/runtime/retraining.py`
- Modify: `src/firce/runtime/inference.py`

**Interfaces:**
- Produces: `firce.runtime.constants._label_column(model_type: ModelType) -> str` (identical behavior to the function it replaces).
- No behavior change — this is a pure move, verified by the existing `test_retraining.py`/`test_inference.py` suites still passing unchanged.

- [ ] **Step 1: Add `_label_column` to `constants.py`**

At the top of `src/firce/runtime/constants.py`, add:

```python
from firce.utils.config import ModelType


def _label_column(model_type: ModelType) -> str:
    """Return the label column name for the given model type."""
    return 'BinLabel' if model_type == ModelType.BINARY else 'MC_Label'
```

(place this before the existing `PRED_THRESHOLD: float = 0.5` line, or anywhere before the module's other constants — order doesn't matter for a module-level function.)

- [ ] **Step 2: Remove the duplicate definition from `retraining.py` and import it instead**

In `src/firce/runtime/retraining.py`, remove:

```python
def _label_column(model_type: ModelType) -> str:
    """Return the label column name for the given model type."""
    return 'BinLabel' if model_type == ModelType.BINARY else 'MC_Label'
```

and change the import from `firce.runtime.constants`:

```python
from firce.runtime.constants import FULL_DROP_COLS
```

to:

```python
from firce.runtime.constants import FULL_DROP_COLS, _label_column
```

- [ ] **Step 3: Update `inference.py`'s import**

Change:

```python
from firce.runtime.retraining import _label_column, retrain_runtime
```

to:

```python
from firce.runtime.constants import _label_column
```

(keep the separate `from firce.runtime.retraining import retrain_runtime` import — check the exact current import block, since `_label_column` and `retrain_runtime` may be on the same line; split them into two lines, one per source module.)

- [ ] **Step 4: Verify imports and run the full suite**

Run: `uv run python -c "import firce.runtime.bootstrap, firce.runtime.retraining, firce.runtime.inference; print('ok')"`
Expected: `ok`

Run: `uv run pytest -q`
Expected: all tests pass, no new failures (this step is a pure refactor — nothing should change behaviorally)

- [ ] **Step 5: Commit**

```bash
git add src/firce/runtime/constants.py src/firce/runtime/retraining.py src/firce/runtime/inference.py
git commit -m "refactor: move _label_column to constants.py to avoid a bootstrap<->retraining import cycle"
```

---

### Task 2: Fix the default rolling-log schema to include `MC_Label`

**Files:**
- Modify: `src/firce/runtime/bootstrap.py`
- Test: `tests/test_config_validation.py`

**Interfaces:**
- `get_rolling_columns(config: SimulationConfig) -> list[str]` — for non-UNSW multiclass configs, returns the schema with `MC_Label` in place of `BinLabel`; binary and UNSW behavior unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_validation.py`:

```python
import numpy as np
import pandas as pd
import torch

from firce.runtime.bootstrap import get_rolling_columns
from firce.utils.config import CEType, ModelType, ModelVariant, MonitorType, SimulationConfig

DEVICE = torch.device('cpu')


def _make_config(tmp_path, **overrides):
    dummy = tmp_path / 'dummy.csv'
    dummy.write_text('a\n1\n')
    defaults = dict(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=dummy,
        flows_path=dummy,
        is_unsw=False,
        seed=0,
        device=DEVICE,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_get_rolling_columns_multiclass_includes_mc_label(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.MULTI)
    columns = get_rolling_columns(config)

    assert 'MC_Label' in columns
    assert 'BinLabel' not in columns


def test_get_rolling_columns_binary_still_includes_bin_label(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)
    columns = get_rolling_columns(config)

    assert 'BinLabel' in columns
    assert 'MC_Label' not in columns
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_config_validation.py -v`
Expected: FAIL — `test_get_rolling_columns_multiclass_includes_mc_label` (`'MC_Label' not in columns`, since `FINAL_LOG_COLUMNS` never included it)

- [ ] **Step 3: Fix `get_rolling_columns`**

In `src/firce/runtime/bootstrap.py`, change:

```python
def get_rolling_columns(config: SimulationConfig) -> list[str]:
    """
    Get the rolling logger schema for the given configuration.

    Args:
        config: Simulation configuration.

    Returns:
        Rolling logger column list.
    """
    if config.is_unsw:
        return ROLLING_COLS.copy()

    drop_before_seed = set(get_seed_drop_columns())
    return [col for col in FINAL_LOG_COLUMNS if col not in drop_before_seed]
```

to:

```python
def get_rolling_columns(config: SimulationConfig) -> list[str]:
    """
    Get the rolling logger schema for the given configuration.

    Args:
        config: Simulation configuration.

    Returns:
        Rolling logger column list.
    """
    if config.is_unsw:
        return ROLLING_COLS.copy()

    drop_before_seed = set(get_seed_drop_columns())
    label_col = _label_column(config.model_type)
    return [label_col if col == 'BinLabel' else col for col in FINAL_LOG_COLUMNS if col not in drop_before_seed]
```

Add `_label_column` to the existing `from firce.runtime.constants import ...` import in `bootstrap.py` (it now lives there per Task 1).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_validation.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 6: Commit**

```bash
git add src/firce/runtime/bootstrap.py tests/test_config_validation.py
git commit -m "fix: include MC_Label in the default rolling-log schema for multiclass runs"
```

---

### Task 3: Guard UNSW + multiclass at the live-simulation entry point

**Files:**
- Modify: `src/firce/runtime/bootstrap.py`
- Test: `tests/test_config_validation.py`

**Interfaces:**
- `initialize_simulation_runtime(config: SimulationConfig) -> SimulationRuntime` — raises `ValueError` immediately for `is_unsw=True, model_type=MULTI` configs, before doing any work. `SimulationConfig` itself is unaffected — it can still be constructed with this combination (needed by #87's own tests, which only exercise `train_ce_multiclass` directly, not live simulation).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config_validation.py`:

```python
import pytest

from firce.runtime.bootstrap import initialize_simulation_runtime


def test_initialize_simulation_runtime_rejects_unsw_multiclass(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.MULTI, is_unsw=True)

    with pytest.raises(ValueError, match='UNSW \\+ multiclass'):
        initialize_simulation_runtime(config)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_config_validation.py -v -k unsw_multiclass`
Expected: FAIL — no `ValueError` is raised; it instead proceeds and fails later (or differently) trying to actually train/load, since nothing currently blocks this combination upfront.

- [ ] **Step 3: Add the guard**

In `src/firce/runtime/bootstrap.py`, change the start of `initialize_simulation_runtime`:

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

to:

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_validation.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 6: Commit**

```bash
git add src/firce/runtime/bootstrap.py tests/test_config_validation.py
git commit -m "fix: reject UNSW + multiclass at the live-simulation entry point with a clear error"
```

---

### Task 4: End-to-end variant coverage for `initialize_simulation_runtime`

**Files:**
- Test: `tests/test_config_validation.py`

**Interfaces:**
- No production code change — this closes out #92's literal ask ("verify `--modelType multi` works correctly ... for every model variant") with a real integration test exercising the *actual* `get_rolling_columns`/`create_rolling_logger`/`seed_rolling_logger` path this time (not a hand-crafted fixture), for every classical variant plus feedforward.

- [ ] **Step 1: Write the test**

Append to `tests/test_config_validation.py`:

```python
def _make_multiclass_csv(tmp_path, n=60, seed=0, dirname='DS'):
    rng = np.random.default_rng(seed)
    ds_dir = tmp_path / dirname
    ds_dir.mkdir(exist_ok=True)
    csv_path = ds_dir / 'flows.csv'
    labels = np.array(['Benign', 'PortScan', 'XMasAttack'])
    idx = rng.integers(0, 3, size=n)
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
        'MC_Label': labels[idx],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.parametrize(
    'model_variant', [ModelVariant.DT, ModelVariant.KNN, ModelVariant.RF, ModelVariant.SVM, ModelVariant.FEEDFORWARD]
)
def test_initialize_simulation_runtime_multiclass_all_variants(tmp_path, monkeypatch, model_variant):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(
        tmp_path,
        model_variant=model_variant,
        aggregated_path=csv_path,
        flows_path=csv_path,
        monitor_type=MonitorType.NONE,
    )

    runtime = initialize_simulation_runtime(config)

    assert runtime.model is not None
    assert runtime.scaler is not None
    assert runtime.label_encoder is not None
    assert 'MC_Label' in runtime.rolling.columns
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_validation.py -v`
Expected: 8 passed (2 from Task 2 + 1 from Task 3 + 5 parametrized here)

- [ ] **Step 3: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 4: Commit**

```bash
git add tests/test_config_validation.py
git commit -m "test: end-to-end initialize_simulation_runtime coverage for all multiclass variants"
```

---

### Task 5: Lint, format, and final regression

**Files:**
- All files touched in Tasks 1-4 (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any issues**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
```

If issues appear, run `uv run ruff check --fix .` and `uv run ruff format .`, then re-run the full suite.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format multiclass config validation changes"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** Issue #92 asks to verify `--modelType multi` works for every variant and add validation. Task 4 delivers the verification (real end-to-end test, not the hand-crafted-fixture version from earlier PRs). Tasks 2-3 deliver the validation half — but what actually needed validating turned out to be a rolling-schema bug (Task 2, fixed per explicit user direction) and a genuinely-unsupported combination (Task 3, guarded with a clear error rather than silently breaking).

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code.

**Type consistency:** `_label_column`'s signature and behavior are unchanged by the move (Task 1) — every call site (`retraining.py`, `inference.py`, now `bootstrap.py`) uses it identically.

**Explicitly out of scope:** #108 (UNSW rolling schema) remains unfixed — Task 3 guards against it with a clear error rather than fixing it. CE evaluator multiclass correctness (#93), CADE compatibility (#94), and `use_mlp`/`use_svm` CE-model flags (orthogonal to `model_variant`/`model_type`, #93 territory) are untouched.
