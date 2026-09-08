# Wire Multiclass into retraining.py (#89) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `src/firce/runtime/retraining.py` hardcodes the binary retraining path everywhere — label column (`BinLabel`), artifact filenames (`scaler_binary.pkl`, `feedforward_model_binary.pt`, `{variant}_model_binary.pkl`), and unconditionally calls `train_ce_binary`. Wire in `model_type`-aware branching so multiclass retraining actually works during a live/streaming run.

**Architecture:** The file has two retrain implementations: a standalone `retrain(...)` function (loose params) and `retrain_runtime(runtime: SimulationRuntime)` (the newer, actually-used API — confirmed via grep, `retrain(` has zero callers anywhere in `src/`, only `retrain_runtime` is imported, by `inference.py`). The standalone `retrain()` is dead code and gets deleted rather than fixed twice. `retrain_runtime` and its three helpers (`_load_retraining_frame`, `_prune_unsw_retraining_frame`, `_load_retrained_artifacts`, `_fit_monitor_on_retrained_data`) get `model_type` branching mirroring `bootstrap.py`'s pattern from #88.

This requires extending `train_ce_multiclass` (`ce_model_training.py`, from #87) with the same `df_log: pd.DataFrame | None = None` retraining parameter `train_ce_binary` already has — #87 explicitly deferred this ("wiring into retraining.py is #89"), and retraining fundamentally can't work without it (there's no other way to train on the in-memory rolling window instead of re-reading a CSV from disk).

**Tech Stack:** Python 3.11+, `torch`, `pandas`, `pytest` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- `xgboost` is intentionally excluded from CI's `torch` dependency group (see #103/#105/#106) — verify `retraining.py` still imports cleanly without xgboost after these changes (its import chain already goes through `bootstrap.py` and `fire.simulations`, both already fixed in #88; deleting `retrain()` also removes retraining.py's only use of `AdaptiveSignificanceController`, simplifying its import chain further).
- `retrain_runtime`'s only real caller is `inference.py:484` — do not modify `inference.py` here. Tracking prediction correctness for multiclass in `inference.py` is #90.
- Artifact retraining directory convention (`Model_<variant>_Retraining_<uuid>/`, stale-directory cleanup via `glob`) must exactly mirror `train_ce_binary`'s existing binary retraining path, just under `multi_class_models/` instead of `binary_models/`.

---

## File Structure

- Modify: `src/firce/ce_model_training.py` — add `df_log` parameter to `train_ce_multiclass`, mirroring `train_ce_binary`'s retraining-directory convention.
- Modify: `src/firce/runtime/retraining.py` — delete dead `retrain()`, add `_label_column` helper, fix `_load_retraining_frame`/`_prune_unsw_retraining_frame`/`retrain_runtime`/`_load_retrained_artifacts`/`_fit_monitor_on_retrained_data` for multiclass.
- Modify: `tests/test_ce_model_training.py` — tests for `train_ce_multiclass`'s new `df_log` path.
- Create: `tests/test_retraining.py` — tests for `retrain_runtime`'s multiclass path.

---

### Task 1: Add `df_log` retraining support to `train_ce_multiclass`

**Files:**
- Modify: `src/firce/ce_model_training.py`
- Test: `tests/test_ce_model_training.py`

**Interfaces:**
- Produces: `train_ce_multiclass(config, flows_csv, variant, use_pca=True, df_log: pd.DataFrame | None = None) -> Path`. When `df_log` is provided, `flows_csv` is only used to determine artifact directory naming context is *not* used at all for reading — training data comes from `df_log`, matching `train_ce_binary`'s exact convention (`df_log`'s caller already prepared the frame; no `clean_data` call happens on this path, mirroring `train_ce_binary`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ce_model_training.py`:

```python
def test_train_ce_multiclass_with_df_log_writes_retraining_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    df_log = pd.read_csv(csv_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)

    outdir = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False, df_log=df_log)

    assert outdir.name.startswith('Model_dt_Retraining_')
    assert (outdir / 'dt_model_multi.pkl').exists()
    assert (outdir / 'label_encoder_multi.pkl').exists()


def test_train_ce_multiclass_with_df_log_cleans_up_old_retraining_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    df_log = pd.read_csv(csv_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)

    first = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False, df_log=df_log)
    second = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False, df_log=df_log)

    assert not first.exists()
    assert second.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ce_model_training.py -v -k df_log`
Expected: FAIL — `train_ce_multiclass() got an unexpected keyword argument 'df_log'`

- [ ] **Step 3: Add the `df_log` parameter**

In `src/firce/ce_model_training.py`, change the `train_ce_multiclass` signature and its first block from:

```python
def train_ce_multiclass(
    config: SimulationConfig,
    flows_csv: str,
    variant: ModelVariant,
    use_pca: bool = True,
) -> Path:
    """
    Train a multiclass CE model on labeled flow data and save all artifacts.

    Mirrors `train_ce_binary`'s preprocessing (numeric feature selection,
    standard scaling, optional PCA) but resolves a categorical multiclass
    label (`MC_Label`, or `Attack` for UNSW-style datasets) instead of a
    binary one, integer-encodes it via a `LabelEncoder`, and reports
    macro-averaged metrics.

    Args:
        config: Simulation configuration (uses `is_unsw`, `seed`, `device`).
        flows_csv: Path to the CSV file containing labeled multiclass flow data.
        variant: Model architecture to use. One of "dt", "knn", "rf", "svm",
            "xgb", "feedforward".
        use_pca: If True, apply PCA to reduce feature space to 95% explained variance.

    Returns:
        The output directory artifacts were written to.

    Raises:
        ValueError: If the dataset has no usable multiclass label column,
            fewer than 2 distinct classes, or an unsupported variant is given.
    """
    logger.info(f'Training multiclass CE model with {flows_csv} dataset')
    df = clean_data(pd.read_csv(flows_csv), config.is_unsw)
    dataset = Path(flows_csv).parent.name
    outdir = Path('multi_class_models') / dataset
    outdir.mkdir(parents=True, exist_ok=True)
```

to:

```python
def train_ce_multiclass(
    config: SimulationConfig,
    flows_csv: str,
    variant: ModelVariant,
    use_pca: bool = True,
    df_log: pd.DataFrame | None = None,
) -> Path:
    """
    Train a multiclass CE model on labeled flow data and save all artifacts.

    Mirrors `train_ce_binary`'s preprocessing (numeric feature selection,
    standard scaling, optional PCA) but resolves a categorical multiclass
    label (`MC_Label`, or `Attack` for UNSW-style datasets) instead of a
    binary one, integer-encodes it via a `LabelEncoder`, and reports
    macro-averaged metrics.

    Args:
        config: Simulation configuration (uses `is_unsw`, `seed`, `device`).
        flows_csv: Path to the CSV file containing labeled multiclass flow data.
            Only used to name the output directory when `df_log` is given
            (matching `train_ce_binary`'s retraining convention) — the data
            itself comes from `df_log` in that case, not from re-reading this path.
        variant: Model architecture to use. One of "dt", "knn", "rf", "svm",
            "xgb", "feedforward".
        use_pca: If True, apply PCA to reduce feature space to 95% explained variance.
        df_log: If given, train on this in-memory dataframe instead of reading
            `flows_csv` from disk (the retraining path — mirrors `train_ce_binary`'s
            `df_log` parameter exactly, including the `Model_<variant>_Retraining_<uuid>/`
            output directory convention and stale-directory cleanup).

    Returns:
        The output directory artifacts were written to.

    Raises:
        ValueError: If the dataset has no usable multiclass label column,
            fewer than 2 distinct classes, or an unsupported variant is given.
    """
    if df_log is None:
        logger.info(f'Training multiclass CE model with {flows_csv} dataset')
        df = clean_data(pd.read_csv(flows_csv), config.is_unsw)
        dataset = Path(flows_csv).parent.name
        outdir = Path('multi_class_models') / dataset
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        df = df_log
        pattern = f'multi_class_models/Model_{variant.value}_Retraining_*'
        for path in glob.glob(pattern):
            if Path(path).is_dir():
                logger.info(f'Removing old retraining directory: {path}')
                shutil.rmtree(path)
        outdir = Path('multi_class_models') / f'Model_{variant.value}_Retraining_{shortuuid.ShortUUID().random(length=8)}'
        outdir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Output directory for multiclass model retraining artifacts: {outdir}')
```

Everything below this block (label resolution, feature prep, training, metrics, artifact saving) is unchanged — it already operates uniformly on `df` regardless of where it came from.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ce_model_training.py -v -k df_log`
Expected: 2 passed

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 6: Commit**

```bash
git add src/firce/ce_model_training.py tests/test_ce_model_training.py
git commit -m "feat: add df_log retraining support to train_ce_multiclass"
```

---

### Task 2: Delete dead `retrain()` function

**Files:**
- Modify: `src/firce/runtime/retraining.py`

**Interfaces:**
- No public API change — `retrain()` has zero callers in `src/` (verified: only `retrain_runtime` is imported, by `inference.py:21`).

- [ ] **Step 1: Verify `retrain()` truly has no callers**

Run: `grep -rn "[^_]retrain(" --include="*.py" src/ tests/`
Expected: only the `def retrain(` definition line itself and `def retrain_runtime(` — no call sites for bare `retrain(`.

- [ ] **Step 2: Delete the function and its now-unused import**

In `src/firce/runtime/retraining.py`, delete everything from `def retrain(` through the duplicate `logger = logging.getLogger(__name__)` line that immediately follows it (i.e. the entire `retrain()` function body, ending right before `def retrain_runtime(`).

Then remove the now-unused import (only used by the deleted function's `_sig_controller` parameter type hint):

```python
from firce.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController
```

- [ ] **Step 3: Verify the module still imports and the full suite still passes**

Run: `uv run python -c "import firce.runtime.retraining; print('ok')"`
Expected: `ok`

Run: `uv run pytest -q`
Expected: all existing tests still pass (nothing called the deleted function)

- [ ] **Step 4: Commit**

```bash
git add src/firce/runtime/retraining.py
git commit -m "refactor: delete dead retrain() function (retrain_runtime is the only live path)"
```

---

### Task 3: Wire multiclass into `retrain_runtime`

**Files:**
- Modify: `src/firce/runtime/retraining.py`
- Test: `tests/test_retraining.py`

**Interfaces:**
- Consumes: `train_ce_multiclass` (extended in Task 1), `FeedForwardMulticlass` (from `firce.models.feedforward_multiclass`).
- Produces: `retrain_runtime(runtime: SimulationRuntime) -> None` — unchanged signature, now trains/loads the correct model_type's artifacts.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retraining.py`:

```python
import numpy as np
import pandas as pd
import torch

from firce.runtime.retraining import retrain_runtime
from firce.runtime.sim_types import SimulationRuntime
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig
from firce.utils.perf_stats import PerformanceStats

DEVICE = torch.device('cpu')


class _StubMonitor:
    """Duck-typed drift monitor stub — decouples this test from CE/CADE's
    unverified multiclass internals (#93/#94), so it only exercises
    retrain_runtime's own model_type branching and artifact loading."""

    def __init__(self):
        self.fit_calls: list = []

    def fit(self, X, y, perf_stats):
        self.fit_calls.append((X, y))

ROLLING_COLUMNS = [
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
    'MC_Label',
]


def _make_multiclass_rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(['Benign', 'PortScan', 'XMasAttack'])
    idx = rng.integers(0, 3, size=n)
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
        'MC_Label': labels[idx],
    })


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


def test_retrain_runtime_multiclass_feedforward(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = _make_runtime(tmp_path, model_variant=ModelVariant.FEEDFORWARD)

    retrain_runtime(runtime)

    assert runtime.model is not None
    x = torch.randn(1, next(runtime.model.parameters()).shape[-1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retraining.py -v`
Expected: FAIL — `KeyError: 'BinLabel'` (or similar) from `_load_retraining_frame` hardcoding the binary label column, since this runtime is configured for `ModelType.MULTI` with an `MC_Label` column and no `BinLabel` at all.

- [ ] **Step 3: Add the `_label_column` helper and fix `_load_retraining_frame`**

Add near the top of `src/firce/runtime/retraining.py` (after the imports, before `retrain_runtime`):

```python
def _label_column(model_type: ModelType) -> str:
    """Return the label column name for the given model type."""
    return 'BinLabel' if model_type == ModelType.BINARY else 'MC_Label'
```

In `_load_retraining_frame`, change:

```python
    values = df_log['BinLabel']
    logger.debug('[pre-clean] BinLabel dtype=%s, n_rows=%d', values.dtype, len(values))
    logger.debug(
        '[pre-clean] BinLabel nunique(excl NaN)=%d, n_nan=%d',
        values.nunique(dropna=True),
        int(values.isna().sum()),
    )
    logger.debug(
        '[pre-clean] BinLabel unique values (raw): %s',
        list(pd.unique(values)),
    )
    return df_log
```

to:

```python
    label_col = _label_column(runtime.config.model_type)
    values = df_log[label_col]
    logger.debug('[pre-clean] %s dtype=%s, n_rows=%d', label_col, values.dtype, len(values))
    logger.debug(
        '[pre-clean] %s nunique(excl NaN)=%d, n_nan=%d',
        label_col,
        values.nunique(dropna=True),
        int(values.isna().sum()),
    )
    logger.debug(
        '[pre-clean] %s unique values (raw): %s',
        label_col,
        list(pd.unique(values)),
    )
    return df_log
```

- [ ] **Step 4: Fix `_prune_unsw_retraining_frame` to preserve multiclass label columns too**

Change:

```python
    to_drop = set(df_log.columns) - set(ce_columns) - {'Label', 'BinLabel'}
    return df_log.drop(columns=list(to_drop))
```

to:

```python
    to_drop = set(df_log.columns) - set(ce_columns) - {'Label', 'BinLabel', 'MC_Label', 'Attack'}
    return df_log.drop(columns=list(to_drop))
```

- [ ] **Step 5: Branch `retrain_runtime` on model type**

Add the import at the top of the file — change:

```python
from firce.ce_model_training import train_ce_binary
```

to:

```python
from firce.ce_model_training import train_ce_binary, train_ce_multiclass
from firce.models.feedforward_multiclass import FeedForwardMulticlass
```

In `retrain_runtime`, change:

```python
    model_dir = train_ce_binary(
        runtime.config,
        runtime.config.log_path.as_posix(),
        runtime.perf_stats,
        df_log,
    )
```

to:

```python
    if runtime.config.model_type == ModelType.BINARY:
        model_dir = train_ce_binary(
            runtime.config,
            runtime.config.log_path.as_posix(),
            runtime.perf_stats,
            df_log,
        )
    else:
        model_dir = train_ce_multiclass(
            runtime.config,
            runtime.config.log_path.as_posix(),
            variant=runtime.config.model_variant,
            use_pca=runtime.config.use_pca,
            df_log=df_log,
        )
```

- [ ] **Step 6: Branch `_load_retrained_artifacts` on model type**

Change:

```python
    scaler = joblib.load(model_dir / 'scaler_binary.pkl')
    pca = joblib.load(model_dir / 'pca_binary.pkl') if runtime.config.use_pca else None

    if runtime.config.model_variant.value == 'feedforward':
        logger.debug(
            'Loading Torch feedforward model from %s',
            model_dir / 'feedforward_model_binary.pt',
        )
        checkpoint = torch.load(
            model_dir / 'feedforward_model_binary.pt',
            map_location='cpu',
        )
        input_dim = int(checkpoint.get('input_dim'))
        p_drop = float(checkpoint.get('dropout', 0.3))
        state_dict = checkpoint['state_dict']

        model = FeedForwardBinary(input_dim=input_dim, p_drop=p_drop)
        model.load_state_dict(state_dict, strict=False)
        model.to(runtime.config.device)
        model.eval()
    else:
        model = joblib.load(model_dir / f'{runtime.config.model_variant.value}_model_binary.pkl')

    return scaler, pca, model
```

to:

```python
    suffix = 'binary' if runtime.config.model_type == ModelType.BINARY else 'multi'
    scaler = joblib.load(model_dir / f'scaler_{suffix}.pkl')
    pca = joblib.load(model_dir / f'pca_{suffix}.pkl') if runtime.config.use_pca else None

    if runtime.config.model_variant.value == 'feedforward':
        ckpt_path = model_dir / f'feedforward_model_{suffix}.pt'
        logger.debug('Loading Torch feedforward model from %s', ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        input_dim = int(checkpoint.get('input_dim'))
        p_drop = float(checkpoint.get('dropout', 0.3))
        state_dict = checkpoint['state_dict']

        if runtime.config.model_type == ModelType.BINARY:
            model = FeedForwardBinary(input_dim=input_dim, p_drop=p_drop)
        else:
            num_classes = int(checkpoint['num_classes'])
            model = FeedForwardMulticlass(input_dim=input_dim, num_classes=num_classes, p_drop=p_drop)
        model.load_state_dict(state_dict, strict=False)
        model.to(runtime.config.device)
        model.eval()
    else:
        model = joblib.load(model_dir / f'{runtime.config.model_variant.value}_model_{suffix}.pkl')

    return scaler, pca, model
```

- [ ] **Step 7: Fix `_fit_monitor_on_retrained_data`'s label column**

Change:

```python
    y = clean_df['BinLabel']
```

to:

```python
    label_col = _label_column(runtime.config.model_type)
    y = clean_df[label_col]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_retraining.py -v`
Expected: 2 passed

- [ ] **Step 9: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 10: Commit**

```bash
git add src/firce/runtime/retraining.py tests/test_retraining.py
git commit -m "fix: wire multiclass model_type branching into retrain_runtime"
```

---

### Task 4: Lint, format, and final regression

**Files:**
- Modify: `src/firce/ce_model_training.py`, `src/firce/runtime/retraining.py`, `tests/test_ce_model_training.py`, `tests/test_retraining.py` (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any formatting issues**

Run:
```bash
uv run ruff check src/firce/ce_model_training.py src/firce/runtime/retraining.py tests/test_ce_model_training.py tests/test_retraining.py
uv run ruff format --check src/firce/ce_model_training.py src/firce/runtime/retraining.py tests/test_ce_model_training.py tests/test_retraining.py
```

If formatting or import-sorting differs, run `uv run ruff check --fix` and `uv run ruff format` on the same file list, then re-run the full suite to confirm nothing broke.

- [ ] **Step 2: Full regression across the whole repo**

Run: `uv run ruff check .` and `uv run ruff format --check .`
Expected: clean

Run: `uv run pytest -q`
Expected: all tests pass

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format retraining.py changes"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** Issue #89 asks to (1) add `model_type` branching parallel to bootstrap's, (2) call `train_ce_multiclass` when appropriate, (3) load multiclass artifacts, (4) replace the `elif len(y.unique()) > 2` no-op with an actual multiclass retrain branch. Points 1-3 are Task 3. Point 4's no-op lived in the now-deleted dead `retrain()` function (Task 2) — `retrain_runtime`, the actually-live path, never had that line at all (it would have crashed with `KeyError` on `BinLabel` for multiclass instead, which Task 3 fixes directly).

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code.

**Explicitly out of scope (belongs to other issues):** `inference.py` (the only caller of `retrain_runtime`) is not modified — prediction-side multiclass handling is #90. CLI/config validation for variant combinations is #92. CE evaluator multiclass correctness (whether `monitor.fit()` actually produces valid drift signals for multiclass labels) is #93/#94 — this plan only ensures the *right* label column and data reach `monitor.fit()`, not that the conformal evaluator's internals are verified for K>2 classes.

**Dead code note:** `retrain()` (standalone function, distinct from `retrain_runtime`) was confirmed to have zero callers anywhere in `src/` or `tests/` before deletion.
