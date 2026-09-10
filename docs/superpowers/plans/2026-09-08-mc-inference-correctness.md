# Multiclass Prediction Correctness Tracking (#90) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `src/firce/runtime/inference.py::_record_prediction_outcome` tracks correctness (`correct_log`, incorrect-prediction logging) for binary but not multiclass. Add parity for multiclass. Issue #90's text describes this as the whole task, but scoping it surfaced that the actual multiclass *prediction* path is broken well before correctness tracking ever runs.

**Architecture (what's actually broken, discovered while scoping):**

1. `fire/simulations.py::load_simulation_objects` has a half-implemented multiclass branch with filename bugs: it looks for `decision_tree_multi.pkl`/`feedforward_multi.pt`, but #87's `train_ce_multiclass` actually saves `dt_model_multi.pkl`/`feedforward_model_multi.pt` (the binary-symmetric naming #87 was explicitly asked to use). A multiclass bootstrap load would `FileNotFoundError` today. It also rebuilds `FeedForwardBinary` unconditionally even for `model_type='multi'`.
2. Nothing loads the `label_encoder_multi.pkl` artifact #87 produces into the runtime at all — not at bootstrap time, not after a retrain. Without it, an integer class prediction from a multiclass model can't be turned back into the string label (`'Benign'`, `'PortScan'`, ...) the rest of the pipeline (rolling log, `MC_Label` column, retraining) expects.
3. `inference.py::predict_row` has a `FeedForwardBinary`-specific branch (sigmoid + threshold) but no `FeedForwardMulticlass` branch (softmax + argmax) — predicting with a multiclass feedforward model during simulation crashes with `TypeError: Unsupported model type for prediction` today, since it isn't `FeedForwardBinary` and multiclass torch modules don't have a bare `.predict()`.
4. `inference.py::_prepare_chunk` only ever extracts ground truth from a `BinLabel` column — for multiclass, ground truth is always `None` today (`MC_Label` is never checked).
5. `inference.py::_record_prediction_outcome`'s multiclass branch is the literal issue: `else: row_to_log['Label'] = prediction` — wrong column name (should be `MC_Label`, per the same convention #87/#88/#89 established) and no correctness tracking.

Fixing 5 without 1-4 would just move the crash around. This plan fixes all five, in dependency order. **Explicitly out of scope:** the UNSW rolling-log schema (`ROLLING_COLS`) has no `MC_Label` slot at all — `_append_unsw_row` would still break for a UNSW+multiclass run after this plan. Filed as #108, not fixed here (it's a separate structural gap, not part of the general/non-UNSW prediction path this issue is about).

**Tech Stack:** Python 3.11+, `torch`, `pandas`, `pytest` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- `load_simulation_objects` keeps its existing 3-tuple return signature (`scaler, pca, model`) — it has three other internal call sites inside `fire/simulations.py`'s own batch pipeline (unrelated to `firce`'s streaming runtime); changing its signature would ripple into those. The label encoder is loaded as a separate, additive step in `bootstrap.py` instead.
- `SimulationRuntime` gains a new `label_encoder: Any | None = None` field — every existing construction call site uses keyword arguments, so this is backward compatible.
- This plan covers the non-UNSW (general/cicflowmeter-schema) prediction path only. UNSW+multiclass rolling-log support is #108.

---

## File Structure

- Modify: `src/firce/runtime/inference.py` — (prerequisite, Task 0) lazy xgboost import; (Task 4) multiclass prediction/correctness fixes.
- Modify: `src/fire/simulations.py` — fix `load_simulation_objects`'s multiclass artifact filenames and `FeedForwardMulticlass` loading.
- Modify: `src/firce/runtime/sim_types.py` — add `label_encoder` field to `SimulationRuntime`.
- Modify: `src/firce/runtime/bootstrap.py` — load the label encoder at bootstrap time.
- Modify: `src/firce/runtime/retraining.py` — keep the label encoder fresh after a retrain.
- Modify: `src/firce/runtime/inference.py` — fix `predict_row`'s multiclass feedforward branch, `_prepare_chunk`'s ground-truth extraction, and `_record_prediction_outcome`'s correctness tracking.
- Create: `tests/test_inference.py` — end-to-end tests covering a real trained multiclass model through prediction and correctness tracking.

---

### Task 0: Lazy xgboost import in `inference.py`

**Files:**
- Modify: `src/firce/runtime/inference.py`

**Interfaces:**
- No public API change. `inference.py` does `import xgboost as xgb` at module level (used in `predict_row`'s type hint and one genuine runtime `isinstance`/`xgb.DMatrix` call, guarded by `config.model_variant == ModelVariant.XGB`) — this blocks importing the module at all without xgboost installed, discovered immediately when setting up this plan's test environment (CI's lean `torch` group excludes xgboost, per #103/#105-#107).

- [ ] **Step 1: Make the import lazy**

Change:

```python
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from sklearn.base import ClassifierMixin
```

to:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from sklearn.base import ClassifierMixin
```

and add, after the last `from firce...`/`from fire...` import (before `logger = logging.getLogger(__name__)`):

```python
if TYPE_CHECKING:
    import xgboost as xgb
```

Then change the one genuine runtime usage inside `predict_row`:

```python
    if config.model_variant == ModelVariant.XGB and isinstance(model, xgb.Booster):
        fnames = [f'f_{i}' for i in range(X_p.shape[1])]
        dtest = xgb.DMatrix(X_p, feature_names=fnames)
```

to:

```python
    if config.model_variant == ModelVariant.XGB:
        import xgboost as xgb

    if config.model_variant == ModelVariant.XGB and isinstance(model, xgb.Booster):
        fnames = [f'f_{i}' for i in range(X_p.shape[1])]
        dtest = xgb.DMatrix(X_p, feature_names=fnames)
```

(the two-`if` shape keeps the diff minimal; ruff/mypy won't complain about the local import shadowing the `TYPE_CHECKING`-only name since they're in different scopes.)

- [ ] **Step 2: Verify the module imports without xgboost and the full suite still passes**

Run: `uv run python -c "import firce.runtime.inference; print('ok')"`
Expected: `ok`

Run: `uv run pytest -q`
Expected: all existing tests still pass

- [ ] **Step 3: Commit**

```bash
git add src/firce/runtime/inference.py
git commit -m "refactor: make inference.py importable without xgboost installed"
```

---

### Task 1: Fix `load_simulation_objects`'s multiclass artifact loading

**Files:**
- Modify: `src/fire/simulations.py`
- Test: `tests/test_ce_model_training.py` (reuse existing fixtures indirectly via a focused new test)

**Interfaces:**
- `load_simulation_objects(aggregated_file, model_type, model_variant, use_pca=True) -> Tuple[StandardScaler, Optional[PCA], ClassifierMixin | xgb.Booster | FeedForwardBinary | FeedForwardMulticlass]` — same signature, same 3-tuple return, fixed multiclass filenames and feedforward model class.

- [ ] **Step 1: Write the failing test**

Create `tests/test_simulations_load.py`:

```python
import numpy as np
import pandas as pd
import torch

from fire.simulations import load_simulation_objects
from firce.ce_model_training import train_ce_multiclass
from firce.models.feedforward_multiclass import FeedForwardMulticlass
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig

DEVICE = torch.device('cpu')


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


def test_load_simulation_objects_multiclass_dt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)
    train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False)

    scaler, pca, model = load_simulation_objects(str(csv_path), 'multi', 'dt', use_pca=False)

    assert scaler is not None
    assert pca is None
    assert hasattr(model, 'predict')


def test_load_simulation_objects_multiclass_feedforward(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.FEEDFORWARD)
    train_ce_multiclass(config, str(csv_path), variant=ModelVariant.FEEDFORWARD, use_pca=False)

    scaler, pca, model = load_simulation_objects(str(csv_path), 'multi', 'feedforward', use_pca=False)

    assert isinstance(model, FeedForwardMulticlass)
    out = model(torch.randn(2, model.trunk[0].in_features))
    assert out.shape == (2, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_simulations_load.py -v`
Expected: FAIL — `test_load_simulation_objects_multiclass_dt` with `FileNotFoundError` (looking for `decision_tree_multi.pkl`, which doesn't exist — `dt_model_multi.pkl` does); `test_load_simulation_objects_multiclass_feedforward` similarly with `FileNotFoundError` (looking for `feedforward_multi.pt` instead of `feedforward_model_multi.pt`).

- [ ] **Step 3: Fix `load_simulation_objects`**

In `src/fire/simulations.py`, add the import (near the other `firce.models` import):

```python
from firce.models.feedforward_multiclass import FeedForwardMulticlass
```

Change:

```python
    else:
        base = os.path.join(os.getcwd(), 'multi_class_models', dataset_name)
        scaler_file = os.path.join(base, 'scaler_multi.pkl')
        pca_file = os.path.join(base, 'pca_multi.pkl')
        mapping = {
            'dt': 'decision_tree_multi.pkl',
            'rf': 'random_forest_multi.pkl',
            'feedforward': 'feedforward_multi.pt',
            'knn': 'knearest_multi.pkl',
            'svm': 'svm_multi.pkl',
            'xgb': 'xgboost_multi.pkl',
        }
        if model_variant not in mapping:
            raise ValueError(f'Unsupported multi-class variant: {model_variant}')
        model_file = os.path.join(base, mapping[model_variant])
```

to:

```python
    else:
        base = os.path.join(os.getcwd(), 'multi_class_models', dataset_name)
        scaler_file = os.path.join(base, 'scaler_multi.pkl')
        pca_file = os.path.join(base, 'pca_multi.pkl')
        if model_variant != 'feedforward':
            model_file = os.path.join(base, f'{model_variant}_model_multi.pkl')
        else:
            model_file = os.path.join(base, 'feedforward_model_multi.pt')
```

Then change:

```python
    logger.debug(f'Rebuilding FeedForwardBinary(input_dim={input_dim}, p_drop={p_drop})')
    torch_model = FeedForwardBinary(input_dim=input_dim, p_drop=p_drop)
    missing, unexpected = torch_model.load_state_dict(state_dict, strict=False)
```

to:

```python
    if model_type == 'binary':
        logger.debug(f'Rebuilding FeedForwardBinary(input_dim={input_dim}, p_drop={p_drop})')
        torch_model = FeedForwardBinary(input_dim=input_dim, p_drop=p_drop)
    else:
        num_classes = int(ckpt['num_classes'])
        logger.debug(
            f'Rebuilding FeedForwardMulticlass(input_dim={input_dim}, num_classes={num_classes}, p_drop={p_drop})'
        )
        torch_model = FeedForwardMulticlass(input_dim=input_dim, num_classes=num_classes, p_drop=p_drop)
    missing, unexpected = torch_model.load_state_dict(state_dict, strict=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_simulations_load.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 6: Commit**

```bash
git add src/fire/simulations.py tests/test_simulations_load.py
git commit -m "fix: correct multiclass artifact filenames and feedforward class in load_simulation_objects"
```

---

### Task 2: Thread the label encoder into `SimulationRuntime`

**Files:**
- Modify: `src/firce/runtime/sim_types.py`
- Modify: `src/firce/runtime/bootstrap.py`

**Interfaces:**
- Produces: `SimulationRuntime.label_encoder: Any | None = None` (new field, defaults `None`, backward compatible with all keyword-arg construction sites). `bootstrap.py::load_label_encoder(config: SimulationConfig) -> Any | None`.

- [ ] **Step 1: Add the field to `SimulationRuntime`**

In `src/firce/runtime/sim_types.py`, change:

```python
from typing import TYPE_CHECKING
```

to:

```python
from typing import TYPE_CHECKING, Any
```

Then change the dataclass body from:

```python
@dataclass
class SimulationRuntime:
    """Mutable runtime state for a simulation run."""

    config: SimulationConfig
    perf_stats: PerformanceStats
    sig_controller: AdaptiveSignificanceController | None
    rolling: RollingCSV | CircularDequeLogger
    scaler: StandardScaler
    pca: PCA | None
    model: ClassifierMixin | xgb.Booster | FeedForwardBinary
    monitor: DriftMonitor | None
    train_df: pd.DataFrame
```

to:

```python
@dataclass
class SimulationRuntime:
    """Mutable runtime state for a simulation run."""

    config: SimulationConfig
    perf_stats: PerformanceStats
    sig_controller: AdaptiveSignificanceController | None
    rolling: RollingCSV | CircularDequeLogger
    scaler: StandardScaler
    pca: PCA | None
    model: ClassifierMixin | xgb.Booster | FeedForwardBinary
    monitor: DriftMonitor | None
    train_df: pd.DataFrame
    label_encoder: Any | None = None
```

- [ ] **Step 2: Load the label encoder at bootstrap time**

In `src/firce/runtime/bootstrap.py`, add near the top (with the other stdlib imports):

```python
import joblib
```

Add a new function after `load_runtime_artifacts`:

```python
def load_label_encoder(config: SimulationConfig) -> Any | None:
    """
    Load the multiclass label encoder if applicable.

    Args:
        config: Simulation configuration.

    Returns:
        Fitted LabelEncoder for multiclass runs, or None for binary runs.
    """
    if config.model_type != ModelType.MULTI:
        return None
    dataset_name = config.aggregated_path.parent.name
    encoder_path = Path('multi_class_models') / dataset_name / 'label_encoder_multi.pkl'
    return joblib.load(encoder_path)
```

(`Path` is already imported in this file via other usages — check the top-of-file import block; if `from pathlib import Path` isn't already present, add it alongside the other stdlib imports.)

In `initialize_simulation_runtime`, change:

```python
    scaler, pca, model = load_runtime_artifacts(config)
    monitor = build_runtime_monitor(
        config=config,
        train_df=train_df,
        scaler=scaler,
        pca=pca,
        model=model,
        sig_controller=sig_controller,
        perf_stats=perf_stats,
    )

    return SimulationRuntime(
        config=config,
        perf_stats=perf_stats,
        sig_controller=sig_controller,
        rolling=rolling,
        scaler=scaler,
        pca=pca,
        model=model,
        monitor=monitor,
        train_df=train_df,
    )
```

to:

```python
    scaler, pca, model = load_runtime_artifacts(config)
    label_encoder = load_label_encoder(config)
    monitor = build_runtime_monitor(
        config=config,
        train_df=train_df,
        scaler=scaler,
        pca=pca,
        model=model,
        sig_controller=sig_controller,
        perf_stats=perf_stats,
    )

    return SimulationRuntime(
        config=config,
        perf_stats=perf_stats,
        sig_controller=sig_controller,
        rolling=rolling,
        scaler=scaler,
        pca=pca,
        model=model,
        label_encoder=label_encoder,
        monitor=monitor,
        train_df=train_df,
    )
```

- [ ] **Step 3: Verify imports and run the full suite**

Run: `uv run python -c "import firce.runtime.bootstrap; print('ok')"`
Expected: `ok`

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 4: Commit**

```bash
git add src/firce/runtime/sim_types.py src/firce/runtime/bootstrap.py
git commit -m "feat: load multiclass label encoder into SimulationRuntime at bootstrap"
```

---

### Task 3: Keep the label encoder fresh after retraining

**Files:**
- Modify: `src/firce/runtime/retraining.py`

**Interfaces:**
- `_load_retrained_artifacts(runtime, model_dir) -> tuple[Any, Any, Any, Any]` — now returns `(scaler, pca, model, label_encoder)`. This is a private function with its only caller in the same file (verified via grep) — safe to change its arity.

- [ ] **Step 1: Update `_load_retrained_artifacts` and its caller**

In `src/firce/runtime/retraining.py`, change:

```python
    suffix = 'binary' if runtime.config.model_type == ModelType.BINARY else 'multi'
    scaler = joblib.load(model_dir / f'scaler_{suffix}.pkl')
    pca = joblib.load(model_dir / f'pca_{suffix}.pkl') if runtime.config.use_pca else None
```

to:

```python
    suffix = 'binary' if runtime.config.model_type == ModelType.BINARY else 'multi'
    scaler = joblib.load(model_dir / f'scaler_{suffix}.pkl')
    pca = joblib.load(model_dir / f'pca_{suffix}.pkl') if runtime.config.use_pca else None
    label_encoder = (
        joblib.load(model_dir / 'label_encoder_multi.pkl') if runtime.config.model_type == ModelType.MULTI else None
    )
```

and change the function's final line from:

```python
    return scaler, pca, model
```

to:

```python
    return scaler, pca, model, label_encoder
```

Then in `retrain_runtime`, change:

```python
    scaler, pca, model = _load_retrained_artifacts(runtime, model_dir)
    runtime.scaler = scaler
    runtime.pca = pca
    runtime.model = model
```

to:

```python
    scaler, pca, model, label_encoder = _load_retrained_artifacts(runtime, model_dir)
    runtime.scaler = scaler
    runtime.pca = pca
    runtime.model = model
    runtime.label_encoder = label_encoder
```

Also update `_load_retrained_artifacts`'s return type hint from `tuple[Any, Any, Any]` to `tuple[Any, Any, Any, Any]`.

- [ ] **Step 2: Run the retraining tests and full suite**

Run: `uv run pytest tests/test_retraining.py -v`
Expected: still 2 passed (existing tests don't assert on `label_encoder`, but must not break)

Run: `uv run pytest -q`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add src/firce/runtime/retraining.py
git commit -m "feat: refresh label encoder on runtime after multiclass retrain"
```

---

### Task 4: Fix `inference.py`'s multiclass prediction and correctness tracking

**Files:**
- Modify: `src/firce/runtime/inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- `predict_row(...) -> int` — now also handles `FeedForwardMulticlass` (softmax + argmax), returning the integer class index (same representation sklearn/xgb multiclass models' `.predict()` already returns, since #87 trained them on `LabelEncoder`-encoded integer labels).
- `_record_prediction_outcome(...)` — multiclass branch now decodes the integer prediction via `runtime.label_encoder.inverse_transform([prediction])[0]`, writes it to `row_to_log['MC_Label']`, and tracks correctness against string ground truth exactly like the binary branch does against `BinLabel`.
- `_prepare_chunk(...)` — ground truth extraction now uses `_label_column(runtime.config.model_type)` (imported from `firce.runtime.retraining`) instead of a hardcoded `'BinLabel'`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_inference.py`:

```python
import numpy as np
import pandas as pd
import pytest
import torch

from firce.ce_model_training import train_ce_multiclass
from firce.runtime.inference import _prepare_chunk, _record_prediction_outcome, predict_row
from firce.runtime.sim_types import SimulationRuntime
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig
from firce.utils.perf_stats import PerformanceStats

DEVICE = torch.device('cpu')

FLOW_COLUMNS = [
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


def _train_and_build_runtime(tmp_path, monkeypatch, model_variant=ModelVariant.DT):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    _make_multiclass_rows().to_csv(csv_path, index=False)

    config = _make_config(tmp_path, model_variant=model_variant)
    import joblib

    model_dir = train_ce_multiclass(config, str(csv_path), variant=model_variant, use_pca=False)
    scaler = joblib.load(model_dir / 'scaler_multi.pkl')
    label_encoder = joblib.load(model_dir / 'label_encoder_multi.pkl')

    if model_variant == ModelVariant.FEEDFORWARD:
        from firce.models.feedforward_multiclass import FeedForwardMulticlass

        ckpt = torch.load(model_dir / 'feedforward_model_multi.pt', map_location='cpu')
        model = FeedForwardMulticlass(input_dim=int(ckpt['input_dim']), num_classes=int(ckpt['num_classes']))
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model.eval()
    else:
        model = joblib.load(model_dir / f'{model_variant.value}_model_multi.pkl')

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=200, columns=FLOW_COLUMNS),
        scaler=scaler,
        pca=None,
        model=model,
        label_encoder=label_encoder,
        monitor=None,
        train_df=pd.DataFrame(),
    )
    return runtime, csv_path


def test_prepare_chunk_extracts_mc_label_ground_truth(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)

    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)

    assert ground_truth is not None
    assert set(ground_truth.tolist()) <= {'Benign', 'PortScan', 'XMasAttack'}
    assert 'MC_Label' not in clean_chunk.columns


@pytest.mark.parametrize('model_variant', [ModelVariant.DT, ModelVariant.FEEDFORWARD])
def test_predict_row_multiclass_returns_valid_class_index(tmp_path, monkeypatch, model_variant):
    runtime, csv_path = _train_and_build_runtime(tmp_path, monkeypatch, model_variant=model_variant)
    chunk = pd.read_csv(csv_path)
    clean_chunk, _ = _prepare_chunk(runtime, chunk)
    row = clean_chunk.iloc[[0]]

    from firce.runtime.constants import DROP_COLS, PRED_THRESHOLD

    prediction = predict_row(row, DROP_COLS, runtime.scaler, runtime.pca, runtime.config, runtime.model, PRED_THRESHOLD)

    assert prediction in range(len(runtime.label_encoder.classes_))


def test_record_prediction_outcome_multiclass_tracks_correctness(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)
    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)
    raw_row = clean_chunk.iloc[0]
    row_to_log = clean_chunk.iloc[[0]].copy()

    true_label = ground_truth.iloc[0]
    true_idx = int(runtime.label_encoder.transform([true_label])[0])

    _record_prediction_outcome(
        runtime=runtime,
        row_index=0,
        raw_row=raw_row,
        row_to_log=row_to_log,
        prediction=true_idx,
        ground_truth=ground_truth,
    )

    assert row_to_log['MC_Label'].iloc[0] == true_label
    assert runtime.perf_stats.correct_log == [True]


def test_record_prediction_outcome_multiclass_tracks_incorrect(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)
    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)
    raw_row = clean_chunk.iloc[0]
    row_to_log = clean_chunk.iloc[[0]].copy()

    true_label = ground_truth.iloc[0]
    true_idx = int(runtime.label_encoder.transform([true_label])[0])
    wrong_idx = (true_idx + 1) % len(runtime.label_encoder.classes_)

    _record_prediction_outcome(
        runtime=runtime,
        row_index=0,
        raw_row=raw_row,
        row_to_log=row_to_log,
        prediction=wrong_idx,
        ground_truth=ground_truth,
    )

    assert row_to_log['MC_Label'].iloc[0] == runtime.label_encoder.classes_[wrong_idx]
    assert runtime.perf_stats.correct_log == [False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_inference.py -v`
Expected: FAIL — `test_prepare_chunk_extracts_mc_label_ground_truth` (ground_truth is `None`, since `_prepare_chunk` only checks `BinLabel`); `test_predict_row_multiclass_returns_valid_class_index[FEEDFORWARD]` (`TypeError: Unsupported model type for prediction`, the DT variant likely already passes since `hasattr(model, 'predict')` already works generically); `test_record_prediction_outcome_multiclass_tracks_correctness`/`_incorrect` (`row_to_log['MC_Label']` KeyError, since the current code writes to `'Label'` instead).

- [ ] **Step 3: Fix `_prepare_chunk`**

In `src/firce/runtime/inference.py`, add the import:

```python
from firce.runtime.retraining import _label_column, retrain_runtime
```

(replacing the existing `from firce.runtime.retraining import retrain_runtime` line)

Change:

```python
    ground_truth = clean_chunk['BinLabel'].reset_index(drop=True) if 'BinLabel' in clean_chunk.columns else None

    if 'BinLabel' in clean_chunk.columns:
        clean_chunk = clean_chunk.drop(columns=['BinLabel'])
```

to:

```python
    label_col = _label_column(runtime.config.model_type)
    ground_truth = clean_chunk[label_col].reset_index(drop=True) if label_col in clean_chunk.columns else None

    if label_col in clean_chunk.columns:
        clean_chunk = clean_chunk.drop(columns=[label_col])
```

Note `_prepare_chunk` needs `runtime` in scope for this — it already takes `runtime: SimulationRuntime` as its first parameter, so `runtime.config.model_type` is available directly.

- [ ] **Step 4: Fix `predict_row`'s multiclass feedforward branch**

Add the import:

```python
from firce.models.feedforward_multiclass import FeedForwardMulticlass
```

Update the type hint on `predict_row` from:

```python
    model: ClassifierMixin | xgb.Booster | FeedForwardBinary,
```

to:

```python
    model: ClassifierMixin | xgb.Booster | FeedForwardBinary | FeedForwardMulticlass,
```

Then add a new branch right after the existing `FeedForwardBinary` branch:

```python
    if config.model_variant == ModelVariant.FEEDFORWARD and isinstance(model, FeedForwardBinary):
        dev = config.device
        xt = torch.from_numpy(np.asarray(X_p, dtype=np.float32)).to(dev)
        model.eval()
        with torch.no_grad():
            logits = model(xt)
            prob = torch.sigmoid(logits).flatten().item()
        logger.debug(f'[predict_row][ff] prob={prob:.6f}, thr={threshold}')
        return int(prob > threshold)
```

by adding immediately after it:

```python
    if config.model_variant == ModelVariant.FEEDFORWARD and isinstance(model, FeedForwardMulticlass):
        dev = config.device
        xt = torch.from_numpy(np.asarray(X_p, dtype=np.float32)).to(dev)
        model.eval()
        with torch.no_grad():
            logits = model(xt)
            pred_idx = int(torch.argmax(logits, dim=-1).item())
        logger.debug(f'[predict_row][ff-multi] pred_idx={pred_idx}')
        return pred_idx
```

- [ ] **Step 5: Fix `_record_prediction_outcome`'s multiclass branch**

Change:

```python
    else:
        row_to_log['Label'] = prediction
```

to:

```python
    else:
        label = (
            runtime.label_encoder.inverse_transform([prediction])[0]
            if runtime.label_encoder is not None
            else prediction
        )
        row_to_log['MC_Label'] = label
        logger.debug('Row %d prediction: %r', row_index, label)

        if ground_truth is not None:
            true_value = ground_truth.iloc[row_index]
            is_correct = label == true_value
            runtime.perf_stats.correct_log.append(is_correct)

            logger.debug(
                '[Index %d] Predicted=%s, Actual=%s',
                row_index,
                label,
                true_value,
            )
            if not is_correct:
                logger.info(
                    '[Incorrect] Predicted=%s, Actual=%s',
                    label,
                    true_value,
                )
                logger.debug('Row %d details: %s', row_index, raw_row.to_json())
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_inference.py -v`
Expected: 5 passed

- [ ] **Step 7: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 8: Commit**

```bash
git add src/firce/runtime/inference.py tests/test_inference.py
git commit -m "fix: track multiclass prediction correctness and fix predict_row/prepare_chunk for multiclass"
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
git commit -m "style: ruff format inference.py multiclass changes"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** Issue #90 literally asks for `correct_log` append + incorrect-prediction logging parity in `_record_prediction_outcome`. That's Task 4, Step 5. Tasks 1-3 are prerequisites discovered while scoping: without them, a multiclass model either fails to load at all (Task 1), can't be decoded back to its string label for comparison/logging (Tasks 2-3), or crashes on prediction before `_record_prediction_outcome` ever runs (Task 4, Steps 3-4).

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code.

**Explicitly out of scope:** UNSW+multiclass rolling-log schema support is #108, filed separately — `_append_unsw_row`'s `_coerce_binary_label(pruned['BinLabel'])` call still assumes binary for UNSW runs after this plan. CE evaluator multiclass correctness (#93) and CADE compatibility (#94) are untouched — this plan only wires the prediction/logging path, not conformal drift detection math.

**Type consistency:** `SimulationRuntime.label_encoder`, `predict_row`'s new `FeedForwardMulticlass` branch, and `_record_prediction_outcome`'s decoding logic all agree on the same contract: predictions are integer class indices (matching what sklearn/xgb `.predict()` already returns for #87-trained models), decoded to string labels only at the logging/correctness-comparison boundary via `label_encoder.inverse_transform`.
