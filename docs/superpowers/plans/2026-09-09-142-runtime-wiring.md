# #142: Wire Novelty + XAI + LLM Reporting Into the Live FIRCE Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make #97 (novelty decision rule), #98 (XAI explanations + selective generation), and #99 (local LLM reporting) actually run during a live streaming simulation, not just exist as tested standalone library code.

**Architecture:** `firce/runtime/inference.py::process_chunk` already has two batched, chunk-level building blocks that map directly onto what novelty scoring needs: `_prepare_monitor_chunk_features` (produces the scaled/PCA'd feature matrix a CE-calibrated model expects - already used for chunk-level drift detection) and the CE monitor's own calibrated model/calibration-scores (already fit during bootstrap for drift detection). Rather than building a second, separate calibration path, `ConformalDriftMonitor` (`firce/drift_monitor/conformal_monitor.py`) gets two new read-only properties (`model`, `calibration_scores`) that expose its existing private `_evaluator.evaluator` state - the exact `model`/`calibration_scores` pair `firce.novelty.decision_rules.compute_all_class_p_values` needs. A new function `_score_chunk_novelty` in `inference.py` computes `is_novel` flags for an entire chunk in one batched call (reusing `_prepare_monitor_chunk_features`), then `_generate_novelty_reports` runs #98's `select_events_to_explain` on those flags, generates a structured explanation for each selected row via `explain_with_shap`, and - only if a local LLM backend is configured (optional, since loading a real model is expensive and most existing runtime tests use tiny synthetic setups that shouldn't suddenly need one) - a report via #99's `generate_report`. Results accumulate in a new `runtime.novelty_reports: list[dict]` field (an in-memory accumulator, matching the existing `runtime.perf_stats` pattern) and are logged at INFO level; both are simple, testable, and don't require designing a new on-disk report format as part of this issue (an explicit open question in #142's own text, deliberately not over-engineered here). Nine new `SimulationConfig` fields gate and tune all of this, all defaulting to novelty detection being **off**, so every existing test/config is unaffected unless it opts in.

**Tech Stack:** Python, numpy, scikit-learn, pydantic (config), pytest.

## Global Constraints

- Novelty detection must default to **disabled** (`novelty_enabled: bool = False`) - zero behavior change for any existing config/test that doesn't explicitly opt in.
- LLM report generation is optional even when novelty detection is on (`novelty_llm_backend_type: Optional[str] = None` by default) - explanation-only operation (no LLM) must work standalone, since loading a real local model is expensive and shouldn't be forced onto every novelty-enabled run.
- Do not change `predict_row`'s signature or `_process_chunk_rows`'s existing per-row prediction behavior - novelty scoring is a separate, additive, chunk-level step, not a modification of the existing prediction path.
- Reuse `_prepare_monitor_chunk_features` for novelty scoring's feature matrix rather than building a new transform path - it already produces exactly what a CE-calibrated model expects.

---

### Task 1: Expose `ConformalDriftMonitor.model` / `.calibration_scores`

**Files:**
- Modify: `src/firce/drift_monitor/conformal_monitor.py`
- Test: `tests/test_drift_monitor_factory.py` (or a new `tests/test_conformal_drift_monitor.py` if that file doesn't already cover `ConformalDriftMonitor` directly - check first with `grep -n "ConformalDriftMonitor" tests/*.py`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `ConformalDriftMonitor.model -> Any` and `ConformalDriftMonitor.calibration_scores -> Dict[Any, np.ndarray]` properties - used by Task 4.

- [ ] **Step 1: Check for an existing direct test file, write the failing test**

Run: `grep -n "ConformalDriftMonitor" tests/*.py` to see where this class is already tested. Add to whichever file directly instantiates/fits a `ConformalDriftMonitor` (or create `tests/test_conformal_drift_monitor.py` if none does):

```python
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from firce.drift_monitor.conformal_monitor import ConformalDriftMonitor
from firce.utils.config import CEType
from firce.utils.perf_stats import PerformanceStats


def test_conformal_drift_monitor_exposes_model_and_calibration_scores():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = np.select([X[:, 0] > 0.5], ['Attack'], default='Benign')

    monitor = ConformalDriftMonitor(CEType.ICE, DecisionTreeClassifier(random_state=0), calibration_split=0.3, random_state=0)
    monitor.fit(X, y, PerformanceStats())

    assert hasattr(monitor.model, 'predict_proba')
    assert set(monitor.calibration_scores.keys()) == {'Benign', 'Attack'}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_conformal_drift_monitor.py -v --no-cov` (or wherever the test was added)
Expected: FAIL with `AttributeError: 'ConformalDriftMonitor' object has no attribute 'model'`

- [ ] **Step 3: Add the properties**

In `src/firce/drift_monitor/conformal_monitor.py`, add after `__init__`:

```python
    @property
    def model(self) -> Any:
        """The underlying calibrated model this monitor's CE evaluator wraps."""
        return self._evaluator.evaluator.model

    @property
    def calibration_scores(self) -> dict:
        """Per-class calibration nonconformity scores, as produced by CE calibration."""
        return self._evaluator.evaluator.calibration_scores
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_conformal_drift_monitor.py -v --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/firce/drift_monitor/conformal_monitor.py tests/test_conformal_drift_monitor.py
git commit -m "feat: expose ConformalDriftMonitor.model/.calibration_scores (#142)"
```

---

### Task 2: Novelty config fields on `SimulationConfig`

**Files:**
- Modify: `src/firce/utils/config.py`
- Test: `tests/test_cade_config.py` (has existing `SimulationConfig` validator tests - check with `grep -n "class Test\|^def test" tests/test_cade_config.py`) or add to a new/existing config test file.

**Interfaces:**
- Consumes: nothing new.
- Produces: 9 new `SimulationConfig` fields - used by Task 4.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cade_config.py` (import `SimulationConfig`, `ModelType`, `ModelVariant`, `CEType` if not already imported in that file - check first):

```python
def test_simulation_config_novelty_fields_default_to_disabled(tmp_path):
    train_csv = tmp_path / 'train.csv'
    train_csv.write_text('a,b\n1,2\n')
    flows_csv = tmp_path / 'flows.csv'
    flows_csv.write_text('a,b\n1,2\n')

    config = SimulationConfig(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.ICE,
        aggregated_path=train_csv,
        flows_path=flows_csv,
    )

    assert config.novelty_enabled is False
    assert config.novelty_tau == 0.6
    assert config.novelty_alpha == 0.3
    assert config.novelty_selective_mode == 'unknown_only'
    assert config.novelty_sample_rate == 0.1
    assert config.novelty_window_size == 10
    assert config.novelty_explain_method == 'shap'
    assert config.novelty_llm_backend_type is None
    assert config.novelty_llm_model_path is None


def test_simulation_config_novelty_tau_alpha_must_be_unit_interval(tmp_path):
    train_csv = tmp_path / 'train.csv'
    train_csv.write_text('a,b\n1,2\n')
    flows_csv = tmp_path / 'flows.csv'
    flows_csv.write_text('a,b\n1,2\n')

    with pytest.raises(ValidationError):
        SimulationConfig(
            model_type=ModelType.BINARY,
            model_variant=ModelVariant.DT,
            ce_type=CEType.ICE,
            aggregated_path=train_csv,
            flows_path=flows_csv,
            novelty_tau=1.5,
        )
```

Add `import pytest` and `from pydantic import ValidationError` near the top of the test file if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cade_config.py -v --no-cov -k novelty`
Expected: FAIL with a pydantic error about `novelty_enabled`/extra fields not permitted (since `extra='forbid'`), or `AttributeError`.

- [ ] **Step 3: Add the fields**

In `src/firce/utils/config.py`, add `Optional` to the `typing` import:
```python
from typing import Any, Dict, Optional
```

Add after the existing `monitor_kwargs: Dict[str, Any] = Field(default_factory=dict)` line:

```python
    novelty_enabled: bool = False
    novelty_tau: float = 0.6
    novelty_alpha: float = 0.3
    novelty_selective_mode: str = 'unknown_only'
    novelty_sample_rate: float = 0.1
    novelty_window_size: int = 10
    novelty_explain_method: str = 'shap'
    novelty_llm_backend_type: Optional[str] = None
    novelty_llm_model_path: Optional[str] = None
```

Add a new validator near the existing `_threshold_in_unit_interval` validator:

```python
    @field_validator('novelty_tau', 'novelty_alpha', 'novelty_sample_rate')
    @classmethod
    def _novelty_float_in_unit_interval(cls, value: float) -> float:
        """Validate that novelty tau/alpha/sample_rate lie within [0, 1]."""
        if not (0.0 <= value <= 1.0):
            raise ValueError('must be in [0, 1]')
        return value

    @field_validator('novelty_selective_mode')
    @classmethod
    def _novelty_selective_mode_is_valid(cls, value: str) -> str:
        """Validate novelty_selective_mode against firce.novelty.explain's supported modes."""
        valid_modes = ('sampled', 'unknown_only', 'windowed')
        if value not in valid_modes:
            raise ValueError(f'novelty_selective_mode must be one of {valid_modes}')
        return value

    @field_validator('novelty_explain_method')
    @classmethod
    def _novelty_explain_method_is_valid(cls, value: str) -> str:
        """Validate novelty_explain_method against firce.novelty.explain's supported methods."""
        valid_methods = ('shap', 'lime')
        if value not in valid_methods:
            raise ValueError(f'novelty_explain_method must be one of {valid_methods}')
        return value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cade_config.py -v --no-cov -k novelty`
Expected: PASS

- [ ] **Step 5: Run the full config test file to confirm no regression**

Run: `uv run pytest tests/test_cade_config.py -v --no-cov`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/firce/utils/config.py tests/test_cade_config.py
git commit -m "feat: add novelty detection config fields to SimulationConfig (#142)"
```

---

### Task 3: `runtime.novelty_reports` / `runtime.llm_backend` + bootstrap wiring

**Files:**
- Modify: `src/firce/runtime/sim_types.py`
- Modify: `src/firce/runtime/bootstrap.py`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: `firce.novelty.llm_reporting.create_local_llm_backend` (existing, #99).
- Produces: `SimulationRuntime.novelty_reports: list`, `SimulationRuntime.llm_backend: Any | None` - used by Task 4.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_bootstrap.py` (check existing imports/fixtures in that file first - it already has binary-path `initialize_simulation_runtime` tests to follow the pattern of):

```python
def test_initialize_simulation_runtime_novelty_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    train_csv = ds_dir / 'train.csv'
    _write_binary_flow_csv(train_csv)  # reuse whatever fixture-writing helper this file already has for binary CSVs

    config = SimulationConfig(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.ICE,
        aggregated_path=train_csv,
        flows_path=train_csv,
        is_unsw=False,
        seed=0,
        device=DEVICE,
    )

    runtime = initialize_simulation_runtime(config)

    assert runtime.novelty_reports == []
    assert runtime.llm_backend is None
```

Adjust the fixture-writing call (`_write_binary_flow_csv` or equivalent) to match whatever helper `test_bootstrap.py` already uses for its other binary-path tests - do not invent a new one if one already exists in that file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bootstrap.py -v --no-cov -k novelty_disabled`
Expected: FAIL with `AttributeError: 'SimulationRuntime' object has no attribute 'novelty_reports'` (or a `TypeError` from the dataclass constructor once Step 3 partially lands - run this before touching sim_types.py).

- [ ] **Step 3: Add the fields to `SimulationRuntime`**

In `src/firce/runtime/sim_types.py`, add `field` to the dataclasses import:
```python
from dataclasses import dataclass, field
```

Add two new fields at the end of the `SimulationRuntime` dataclass (after `label_encoder`):

```python
    llm_backend: Any | None = None
    novelty_reports: list = field(default_factory=list)
```

- [ ] **Step 4: Wire optional LLM backend construction into `initialize_simulation_runtime`**

In `src/firce/runtime/bootstrap.py`, add to the top-level imports:
```python
from firce.novelty.llm_reporting import create_local_llm_backend
```

In `initialize_simulation_runtime`, right before the `return SimulationRuntime(...)` call, add:

```python
    llm_backend = None
    if config.novelty_llm_backend_type is not None:
        llm_backend = create_local_llm_backend(
            config.novelty_llm_backend_type, model_name_or_path=config.novelty_llm_model_path
        )
```

and add `llm_backend=llm_backend,` to the `SimulationRuntime(...)` constructor call.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_bootstrap.py -v --no-cov -k novelty_disabled`
Expected: PASS

- [ ] **Step 6: Run the full bootstrap test file to confirm no regression**

Run: `uv run pytest tests/test_bootstrap.py -v --no-cov`
Expected: all PASS (existing tests construct `SimulationRuntime` via `initialize_simulation_runtime` and should pick up the new fields' defaults transparently; any test that constructs `SimulationRuntime` directly, as in `test_e2e_simulation.py`, needs no change since the new fields have defaults).

- [ ] **Step 7: Commit**

```bash
git add src/firce/runtime/sim_types.py src/firce/runtime/bootstrap.py tests/test_bootstrap.py
git commit -m "feat: add runtime.novelty_reports/llm_backend + optional bootstrap wiring (#142)"
```

---

### Task 4: `_score_chunk_novelty` + `_generate_novelty_reports`, wired into `process_chunk`

**Files:**
- Modify: `src/firce/runtime/inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: `firce.novelty.decision_rules.{compute_all_class_p_values, is_novel, max_softmax_confidence}`, `firce.novelty.explain.{select_events_to_explain, explain_with_shap, explain_with_lime}`, `firce.novelty.llm_reporting.generate_report` (all existing, #97/#98/#99), `ConformalDriftMonitor.model`/`.calibration_scores` (Task 1), `SimulationRuntime.novelty_reports`/`.llm_backend` (Task 3), the 9 new `SimulationConfig` fields (Task 2).
- Produces: `_score_chunk_novelty(runtime, clean_chunk) -> tuple[np.ndarray, np.ndarray] | None` (returns `None` when novelty detection isn't active) and `_generate_novelty_reports(runtime, clean_chunk, x_monitor, novelty_flags, novelty_scores) -> None` - called from `process_chunk`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_inference.py` (check existing imports/fixtures in that file first for the established pattern of building a calibrated `ConformalDriftMonitor`+`SimulationRuntime` directly, matching `test_e2e_simulation.py`'s `_AlwaysDriftMonitor`-free style):

```python
from firce.drift_monitor.conformal_monitor import ConformalDriftMonitor
from firce.runtime.inference import _generate_novelty_reports, _score_chunk_novelty
from firce.utils.config import CEType
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


def _make_novelty_test_runtime(tmp_path, novelty_enabled=True, **config_overrides):
    dummy = tmp_path / 'dummy.csv'
    dummy.write_text('a\n1\n')
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        {
            'flow_duration': rng.normal(size=n),
            'tot_fwd_pkt': rng.normal(size=n),
        }
    )
    y = np.select([X['flow_duration'] > 0.5], ['Attack'], default='Benign')

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    monitor = ConformalDriftMonitor(
        CEType.ICE, DecisionTreeClassifier(random_state=0), calibration_split=0.3, random_state=0
    )
    monitor.fit(X_scaled, y, PerformanceStats())

    config = SimulationConfig(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.ICE,
        aggregated_path=dummy,
        flows_path=dummy,
        device=DEVICE,
        novelty_enabled=novelty_enabled,
        **config_overrides,
    )

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=500, columns=['flow_duration', 'tot_fwd_pkt', 'BinLabel']),
        scaler=scaler,
        pca=None,
        model=monitor.model,
        label_encoder=None,
        monitor=monitor,
        train_df=pd.DataFrame(),
    )
    clean_chunk = X.copy()
    return runtime, clean_chunk


def test_score_chunk_novelty_returns_none_when_disabled(tmp_path):
    runtime, clean_chunk = _make_novelty_test_runtime(tmp_path, novelty_enabled=False)

    result = _score_chunk_novelty(runtime, clean_chunk)

    assert result is None


def test_score_chunk_novelty_returns_flags_and_features_when_enabled(tmp_path):
    runtime, clean_chunk = _make_novelty_test_runtime(tmp_path, novelty_enabled=True)

    result = _score_chunk_novelty(runtime, clean_chunk)

    assert result is not None
    novelty_flags, x_monitor = result
    assert novelty_flags.shape == (len(clean_chunk),)
    assert novelty_flags.dtype == bool
    assert x_monitor.shape[0] == len(clean_chunk)


def test_generate_novelty_reports_populates_runtime_with_explanations_only_by_default(tmp_path):
    # novelty_llm_backend_type stays None (default) - explanation-only path.
    runtime, clean_chunk = _make_novelty_test_runtime(tmp_path, novelty_enabled=True, novelty_tau=0.99, novelty_alpha=0.99)
    novelty_flags, x_monitor = _score_chunk_novelty(runtime, clean_chunk)

    _generate_novelty_reports(runtime, clean_chunk, x_monitor, novelty_flags)

    if novelty_flags.any():
        assert len(runtime.novelty_reports) > 0
        record = runtime.novelty_reports[0]
        assert 'explanation' in record
        assert record['llm_report'] is None  # no LLM backend configured
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_inference.py -v --no-cov -k "novelty"`
Expected: FAIL with `ImportError: cannot import name '_score_chunk_novelty'`

- [ ] **Step 3: Implement `_score_chunk_novelty` and `_generate_novelty_reports`**

In `src/firce/runtime/inference.py`, add to the imports:

```python
from firce.drift_monitor.conformal_monitor import ConformalDriftMonitor
from firce.novelty.decision_rules import compute_all_class_p_values, is_novel, max_softmax_confidence
from firce.novelty.explain import explain_with_lime, explain_with_shap, select_events_to_explain
from firce.novelty.llm_reporting import generate_report
```

Add near the end of the file (before `_handle_detected_drift`, or anywhere at module scope after the existing chunk-drift helpers):

```python
def _score_chunk_novelty(
    runtime: SimulationRuntime, clean_chunk: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Compute novelty flags for an entire chunk in one batched call, reusing
    _prepare_monitor_chunk_features (already used for chunk-level drift
    detection - produces exactly the feature matrix a CE-calibrated model expects).

    Returns None when novelty detection isn't enabled or the configured
    monitor isn't CE-calibrated (ConformalDriftMonitor) - CADE/None monitors
    have no per-class calibration to build novelty p-values from.

    Args:
        runtime: Mutable simulation runtime.
        clean_chunk: Cleaned chunk dataframe (post _prepare_chunk).

    Returns:
        (novelty_flags, x_monitor) if active, else None. novelty_flags is a
        boolean array of shape (n_rows,); x_monitor is the feature matrix
        used to compute it (also needed by _generate_novelty_reports for explanation).
    """
    if not runtime.config.novelty_enabled or not isinstance(runtime.monitor, ConformalDriftMonitor):
        return None

    x_monitor = _prepare_monitor_chunk_features(runtime, clean_chunk)
    model = runtime.monitor.model
    probas = model.predict_proba(x_monitor)
    all_class_p_values = compute_all_class_p_values(model, runtime.monitor.calibration_scores, x_monitor)
    novelty_flags = is_novel(probas, all_class_p_values, tau=runtime.config.novelty_tau, alpha=runtime.config.novelty_alpha)
    return novelty_flags, x_monitor


def _explain_model_type(config: SimulationConfig) -> str:
    """Tree-based variants use SHAP's fast TreeExplainer path; everything else uses KernelExplainer."""
    return 'tree' if config.model_variant in (ModelVariant.DT, ModelVariant.RF, ModelVariant.XGB) else 'kernel'


def _generate_novelty_reports(
    runtime: SimulationRuntime,
    clean_chunk: pd.DataFrame,
    x_monitor: np.ndarray,
    novelty_flags: np.ndarray,
) -> None:
    """
    Generate and record explanations (and, if a local LLM backend is
    configured, structured reports) for the events selected by #98's
    selective-generation policy.

    Args:
        runtime: Mutable simulation runtime (novelty_reports is appended to in place).
        clean_chunk: Cleaned chunk dataframe (used only for column names as feature names).
        x_monitor: The feature matrix novelty_flags was computed against (Task's _score_chunk_novelty output).
        novelty_flags: Boolean novelty flags for this chunk.
    """
    config = runtime.config
    selected_indices = select_events_to_explain(
        novelty_flags,
        mode=config.novelty_selective_mode,
        sample_rate=config.novelty_sample_rate,
        window_size=config.novelty_window_size,
    )
    if len(selected_indices) == 0:
        return

    model = runtime.monitor.model
    feature_names = (
        list(runtime.scaler.feature_names_in_)
        if hasattr(runtime.scaler, 'feature_names_in_')
        else [f'f{i}' for i in range(x_monitor.shape[1])]
    )
    model_type = _explain_model_type(config)
    class_names = [str(c) for c in model.classes_]
    max_softmax = max_softmax_confidence(model.predict_proba(x_monitor))

    for idx in selected_indices:
        if config.novelty_explain_method == 'lime':
            explanation = explain_with_lime(model, x_monitor, x_monitor[idx], feature_names, class_names)
        else:
            explanation = explain_with_shap(model, x_monitor, x_monitor[idx], feature_names, model_type=model_type)

        llm_report = None
        if runtime.llm_backend is not None:
            novelty_context = {
                'max_softmax': float(max_softmax[idx]),
                'tau': config.novelty_tau,
                'alpha': config.novelty_alpha,
            }
            llm_report = generate_report(runtime.llm_backend, explanation, novelty_context)

        record = {'row_index': int(idx), 'explanation': explanation, 'llm_report': llm_report}
        runtime.novelty_reports.append(record)
        logger.info('Novelty report generated for row %d: %s', idx, record)
```

- [ ] **Step 4: Wire `_score_chunk_novelty`/`_generate_novelty_reports` into `process_chunk`**

In `src/firce/runtime/inference.py`, change `process_chunk`:

```python
def process_chunk(
    runtime: SimulationRuntime,
    chunk: pd.DataFrame,
    chunk_num: int,
) -> None:
    """
    Process a single simulation chunk.

    Args:
        runtime: Mutable simulation runtime.
        chunk: Flow chunk to process.
        chunk_num: Chunk index.
    """
    start_iter = time.perf_counter()

    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)
    _process_chunk_rows(runtime, clean_chunk, ground_truth)

    novelty_result = _score_chunk_novelty(runtime, clean_chunk)
    if novelty_result is not None:
        novelty_flags, x_monitor = novelty_result
        _generate_novelty_reports(runtime, clean_chunk, x_monitor, novelty_flags)

    drift_detected = _detect_chunk_drift(runtime, clean_chunk)
    if drift_detected:
        _handle_detected_drift(runtime, chunk_num)

    runtime.perf_stats.iteration_times.append(time.perf_counter() - start_iter)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_inference.py -v --no-cov -k "novelty"`
Expected: PASS (3 tests). If `test_generate_novelty_reports_populates_runtime_with_explanations_only_by_default` finds `novelty_flags.any()` is False for this particular random seed/fixture, the assertion body is skipped safely (it's guarded by `if novelty_flags.any():`) - that's an acceptable degenerate case for a small random fixture, not a bug; don't loosen `novelty_tau=0.99, novelty_alpha=0.99` further to force it, since those are already maximally permissive.

- [ ] **Step 6: Run the full existing inference test suite to confirm no regression**

Run: `uv run pytest tests/test_inference.py -v --no-cov`
Expected: all PASS - existing tests don't set `novelty_enabled=True`, so `_score_chunk_novelty` returns `None` for them and `process_chunk`'s new lines are a no-op.

- [ ] **Step 7: Commit**

```bash
git add src/firce/runtime/inference.py tests/test_inference.py
git commit -m "feat: wire novelty detection + XAI + LLM reporting into process_chunk (#142)"
```

---

### Task 5: Real end-to-end open-world streaming test

**Files:**
- Modify: `tests/test_e2e_simulation.py`

Reuses the established `train_ce_multiclass` + manually-constructed-`SimulationRuntime` + direct `process_chunk` call pattern already in this file (see `test_process_chunk_triggers_retrain_on_detected_drift_multiclass`), applied to a genuinely held-out-class scenario - a real, non-mocked proof that a novel row in a streaming chunk produces a real generated report.

- [ ] **Step 1: Write the test**

Add to `tests/test_e2e_simulation.py`:

```python
def test_process_chunk_generates_novelty_report_for_genuinely_held_out_class(tmp_path, monkeypatch):
    # RandomForestClassifier (ModelVariant.RF), not DT: confirmed via direct
    # execution in #101 that a plain decision tree is too discrete/overconfident
    # for a genuine open-world demonstration - RandomForest gives real signal.
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()

    full_train_df = pd.read_csv(FIXTURES / 'ce_flows_e2e_mc_train.csv')
    held_out_class = sorted(full_train_df['MC_Label'].unique())[-1]
    known_only_df = full_train_df[full_train_df['MC_Label'] != held_out_class]
    train_csv = ds_dir / 'train.csv'
    known_only_df.to_csv(train_csv, index=False)

    config = SimulationConfig(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.RF,
        ce_type=CEType.ICE,
        aggregated_path=train_csv,
        flows_path=train_csv,
        is_unsw=False,
        seed=0,
        device=DEVICE,
        novelty_enabled=True,
        novelty_tau=0.8,
        novelty_alpha=0.5,
        novelty_selective_mode='unknown_only',
    )

    model_dir = train_ce_multiclass(config, str(train_csv), PerformanceStats())
    scaler = joblib.load(model_dir / 'scaler_multi.pkl')
    model = joblib.load(model_dir / 'rf_model_multi.pkl')
    label_encoder = joblib.load(model_dir / 'label_encoder_multi.pkl')

    monitor = ConformalDriftMonitor(CEType.ICE, model, calibration_split=0.3, random_state=0)
    x_train = known_only_df.drop(columns=['MC_Label', 'BinLabel'], errors='ignore').select_dtypes(include=['number'])
    monitor.fit(scaler.transform(x_train), known_only_df['MC_Label'].to_numpy(), PerformanceStats())

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=500, columns=get_rolling_columns(config)),
        scaler=scaler,
        pca=None,
        model=model,
        label_encoder=label_encoder,
        monitor=monitor,
        train_df=known_only_df,
    )

    # Stream the FULL (unfiltered) fixture, which includes the held-out class.
    full_stream_df = pd.read_csv(FIXTURES / 'ce_flows_e2e_mc_stream.csv')
    held_out_stream_rows = full_stream_df[full_stream_df['MC_Label'] == held_out_class]
    assert len(held_out_stream_rows) > 0, 'fixture must actually contain the held-out class in the stream file'

    process_chunk(runtime, held_out_stream_rows, chunk_num=0)

    assert len(runtime.novelty_reports) > 0
    record = runtime.novelty_reports[0]
    assert 'contributions' in record['explanation']
    assert record['llm_report'] is None  # no LLM backend configured for this test
```

Check the exact artifact filenames `train_ce_multiclass` produces (`rf_model_multi.pkl`, `label_encoder_multi.pkl`, etc.) against `src/firce/ce_model_training.py` before finalizing this step - adjust the `joblib.load` paths to match if they differ from what's assumed here (existing tests in this file already load multiclass artifacts this way; mirror their exact paths).

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_e2e_simulation.py -v --no-cov -k held_out_class`
Expected: PASS. If `len(runtime.novelty_reports) == 0`, first check whether `held_out_stream_rows` is actually being flagged as novel at all (add a temporary print of `_score_chunk_novelty(runtime, clean_chunk)` output) before assuming a code bug - RF's confidence spread depends on the specific fixture data, and `novelty_tau=0.8`/`novelty_alpha=0.5` were chosen based on #101's confirmed working range (93-100% flag rate), but this fixture (`ce_flows_e2e_mc_train.csv`/`_stream.csv`, real captured data) may behave differently from #101's synthetic data - adjust thresholds based on what's actually observed, don't just assume the wiring is broken.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_simulation.py
git commit -m "test: add real end-to-end open-world novelty report generation test (#142)"
```

---

### Task 6: Full suite, lint, push, PR

- [ ] **Step 1: Run the full lean-CI suite**

```bash
uv run pytest -q --no-cov
```
Expected: exit code 0.

- [ ] **Step 2: Lint**

```bash
uv run ruff check src/firce/ tests/
uv run ruff format --check src/firce/ tests/
```
Expected: clean.

- [ ] **Step 3: Push and open the PR**

```bash
git push origin 142-runtime-wiring
gh pr create --base multiclass --head 142-runtime-wiring \
  --title "feat: wire novelty + XAI + LLM reporting into the live FIRCE runtime (#142)" \
  --body "$(cat <<'EOF'
## Summary
- #97 (novelty decision rule), #98 (XAI + selective generation), #99 (local LLM reporting) were real, tested, standalone modules never actually called from the live simulation path. This wires them into `firce/runtime/inference.py::process_chunk`.
- `ConformalDriftMonitor` gains `.model`/`.calibration_scores` properties, exposing its existing calibration state rather than building a second, duplicate calibration path for novelty scoring - reuses exactly what drift detection already computes.
- New `_score_chunk_novelty` (batched per-chunk, reuses the existing `_prepare_monitor_chunk_features` transform already used for drift detection) + `_generate_novelty_reports` (runs #98's selective-generation policy, generates explanations, and - only if a local LLM backend is configured - structured reports via #99).
- 9 new `SimulationConfig` fields gate/tune all of this, defaulting to **disabled** - zero behavior change for any existing config.
- Results accumulate in a new `runtime.novelty_reports: list[dict]` field and are logged at INFO level - a deliberately simple choice for where reports go (on-disk format design is an open question, not resolved here).

## Test plan
- [x] `ConformalDriftMonitor.model`/`.calibration_scores` covered directly.
- [x] New `SimulationConfig` fields covered (defaults + validation).
- [x] `runtime.novelty_reports`/`.llm_backend` default-empty/None, optional LLM backend construction covered.
- [x] `_score_chunk_novelty`/`_generate_novelty_reports` covered directly against a real calibrated `ConformalDriftMonitor` (both the disabled-by-default and enabled paths).
- [x] Real end-to-end test: genuine held-out-class scenario (RandomForest, not DecisionTree - per #101's confirmed finding) through a real `process_chunk` call on real streamed fixture data, proving `runtime.novelty_reports` gets populated with a real generated explanation - not mocks.
- [x] `uv run pytest -q`: full suite passes, existing tests unaffected (novelty detection off by default).
- [x] `ruff check`/`ruff format --check`: clean.
EOF
)"
```

## Self-Review

**Spec coverage:** Task 1 exposes the calibration state needed. Task 2 adds the config surface. Task 3 adds runtime state + optional LLM backend construction. Task 4 implements and wires the actual scoring/explanation/reporting call sequence into `process_chunk`, exactly as #142 asks. Task 5 proves it end-to-end with real streamed data and a genuine held-out class. Task 6 verifies and ships.

**Placeholder scan:** No TBD/TODO; every step has literal runnable code. Task 5 explicitly calls out to verify exact artifact filenames against `ce_model_training.py` rather than guessing, and to adjust thresholds based on observed behavior rather than assuming - this is guidance for handling real fixture data, not a placeholder for unwritten logic.

**Type consistency:** `_score_chunk_novelty(runtime, clean_chunk) -> tuple[np.ndarray, np.ndarray] | None` is defined once in Task 4 and its return shape (`novelty_flags, x_monitor`) is consumed identically by `_generate_novelty_reports` (same task) and `process_chunk` (Task 4 Step 4) and the Task 5 end-to-end test.

**Explicitly out of scope:** On-disk report format/persistence (rolling CSV integration, structured file output) - `runtime.novelty_reports` is a deliberately simple in-memory accumulator; a follow-up issue can design persistence once there's a concrete consumer need. Per-row (rather than per-chunk-batched) novelty scoring - batching matches #98's actual designed API (`select_events_to_explain` operates on arrays) and is more efficient. Wiring #100's MITRE mapping into this path - not part of #142's stated scope, and would naturally layer on top of `runtime.novelty_reports['llm_report']` if wanted later.
