# General Test Coverage Pt.3: Binary conformalEval + drift_monitor Tests (#104) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue #104 (Part 3 of several — Part 1 is `2026-09-08-general-test-coverage-pt1-conftest-binary-training.md`, Part 2 is `2026-09-08-general-test-coverage-pt2-binary-component-tests.md`, both merged). This part covers the issue's remaining `conformalEval`/`drift_monitor` asks: "Unit tests for... the binary path of `conformalEval/{ice,cce,tce,approx_cce}.py`" and "`src/firce/drift_monitor/*.py` — no tests at all."

**Architecture:** `tests/test_conformal_eval_multiclass.py` (added in #93/PR#111) already proves ICE/CCE/Approx-TCE/Approx-CCE work correctly with K=3 string-labeled classes, plus the `ConformalEvaluator` wrapper and `ConformalDriftMonitor` adapter. This plan adds a close mirror, `tests/test_conformal_eval_binary.py`, using K=2 int-labeled classes (`0`/`1`, matching `BinLabel`'s real type) — the evaluators are class-count-agnostic (confirmed in #93), so this is about locking down the *binary-specific* label type/shape, not re-deriving new test design. `firce/drift_monitor/factory.py::build_monitor` (CE/NONE/CADE dispatch) has zero tests — added in `tests/test_drift_monitor_factory.py`. `firce/drift_monitor/cade_config.py::CadeMonitorConfig` is pure pydantic validation with no `cade`/`tensorflow` import required — added in `tests/test_cade_config.py`, runnable in the lean CI environment. `CadeDriftMonitor`/`CadeRuntimeDetector` themselves require the optional `cade` dependency group (pulls in `tensorflow`, not installed in the lean `torch` CI group) — the one `build_monitor` CADE-dispatch test that needs it is gated behind `pytest.importorskip('cade')`, matching the existing xgboost-skip pattern, so it documents intent without requiring a heavy install in this pass.

**Tech Stack:** Python 3.11+, `pytest`, `pydantic`, `scikit-learn`, `numpy` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- Do not install the `cade` dependency group in this pass — it pulls in `tensorflow`, which CI's `pytest` job (lean `torch` group) does not have. Any test needing real `CadeDriftMonitor`/`CadeRuntimeDetector` behavior must use `pytest.importorskip('cade')` so it skips cleanly rather than failing collection.
- `MonitorType` only has three values (`CE`/`NONE`/`CADE`, all handled by `build_monitor`) — the function's final `raise ValueError(f'Unsupported monitor_type: ...')` branch is unreachable through any validated `SimulationConfig` and is not tested (would require an artificial, unrepresentative setup to exercise).

---

## File Structure

- Create: `tests/test_conformal_eval_binary.py` — mirrors `test_conformal_eval_multiclass.py` with K=2 int-labeled (`BinLabel`-shaped) data.
- Create: `tests/test_drift_monitor_factory.py` — `build_monitor` dispatch tests (CE, NONE, CADE via importorskip).
- Create: `tests/test_cade_config.py` — `CadeMonitorConfig` pydantic validator tests.

---

### Task 1: Binary-path conformalEval tests

**Files:**
- Create: `tests/test_conformal_eval_binary.py`

**Interfaces:**
- Consumes: `InductiveConformalEvaluator`, `CrossConformalEvaluator`, `ApproximateTransductiveConformalEvaluator`, `ApproxCrossConformalEvaluator` (`firce.conformalEval.*`), `ConformalEvaluator` (`firce.conformalEval.conformal_evaluators`), `ConformalDriftMonitor` (`firce.drift_monitor.conformal_monitor`), `CEType` (`firce.utils.config`), `PerformanceStats` (`firce.utils.perf_stats`) — same imports as `tests/test_conformal_eval_multiclass.py`.

- [ ] **Step 1: Write the tests**

```python
import numpy as np
import pytest

from sklearn.tree import DecisionTreeClassifier

from firce.conformalEval.approx_cce import ApproxCrossConformalEvaluator
from firce.conformalEval.cce import CrossConformalEvaluator
from firce.conformalEval.conformal_evaluators import ConformalEvaluator
from firce.conformalEval.ice import InductiveConformalEvaluator
from firce.conformalEval.tce import ApproximateTransductiveConformalEvaluator
from firce.drift_monitor.conformal_monitor import ConformalDriftMonitor
from firce.utils.config import CEType
from firce.utils.perf_stats import PerformanceStats


def _make_binary_data(n=90, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float64)
    y = (X[:, 0] > 0.0).astype(int)
    return X, y


EVALUATOR_FACTORIES = {
    'ice': lambda: InductiveConformalEvaluator(
        DecisionTreeClassifier(random_state=0), calibration_split=0.3, random_state=0
    ),
    'cce': lambda: CrossConformalEvaluator(DecisionTreeClassifier(random_state=0), folds=3, random_state=0),
    'approx_tce': lambda: ApproximateTransductiveConformalEvaluator(DecisionTreeClassifier(random_state=0)),
    'approx_cce': lambda: ApproxCrossConformalEvaluator(
        DecisionTreeClassifier(random_state=0), folds=3, random_state=0
    ),
}


@pytest.mark.parametrize('evaluator_name', list(EVALUATOR_FACTORIES))
def test_calibrate_produces_per_class_thresholds(evaluator_name):
    X, y = _make_binary_data()
    evaluator = EVALUATOR_FACTORIES[evaluator_name]()

    evaluator.calibrate(X, y, PerformanceStats())
    thresholds = evaluator.get_thresholds()

    assert set(thresholds.keys()) == {0, 1}
    for thresh in thresholds.values():
        assert 0.0 <= thresh <= 1.0


@pytest.mark.parametrize('evaluator_name', list(EVALUATOR_FACTORIES))
def test_predict_p_values_are_valid_and_match_known_classes(evaluator_name):
    X, y = _make_binary_data()
    evaluator = EVALUATOR_FACTORIES[evaluator_name]()
    evaluator.calibrate(X, y, PerformanceStats())

    result = evaluator.predict_p_values(X[:10])

    assert set(result.keys()) == {'class', 'p_value'}
    assert len(result['class']) == 10
    assert len(result['p_value']) == 10
    assert set(np.asarray(result['class']).tolist()) <= {0, 1}
    assert np.all((np.asarray(result['p_value']) >= 0.0) & (np.asarray(result['p_value']) <= 1.0))


@pytest.mark.parametrize(
    'ce_type,extra_kwargs',
    [
        (CEType.ICE, {'calibration_split': 0.3, 'random_state': 0}),
        (CEType.CCE, {'folds': 3, 'random_state': 0}),
        (CEType.APPROX_TCE, {}),
        (CEType.APPROX_CCE, {'folds': 3, 'random_state': 0}),
    ],
)
def test_conformal_evaluator_wrapper_binary_detect_drift(ce_type, extra_kwargs):
    X, y = _make_binary_data()
    ce = ConformalEvaluator(ce_type, DecisionTreeClassifier(random_state=0), **extra_kwargs)
    ce.calibrate(X, y, PerformanceStats())

    single_row = X[[0]]
    drift_flags = ce.detect_drift(single_row)

    assert drift_flags.shape == (1,)
    assert drift_flags.dtype == bool


@pytest.mark.parametrize(
    'ce_type,extra_kwargs',
    [
        (CEType.ICE, {'calibration_split': 0.3, 'random_state': 0}),
        (CEType.CCE, {'folds': 3, 'random_state': 0}),
        (CEType.APPROX_TCE, {}),
        (CEType.APPROX_CCE, {'folds': 3, 'random_state': 0}),
    ],
)
def test_conformal_drift_monitor_binary_fit_and_detect(ce_type, extra_kwargs):
    X, y = _make_binary_data()
    monitor = ConformalDriftMonitor(ce_type, DecisionTreeClassifier(random_state=0), **extra_kwargs)

    monitor.fit(X, y, PerformanceStats())
    result = monitor.detect(X[[0]])

    assert result.row_flags.shape == (1,)
    assert result.row_flags.dtype == bool
    assert result.chunk_drift == bool(result.row_flags.any())
    assert result.metadata['chunk_size'] == 1
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_conformal_eval_binary.py -v`
Expected: all 16 pass — the evaluators are already proven class-count-agnostic in #93; this locks down the int-label (`BinLabel`) case specifically, including the same `CrossConformalEvaluator` majority-vote path fixed for strings in #93 (`Counter`-based, works for ints too — this test would have caught a regression if that fix had broken the int case).

- [ ] **Step 3: Commit**

```bash
git add tests/test_conformal_eval_binary.py
git commit -m "test: verify ICE/CCE/Approx-TCE/Approx-CCE work correctly with binary (int) labels"
```

---

### Task 2: `drift_monitor/factory.py::build_monitor` dispatch tests

**Files:**
- Create: `tests/test_drift_monitor_factory.py`

**Interfaces:**
- Consumes: `build_monitor` (`firce.drift_monitor.factory`), `ConformalDriftMonitor` (`firce.drift_monitor.conformal_monitor`), `SimulationConfig`/`CEType`/`ModelType`/`ModelVariant`/`MonitorType` (`firce.utils.config`).

- [ ] **Step 1: Write the tests**

```python
import pytest

from sklearn.tree import DecisionTreeClassifier

from firce.drift_monitor.conformal_monitor import ConformalDriftMonitor
from firce.drift_monitor.factory import build_monitor
from firce.utils.config import CEType, ModelType, ModelVariant, MonitorType, SimulationConfig


def _make_config(tmp_path, **overrides):
    dummy = tmp_path / 'dummy.csv'
    dummy.write_text('a\n1\n')
    defaults = dict(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=dummy,
        flows_path=dummy,
        is_unsw=False,
        seed=0,
        monitor_type=MonitorType.NONE,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_build_monitor_returns_none_when_disabled(tmp_path):
    config = _make_config(tmp_path, monitor_type=MonitorType.NONE)

    monitor = build_monitor(config, model=DecisionTreeClassifier())

    assert monitor is None


def test_build_monitor_returns_conformal_drift_monitor_for_ce(tmp_path):
    config = _make_config(tmp_path, monitor_type=MonitorType.CE, ce_type=CEType.ICE)

    monitor = build_monitor(config, model=DecisionTreeClassifier(random_state=0))

    assert isinstance(monitor, ConformalDriftMonitor)


def test_build_monitor_dispatches_cade(tmp_path):
    pytest.importorskip('cade')
    config = _make_config(
        tmp_path,
        monitor_type=MonitorType.CADE,
        monitor_kwargs={'dims': [4, 2]},
    )

    from firce.drift_monitor.cade_monitor import CadeDriftMonitor

    monitor = build_monitor(config, model=None)

    assert isinstance(monitor, CadeDriftMonitor)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_drift_monitor_factory.py -v`
Expected: 3 pass, with `test_build_monitor_dispatches_cade` skipped in this environment (`cade` not installed in the lean `torch` group) — that's expected, not a failure.

- [ ] **Step 3: Commit**

```bash
git add tests/test_drift_monitor_factory.py
git commit -m "test: add build_monitor dispatch tests (CE, NONE, CADE)"
```

---

### Task 3: `cade_config.py::CadeMonitorConfig` validator tests

**Files:**
- Create: `tests/test_cade_config.py`

**Interfaces:**
- Consumes: `CadeMonitorConfig` (`firce.drift_monitor.cade_config`) — pure pydantic, no `cade` package import required.

- [ ] **Step 1: Write the tests**

```python
import pytest
from pydantic import ValidationError

from firce.drift_monitor.cade_config import CadeMonitorConfig


def test_cade_monitor_config_accepts_valid_dims():
    config = CadeMonitorConfig(dims=[10, 5, 2])

    assert config.dims == [10, 5, 2]
    assert config.batch_size == 64
    assert config.mad_threshold == 3.5


def test_cade_monitor_config_rejects_single_element_dims():
    # pydantic's own Field(min_length=2) constraint fires before the custom
    # field_validator, so its built-in "too_short" message wins here.
    with pytest.raises(ValidationError, match='at least 2 items'):
        CadeMonitorConfig(dims=[10])


def test_cade_monitor_config_rejects_non_positive_dims():
    with pytest.raises(ValidationError, match='must be positive'):
        CadeMonitorConfig(dims=[10, 0])


def test_cade_monitor_config_rejects_batch_size_not_multiple_of_four():
    with pytest.raises(ValidationError, match='multiple of 4'):
        CadeMonitorConfig(dims=[10, 2], batch_size=5)


def test_cade_monitor_config_rejects_out_of_range_ratio():
    with pytest.raises(ValidationError, match=r'must be in \[0, 1\]'):
        CadeMonitorConfig(dims=[10, 2], min_drift_ratio=1.5)


def test_cade_monitor_config_rejects_unknown_field():
    with pytest.raises(ValidationError):
        CadeMonitorConfig(dims=[10, 2], not_a_real_field=1)


def test_cade_monitor_config_is_frozen():
    config = CadeMonitorConfig(dims=[10, 2])

    with pytest.raises(ValidationError):
        config.batch_size = 128
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_cade_config.py -v`
Expected: all 7 pass. If `test_cade_monitor_config_is_frozen` fails because pydantic raises a different exception type for frozen-model mutation (e.g. `TypeError` in some pydantic versions), diagnose via the traceback and adjust the `pytest.raises(...)` type to match — don't guess, check what pydantic v2's `ConfigDict(frozen=True)` actually raises in this project's pinned version.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cade_config.py
git commit -m "test: add CadeMonitorConfig validator tests"
```

---

### Task 4: Lint, format, and final regression

**Files:**
- All files created in Tasks 1-3 (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any issues**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
```

If issues appear, run `uv run ruff check --fix .` and `uv run ruff format .`, then re-run the full suite.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures beyond the pre-existing/expected skips (xgboost/cade not installed in lean group, unrelated MC rolling-logger skips)

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format binary CE / drift_monitor test additions"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** Covers #104's remaining `conformalEval`/`drift_monitor` asks: binary-path `conformalEval/{ice,cce,tce,approx_cce}.py` (Task 1), `drift_monitor/*.py` (Tasks 2-3 — `factory.py` and `cade_config.py`; `conformal_monitor.py` already covered by Task 1's `ConformalDriftMonitor` tests and #93's multiclass equivalent).

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code. Task 3 Step 2 has an explicit, reasoned contingency for a genuinely version-dependent pydantic behavior, not a placeholder.

**Explicitly out of scope for this PR (remaining #104 work):** `CadeDriftMonitor`/`CadeRuntimeDetector` real behavior (needs the `cade` group installed — `pytest.importorskip`-gated, not exercised in this environment), the full e2e test, dead test file triage (`test_simulations.py`/`test_main.py`/`test_models.py`), `is_unsw=True` path for `train_ce_binary` (deferred since Part 1). This should be the second-to-last part of #104 — only the e2e test and dead-file triage remain after this merges.
