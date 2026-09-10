# Multiclass CE Evaluator Test Coverage (#93) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `src/firce/conformalEval/{ice,cce,tce,approx_cce}.py` look class-count agnostic by reading the code, but no multiclass test exists today. Add tests proving ICE/CCE/Approx-TCE/Approx-CCE actually work correctly with >2 classes. If gaps are found, fix them here.

**Architecture:** Unlike the runtime wiring work (#87-#92), which repeatedly found real binary-only bugs on inspection, all four evaluators in `conformalEval/` were read in full and appear to have been written with class-count-agnosticism as an explicit design goal from the start: every score/threshold computation goes through `model.classes_` and per-class dict comprehensions over `np.unique(y)`, and every metrics call already branches `average='binary' if len(np.unique(y)) == 2 else 'weighted'`. This plan's job is to *prove* that empirically with real multiclass tests, not to re-derive a design. `tce.py`'s class is `ApproximateTransductiveConformalEvaluator` (`CEType.APPROX_TCE`) — there is no separate true-TCE implementation; `CEType` only has `ICE | CCE | APPROX_TCE | APPROX_CCE | NONE`, so "TCE" in the issue title means Approx-TCE, the only TCE variant that exists in this codebase.

All four evaluators share the same calibrate/predict interface (`calibrate(X, y, perf_stats)`, `predict_p_values(X) -> {'class': ..., 'p_value': ...}`, `get_thresholds() -> dict`), differing only in constructor kwargs (`calibration_split` for ICE, `folds` for CCE/Approx-CCE). Tests are written as one parametrized suite across all four rather than four separate near-duplicate files.

**Tech Stack:** Python 3.11+, `scikit-learn`, `numpy`, `pytest` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- `StratifiedKFold` (used by CCE and Approx-CCE) requires at least `folds` samples per class in the calibration data — keep test datasets large enough per class (≥30/class with `folds=3`) to avoid spurious `ValueError`s unrelated to the actual thing being tested.
- Do not use a `significance_controller` in these tests — that's a separate, adaptive-significance-specific concern (`adaptive_sig_ctlr.py`), not part of "does the core multiclass calibration/p-value math work."
- `ConformalEvaluator.detect_drift` (the runtime-facing wrapper) is documented and implemented for single-sample input (`X.shape == (1, n_features)`) — test it that way, matching its actual contract. Whether multi-row chunks passed to it only evaluate row 0 is a pre-existing question that affects binary equally; it is not multiclass-specific and is out of scope here.

---

## File Structure

- Create: `tests/test_conformal_eval_multiclass.py` — parametrized multiclass tests across ICE/CCE/Approx-TCE/Approx-CCE, the `ConformalEvaluator` wrapper, and the `ConformalDriftMonitor` runtime adapter.

---

### Task 1: Multiclass tests for all four CE evaluators + the runtime wrapper

**Files:**
- Create: `tests/test_conformal_eval_multiclass.py`

**Interfaces:**
- No production code interfaces produced — this task is pure test-writing against the existing `InductiveConformalEvaluator`, `CrossConformalEvaluator`, `ApproximateTransductiveConformalEvaluator`, `ApproxCrossConformalEvaluator`, `ConformalEvaluator`, and `ConformalDriftMonitor` classes.

- [ ] **Step 1: Write the tests**

Create `tests/test_conformal_eval_multiclass.py`:

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


def _make_multiclass_data(n=90, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float64)
    y = np.select(
        [X[:, 0] > 0.5, X[:, 1] > 0.5],
        ['PortScan', 'XMasAttack'],
        default='Benign',
    )
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
    X, y = _make_multiclass_data()
    evaluator = EVALUATOR_FACTORIES[evaluator_name]()

    evaluator.calibrate(X, y, PerformanceStats())
    thresholds = evaluator.get_thresholds()

    assert set(thresholds.keys()) == set(np.unique(y).tolist())
    for thresh in thresholds.values():
        assert 0.0 <= thresh <= 1.0


@pytest.mark.parametrize('evaluator_name', list(EVALUATOR_FACTORIES))
def test_predict_p_values_are_valid_and_match_known_classes(evaluator_name):
    X, y = _make_multiclass_data()
    evaluator = EVALUATOR_FACTORIES[evaluator_name]()
    evaluator.calibrate(X, y, PerformanceStats())

    result = evaluator.predict_p_values(X[:10])

    assert set(result.keys()) == {'class', 'p_value'}
    assert len(result['class']) == 10
    assert len(result['p_value']) == 10
    assert set(np.asarray(result['class']).tolist()) <= set(np.unique(y).tolist())
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
def test_conformal_evaluator_wrapper_multiclass_detect_drift(ce_type, extra_kwargs):
    X, y = _make_multiclass_data()
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
def test_conformal_drift_monitor_multiclass_fit_and_detect(ce_type, extra_kwargs):
    X, y = _make_multiclass_data()
    monitor = ConformalDriftMonitor(ce_type, DecisionTreeClassifier(random_state=0), **extra_kwargs)

    monitor.fit(X, y, PerformanceStats())
    result = monitor.detect(X[[0]])

    assert result.row_flags.shape == (1,)
    assert result.row_flags.dtype == bool
    assert result.chunk_drift == bool(result.row_flags.any())
    assert result.metadata['chunk_size'] == 1
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_conformal_eval_multiclass.py -v`
Expected: based on reading all four evaluator implementations in full, every score/threshold/metric computation already routes through `model.classes_` and per-class dict comprehensions with no binary-specific hardcoding — expect all tests to pass. If any fail, diagnose the actual failure (do not guess): read the traceback, identify which specific line makes a binary assumption, and fix it there, matching the class-agnostic pattern already used everywhere else in that file (e.g. `average='binary' if len(np.unique(y)) == 2 else 'weighted'`, `{cls: ... for cls in np.unique(y)}`).

- [ ] **Step 3: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 4: Commit**

```bash
git add tests/test_conformal_eval_multiclass.py
git commit -m "test: verify ICE/CCE/Approx-TCE/Approx-CCE work correctly with multiclass labels"
```

(If Step 2 required a fix, that fix goes in this same commit alongside the tests, or a preceding commit if the fix is substantial enough to warrant its own — use judgment at execution time based on what's actually found.)

---

### Task 2: Lint, format, and final regression

**Files:**
- `tests/test_conformal_eval_multiclass.py` (formatting only, if needed)

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
git commit -m "style: ruff format multiclass CE evaluator tests"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** Issue #93 asks to add tests proving ICE/CCE/TCE/Approx-CCE work correctly with >2 classes, and fix gaps if found. Task 1 covers all four evaluators plus the two layers that actually sit between them and the live runtime (`ConformalEvaluator` wrapper, `ConformalDriftMonitor` adapter) — closing the loop all the way to what `inference.py::_detect_chunk_drift` actually calls.

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code. The one open contingency (Step 2 of Task 1: "if any fail, diagnose and fix") is explicitly not a placeholder — it's an honest acknowledgment that a plan cannot pre-write a fix for a bug not yet confirmed to exist, unlike #87-#92 where the bugs were already confirmed by direct code inspection before the plan was written.

**Explicitly out of scope:** CADE drift-monitor multiclass compatibility is a separate investigation-only spike (#94). Whether `ConformalDriftMonitor.detect()` correctly handles multi-row chunks (vs. only evaluating row 0) is a pre-existing question affecting binary equally — not fixed or further investigated here since it isn't multiclass-specific. General multiclass test coverage beyond the CE evaluators (#95) and the broader firce test-coverage gap (#104) are separate.
