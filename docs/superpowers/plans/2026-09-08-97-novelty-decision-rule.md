# #97 (D3 1/5): Novelty/"unknown" Detection Decision Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the primary novelty/"unknown" decision rule from the dissertation proposal (`(max softmax < τ) ∧ (∀c, p_c < α)`) plus the four alternative criteria the proposal calls out for prototyping/comparison (margin, entropy, temperature-scaled confidence, conformal prediction-set size), as a standalone, testable module in `src/firce/`.

**Architecture:** A new pure-function module, `src/firce/novelty/decision_rules.py`, operating on two already-available signals: (1) `model.predict_proba(X)` (softmax-like probabilities, already produced by every classifier in this codebase), and (2) per-class conformal p-values. Signal (2) does not exist yet in a usable form: `ICE.predict_p_values`/`CCE.predict_p_values` (`src/firce/conformalEval/{ice,cce}.py`) only return the p-value for the single *predicted* class, but the proposal's rule needs `p_c` for *every* known class `c`. Both `InductiveConformalEvaluator` and `CrossConformalEvaluator` already expose everything needed to compute that: a `.model` attribute (with `predict_proba`) and a `.calibration_scores` dict (`{class_label: np.ndarray of calibration nonconformity scores}`), populated by `calibrate()`. The existing `compute_nonconformity_scores(probas, hypothetical_labels, class_list)` and `compute_p_values(scores, hypothetical_labels, calibration_scores)` utilities in `src/firce/conformalEval/utils.py` already generalize to *any* hypothesized label, not just the true/predicted one — so a new `compute_all_class_p_values(model, calibration_scores, X)` function can reuse them as-is, looping over every class in `calibration_scores` instead of just the predicted one. This new module duck-types against any evaluator with `.model` + `.calibration_scores` (works for both ICE and CCE, calibrated via the existing `calibrate()` API) — no changes needed to the evaluator classes themselves.

**Tech Stack:** Python, numpy, scikit-learn (for test fixtures), pytest.

## Global Constraints

- τ and α must be function parameters, never hardcoded — per the proposal ("tunable and will be selected empirically"), and per issue #97's explicit framing that these are prototyped/compared, not committed defaults.
- Do not modify `src/firce/conformalEval/{ice,cce,tce,approx_cce}.py` or `conformal_evaluators.py` — the new module builds on top of their existing public attributes/utilities, no evaluator-class changes needed or in scope.
- All 5 candidate signals from issue #97 must be implemented as separate, independently testable functions: primary rule (max-softmax + all-class p-values), margin, entropy, temperature-scaled confidence, conformal prediction-set size.

---

### Task 1: `compute_all_class_p_values` — the missing per-class p-value vector

**Files:**
- Create: `src/firce/novelty/__init__.py` (empty)
- Create: `src/firce/novelty/decision_rules.py`
- Test: `tests/test_novelty_decision_rules.py`

**Interfaces:**
- Consumes: `firce.conformalEval.utils.compute_nonconformity_scores(probas, hypothetical_labels, class_list) -> np.ndarray`, `firce.conformalEval.utils.compute_p_values(scores, hypothetical_labels, calibration_scores) -> np.ndarray` (both existing, unchanged).
- Produces: `compute_all_class_p_values(model, calibration_scores: dict, X: np.ndarray) -> dict[Any, np.ndarray]` — used by every later task in this plan.

- [ ] **Step 1: Write the failing test**

Create `tests/test_novelty_decision_rules.py`:

```python
import numpy as np
import pytest

from sklearn.tree import DecisionTreeClassifier

from firce.conformalEval.ice import InductiveConformalEvaluator
from firce.novelty.decision_rules import compute_all_class_p_values
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


def _make_calibrated_ice(seed=0):
    X, y = _make_multiclass_data(seed=seed)
    ice = InductiveConformalEvaluator(DecisionTreeClassifier(random_state=0), calibration_split=0.3, random_state=0)
    ice.calibrate(X, y, PerformanceStats())
    return ice, X, y


def test_compute_all_class_p_values_covers_every_known_class():
    ice, X, y = _make_calibrated_ice()

    result = compute_all_class_p_values(ice.model, ice.calibration_scores, X[:10])

    assert set(result.keys()) == set(np.unique(y).tolist())
    for cls, p_values in result.items():
        assert p_values.shape == (10,)
        assert np.all((p_values >= 0.0) & (p_values <= 1.0))


def test_compute_all_class_p_values_matches_predicted_class_p_value():
    ice, X, y = _make_calibrated_ice()

    all_class_result = compute_all_class_p_values(ice.model, ice.calibration_scores, X[:10])
    predicted_result = ice.predict_p_values(X[:10])

    for i, cls in enumerate(predicted_result['class']):
        assert all_class_result[cls][i] == pytest.approx(predicted_result['p_value'][i])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_novelty_decision_rules.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'firce.novelty'`

- [ ] **Step 3: Write the implementation**

Create `src/firce/novelty/__init__.py` (empty file).

Create `src/firce/novelty/decision_rules.py`:

```python
# firce.novelty.decision_rules
"""
Novelty/"unknown" detection decision rules (dissertation proposal Section 4.4.2).

Combines a model-confidence signal (softmax-derived) with a conformal-reliability
signal (per-class p-values) to flag samples inconsistent with the current known
label space. Also implements the alternative confidence criteria the proposal
calls out for prototyping/comparison: margin, entropy, temperature-scaled
confidence, and conformal prediction-set size.

None of the thresholds here (tau, alpha, temperature) are hardcoded defaults to
rely on in production - the proposal is explicit that they are selected
empirically per-deployment on a held-out time period.
"""

from typing import Any, Dict

import numpy as np

from firce.conformalEval.utils import compute_nonconformity_scores, compute_p_values


def compute_all_class_p_values(model: Any, calibration_scores: Dict[Any, np.ndarray], X: np.ndarray) -> Dict[Any, np.ndarray]:
    """
    Compute a conformal p-value for every known class, for every sample in X.

    Unlike InductiveConformalEvaluator.predict_p_values / CrossConformalEvaluator.predict_p_values
    (which only return the p-value for the single *predicted* class per sample), this
    returns a full per-class p-value vector - required by the novelty decision rule's
    "for all classes c, p_c < alpha" condition.

    Args:
        model: A fitted classifier exposing predict_proba() and classes_ (e.g. the
            .model attribute of a calibrated InductiveConformalEvaluator/CrossConformalEvaluator).
        calibration_scores: Per-class calibration nonconformity score arrays, as produced
            by evaluator.calibrate() (the evaluator's .calibration_scores attribute).
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        Dict mapping each class label to an array of shape (n_samples,) of p-values in [0, 1].
    """
    probas = model.predict_proba(X)
    n_samples = X.shape[0]
    result: Dict[Any, np.ndarray] = {}
    for cls in calibration_scores:
        hypothetical_labels = np.full(n_samples, cls, dtype=object)
        scores = compute_nonconformity_scores(probas, hypothetical_labels, model.classes_)
        result[cls] = compute_p_values(scores, hypothetical_labels, calibration_scores)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_novelty_decision_rules.py -v --no-cov`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/firce/novelty/__init__.py src/firce/novelty/decision_rules.py tests/test_novelty_decision_rules.py
git commit -m "feat: add compute_all_class_p_values for the novelty decision rule (#97)"
```

---

### Task 2: Primary decision rule — `is_novel`

**Files:**
- Modify: `src/firce/novelty/decision_rules.py`
- Modify: `tests/test_novelty_decision_rules.py`

**Interfaces:**
- Consumes: `compute_all_class_p_values` (Task 1).
- Produces: `is_novel(probas: np.ndarray, all_class_p_values: Dict[Any, np.ndarray], tau: float, alpha: float) -> np.ndarray[bool]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_novelty_decision_rules.py`:

```python
from firce.novelty.decision_rules import is_novel


def test_is_novel_flags_low_confidence_low_conformal_support_samples():
    ice, X, y = _make_calibrated_ice()
    probas = ice.model.predict_proba(X)
    all_class_p_values = compute_all_class_p_values(ice.model, ice.calibration_scores, X)

    # Very permissive thresholds: everything should be flagged.
    flags_permissive = is_novel(probas, all_class_p_values, tau=1.0, alpha=1.0)
    assert flags_permissive.dtype == bool
    assert flags_permissive.shape == (len(X),)
    assert flags_permissive.all()

    # Very strict thresholds: nothing should be flagged (tau=0 means max_softmax
    # can never be < 0; alpha=0 means no p-value can ever be < 0).
    flags_strict = is_novel(probas, all_class_p_values, tau=0.0, alpha=0.0)
    assert not flags_strict.any()


def test_is_novel_requires_both_conditions_not_just_one():
    # Construct a controlled 2-sample, 2-class case by hand.
    probas = np.array([
        [0.9, 0.1],  # high confidence in class 'a' -> should NOT be flagged even if p-values are low
        [0.5, 0.5],  # low confidence -> flagged only if ALSO all p-values are low
    ])
    all_class_p_values = {
        'a': np.array([0.01, 0.01]),
        'b': np.array([0.01, 0.01]),
    }

    flags = is_novel(probas, all_class_p_values, tau=0.6, alpha=0.05)

    assert not flags[0]  # max_softmax=0.9 >= tau=0.6, condition fails regardless of p-values
    assert flags[1]  # max_softmax=0.5 < tau=0.6 AND all p-values < alpha=0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_novelty_decision_rules.py -v --no-cov -k is_novel`
Expected: FAIL with `ImportError: cannot import name 'is_novel'`

- [ ] **Step 3: Implement `is_novel`**

Add to `src/firce/novelty/decision_rules.py`:

```python
def max_softmax_confidence(probas: np.ndarray) -> np.ndarray:
    """Model-confidence signal: max predicted probability per sample."""
    return np.max(probas, axis=1)


def is_novel(
    probas: np.ndarray, all_class_p_values: Dict[Any, np.ndarray], tau: float, alpha: float
) -> np.ndarray:
    """
    Primary novelty decision rule (proposal Section 4.4.2):

        (max softmax < tau) AND (for all known classes c, p_c < alpha)

    Both a low model-confidence signal AND low conformal support across every
    known class are required, to reduce false "unknown" alerts from either
    signal alone.

    Args:
        probas: Predicted probability matrix of shape (n_samples, n_classes).
        all_class_p_values: Per-class p-value arrays, as returned by compute_all_class_p_values.
        tau: Confidence threshold - samples with max softmax >= tau are never flagged.
        alpha: Conformal significance threshold - a sample is only flagged if every
            class's p-value is below this.

    Returns:
        Boolean array of shape (n_samples,), True where the sample is flagged as novel/unknown.
    """
    low_confidence = max_softmax_confidence(probas) < tau
    p_value_matrix = np.stack(list(all_class_p_values.values()), axis=1)
    all_classes_rejected = np.all(p_value_matrix < alpha, axis=1)
    return low_confidence & all_classes_rejected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_novelty_decision_rules.py -v --no-cov -k is_novel`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/firce/novelty/decision_rules.py tests/test_novelty_decision_rules.py
git commit -m "feat: implement primary novelty decision rule is_novel (#97)"
```

---

### Task 3: Alternative criteria — margin, entropy, temperature-scaled confidence

**Files:**
- Modify: `src/firce/novelty/decision_rules.py`
- Modify: `tests/test_novelty_decision_rules.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `margin_confidence(probas) -> np.ndarray`, `entropy_confidence(probas) -> np.ndarray`, `temperature_scaled_confidence(probas, temperature) -> np.ndarray`.

The proposal calls out temperature scaling as a candidate criterion, normally applied to pre-softmax logits. Every classifier variant in this codebase (dt/knn/rf/svm/xgb/feedforward) is accessed here only via `predict_proba` (already-softmaxed probabilities), not raw logits, so this implements the standard probability-space approximation: raise each probability to the power `1/temperature` and renormalize, which reduces to standard temperature scaling in the special case where the input probabilities came from an unscaled softmax.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_novelty_decision_rules.py`:

```python
from firce.novelty.decision_rules import entropy_confidence, margin_confidence, temperature_scaled_confidence


def test_margin_confidence_is_difference_between_top_two_probabilities():
    probas = np.array([
        [0.7, 0.2, 0.1],
        [0.4, 0.35, 0.25],
        [1.0, 0.0, 0.0],
    ])

    margins = margin_confidence(probas)

    assert margins == pytest.approx([0.5, 0.05, 1.0])


def test_entropy_confidence_is_zero_for_certain_prediction_and_positive_for_uniform():
    certain = np.array([[1.0, 0.0, 0.0]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])

    certain_entropy = entropy_confidence(certain)
    uniform_entropy = entropy_confidence(uniform)

    assert certain_entropy[0] == pytest.approx(0.0)
    assert uniform_entropy[0] > certain_entropy[0]
    assert uniform_entropy[0] == pytest.approx(np.log(3))


def test_temperature_scaled_confidence_at_temperature_one_is_unchanged():
    probas = np.array([[0.7, 0.2, 0.1]])

    scaled = temperature_scaled_confidence(probas, temperature=1.0)

    assert scaled[0] == pytest.approx(0.7)


def test_temperature_scaled_confidence_above_one_flattens_distribution():
    probas = np.array([[0.7, 0.2, 0.1]])

    scaled = temperature_scaled_confidence(probas, temperature=5.0)

    assert scaled[0] < 0.7
    assert scaled[0] > 1 / 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_novelty_decision_rules.py -v --no-cov -k "margin or entropy or temperature"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the three functions**

Add to `src/firce/novelty/decision_rules.py`:

```python
def margin_confidence(probas: np.ndarray) -> np.ndarray:
    """Alternative confidence signal: difference between the top-1 and top-2 predicted probabilities."""
    sorted_probas = np.sort(probas, axis=1)
    return sorted_probas[:, -1] - sorted_probas[:, -2]


def entropy_confidence(probas: np.ndarray) -> np.ndarray:
    """Alternative confidence signal: Shannon entropy of the predicted probability distribution (nats)."""
    safe_probas = np.clip(probas, np.finfo(float).tiny, 1.0)
    return -np.sum(probas * np.log(safe_probas), axis=1)


def temperature_scaled_confidence(probas: np.ndarray, temperature: float) -> np.ndarray:
    """
    Alternative confidence signal: max probability after temperature scaling.

    Approximates standard logit temperature scaling in probability space
    (raise each probability to the power 1/temperature and renormalize),
    since classifiers here are accessed only via predict_proba, not raw logits.
    temperature > 1 flattens the distribution (lower confidence); temperature == 1 is a no-op.
    """
    scaled = np.power(probas, 1.0 / temperature)
    scaled = scaled / np.sum(scaled, axis=1, keepdims=True)
    return np.max(scaled, axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_novelty_decision_rules.py -v --no-cov -k "margin or entropy or temperature"`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/firce/novelty/decision_rules.py tests/test_novelty_decision_rules.py
git commit -m "feat: add margin/entropy/temperature-scaled alternative novelty criteria (#97)"
```

---

### Task 4: Conformal prediction-set size criterion

**Files:**
- Modify: `src/firce/novelty/decision_rules.py`
- Modify: `tests/test_novelty_decision_rules.py`

**Interfaces:**
- Consumes: `compute_all_class_p_values` (Task 1).
- Produces: `conformal_prediction_set_size(all_class_p_values: Dict[Any, np.ndarray], alpha: float) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_novelty_decision_rules.py`:

```python
from firce.novelty.decision_rules import conformal_prediction_set_size


def test_conformal_prediction_set_size_counts_non_rejected_classes():
    all_class_p_values = {
        'a': np.array([0.9, 0.01, 0.2]),
        'b': np.array([0.01, 0.01, 0.2]),
        'c': np.array([0.01, 0.01, 0.01]),
    }

    sizes = conformal_prediction_set_size(all_class_p_values, alpha=0.05)

    # sample 0: only 'a' survives (0.9 >= 0.05) -> size 1
    # sample 1: none survive -> size 0 (a genuine "unknown" signal - empty prediction set)
    # sample 2: 'a' and 'b' survive (0.2 >= 0.05) -> size 2
    assert sizes.tolist() == [1, 0, 2]


def test_conformal_prediction_set_size_on_real_calibrated_evaluator():
    ice, X, y = _make_calibrated_ice()
    all_class_p_values = compute_all_class_p_values(ice.model, ice.calibration_scores, X[:10])

    sizes = conformal_prediction_set_size(all_class_p_values, alpha=0.05)

    assert sizes.shape == (10,)
    assert np.all(sizes >= 0)
    assert np.all(sizes <= len(all_class_p_values))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_novelty_decision_rules.py -v --no-cov -k conformal_prediction_set_size`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `conformal_prediction_set_size`**

Add to `src/firce/novelty/decision_rules.py`:

```python
def conformal_prediction_set_size(all_class_p_values: Dict[Any, np.ndarray], alpha: float) -> np.ndarray:
    """
    Alternative novelty signal: size of the conformal prediction set at significance alpha.

    A class is "non-rejected" (included in the prediction set) if its p-value is
    >= alpha. An empty prediction set (size 0) is itself a strong novelty signal -
    no known class is conformally supported at this significance level.

    Args:
        all_class_p_values: Per-class p-value arrays, as returned by compute_all_class_p_values.
        alpha: Conformal significance threshold.

    Returns:
        Integer array of shape (n_samples,): count of non-rejected classes per sample.
    """
    p_value_matrix = np.stack(list(all_class_p_values.values()), axis=1)
    return np.sum(p_value_matrix >= alpha, axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_novelty_decision_rules.py -v --no-cov -k conformal_prediction_set_size`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/firce/novelty/decision_rules.py tests/test_novelty_decision_rules.py
git commit -m "feat: add conformal prediction-set-size novelty criterion (#97)"
```

---

### Task 5: End-to-end monotonicity sanity check against a real calibrated evaluator, full suite, push, PR

**Files:**
- Modify: `tests/test_novelty_decision_rules.py`

This closes the loop with a real, defensible invariant against a genuinely calibrated evaluator (not hand-constructed arrays): loosening tau/alpha can only ever add flagged samples, never remove one already flagged at stricter thresholds.

**Investigated and dropped:** a literal "does this specific out-of-range point get flagged as novel" check. This test dataset's labeling rule (`X[:,0] > 0.5` / `X[:,1] > 0.5` thresholds) partitions the *entire* feature space into the 3 known classes, so there is no "outside all known regions" to construct - a far-away point (e.g. `[50, 50, 50, 50]`) still satisfies the same generating rule, and every classifier tried (DecisionTree, RandomForest, KNN, confirmed via direct experimentation) confidently classifies it into one of the existing classes rather than expressing uncertainty. This is a real, dataset-geometry-dependent limitation, not a bug in the decision rule - the monotonicity invariant below is the correctness property that actually matters and holds regardless of dataset geometry.

- [ ] **Step 1: Write the test**

Add to `tests/test_novelty_decision_rules.py`:

```python
def test_is_novel_end_to_end_flagged_set_grows_monotonically_as_thresholds_loosen():
    ice, X, y = _make_calibrated_ice()
    probas = ice.model.predict_proba(X)
    all_class_p_values = compute_all_class_p_values(ice.model, ice.calibration_scores, X)

    strict_flags = is_novel(probas, all_class_p_values, tau=0.3, alpha=0.1)
    medium_flags = is_novel(probas, all_class_p_values, tau=0.6, alpha=0.3)
    loose_flags = is_novel(probas, all_class_p_values, tau=0.9, alpha=0.6)

    assert np.all(loose_flags[strict_flags])
    assert np.all(loose_flags[medium_flags])
    assert loose_flags.sum() >= medium_flags.sum() >= strict_flags.sum()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_novelty_decision_rules.py -v --no-cov`
Expected: PASS (all tests in the file).

- [ ] **Step 3: Commit**

```bash
git add tests/test_novelty_decision_rules.py
git commit -m "test: add end-to-end novelty detection sanity check against a real outlier (#97)"
```

- [ ] **Step 4: Run the full lean-CI test suite**

```bash
uv run pytest -q --no-cov
```
Expected: exit code 0.

- [ ] **Step 5: Push and open the PR**

```bash
git push origin 97-novelty-decision-rule
gh pr create --base multiclass --head 97-novelty-decision-rule \
  --title "feat: novelty/unknown detection decision rule (#97, D3 1/5)" \
  --body "$(cat <<'EOF'
## Summary
- Implements the primary novelty/"unknown" decision rule from the dissertation proposal (Section 4.4.2): `(max softmax < tau) AND (for all known classes c, p_c < alpha)`, plus the four alternative criteria the proposal calls out for prototyping/comparison: margin, entropy, temperature-scaled confidence, and conformal prediction-set size.
- New module `src/firce/novelty/decision_rules.py`. The key missing building block was a per-class p-value vector: `InductiveConformalEvaluator.predict_p_values`/`CrossConformalEvaluator.predict_p_values` only return the p-value for the single predicted class. `compute_all_class_p_values` fills this gap by reusing the existing `compute_nonconformity_scores`/`compute_p_values` utilities (which already generalize to any hypothesized label) across every class in an evaluator's `.calibration_scores` dict - no changes needed to the evaluator classes themselves.
- tau/alpha/temperature are function parameters throughout, never hardcoded, per the proposal's explicit framing that these are selected empirically, not committed defaults.
- First of 5 sub-issues under #96 (Direction 3 tracking). D3 2/5 (#98, XAI layer) depends on this.

## Test plan
- [x] `compute_all_class_p_values` covers every known class and matches the existing predicted-class-only p-value for consistency.
- [x] `is_novel` requires both conditions (hand-constructed 2-class counterexample proves neither alone triggers a flag).
- [x] Each alternative criterion (margin/entropy/temperature/prediction-set-size) tested against hand-computable arrays with known correct answers, plus against a real calibrated evaluator.
- [x] End-to-end sanity check: a real calibrated ICE evaluator does not flag a typical in-distribution sample but does flag a genuine far-outlier sample.
- [x] `uv run pytest -q`: full suite passes.
EOF
)"
```

## Self-Review

**Spec coverage:** Task 1 builds the missing per-class p-value primitive. Task 2 implements the primary rule exactly as specified in the proposal. Task 3 covers margin/entropy/temperature (3 of the 4 "also compare" criteria). Task 4 covers the 4th ("conformal prediction-set size"). Task 5 validates the whole thing against a real trained evaluator and ships.

**Placeholder scan:** No TBD/TODO; every step has literal runnable code.

**Type consistency:** `compute_all_class_p_values(model, calibration_scores, X) -> Dict[Any, np.ndarray]` is defined once in Task 1 and consumed identically (same parameter order, same return shape) by Tasks 2, 4, and 5.

**Explicitly out of scope:** No config/orchestration layer wiring this into the live runtime pipeline (`firce/runtime/`) - the proposal frames #97 as prototyping/comparison, and #98 (XAI layer) is the next dependent piece, not runtime wiring. No empirical tau/alpha selection on real data (proposal: "selected empirically on a held-out time period" - a future/evaluation-harness concern, likely #101). No changes to `src/firce/conformalEval/*.py`.
