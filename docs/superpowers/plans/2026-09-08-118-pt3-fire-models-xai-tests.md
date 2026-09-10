# #118 Pt.3: fire.models XAI Helper Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue #118 (Part 1 lazy imports/merged, Part 2 `fire.simulations` tests/merged). This part adds real tests for `fire/models.py`'s two XAI helper functions, `_explain_with_lime` and `_explain_with_shap`.

**Architecture:** Unlike `run_binary_classification`/`run_multiclass_classification` (which need `xgboost`/`tensorflow`, not in the lean CI group), these two helpers only need `shap`/`lime` — both already in the lean group — so these tests genuinely execute in CI, not just locally. Both take an already-fitted sklearn-compatible model and produce an output file (`_explain_with_lime` → an HTML instance explanation; `_explain_with_shap` → a PNG summary plot). Manually validated during this plan's design: a small `DecisionTreeClassifier` fit on synthetic data works correctly with both, producing the exact filenames the plan asserts on.

**Tech Stack:** Python 3.11+, `pytest`, `numpy`, `scikit-learn`, `shap`, `lime` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q`. Add to the existing `tests/test_models.py`.
- These tests must NOT need `pytest.importorskip('xgboost')`/`pytest.importorskip('tensorflow')` — importing `_explain_with_lime`/`_explain_with_shap` alongside anything else from `fire.models` still triggers `fire.models`'s module-level `import shap`/`lime` imports only (xgboost/tensorflow are lazy inside the two training functions since #118 Part 1), so this file's existing top-level `from fire.models import ...` import continues to work without new guards.

---

## File Structure

- Modify: `tests/test_models.py` — add tests for `_explain_with_lime`/`_explain_with_shap`.

---

### Task 1: XAI helper tests

**Files:**
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `_explain_with_lime`, `_explain_with_shap` (`fire.models`), `DecisionTreeClassifier` (`sklearn.tree`).

- [ ] **Step 1: Add imports and the tests**

At the top of `tests/test_models.py`, change:

```python
from fire.models import _parse_args, run_binary_classification, run_multiclass_classification
```

to:

```python
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from fire.models import _explain_with_lime, _explain_with_shap, _parse_args, run_binary_classification, run_multiclass_classification
```

Add at the end of the file:

```python
def _make_xai_fixture():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(40, 4))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(5, 4))
    model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    return model, X_train, X_test


def test_explain_with_lime_writes_instance_html(tmp_path):
    model, X_train, X_test = _make_xai_fixture()

    _explain_with_lime(
        model,
        X_train,
        X_test,
        feature_names=['f0', 'f1', 'f2', 'f3'],
        class_names=['Benign', 'Attack'],
        outputPath=str(tmp_path),
        output_prefix='lime_test',
    )

    assert (tmp_path / 'lime_test_instance.html').exists()


def test_explain_with_shap_writes_summary_png(tmp_path):
    model, X_train, _ = _make_xai_fixture()

    _explain_with_shap(
        model,
        X_train[:10],
        outputPath=str(tmp_path),
        feature_names=['f0', 'f1', 'f2', 'f3'],
        model_type='tree',
        output_prefix='shap_test',
    )

    assert (tmp_path / 'shap_test_summary.png').exists()
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_models.py -v`
Expected: all pass (2 existing `_parse_args` + 2 existing collision-check + 2 new XAI tests = 6 total), no `importorskip` skips — manually validated during this plan's design with this exact fixture shape.

- [ ] **Step 3: Commit**

```bash
git add tests/test_models.py
git commit -m "test: add fire.models XAI helper tests (_explain_with_lime, _explain_with_shap)"
```

---

### Task 2: Lint, format, and final regression

**Files:**
- `tests/test_models.py` (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any issues**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
```

If issues appear, run `uv run ruff check --fix .` and `uv run ruff format .`, then re-run the full suite.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures.

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format fire.models XAI helper tests"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** Covers 2 of #118's 4 remaining `fire.models` targets (`_explain_with_lime`, `_explain_with_shap`). `run_binary_classification`/`run_multiclass_classification` (needing xgboost+tensorflow) and `fire.main::main()` remain for later parts.

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code, validated by a real manual run during design.

**Explicitly out of scope:** `run_binary_classification`/`run_multiclass_classification`/`fire.main::main()` — later parts of #118.
