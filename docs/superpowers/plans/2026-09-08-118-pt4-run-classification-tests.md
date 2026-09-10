# #118 Part 4: fix multiclass FNN crash + tests for run_binary_classification/run_multiclass_classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a confirmed `UnboundLocalError` crash in `fire.models.run_multiclass_classification` (any call with `isPCA=False`) and add real, execution-based tests for both `run_binary_classification` and `run_multiclass_classification`, the last remaining `fire.models` scope item for xseciot issue #118.

**Architecture:** One-line bug fix (`X1_pca` → `X1_final` in the FNN train/test split, matching every other classifier branch in the same function which already correctly branches on `isPCA`). Tests call both functions end-to-end with tiny synthetic CSV fixtures and `tmp_path`-scoped cwd, asserting the expected artifact files land on disk. Both functions require `xgboost` and `tensorflow`, which conflict with the `torch` dependency group used everywhere else in this repo's test suite (see `pyproject.toml` `[tool.uv] conflicts`), so the new tests are gated behind `pytest.importorskip('xgboost')` / `pytest.importorskip('tensorflow')` — they will skip in the lean CI `torch` environment (matching the established convention already used for other xgboost/cade-dependent tests in this codebase) but run for real in any environment with the `tensorflow`/`xgboost` groups installed.

**Tech Stack:** Python, pytest, scikit-learn, xgboost, tensorflow/keras, pandas, numpy.

## Global Constraints

- Do not touch `binary_models_dir`/`multi_models_dir` naming (already fixed under #126 — `binary_models_legacy_fire`/`multi_class_models_legacy_fire`). Do not reintroduce the old names.
- Do not add `isPCA`/shap/lime toggles to the function signatures — out of scope, not required by the bug fix or the tests.
- Keep fixture row counts as small as correctness allows (tests must complete in reasonable CI time) — 5-fold cross-validation requires at least 5 rows per class; use ~10-15 rows per class.

---

### Task 1: Fix the `X1_pca` UnboundLocalError in `run_multiclass_classification`

**Files:**
- Modify: `src/fire/models.py:565`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `run_multiclass_classification(aggregated_file: str, isUNSW: bool, isPCA: bool) -> None` no longer crashes when `isPCA=False`.

**Confirmed bug (real execution, not guessed):** calling `run_multiclass_classification(csv_path, isUNSW=False, isPCA=False)` against a real 60-row 3-class synthetic dataset ran RF/KNN/DT/SVM/XGBoost training and evaluation successfully, then crashed:
```
Traceback (most recent call last):
  File "<string>", line 30, in <module>
  File "src/fire/models.py", line 565, in run_multiclass_classification
    X1_pca,
    ^^^^^^
UnboundLocalError: cannot access local variable 'X1_pca' where it is not associated with a value
```
`X1_pca` (`src/fire/models.py:446`) is only assigned inside `if isPCA:` (line 444-447); every other classifier in this function correctly branches with `if isPCA: ... X1_pca ... else: ... X1_final ...` (e.g. lines 454-457, 459-469, 480-483, 495-498, 510-513, 526-535, 541-544) but the FNN section's `train_test_split` call at line 564-569 uses `X1_pca` unconditionally. `X1_final` (line 447/449) is the variable set in both branches for exactly this purpose.

- [ ] **Step 1: Write the failing regression test**

Add to `tests/test_models.py` (near the top, after existing imports — add `import pytest` if not already imported):

```python
import pytest


def _make_multiclass_fixture_csv(tmp_path, n_per_class=12):
    rng = np.random.default_rng(0)
    rows = []
    for label, offset in [('Benign', 0.0), ('PortScan', 3.0), ('XMasAttack', -3.0)]:
        for _ in range(n_per_class):
            rows.append({
                'flow_duration': rng.normal(offset, 1.0),
                'tot_fwd_pkt': rng.normal(offset, 1.0),
                'tot_bwd_pkts': rng.normal(offset, 1.0),
                'totlen_fwd_pkts': rng.normal(offset, 1.0),
                'totlen_bwd_pkts': rng.normal(offset, 1.0),
                'Label': label,
            })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    dataset_dir = tmp_path / 'dataset'
    dataset_dir.mkdir()
    csv_path = dataset_dir / 'aggregated.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def test_run_multiclass_classification_no_pca_does_not_crash(tmp_path, monkeypatch):
    pytest.importorskip('xgboost')
    pytest.importorskip('tensorflow')
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_fixture_csv(tmp_path)

    run_multiclass_classification(str(csv_path), isUNSW=False, isPCA=False)

    models_dir = tmp_path / 'multi_class_models_legacy_fire' / 'dataset'
    assert (models_dir / 'random_forest_multi.pkl').exists()
    assert (models_dir / 'feedforward_multi.pkl').exists()
    assert (models_dir / 'scaler_multi.pkl').exists()
    assert not (models_dir / 'pca_multi.pkl').exists()
```

Also add `import pandas as pd` near the top of `tests/test_models.py` if not already present (check the existing import block first — `numpy as np` is already imported).

- [ ] **Step 2: Run test to verify it fails with the exact UnboundLocalError**

Run (from the worktree root, using the isolated tensorflow/xgboost venv):
```bash
/tmp/claude-1001/-home-claude-dfair/3d069162-b78a-4238-91bc-37c15564913a/scratchpad/tf_venv/.venv/bin/python -m pytest tests/test_models.py::test_run_multiclass_classification_no_pca_does_not_crash -v
```
Expected: FAIL with `UnboundLocalError: cannot access local variable 'X1_pca' where it is not associated with a value`

- [ ] **Step 3: Fix the bug**

In `src/fire/models.py`, change line 565 from:
```python
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
        X1_pca,
        y1_categorical,
        test_size=0.2,
        random_state=42,  # type: ignore
    )
```
to:
```python
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
        X1_final,
        y1_categorical,
        test_size=0.2,
        random_state=42,  # type: ignore
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
/tmp/claude-1001/-home-claude-dfair/3d069162-b78a-4238-91bc-37c15564913a/scratchpad/tf_venv/.venv/bin/python -m pytest tests/test_models.py::test_run_multiclass_classification_no_pca_does_not_crash -v
```
Expected: PASS (will take roughly a minute — trains 6 classifiers including a 20-epoch Keras FNN and runs SHAP/LIME explanations for RF/XGB/DT/KNN).

- [ ] **Step 5: Commit**

```bash
git add src/fire/models.py tests/test_models.py
git commit -m "fix: use X1_final instead of undefined X1_pca in multiclass FNN split"
```

---

### Task 2: Add end-to-end test for `run_multiclass_classification` with PCA enabled

**Files:**
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `_make_multiclass_fixture_csv` from Task 1.
- Produces: nothing new consumed elsewhere.

- [ ] **Step 1: Write the test**

Add to `tests/test_models.py`:

```python
def test_run_multiclass_classification_with_pca_writes_pca_artifact(tmp_path, monkeypatch):
    pytest.importorskip('xgboost')
    pytest.importorskip('tensorflow')
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_fixture_csv(tmp_path)

    run_multiclass_classification(str(csv_path), isUNSW=False, isPCA=True)

    models_dir = tmp_path / 'multi_class_models_legacy_fire' / 'dataset'
    assert (models_dir / 'pca_multi.pkl').exists()
    assert (models_dir / 'random_forest_multi.pkl').exists()
```

- [ ] **Step 2: Run test to verify it passes**

Run:
```bash
/tmp/claude-1001/-home-claude-dfair/3d069162-b78a-4238-91bc-37c15564913a/scratchpad/tf_venv/.venv/bin/python -m pytest tests/test_models.py::test_run_multiclass_classification_with_pca_writes_pca_artifact -v
```
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_models.py
git commit -m "test: cover run_multiclass_classification with isPCA=True"
```

---

### Task 3: Add end-to-end test for `run_binary_classification`

**Files:**
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new consumed elsewhere.

This function was already manually validated (no bugs found) via the isolated venv during investigation — it produces `scaler_binary.pkl`, `feedforward_model_binary.pkl`, `xgb_model_binary.pkl`, `svm_model_binary.pkl`, `dt_model_binary.pkl`, `knn_model_binary.pkl`, `rf_model_binary.pkl` into `binary_models_legacy_fire/<dataset>/`.

- [ ] **Step 1: Write the test**

Add to `tests/test_models.py`:

```python
def _make_binary_fixture_csv(tmp_path, n_per_class=15):
    rng = np.random.default_rng(1)
    rows = []
    for label, offset in [('Benign', 0.0), ('PortScan', 3.0)]:
        for _ in range(n_per_class):
            rows.append({
                'flow_duration': rng.normal(offset, 1.0),
                'tot_fwd_pkt': rng.normal(offset, 1.0),
                'tot_bwd_pkts': rng.normal(offset, 1.0),
                'totlen_fwd_pkts': rng.normal(offset, 1.0),
                'totlen_bwd_pkts': rng.normal(offset, 1.0),
                'Label': label,
            })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    dataset_dir = tmp_path / 'bindataset'
    dataset_dir.mkdir()
    csv_path = dataset_dir / 'aggregated.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def test_run_binary_classification_writes_expected_artifacts(tmp_path, monkeypatch):
    pytest.importorskip('xgboost')
    pytest.importorskip('tensorflow')
    monkeypatch.chdir(tmp_path)
    csv_path = _make_binary_fixture_csv(tmp_path)

    run_binary_classification(str(csv_path), isUNSW=False, isPCA=False)

    models_dir = tmp_path / 'binary_models_legacy_fire' / 'bindataset'
    for filename in (
        'scaler_binary.pkl',
        'feedforward_model_binary.pkl',
        'xgb_model_binary.pkl',
        'svm_model_binary.pkl',
        'dt_model_binary.pkl',
        'knn_model_binary.pkl',
        'rf_model_binary.pkl',
    ):
        assert (models_dir / filename).exists(), filename
    assert not (models_dir / 'pca_binary.pkl').exists()
```

- [ ] **Step 2: Run test to verify it passes**

Run:
```bash
/tmp/claude-1001/-home-claude-dfair/3d069162-b78a-4238-91bc-37c15564913a/scratchpad/tf_venv/.venv/bin/python -m pytest tests/test_models.py::test_run_binary_classification_writes_expected_artifacts -v
```
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_models.py
git commit -m "test: add end-to-end coverage for run_binary_classification"
```

---

### Task 4: Full test suite sanity check and push

**Files:** none (verification only)

- [ ] **Step 1: Run the full lean CI test suite (torch group) to confirm nothing broke and the new tests skip cleanly**

From the worktree root, using the standard per-worktree torch venv (not the isolated tf_venv):
```bash
uv run pytest -q
```
Expected: all previously-passing tests still pass; the four new `run_binary_classification`/`run_multiclass_classification` tests report `SKIPPED` (no xgboost/tensorflow in this env).

- [ ] **Step 2: Run the full suite in the isolated tensorflow/xgboost venv to confirm the new tests genuinely pass there too**

```bash
/tmp/claude-1001/-home-claude-dfair/3d069162-b78a-4238-91bc-37c15564913a/scratchpad/tf_venv/.venv/bin/python -m pytest tests/test_models.py -v
```
Expected: all tests in `test_models.py` PASS, none skipped.

- [ ] **Step 3: Push the branch**

```bash
git push origin fix-118-run-classification
```

- [ ] **Step 4: Open the PR**

```bash
gh pr create --base multiclass --head fix-118-run-classification \
  --title "fix: X1_pca UnboundLocalError + tests for run_binary/multiclass_classification (#118)" \
  --body "$(cat <<'EOF'
## Summary
- Fixes a real crash in `run_multiclass_classification`: the FNN section's train/test split unconditionally referenced `X1_pca`, which is only assigned when `isPCA=True`, causing `UnboundLocalError` on every `isPCA=False` call (confirmed via direct execution against a real dataset).
- Adds end-to-end tests (real training runs, not mocks) for `run_binary_classification` and `run_multiclass_classification` (both `isPCA=True` and `isPCA=False`), gated behind `pytest.importorskip('xgboost')`/`pytest.importorskip('tensorflow')` since these two functions are the only ones in the codebase needing both — they skip in the lean CI `torch` environment (matching existing convention for xgboost/cade-dependent tests) but were verified to genuinely pass in an isolated venv with `tensorflow`+`xgboost` installed.
- Completes the last remaining `fire.models` item of #118's scope list (`fire.main::main()` is being tracked as a separate, final part).

## Test plan
- [x] New regression test reproduces the crash before the fix, passes after.
- [x] `uv run pytest -q` (lean CI env): full suite passes, new tests skip cleanly.
- [x] Isolated tensorflow/xgboost venv: `test_models.py` fully passes, no skips.
EOF
)"
```

## Self-Review

**Spec coverage:** Task 1 fixes the confirmed crash. Tasks 2-3 cover both PCA/no-PCA paths for multiclass and the binary path (already manually validated, now regression-tested). Task 4 verifies both the lean-CI skip path and the full-dependency pass path, then ships the PR.

**Placeholder scan:** No TBD/TODO; every step has literal runnable code and exact commands.

**Type consistency:** `run_binary_classification(aggregated_file: str, isUNSW: bool, isPCA: bool) -> None` and `run_multiclass_classification(aggregated_file: str, isUNSW: bool, isPCA: bool) -> None` match their existing signatures in `src/fire/models.py:127` and `src/fire/models.py:388` exactly — no signature changes made.

**Explicitly out of scope:** `fire.main::main()` (tracked as #118's final remaining part, to be planned separately after this PR merges); adding `isPCA`/shap/lime CLI wiring changes; removing the commented-out FNN SHAP code blocks (pre-existing, unrelated to this fix).
