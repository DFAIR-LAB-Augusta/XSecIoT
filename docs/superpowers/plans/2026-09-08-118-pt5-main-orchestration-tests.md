# #118 Part 5: fire.main::main() orchestration tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real test coverage for `fire.main::main()`, the final remaining item in xseciot issue #118's scope, so that #118 can be closed.

**Architecture:** `main()` orchestrates four already-separately-tested leaf functions: `run_preprocessing` (tested in `test_preprocessing.py`), `run_binary_classification`/`run_multiclass_classification`/`run_feature_engineering` (tested in `test_models.py`, including #118 Part 4's new end-to-end tests), and `sequential_simulation`/`continuous_simulation`/`parallel_simulation` (tested in `test_simulations.py`, #118 Part 2). `main()` itself has zero coverage of its own orchestration logic: which functions it calls, in what order, with what arguments, and how the `--noPre`/`--noMod`/`--noSim` flags gate each stage. A real, full, un-mocked execution of `main()` would run preprocessing plus both modeling functions plus a 2 (model_types) x 5 (variants) x 3 (modes) = 30-combination simulation sweep — computationally infeasible for a test. Since every leaf function already has real, execution-based coverage elsewhere, `main()`'s own tests use `monkeypatch.setattr` to replace the four imported leaf functions with lightweight recording stubs, and assert on call order, call counts, and the exact arguments passed — this is testing the orchestrator's control flow, not re-testing already-covered leaf behavior.

**Tech Stack:** Python, pytest, `monkeypatch`.

## Global Constraints

- Do not modify `fire/main.py`'s actual logic — this is a test-only PR, no production code changes expected (no bug found in `main()` itself during investigation).
- Stub the four names as imported into `fire.main`'s own namespace (`fire.main.run_preprocessing`, `fire.main.run_binary_classification`, `fire.main.run_multiclass_classification`, `fire.main.run_feature_engineering`, `fire.main.sequential_simulation`, `fire.main.continuous_simulation`, `fire.main.parallel_simulation`) — `from .models import ...`/`from .simulations import ...` binds these names directly into `fire.main`'s module namespace, so patching `fire.models.run_preprocessing` would not affect what `fire.main.main()` actually calls.

---

### Task 1: Test the noPre/noMod/noSim flags each skip their stage

**Files:**
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `fire.main.main`, `fire.main._parse_args` (existing).
- Produces: nothing new consumed elsewhere.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_main.py`:

```python
import fire.main as fire_main


def _stub_leaf_functions(monkeypatch, calls):
    monkeypatch.setattr(fire_main, 'run_preprocessing', lambda *a, **kw: calls.append(('run_preprocessing', a, kw)))
    monkeypatch.setattr(
        fire_main, 'run_binary_classification', lambda *a, **kw: calls.append(('run_binary_classification', a, kw))
    )
    monkeypatch.setattr(
        fire_main,
        'run_multiclass_classification',
        lambda *a, **kw: calls.append(('run_multiclass_classification', a, kw)),
    )
    monkeypatch.setattr(
        fire_main, 'run_feature_engineering', lambda *a, **kw: calls.append(('run_feature_engineering', a, kw))
    )
    monkeypatch.setattr(
        fire_main, 'sequential_simulation', lambda *a, **kw: calls.append(('sequential_simulation', a, kw)) or []
    )
    monkeypatch.setattr(
        fire_main,
        'continuous_simulation',
        lambda *a, **kw: calls.append(('continuous_simulation', a, kw)) or (None, []),
    )
    monkeypatch.setattr(
        fire_main, 'parallel_simulation', lambda *a, **kw: calls.append(('parallel_simulation', a, kw)) or []
    )


def test_main_no_pre_skips_preprocessing(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    aggregated = tmp_path / 'aggregated_data.csv'
    aggregated.write_text('placeholder')
    monkeypatch.setattr(
        sys, 'argv', ['prog', str(dataset_path), '--noPre', '--noMod', '--noSim']
    )

    fire_main.main()

    assert calls == []


def test_main_no_mod_skips_modeling(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    monkeypatch.setattr(sys, 'argv', ['prog', str(dataset_path), '--noMod', '--noSim'])

    fire_main.main()

    names = [c[0] for c in calls]
    assert names == ['run_preprocessing']


def test_main_no_sim_skips_simulations(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    aggregated = tmp_path / 'aggregated_data.csv'
    aggregated.write_text('placeholder')
    monkeypatch.setattr(sys, 'argv', ['prog', str(dataset_path), '--noPre', '--noSim'])

    fire_main.main()

    names = [c[0] for c in calls]
    assert names == ['run_binary_classification', 'run_multiclass_classification', 'run_feature_engineering']
```

- [ ] **Step 2: Run tests to verify they fail or pass appropriately**

Run: `uv run pytest tests/test_main.py -v --no-cov`
Expected: these three new tests should already PASS against the current, unmodified `fire/main.py` (this task adds coverage, no bug is being fixed) — if any fails, investigate before proceeding since it would indicate a real behavior mismatch between the code and this plan's understanding of it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_main.py
git commit -m "test: cover fire.main main() stage-skip flags (noPre/noMod/noSim)"
```

---

### Task 2: Test `--noPre` with a missing aggregated file exits with an error

**Files:**
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `_stub_leaf_functions` from Task 1.
- Produces: nothing new consumed elsewhere.

`main()` has an explicit guard (`src/fire/main.py:41-43`): if `--noPre` is set and modeling is not skipped, but the expected `aggregated_data.csv` doesn't exist next to the dataset path, it prints an error and calls `sys.exit(1)`.

- [ ] **Step 1: Write the test**

Add to `tests/test_main.py`:

```python
def test_main_no_pre_without_aggregated_file_exits(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    monkeypatch.setattr(sys, 'argv', ['prog', str(dataset_path), '--noPre'])

    with pytest.raises(SystemExit) as exc_info:
        fire_main.main()

    assert exc_info.value.code == 1
    assert calls == []
```

Add `import pytest` near the top of `tests/test_main.py` if not already present.

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_main.py::test_main_no_pre_without_aggregated_file_exits -v --no-cov`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_main.py
git commit -m "test: cover fire.main main() missing-aggregated-file exit path"
```

---

### Task 3: Test the modeling stage passes correct arguments (unsw/pca flags threaded through)

**Files:**
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `_stub_leaf_functions` from Task 1.
- Produces: nothing new consumed elsewhere.

- [ ] **Step 1: Write the test**

Add to `tests/test_main.py`:

```python
def test_main_modeling_stage_threads_unsw_and_pca_flags(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    aggregated = tmp_path / 'aggregated_data.csv'
    aggregated.write_text('placeholder')
    monkeypatch.setattr(sys, 'argv', ['prog', str(dataset_path), '--noPre', '--noSim', '--unsw', '--pca'])

    fire_main.main()

    call_map = {name: (args, kwargs) for name, args, kwargs in calls}
    assert call_map['run_binary_classification'][0] == (str(aggregated), True, True)
    assert call_map['run_multiclass_classification'][0] == (str(aggregated), True, True)
    assert call_map['run_feature_engineering'][0] == (str(aggregated),)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_main.py::test_main_modeling_stage_threads_unsw_and_pca_flags -v --no-cov`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_main.py
git commit -m "test: cover fire.main modeling stage argument threading"
```

---

### Task 4: Test the simulation sweep covers every model_type/variant/mode combination

**Files:**
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `_stub_leaf_functions` from Task 1.
- Produces: nothing new consumed elsewhere.

`main()`'s simulation stage (`src/fire/main.py:60-92`) iterates `itertools.product(model_types, variants, modes)` where `model_types = ['binary', 'multi']`, `variants = ['dt', 'knn', 'rf', 'feedforward', 'xgb']`, `modes = ['sequential', 'continuous', 'parallel']` — 2 x 5 x 3 = 30 total calls, dispatched to one of `sequential_simulation`/`continuous_simulation`/`parallel_simulation` depending on `mode`.

- [ ] **Step 1: Write the test**

Add to `tests/test_main.py`:

```python
def test_main_simulation_stage_covers_full_sweep(tmp_path, monkeypatch):
    calls = []
    _stub_leaf_functions(monkeypatch, calls)
    dataset_path = tmp_path / 'data.csv'
    aggregated = tmp_path / 'aggregated_data.csv'
    aggregated.write_text('placeholder')
    monkeypatch.setattr(sys, 'argv', ['prog', str(dataset_path), '--noPre', '--noMod'])

    fire_main.main()

    sim_calls = [c for c in calls if c[0] in ('sequential_simulation', 'continuous_simulation', 'parallel_simulation')]
    assert len(sim_calls) == 30

    seen_combos = set()
    for name, _args, kwargs in sim_calls:
        assert kwargs['aggregated_file'] == str(aggregated)
        assert kwargs['threshold'] == 0.5
        assert kwargs['isUNSW'] is False
        seen_combos.add((kwargs['model_type'], kwargs['model_variant'], name))

    expected_modes = {
        'sequential': 'sequential_simulation',
        'continuous': 'continuous_simulation',
        'parallel': 'parallel_simulation',
    }
    expected_combos = {
        (model_type, variant, expected_modes[mode])
        for model_type in ('binary', 'multi')
        for variant in ('dt', 'knn', 'rf', 'feedforward', 'xgb')
        for mode in ('sequential', 'continuous', 'parallel')
    }
    assert seen_combos == expected_combos
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_main.py::test_main_simulation_stage_covers_full_sweep -v --no-cov`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_main.py
git commit -m "test: cover fire.main simulation sweep dispatches all 30 combinations"
```

---

### Task 5: Full suite check, push, PR, close #118

**Files:** none (verification only)

- [ ] **Step 1: Run the full lean-CI suite**

```bash
uv run pytest -q
```
Expected: exit code 0, all tests pass or skip as before, all new `test_main.py` tests pass (no xgboost/tensorflow needed — leaf functions are stubbed).

- [ ] **Step 2: Push and open the PR**

```bash
git push origin fix-118-main-orchestration
gh pr create --base multiclass --head fix-118-main-orchestration \
  --title "test: add fire.main main() orchestration coverage (#118, final part)" \
  --body "$(cat <<'EOF'
## Summary
- Adds real test coverage for `fire.main::main()`'s own orchestration logic (stage-skip flags, the noPre/missing-aggregated-file exit guard, argument threading into the modeling stage, and the full 30-combination simulation sweep dispatch) by stubbing the four already-separately-tested leaf functions it calls.
- This is the final remaining item of #118's scope (`fire/simulations.py` tests landed in PR #130, `_explain_with_lime`/`_explain_with_shap` in PR #131, `run_binary_classification`/`run_multiclass_classification` + a real crash fix in PR #132). Closes #118 once merged.

## Test plan
- [x] `uv run pytest -q`: full suite passes.
- [x] New tests exercise every `--noPre`/`--noMod`/`--noSim` combination, the missing-aggregated-file SystemExit(1) guard, and assert the exact 30 (model_type, variant, mode) combinations the simulation sweep is expected to dispatch.
EOF
)"
```

- [ ] **Step 3: Verify CI, wait for user merge, then close #118**

After the user confirms the PR is merged, close xseciot issue #118 with a comment summarizing all 5 parts (PRs #124, #130, #131, #132, and this one) and sync/clean up the worktree per the established pattern.

## Self-Review

**Spec coverage:** Task 1 covers all three skip flags. Task 2 covers the missing-file guard. Task 3 covers argument threading (unsw/pca). Task 4 covers the full simulation sweep. Task 5 verifies, ships, and closes out #118.

**Placeholder scan:** No TBD/TODO; every step has literal runnable code.

**Type consistency:** `_stub_leaf_functions(monkeypatch, calls)` signature is identical across all tasks that use it. Stub call-recording tuples are consistently `(name, args, kwargs)` throughout.

**Explicitly out of scope:** No changes to `fire/main.py` itself (no bug found); no attempt to run the real 30-combination sweep or real modeling functions from within `test_main.py` (already covered with real execution elsewhere per #118 Parts 1-4).
