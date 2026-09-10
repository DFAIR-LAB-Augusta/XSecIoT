# #118 Pt.2: fire.simulations Orchestration Function Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue #118 (Part 1, lazy xgboost/tensorflow imports, is PR #124/merged). This part adds real tests for `fire/simulations.py`'s three batch-simulation orchestration functions (`sequential_simulation`, `continuous_simulation`, `parallel_simulation`) — the older pipeline, still used by `fire/main.py`'s CLI sweep, distinct from `firce.pipelines.simulation_pipeline.run_simulation_pipeline` (already covered by #104/#125's e2e tests).

**Architecture:** All three functions call `load_simulation_objects(aggregated_file, model_type, model_variant)` (from the same module) to load pre-trained artifacts from `binary_models/<dataset>/` or `multi_class_models/<dataset>/` — none of them train anything themselves, they only predict against already-trained models. Critically, none of them pass `use_pca` explicitly to `load_simulation_objects`, so it uses that function's default (`use_pca=True`) — meaning the fixture must be trained **with** PCA (`train_ce_binary(..., use_pca=True)` via a `SimulationConfig` with `use_pca=True`), or `load_simulation_objects` crashes trying to load a nonexistent `pca_binary.pkl`. Manually validated during this plan's design (all three functions run correctly end-to-end against a small, freshly-trained fixture):
- `sequential_simulation` streams the CSV in fixed chunks, predicts each chunk independently via the module's own `process_chunk` (a distinct function from `firce.runtime.inference.process_chunk`, despite the identical name), returns a flat list of `'Benign'`/`'Attack'` string predictions (one per row).
- `continuous_simulation` maintains a rolling time window keyed on an `end_time_x` datetime column, re-predicting the *entire current window* on every chunk (not just new rows) — so with a `window_duration` long enough to never trim anything (as in a tiny, fast-arriving test fixture), the returned `true_labels`/`preds` lists are longer than the row count (confirmed: 30 rows, 6 chunks of 5, cumulative window sizes 5+10+...+30 = 105 total predictions) — this is the function's actual designed behavior, not a bug, and the test asserts on this exact cumulative-sum relationship rather than a naive row-count match.
- `parallel_simulation` splits the CSV into chunks and predicts them via `multiprocessing.Pool` (sklearn model/scaler/pca objects pickle cleanly), returns a flat list matching the original row count.

**Tech Stack:** Python 3.11+, `pytest`, `pandas`, `scikit-learn` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`). Add to the existing `tests/test_simulations.py` (revived in #104 Part 4 with `_parse_args`/`_get_dataset_name`/`preprocess_chunk` tests) rather than a new file.
- Always pass `delay=0` to `sequential_simulation`/`continuous_simulation` in tests — both default to `delay=1.0` (a real `time.sleep()` per chunk), which would make tests slow without changing correctness.
- Train fixtures with `use_pca=True` — required since none of these three functions override `load_simulation_objects`'s `use_pca=True` default.
- Use `num_processes=1` for the `parallel_simulation` test to avoid subprocess-related CI flakiness/slowness while still exercising the real `multiprocessing.Pool` code path (a pool with 1 worker still pickles/unpickles the worker function and runs through the same code, just without true parallelism).

---

## File Structure

- Modify: `tests/test_simulations.py` — add binary-path tests for all three orchestration functions, using a shared fixture helper.

---

### Task 1: Shared fixture helper + `sequential_simulation` test

**Files:**
- Modify: `tests/test_simulations.py`

**Interfaces:**
- Consumes: `sequential_simulation` (`fire.simulations`), `train_ce_binary` (`firce.ce_model_training`), `SimulationConfig`/`ModelType`/`ModelVariant`/`CEType` (`firce.utils.config`), `PerformanceStats` (`firce.utils.perf_stats`).
- Produces: `_train_binary_fixture(tmp_path, csv_path) -> None` helper, reused by Tasks 2-3.

- [ ] **Step 1: Add imports and the shared fixture helper**

At the top of `tests/test_simulations.py`, add:

```python
import torch

from firce.ce_model_training import train_ce_binary
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig
from firce.utils.perf_stats import PerformanceStats
from fire.simulations import continuous_simulation, parallel_simulation, sequential_simulation

DEVICE = torch.device('cpu')


def _train_binary_fixture(tmp_path, csv_path):
    config = SimulationConfig(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=csv_path,
        flows_path=csv_path,
        is_unsw=False,
        seed=0,
        device=DEVICE,
        use_pca=True,
    )
    train_ce_binary(config, str(csv_path), PerformanceStats())
```

- [ ] **Step 2: Add the `sequential_simulation` test**

```python
def test_sequential_simulation_binary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    n = 30
    rng = np.random.default_rng(0)
    labels = np.array(['Benign', 'Attack'])
    idx = rng.integers(0, 2, size=n)
    pd.DataFrame({
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
        'Label': labels[idx],
    }).to_csv(csv_path, index=False)

    _train_binary_fixture(tmp_path, csv_path)

    preds = sequential_simulation(str(csv_path), model_type='binary', model_variant='dt', chunk_size=5, delay=0)

    assert len(preds) == n
    assert set(preds) <= {'Benign', 'Attack'}
```

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/test_simulations.py::test_sequential_simulation_binary -v`
Expected: PASS — manually validated during this plan's design with this exact fixture shape.

- [ ] **Step 4: Commit**

```bash
git add tests/test_simulations.py
git commit -m "test: add sequential_simulation binary test"
```

---

### Task 2: `continuous_simulation` test

**Files:**
- Modify: `tests/test_simulations.py`

- [ ] **Step 1: Add the test**

```python
def test_continuous_simulation_binary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    n = 30
    rng = np.random.default_rng(0)
    labels = np.array(['Benign', 'Attack'])
    idx = rng.integers(0, 2, size=n)
    pd.DataFrame({
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
        'Label': labels[idx],
        'end_time_x': pd.date_range('2020-01-01', periods=n, freq='s').astype(str),
    }).to_csv(csv_path, index=False)

    _train_binary_fixture(tmp_path, csv_path)

    true_labels, preds = continuous_simulation(
        str(csv_path), model_type='binary', model_variant='dt', chunk_size=5, window_duration=300, delay=0
    )

    # window_duration=300s never trims this 30-row/30-second fixture, so each
    # of the 6 chunks re-predicts the entire accumulated window (5, 10, 15,
    # 20, 25, 30 rows) rather than just new rows - this is the function's
    # actual designed behavior (see plan Architecture section), not a bug.
    expected_total = sum(range(5, n + 1, 5))
    assert len(true_labels) == expected_total
    assert len(preds) == expected_total
    assert set(preds) <= {'Benign', 'Attack'}
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_simulations.py::test_continuous_simulation_binary -v`
Expected: PASS — manually validated during this plan's design (`true_labels`/`preds` both length 105 for this exact 30-row/chunk_size=5 fixture).

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulations.py
git commit -m "test: add continuous_simulation binary test"
```

---

### Task 3: `parallel_simulation` test

**Files:**
- Modify: `tests/test_simulations.py`

- [ ] **Step 1: Add the test**

```python
def test_parallel_simulation_binary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    n = 30
    rng = np.random.default_rng(0)
    labels = np.array(['Benign', 'Attack'])
    idx = rng.integers(0, 2, size=n)
    pd.DataFrame({
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
        'Label': labels[idx],
    }).to_csv(csv_path, index=False)

    _train_binary_fixture(tmp_path, csv_path)

    preds = parallel_simulation(str(csv_path), model_type='binary', model_variant='dt', chunk_size=10, num_processes=1)

    assert len(preds) == n
    assert set(preds) <= {'Benign', 'Attack'}
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_simulations.py::test_parallel_simulation_binary -v`
Expected: PASS — manually validated during this plan's design (with `num_processes=2`; `num_processes=1` exercises the identical `multiprocessing.Pool` code path with less CI overhead).

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulations.py
git commit -m "test: add parallel_simulation binary test"
```

---

### Task 4: Lint, format, and final regression

**Files:**
- `tests/test_simulations.py` (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any issues**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
```

If issues appear, run `uv run ruff check --fix .` and `uv run ruff format .`, then re-run the full suite.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures beyond the pre-existing/expected skips.

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format fire.simulations orchestration tests"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** #118 asks for real tests on `fire.simulations`'s `sequential_simulation`/`continuous_simulation`/`parallel_simulation`. All three covered, each validated by a real manual run during this plan's design before being written into the plan (not guessed).

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code.

**Type consistency:** `_train_binary_fixture(tmp_path, csv_path)` signature matches its three call sites exactly.

**Explicitly out of scope:** Multiclass paths for these three functions (they predate the `MC_Label` convention and `continuous_simulation`'s multi-branch uses raw `Label`, not `MC_Label` — a pre-existing staleness not chased here), `fire.main::main()`, and `fire.models`'s `run_binary_classification`/`run_multiclass_classification`/`_explain_with_lime`/`_explain_with_shap` remain for later parts of #118.
