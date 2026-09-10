# General Test Coverage Pt.1: conftest.py + train_ce_binary Tests (#104) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Issue #104 asks for a full test pyramid on firce's pre-existing binary runtime (which predates the multiclass work and has never had real coverage). This is the first of several PRs closing that issue — see the issue body's checklist. This PR delivers the foundational `tests/conftest.py` shared fixtures the issue explicitly asks for, and uses them to write the first genuinely-missing unit test suite: `train_ce_binary` (the only currently-working training path in the repo before this issue, per the issue body, yet completely untested).

**Architecture:** `train_ce_binary` (`src/firce/ce_model_training.py:202-537`) has several branches worth locking down: classical-variant happy paths (dt/knn/rf/svm/xgb), the feedforward path, PCA on/off, the object-dtype `Label`→`BinLabel` string-label normalization map (`'Benign'→0`, `'Attack'→1`, case variants), the already-numeric-`BinLabel` object-dtype normalization branch, the missing-label-column error, and the `df_log`-driven retraining directory path (mirroring `train_ce_multiclass`'s equivalent, already tested in `tests/test_ce_model_training.py`). This plan adds `tests/conftest.py` with fixtures generalized from the `_make_config`/`_make_*_csv` helper pattern already duplicated across `test_config_validation.py`, `test_retraining.py`, and `test_ce_model_training.py`, then writes binary tests in `test_ce_model_training.py` (the natural home — same module under test) using those new fixtures. Existing test files' local helpers are left as-is; migrating them to conftest is not required by this issue and is out of scope (avoids touching passing tests unnecessarily).

**Tech Stack:** Python 3.11+, `pytest`, `torch`, `pandas`, `scikit-learn` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`). `tests/conftest.py` is auto-discovered by pytest, no import needed.
- New tests are added to the existing `tests/test_ce_model_training.py` (which already tests `train_ce_multiclass`), not a new file — same module under test.
- Do not test the `is_unsw=True` path for `train_ce_binary` in this PR — it has a strict `FINAL_LOG_COLUMNS` superset guard (`raise RuntimeError('Diagnose this for retraining to work properly.')`) that requires a much more carefully constructed fixture than the non-UNSW paths; it's pre-existing code (not new/regressed), and is lower-value than the branches this plan targets. Leave a `# TODO` is explicitly disallowed by the no-placeholders rule — instead, this is simply not claimed as covered; the PR description will note it as a known remaining gap for a follow-up PR.

---

## File Structure

- Create: `tests/conftest.py` — shared fixtures: `device` (CPU torch device), `sim_config_factory` (builder function for a minimal valid `SimulationConfig`), `binary_flow_csv_factory`, `multiclass_flow_csv_factory` (cicflowmeter-schema synthetic datasets).
- Modify: `tests/test_ce_model_training.py` — add `train_ce_binary` unit tests using the new conftest fixtures.

---

### Task 1: `tests/conftest.py` shared fixtures

**Files:**
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `device` fixture → `torch.device('cpu')`.
- Produces: `sim_config_factory` fixture → callable `(tmp_path, **overrides) -> SimulationConfig`, matching the exact shape of the `_make_config` helper already duplicated in `test_config_validation.py`/`test_retraining.py`/`test_ce_model_training.py` (same defaults: `model_type=ModelType.MULTI, model_variant=ModelVariant.DT, ce_type=CEType.NONE, is_unsw=False, seed=0, device=<the device fixture>`, `aggregated_path`/`flows_path` pointing at a dummy placeholder CSV under `tmp_path`).
- Produces: `binary_flow_csv_factory` fixture → callable `(tmp_path, n=60, seed=0, dirname='DS') -> Path`, writing a cicflowmeter-schema CSV with a string `Label` column (`'Benign'`/`'Attack'`), matching the column shape of `_make_multiclass_csv` in `test_ce_model_training.py` but with `Label` instead of `MC_Label`.
- Produces: `multiclass_flow_csv_factory` fixture → callable with the same signature, writing the existing `MC_Label` 3-class shape (`'Benign'`/`'PortScan'`/`'XMasAttack'`) — a direct port of `_make_multiclass_csv`, made available for other test files to adopt later without duplicating it again.

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import numpy as np
import pandas as pd
import pytest
import torch

from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig


@pytest.fixture
def device() -> torch.device:
    return torch.device('cpu')


@pytest.fixture
def sim_config_factory(device):
    def _make(tmp_path, **overrides):
        dummy = tmp_path / 'dummy.csv'
        if not dummy.exists():
            dummy.write_text('a\n1\n')
        defaults = dict(
            model_type=ModelType.MULTI,
            model_variant=ModelVariant.DT,
            ce_type=CEType.NONE,
            aggregated_path=dummy,
            flows_path=dummy,
            is_unsw=False,
            seed=0,
            device=device,
        )
        defaults.update(overrides)
        return SimulationConfig(**defaults)

    return _make


@pytest.fixture
def binary_flow_csv_factory():
    def _make(tmp_path, n=60, seed=0, dirname='DS'):
        rng = np.random.default_rng(seed)
        ds_dir = tmp_path / dirname
        ds_dir.mkdir(exist_ok=True)
        csv_path = ds_dir / 'flows.csv'
        labels = np.array(['Benign', 'Attack'])
        idx = rng.integers(0, 2, size=n)
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
            'Label': labels[idx],
        })
        df.to_csv(csv_path, index=False)
        return csv_path

    return _make


@pytest.fixture
def multiclass_flow_csv_factory():
    def _make(tmp_path, n=60, seed=0, dirname='DS'):
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

    return _make
```

- [ ] **Step 2: Verify pytest picks up the new conftest without breaking collection**

Run: `uv run pytest --collect-only -q 2>&1 | tail -5`
Expected: collection succeeds, no import errors, same total test count as before plus the new file's tests once Task 2 lands (right now, 0 new tests yet — just confirming `conftest.py` itself doesn't break anything).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add tests/conftest.py with shared SimulationConfig/flow-CSV fixtures"
```

---

### Task 2: `train_ce_binary` unit tests

**Files:**
- Modify: `tests/test_ce_model_training.py`

**Interfaces:**
- Consumes: `train_ce_binary` (`firce.ce_model_training`), `sim_config_factory` and `binary_flow_csv_factory` fixtures from `tests/conftest.py` (Task 1).

- [ ] **Step 1: Add `import train_ce_binary` and write the classical-variant + PCA + feedforward + xgb tests**

In `tests/test_ce_model_training.py`, change the import line:

```python
from firce.ce_model_training import train_ce_multiclass
```

to:

```python
from firce.ce_model_training import train_ce_binary, train_ce_multiclass
```

Add these tests at the end of the file:

```python
@pytest.mark.parametrize('variant', [ModelVariant.DT, ModelVariant.KNN, ModelVariant.RF, ModelVariant.SVM])
def test_train_ce_binary_classical_variants(tmp_path, monkeypatch, sim_config_factory, binary_flow_csv_factory, variant):
    monkeypatch.chdir(tmp_path)
    csv_path = binary_flow_csv_factory(tmp_path)
    config = sim_config_factory(tmp_path, model_type=ModelType.BINARY, model_variant=variant)

    outdir = train_ce_binary(config, str(csv_path), PerformanceStats())

    assert outdir.resolve() == (tmp_path / 'binary_models' / 'DS').resolve()
    assert (outdir / 'scaler_binary.pkl').exists()
    assert (outdir / f'{variant.value}_model_binary.pkl').exists()
    assert not (outdir / 'pca_binary.pkl').exists()


def test_train_ce_binary_with_pca_writes_pca_artifact(tmp_path, monkeypatch, sim_config_factory, binary_flow_csv_factory):
    monkeypatch.chdir(tmp_path)
    csv_path = binary_flow_csv_factory(tmp_path)
    config = sim_config_factory(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT, use_pca=True)

    outdir = train_ce_binary(config, str(csv_path), PerformanceStats())

    assert (outdir / 'pca_binary.pkl').exists()


def test_train_ce_binary_feedforward(tmp_path, monkeypatch, sim_config_factory, binary_flow_csv_factory):
    monkeypatch.chdir(tmp_path)
    csv_path = binary_flow_csv_factory(tmp_path)
    config = sim_config_factory(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.FEEDFORWARD)

    outdir = train_ce_binary(config, str(csv_path), PerformanceStats())

    ckpt_path = outdir / 'feedforward_model_binary.pt'
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert ckpt['input_dim'] > 0


def test_train_ce_binary_xgb(tmp_path, monkeypatch, sim_config_factory, binary_flow_csv_factory):
    pytest.importorskip('xgboost')
    monkeypatch.chdir(tmp_path)
    csv_path = binary_flow_csv_factory(tmp_path)
    config = sim_config_factory(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.XGB)

    outdir = train_ce_binary(config, str(csv_path), PerformanceStats())

    assert (outdir / 'xgb_model_binary.pkl').exists()
```

- [ ] **Step 2: Add label-normalization branch tests**

Add to the same file:

```python
def test_train_ce_binary_normalizes_string_labels(tmp_path, monkeypatch, sim_config_factory):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    n = 40
    rng = np.random.default_rng(0)
    pd.DataFrame({
        'flow_duration': rng.random(n) * 100,
        'tot_fwd_pkt': rng.integers(1, 50, size=n),
        'Label': ['Benign' if i % 2 == 0 else 'Attack' for i in range(n)],
    }).to_csv(csv_path, index=False)
    config = sim_config_factory(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)

    outdir = train_ce_binary(config, str(csv_path), PerformanceStats())

    import joblib

    model = joblib.load(outdir / 'dt_model_binary.pkl')
    assert set(model.classes_.tolist()) == {0, 1}


def test_train_ce_binary_string_binlabel_crashes_under_pandas3(tmp_path, monkeypatch, sim_config_factory):
    # KNOWN BUG (#114): pandas 3's default infer_string=True means CSV-read
    # string columns are no longer literally `dtype == object`, so the
    # `if df['BinLabel'].dtype == object:` normalization check never fires,
    # and the code falls through to np.isfinite() on the raw string column,
    # which crashes. This test documents actual current behavior; it is not
    # asserting this is correct - see #114 for the fix.
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    n = 40
    rng = np.random.default_rng(0)
    pd.DataFrame({
        'flow_duration': rng.random(n) * 100,
        'tot_fwd_pkt': rng.integers(1, 50, size=n),
        'BinLabel': ['Benign' if i % 2 == 0 else 'Attack' for i in range(n)],
    }).to_csv(csv_path, index=False)
    config = sim_config_factory(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)

    with pytest.raises(TypeError, match='isfinite'):
        train_ce_binary(config, str(csv_path), PerformanceStats())


def test_train_ce_binary_missing_label_column_raises(tmp_path, monkeypatch, sim_config_factory):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    pd.DataFrame({'flow_duration': [1, 2, 3], 'tot_fwd_pkt': [1, 2, 3]}).to_csv(csv_path, index=False)
    config = sim_config_factory(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)

    with pytest.raises(ValueError, match="must contain either 'Label' or 'BinLabel'"):
        train_ce_binary(config, str(csv_path), PerformanceStats())
```

- [ ] **Step 3: Add the `df_log` retraining-directory test**

Add to the same file:

```python
def test_train_ce_binary_with_df_log_writes_retraining_dir(tmp_path, monkeypatch, sim_config_factory, binary_flow_csv_factory):
    monkeypatch.chdir(tmp_path)
    csv_path = binary_flow_csv_factory(tmp_path)
    df_log = pd.read_csv(csv_path)
    config = sim_config_factory(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)

    outdir = train_ce_binary(config, str(csv_path), PerformanceStats(), df_log=df_log)

    assert outdir.name.startswith('Model_dt_Retraining_')
    assert (outdir / 'dt_model_binary.pkl').exists()
```

- [ ] **Step 4: Add the `PerformanceStats` import and run the new tests**

At the top of `tests/test_ce_model_training.py`, add:

```python
from firce.utils.perf_stats import PerformanceStats
```

Run: `uv run pytest tests/test_ce_model_training.py -v`
Expected: all existing multiclass tests still pass, plus the 9 new binary tests (4 classical variants + pca + feedforward + xgb-skippable + 2 label-normalization + missing-column + df_log = 9 new test functions, one parametrized 4 ways = 12 new test cases total).

- [ ] **Step 5: Commit**

```bash
git add tests/test_ce_model_training.py
git commit -m "test: add train_ce_binary unit tests (classical variants, PCA, feedforward, xgb, label normalization, df_log retraining)"
```

---

### Task 3: Lint, format, and final regression

**Files:**
- `tests/conftest.py`, `tests/test_ce_model_training.py` (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any issues**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
```

If issues appear, run `uv run ruff check --fix .` and `uv run ruff format .`, then re-run the full suite.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures beyond the pre-existing/expected skips (xgboost not installed in lean group, unrelated MC rolling-logger skips)

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format conftest and train_ce_binary tests"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** This is Part 1 of #104's larger checklist. It delivers exactly the `tests/conftest.py` piece the issue asks for (minimal `SimulationConfig` builder, synthetic binary + multiclass flow-dataset fixtures) and the `train_ce_binary` unit-test piece ("happy path + the various label-normalization branches already in the function"). Remaining checklist items — component/integration tests for `bootstrap.py`/`retraining.py`/`inference.py`/`monitoring.py`, binary-path `conformalEval`/`drift_monitor` tests, the full e2e test, and the dead-file triage for `test_simulations.py`/`test_main.py` — are follow-up PRs against the same issue, not claimed as done here.

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code. The `is_unsw` binary path is explicitly named as *not* covered in Global Constraints rather than stubbed with a placeholder test.

**Type consistency:** `sim_config_factory(tmp_path, **overrides)` and `binary_flow_csv_factory(tmp_path, n=60, seed=0, dirname='DS')` signatures in Task 2's tests match Task 1's fixture definitions exactly.

**Explicitly out of scope for this PR:** `is_unsw=True` binary path, component/integration tests, e2e test, dead test file triage, drift_monitor tests, binary conformalEval tests — all tracked as remaining #104 scope, to be picked up in subsequent PRs.
