# General Test Coverage Pt.4: Full E2E Test + Dead-File Triage (#104) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Final part of #104 (Parts 1-3 are `2026-09-08-general-test-coverage-pt{1,2,3}-*.md`, all merged). Delivers the issue's last two asks: "At least one full end-to-end test" and "Either delete or revive the dead/commented-out test files (`test_simulations.py`, `test_main.py`, `test_models.py`)." This is the part that closes #104.

**Architecture:**

*E2E test.* Investigation confirmed the real, current, top-level live-simulation driver is `firce.pipelines.simulation_pipeline.run_simulation_pipeline(config)` — it calls `initialize_simulation_runtime`, then streams `config.flows_path` in chunks through `firce.runtime.inference.process_chunk` (predict → log → detect drift → retrain-on-drift), then `plot_results`. This is a *different, newer* pipeline than `fire.simulations`'s `sequential_simulation`/`continuous_simulation`/`parallel_simulation` (which is what `fire.main`'s batch CLI still uses — see dead-file section below). `run_simulation_pipeline` was manually smoke-tested against real data during this plan's design and confirmed to: complete without error, write `binary_models/<dataset>/{dt_model_binary.pkl,scaler_binary.pkl}`, and write `logging/chunk_size_<N>/<DS>/dt_ice_binary_0_0_accuracy_plot.png` (`plot_results` needs `config.aggregated_path`'s string to contain `'CETrain'`, `'UNSW_NB15'`, or `'CIC_UNSW'` — `_resolve_dataset_type` raises otherwise).

Manually testing showed that a distribution shift alone (scaling real feature values ×100) does *not* reliably trigger the real CE drift monitor — a `DecisionTreeClassifier` can stay confidently (and by luck, correctly) classified even on wildly out-of-range inputs, so nonconformity scores stay low. Rather than fight for a naturally-drift-inducing fixture, drift→retrain wiring is tested *separately and deterministically*: a stub monitor whose `detect()` always reports `chunk_drift=True` is wired into a real, trained `SimulationRuntime`, and `firce.runtime.inference.process_chunk` is called directly with `retrain_runtime` monkeypatched to a call-recording spy — decoupling "does drift correctly trigger retraining" from "does real CE calibration detect this specific distribution shift" (the same kind of decoupling `_StubMonitor` already uses in `tests/test_retraining.py`). This also closes a real, previously-uncovered gap: `process_chunk`/`_detect_chunk_drift`/`_handle_detected_drift` (the streaming orchestration glue) had zero tests before this plan — Part 2's `inference.py` tests only covered the lower-level `_prepare_chunk`/`predict_row`/`_record_prediction_outcome` functions directly.

Both tests use two small (~150/200-row) fixture CSVs sampled from the real `CEFlows2_merged.csv` dataset (checked into `~/dfair/repos/`, *outside* this git repo and therefore unavailable in CI) rather than a hand-built synthetic CSV — hand-building one is infeasible for the non-UNSW rolling-log path, because `_append_row_to_rolling_log` does a *strict-width* `CircularDequeLogger.append()` (raises `ValueError` on any column-count mismatch against `get_rolling_columns(config)`, which resolves to a specific 77-column slice of the real 83-column cicflowmeter schema for non-UNSW). The real dataset already has exactly this schema, so sampling it sidesteps the entire column-matching problem. The two fixture CSVs are committed to `tests/fixtures/`.

*Dead-file triage.* All three commented-out files' targets were checked against current source:
- `test_simulations.py` → `fire/simulations.py`'s `_parse_args`, `_get_dataset_name`, `preprocess_chunk`, `sequential_simulation`, `continuous_simulation`, `parallel_simulation` **all still exist**.
- `test_main.py` → `fire/main.py`'s `_parse_args`, `main` **still exist**.
- `test_models.py` → `fire/models.py`'s `_parse_args`, `_explain_with_lime`, `_explain_with_shap`, `run_feature_engineering`, plus `run_binary_classification`/`run_multiclass_classification` **all still exist**.

None are deletion candidates — all three are revival candidates. This plan revives each file with real, passing tests for the small, pure, low-risk functions (`_parse_args`, `_get_dataset_name`) that answer the issue's literal "does the code still exist in this shape" question and replace the fully-commented dead state with a real active test file. The heavier orchestration functions (`sequential_simulation`/`continuous_simulation`/`parallel_simulation`, `main()`, `run_binary_classification`/`run_multiclass_classification`, `_explain_with_lime`/`_explain_with_shap`) are a substantially larger undertaking (each needs its own realistic fixture/mocking setup, similar in scope to this plan's own e2e investigation) and are filed as a follow-up issue rather than attempted here — this plan does not claim full revival, only "confirmed alive, not dead, and no longer fully commented out."

**Tech Stack:** Python 3.11+, `pytest`, `torch`, `pandas`, `scikit-learn` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`). Fixture data goes in `tests/fixtures/`.
- Do not reference `/home/claude/dfair/repos/CEFlows2_merged.csv` (or any path outside this git repo) from test code — it does not exist in CI. Only the committed `tests/fixtures/*.csv` files may be read at test time.
- `_resolve_dataset_type` (`firce/utils/plotter.py`) requires `str(config.aggregated_path)` to contain `'CETrain'`, `'UNSW_NB15'`, or `'CIC_UNSW'` — name the e2e test's dataset directory `CETrain_e2e` to satisfy this.
- This is the last part of #104 — after this merges, close the issue.

---

## File Structure

- Create: `tests/fixtures/ce_flows_e2e_train.csv` (150 rows, mixed `BinLabel` 0/1, real cicflowmeter-schema data) and `tests/fixtures/ce_flows_e2e_stream.csv` (200 rows, `BinLabel`=0, same schema) — already generated and validated during this plan's design; committing them is Task 1.
- Create: `tests/test_e2e_simulation.py` — the full pipeline e2e test and the drift→retrain wiring test.
- Modify: `tests/test_simulations.py` — replace fully-commented content with real `_parse_args`/`_get_dataset_name`/`preprocess_chunk` tests.
- Modify: `tests/test_main.py` — replace fully-commented content with a real `_parse_args` test.
- Modify: `tests/test_models.py` — replace fully-commented content with a real `_parse_args` test.

---

### Task 1: Commit the e2e fixture CSVs

**Files:**
- Create: `tests/fixtures/ce_flows_e2e_train.csv`
- Create: `tests/fixtures/ce_flows_e2e_stream.csv`

These files already exist in the current worktree (generated during this plan's design/validation phase, sampled from the real `CEFlows2_merged.csv`: 150 rows — 100 `BinLabel=0` + 50 `BinLabel=1` — for training; 200 `BinLabel=0` rows for streaming). No regeneration needed.

- [ ] **Step 1: Verify the fixture files are present and correctly shaped**

Run:
```bash
uv run python -c "
import pandas as pd
train = pd.read_csv('tests/fixtures/ce_flows_e2e_train.csv')
stream = pd.read_csv('tests/fixtures/ce_flows_e2e_stream.csv')
assert train.shape == (150, 83), train.shape
assert stream.shape == (200, 83), stream.shape
assert set(train['BinLabel'].unique()) == {0, 1}
assert set(stream['BinLabel'].unique()) == {0}
assert not train.isna().any().any()
assert not stream.isna().any().any()
print('fixtures OK')
"
```
Expected: `fixtures OK`

- [ ] **Step 2: Commit**

```bash
git add tests/fixtures/ce_flows_e2e_train.csv tests/fixtures/ce_flows_e2e_stream.csv
git commit -m "test: add e2e fixture CSVs sampled from real CEFlows2_merged data"
```

---

### Task 2: Full pipeline e2e test

**Files:**
- Create: `tests/test_e2e_simulation.py`

**Interfaces:**
- Consumes: `run_simulation_pipeline` (`firce.pipelines.simulation_pipeline`), `SimulationConfig`/`CEType`/`ModelType`/`ModelVariant`/`MonitorType` (`firce.utils.config`).

- [ ] **Step 1: Write the test**

```python
import shutil

from pathlib import Path

import torch

from firce.pipelines.simulation_pipeline import run_simulation_pipeline
from firce.utils.config import CEType, ModelType, ModelVariant, MonitorType, SimulationConfig

DEVICE = torch.device('cpu')
FIXTURES = Path(__file__).parent / 'fixtures'


def test_run_simulation_pipeline_end_to_end_binary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'CETrain_e2e'
    ds_dir.mkdir()
    train_csv = ds_dir / 'train.csv'
    stream_csv = ds_dir / 'stream.csv'
    shutil.copy(FIXTURES / 'ce_flows_e2e_train.csv', train_csv)
    shutil.copy(FIXTURES / 'ce_flows_e2e_stream.csv', stream_csv)

    config = SimulationConfig(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.ICE,
        aggregated_path=train_csv,
        flows_path=stream_csv,
        is_unsw=False,
        seed=0,
        device=DEVICE,
        monitor_type=MonitorType.CE,
        chunk_size=50,
        use_circular_logger=True,
        use_pca=False,
    )

    run_simulation_pipeline(config)

    model_dir = tmp_path / 'binary_models' / 'CETrain_e2e'
    assert (model_dir / 'dt_model_binary.pkl').exists()
    assert (model_dir / 'scaler_binary.pkl').exists()

    plot_dir = tmp_path / 'logging' / 'chunk_size_50' / 'DFAIR'
    assert (plot_dir / 'dt_ice_binary_0_0_accuracy_plot.png').exists()
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_e2e_simulation.py::test_run_simulation_pipeline_end_to_end_binary -v`
Expected: PASS — this was manually smoke-tested during this plan's design with these exact fixtures/config and confirmed to complete, write both model artifacts, and write the accuracy plot.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_simulation.py
git commit -m "test: add full end-to-end simulation pipeline test"
```

---

### Task 3: Drift-detection → retraining wiring test

**Files:**
- Modify: `tests/test_e2e_simulation.py`

**Interfaces:**
- Consumes: `train_ce_binary` (`firce.ce_model_training`), `process_chunk` (`firce.runtime.inference`), `get_rolling_columns` (`firce.runtime.bootstrap`), `SimulationRuntime` (`firce.runtime.sim_types`), `DriftDetectionResult` (`firce.drift_monitor.base`), `CircularDequeLogger` (`firce.utils.circular_logger`), `PerformanceStats` (`firce.utils.perf_stats`).

- [ ] **Step 1: Add the test**

Add to `tests/test_e2e_simulation.py`:

```python
import joblib
import numpy as np
import pandas as pd

from firce.ce_model_training import train_ce_binary
from firce.drift_monitor.base import DriftDetectionResult
from firce.runtime.bootstrap import get_rolling_columns
from firce.runtime.inference import process_chunk
from firce.runtime.sim_types import SimulationRuntime
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.perf_stats import PerformanceStats


class _AlwaysDriftMonitor:
    """Duck-typed drift monitor stub that always reports drift - decouples
    this test from real CE calibration internals (already covered by #93),
    so it only exercises process_chunk's own drift -> retrain wiring."""

    def __init__(self):
        self.fit_calls: list = []

    def fit(self, X, y, perf_stats):
        self.fit_calls.append((X, y))

    def detect(self, X):
        return DriftDetectionResult(row_flags=np.ones(len(X), dtype=bool), chunk_drift=True)


def test_process_chunk_triggers_retrain_on_detected_drift(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    train_csv = ds_dir / 'train.csv'
    shutil.copy(FIXTURES / 'ce_flows_e2e_train.csv', train_csv)

    config = SimulationConfig(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=train_csv,
        flows_path=train_csv,
        is_unsw=False,
        seed=0,
        device=DEVICE,
    )

    model_dir = train_ce_binary(config, str(train_csv), PerformanceStats())
    scaler = joblib.load(model_dir / 'scaler_binary.pkl')
    model = joblib.load(model_dir / 'dt_model_binary.pkl')

    retrain_calls = []
    monkeypatch.setattr(
        'firce.runtime.inference.retrain_runtime',
        lambda runtime: retrain_calls.append(runtime),
    )

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=500, columns=get_rolling_columns(config)),
        scaler=scaler,
        pca=None,
        model=model,
        label_encoder=None,
        monitor=_AlwaysDriftMonitor(),
        train_df=pd.DataFrame(),
    )

    chunk = pd.read_csv(FIXTURES / 'ce_flows_e2e_stream.csv').head(20)
    process_chunk(runtime, chunk, chunk_num=0)

    assert len(retrain_calls) == 1
    assert retrain_calls[0] is runtime
    assert runtime.perf_stats.drift_detected_indices == [0]
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_e2e_simulation.py -v`
Expected: both tests pass (2 total).

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_simulation.py
git commit -m "test: add drift-detection to retraining wiring test for process_chunk"
```

---

**Discovered during Task 5/6 execution**: `fire/main.py` imports `fire.models` at module level, and `fire/models.py` itself has *four* eager top-level heavy imports (`shap`, `lime`, `xgboost`, `tensorflow`) — of which `shap`/`lime` happen to already be in the lean CI group, but `xgboost`/`tensorflow` are not. This blocks collection of both `test_main.py` and `test_models.py` in the lean environment regardless of which specific function is under test. Making all four imports lazy (matching the established `firce/`-side pattern from #88/#89) is a substantially bigger undertaking than this plan's scope — both files are gated with module-level `pytest.importorskip('xgboost')`/`pytest.importorskip('tensorflow')` instead, matching the same honest-skip pattern used for `cade` in Part 3. This is folded into Task 7's follow-up issue as an additional, more foundational item.

### Task 4: Revive `test_simulations.py`

**Files:**
- Modify: `tests/test_simulations.py` (currently 155 lines, fully commented out)

**Interfaces:**
- Consumes: `_parse_args`, `_get_dataset_name`, `preprocess_chunk` (`fire.simulations`).

- [ ] **Step 1: Replace the file's content**

Replace the entire contents of `tests/test_simulations.py` with:

```python
import sys

import numpy as np
import pandas as pd

from fire.simulations import _get_dataset_name, _parse_args, preprocess_chunk


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'agg.csv'])
    args = _parse_args()

    assert args.aggregated_file == 'agg.csv'
    assert args.mode == 'sequential'
    assert args.model_type == 'binary'
    assert args.model_variant == 'dt'
    assert args.chunk_size == 1000
    assert not args.unsw


def test_parse_args_overrides(monkeypatch):
    monkeypatch.setattr(
        sys, 'argv', ['prog', 'agg.csv', '--mode', 'parallel', '--model_type', 'multi', '--model_variant', 'rf', '--unsw']
    )
    args = _parse_args()

    assert args.mode == 'parallel'
    assert args.model_type == 'multi'
    assert args.model_variant == 'rf'
    assert args.unsw


def test_get_dataset_name():
    assert _get_dataset_name('/foo/BAR/agg.csv') == 'BAR'


def test_preprocess_chunk_drops_columns_and_fills_na():
    df = pd.DataFrame({'a': [1, np.nan], 'b': [np.nan, 2], 'drop': [9, 9]})

    cleaned = preprocess_chunk(df, ['drop'])

    assert 'drop' not in cleaned.columns
    assert not cleaned.isna().any().any()
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_simulations.py -v`
Expected: all 4 pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulations.py
git commit -m "test: revive test_simulations.py with real _parse_args/_get_dataset_name/preprocess_chunk tests"
```

---

### Task 5: Revive `test_main.py`

**Files:**
- Modify: `tests/test_main.py` (currently 153 lines, fully commented out)

**Interfaces:**
- Consumes: `_parse_args` (`fire.main`).

- [ ] **Step 1: Replace the file's content**

Replace the entire contents of `tests/test_main.py` with:

```python
import sys

import pytest

pytest.importorskip('xgboost')
pytest.importorskip('tensorflow')

from fire.main import _parse_args


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'data.csv'])
    args = _parse_args()

    assert args.dataset_path == 'data.csv'
    assert args.window_size == '5s'
    assert args.step_size == '1s'
    assert not args.unsw
    assert not args.pca
    assert not args.noPre
    assert not args.noMod
    assert not args.noSim


def test_parse_args_skip_flags(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'data.csv', '--noPre', '--noMod', '--noSim', '--unsw'])
    args = _parse_args()

    assert args.noPre
    assert args.noMod
    assert args.noSim
    assert args.unsw
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_main.py -v`
Expected: both pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_main.py
git commit -m "test: revive test_main.py with real _parse_args tests"
```

---

### Task 6: Revive `test_models.py`

**Files:**
- Modify: `tests/test_models.py` (currently 109 lines, fully commented out)

**Interfaces:**
- Consumes: `_parse_args` (`fire.models`).

- [ ] **Step 1: Replace the file's content**

Replace the entire contents of `tests/test_models.py` with:

```python
import sys

import pytest

pytest.importorskip('xgboost')
pytest.importorskip('tensorflow')

from fire.models import _parse_args


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'data.csv'])
    args = _parse_args()

    assert args.aggregated_file == 'data.csv'
    assert not args.unsw
    assert not args.pca
    assert not args.shap
    assert not args.lime


def test_parse_args_xai_flags(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'data.csv', '--shap', '--lime', '--pca'])
    args = _parse_args()

    assert args.shap
    assert args.lime
    assert args.pca
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_models.py -v`
Expected: both pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_models.py
git commit -m "test: revive test_models.py with real _parse_args tests"
```

---

### Task 7: File a follow-up issue for deeper orchestration-function coverage

**Files:** None (GitHub issue only)

- [ ] **Step 1: File the issue**

```bash
export GH_TOKEN=$(cat /home/claude/dfair_pat.env | tr -d '\n')
gh issue create --repo DFAIR-LAB-Augusta/XSecIoT --title "Deeper test coverage for fire.simulations/fire.main/fire.models orchestration functions" --body "$(cat <<'BODYEOF'
## Problem
#104's dead-file triage (test_simulations.py, test_main.py, test_models.py) confirmed all three target modules still exist in a testable shape, and revived each file with tests for the small, pure functions (_parse_args, _get_dataset_name, preprocess_chunk). The heavier orchestration functions in the same modules remain untested:

- fire/simulations.py: sequential_simulation, continuous_simulation, parallel_simulation (the older batch-simulation pipeline, still used by fire/main.py's CLI sweep - distinct from firce.pipelines.simulation_pipeline.run_simulation_pipeline, which is the newer drift-aware online pipeline covered by #104's e2e test).
- fire/main.py: main() (full preprocessing -> modeling -> simulation-sweep orchestration).
- fire/models.py: run_binary_classification, run_multiclass_classification, _explain_with_lime, _explain_with_shap.

## What to do
Add real tests for these, following the same stubbed-artifact-directory pattern used throughout #104's PRs. Each of these needs its own realistic fixture/mocking setup (pre-trained model artifacts on disk, synthetic datasets matching each function's expected schema) - comparable in scope to #104's own e2e test investigation, hence tracked separately rather than folded into #104's dead-file triage.

## Depends on
Nothing - can start independently.
BODYEOF
)"
```

- [ ] **Step 2: No verification needed** (issue creation is fire-and-forget; confirm the returned URL looks correct)

---

### Task 8: Lint, format, and final regression

**Files:**
- All files created/modified in Tasks 1-6 (formatting only, if needed)

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
git commit -m "style: ruff format e2e and dead-file-revival test additions"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** Delivers #104's final two literal asks: "at least one full end-to-end test" (Task 2, run through the real `run_simulation_pipeline`, confirmed via manual smoke-test to complete and produce expected artifacts/logs) plus a dedicated, deterministic test proving drift correctly triggers retraining (Task 3 — the issue's parenthetical "(if a drift-inducing chunk is included) triggers retraining" is satisfied via a stub-monitor wiring test rather than a naturally-occurring drift chunk, since manual investigation showed real CE calibration doesn't reliably flag a scaled-feature chunk as drift with a `DecisionTreeClassifier`). "Either delete or revive the dead test files" (Tasks 4-6 — all revived, since all three targets are confirmed alive; the heavier remaining functions are tracked in Task 7's follow-up issue rather than silently left uncovered).

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code, validated against a real manual smoke test run during this plan's design (not just theorized).

**Type consistency:** `SimulationRuntime` field names/kwargs in Task 3 match the established pattern from `tests/test_inference.py`/`tests/test_retraining.py`. `get_rolling_columns(config)` is used directly (not hand-duplicated) to guarantee the rolling logger's column schema exactly matches what `process_chunk`'s real row-building path produces.

**Explicitly out of scope:** Deeper coverage for `sequential_simulation`/`continuous_simulation`/`parallel_simulation`/`main()`/`run_binary_classification`/`run_multiclass_classification`/`_explain_with_lime`/`_explain_with_shap` — tracked in Task 7's follow-up issue, not attempted here. This plan, once merged, closes #104 — the issue's core ask (test pyramid: conftest, component/integration tests, unit tests, e2e test, dead-file decision) is fully addressed across Parts 1-4.
