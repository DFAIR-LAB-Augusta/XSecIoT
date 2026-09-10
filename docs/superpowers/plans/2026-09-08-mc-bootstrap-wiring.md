# Wire Multiclass into bootstrap.py (#88) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `src/firce/runtime/bootstrap.py::ensure_model_artifacts` currently skips multiclass training entirely for `--modelVariant feedforward` (an explicit guard) and silently swallows `NotImplementedError` for every other variant (dead code now that #87 landed — `train_ce_multiclass` no longer raises it). Fix both so `model_type=multi` actually trains, for every variant including feedforward, with errors propagating instead of being silently logged away.

**Architecture:** Delete the guard and the try/except; make the multiclass branch structurally symmetric with the binary branch two lines above it (which has no guard and no swallow-catch — errors from `train_ce_binary` already propagate today). `bootstrap.py` also has an `import xgboost as xgb` at module level used only in type annotations (`xgb.Booster`, never at runtime) — this blocks any test from importing the module without xgboost installed, the same problem fixed in `ce_model_training.py` for #87. Fix it here via `from __future__ import annotations` (PEP 563 lazy annotations) rather than a local import, since there's no runtime usage to scope a local import to — this makes the whole module importable without xgboost, not just one branch.

**Tech Stack:** Python 3.11+, `pytest` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- `xgboost` is intentionally excluded from CI's `torch` dependency group (see PR #103) — `bootstrap.py` must remain importable without it installed.
- This is scoped to `bootstrap.py`'s multiclass training call only. Wiring into `retraining.py` is #89 — not touched here.

---

## File Structure

- Modify: `src/firce/runtime/bootstrap.py` — add `from __future__ import annotations`, remove the now-unneeded `import xgboost as xgb`, fix `ensure_model_artifacts`'s multiclass branch.
- Create: `tests/test_bootstrap.py` — tests for `ensure_model_artifacts`'s multiclass path.

---

### Task 1: Lazy type annotations (drop the xgboost import requirement)

**Files:**
- Modify: `src/firce/runtime/bootstrap.py`
- Modify: `src/firce/conformalEval/utils.py` (transitive import chain: `bootstrap.py` → `adaptive_sig_ctlr.py` → this file)
- Modify: `src/firce/runtime/sim_types.py` (transitive: `bootstrap.py` imports `SimulationRuntime` from here)
- Modify: `src/fire/simulations.py` (transitive: `bootstrap.py` imports `load_simulation_objects`/`preprocess_chunk` from here)

**Interfaces:**
- No public API change. Discovered while implementing: `bootstrap.py`'s import chain is deeper than one file — `conformalEval/utils.py` does `from xgboost import XGBClassifier` unconditionally despite already having an `if XGBClassifier is not None` runtime guard (clearly the original intent was graceful degradation, just never wired up); `sim_types.py` uses `xgb.Booster` in a dataclass field annotation only; `fire/simulations.py` uses `xgb.Booster` in type annotations *and* has a genuine runtime `isinstance(model, xgb.Booster)` / `xgb.DMatrix(...)` call inside `process_chunk`, guarded by `model_variant.startswith('xgb')`. Each gets the fix appropriate to its usage: annotation-only usages get `from __future__ import annotations` + `TYPE_CHECKING`; the one genuine runtime call gets a local import inside its guarding `if` branch (same pattern as #87's `train_ce_multiclass`); the already-guarded `XGBClassifier` import gets a `try/except ImportError`.

- [ ] **Step 1: Add postponed annotation evaluation and drop the xgboost import**

In `src/firce/runtime/bootstrap.py`, change the top of the file from:

```python
import logging
import time

from typing import Any

import pandas as pd
import xgboost as xgb

from sklearn.base import ClassifierMixin
```

to:

```python
from __future__ import annotations

import logging
import time

from typing import TYPE_CHECKING, Any

import pandas as pd

from sklearn.base import ClassifierMixin
```

Then add, right after the existing import block (after `from fire.simulations import load_simulation_objects, preprocess_chunk`, before `logger = logging.getLogger(__name__)`):

```python
if TYPE_CHECKING:
    import xgboost as xgb
```

- [ ] **Step 2: Fix `src/firce/conformalEval/utils.py`'s unconditional XGBClassifier import**

Change:
```python
from xgboost import XGBClassifier
```
to:
```python
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
```
This completes the graceful-degradation the file already implements (`if XGBClassifier is not None and isinstance(model, XGBClassifier):` further down) but never actually wired up.

- [ ] **Step 3: Fix `src/firce/runtime/sim_types.py`**

Change the top of the file from:
```python
from dataclasses import dataclass

import pandas as pd
import xgboost as xgb

from sklearn.base import ClassifierMixin
```
to:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from sklearn.base import ClassifierMixin
```
and add, after the last `from firce...`/`from fire...` import (before the `@dataclass` class):
```python
if TYPE_CHECKING:
    import xgboost as xgb
```

- [ ] **Step 4: Fix `src/fire/simulations.py`**

Change the top of the file from:
```python
# fire.simulations

import argparse
import logging
import multiprocessing as mp
import os
import time

from functools import partial
from typing import List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from sklearn.base import ClassifierMixin
```
to:
```python
# fire.simulations

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import time

from functools import partial
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.base import ClassifierMixin
```
and add, after the `from firce...` imports (before `logger = logging.getLogger(__name__)`):
```python
if TYPE_CHECKING:
    import xgboost as xgb
```

Then fix the one genuine runtime usage inside `process_chunk`. Change:
```python
    # 4) predict
    if model_variant.startswith('xgb') and isinstance(model, xgb.Booster):
        fnames = [f'f_{i}' for i in range(X_p.shape[1])]
        dtest = xgb.DMatrix(X_p, feature_names=fnames)
        preds = model.predict(dtest)
```
to:
```python
    # 4) predict
    if model_variant.startswith('xgb'):
        import xgboost as xgb

        if isinstance(model, xgb.Booster):
            fnames = [f'f_{i}' for i in range(X_p.shape[1])]
            dtest = xgb.DMatrix(X_p, feature_names=fnames)
            preds = model.predict(dtest)
        else:
            preds = model.predict(X_p)  # type: ignore
```
(the existing `else:` branch below, for non-xgb variants, is untouched.)

- [ ] **Step 5: Verify the module imports without xgboost and the full suite still passes**

Run: `uv run python -c "import firce.runtime.bootstrap; print('ok')"`
Expected: `ok`

Run: `uv run pytest -q`
Expected: all existing tests still pass (no behavior change)

- [ ] **Step 6: Commit**

```bash
git add src/firce/runtime/bootstrap.py src/firce/conformalEval/utils.py src/firce/runtime/sim_types.py src/fire/simulations.py
git commit -m "refactor: make bootstrap.py's import chain work without xgboost installed"
```

---

### Task 2: Fix the multiclass training branch

**Files:**
- Modify: `src/firce/runtime/bootstrap.py`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: `train_ce_multiclass` (from `firce.ce_model_training`, implemented in #87).
- Produces: `ensure_model_artifacts(config: SimulationConfig, perf_stats: PerformanceStats) -> None` — unchanged signature, fixed behavior: multiclass training now runs for every `ModelVariant` (including feedforward) and raises on real errors instead of swallowing them.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bootstrap.py`:

```python
import numpy as np
import pandas as pd
import pytest
import torch

from firce.runtime.bootstrap import ensure_model_artifacts
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig
from firce.utils.perf_stats import PerformanceStats

DEVICE = torch.device('cpu')


def _make_multiclass_csv(tmp_path, n=60, seed=0, dirname='DS'):
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


def _make_config(csv_path, **overrides):
    defaults = dict(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=csv_path,
        flows_path=csv_path,
        is_unsw=False,
        seed=0,
        device=DEVICE,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_ensure_model_artifacts_multiclass_trains(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(csv_path, model_variant=ModelVariant.DT)

    ensure_model_artifacts(config, PerformanceStats())

    assert (tmp_path / 'multi_class_models' / 'DS' / 'dt_model_multi.pkl').exists()


def test_ensure_model_artifacts_multiclass_feedforward_now_trains(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(csv_path, model_variant=ModelVariant.FEEDFORWARD)

    ensure_model_artifacts(config, PerformanceStats())

    assert (tmp_path / 'multi_class_models' / 'DS' / 'feedforward_model_multi.pt').exists()


def test_ensure_model_artifacts_multiclass_propagates_real_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    pd.DataFrame({
        'flow_duration': [1, 2, 3],
        'tot_fwd_pkt': [1, 2, 3],
        'MC_Label': ['Benign', 'Benign', 'Benign'],
    }).to_csv(csv_path, index=False)
    config = _make_config(csv_path, model_variant=ModelVariant.DT)

    with pytest.raises(ValueError, match='at least 2 distinct MC_Label classes'):
        ensure_model_artifacts(config, PerformanceStats())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: `test_ensure_model_artifacts_multiclass_feedforward_now_trains` FAILs (feedforward is currently skipped, no artifact written); `test_ensure_model_artifacts_multiclass_propagates_real_errors` FAILs (the error is currently swallowed as a warning, not raised — though note this dataset's error is a `ValueError`, not `NotImplementedError`, so it would actually already propagate today; this test is really guarding against a future regression back to a swallow-catch). `test_ensure_model_artifacts_multiclass_trains` likely already PASSES today since the DT path was never guarded — confirm this before Step 3 so you know which failures are real.

- [ ] **Step 3: Fix `ensure_model_artifacts`**

In `src/firce/runtime/bootstrap.py`, replace:

```python
    if config.model_variant != ModelVariant.FEEDFORWARD and config.model_type == ModelType.MULTI:
        logger.info(
            "CE multiclass artifacts missing for '%s'; training now...",
            dataset_name,
        )
        start = time.perf_counter()
        try:
            train_ce_multiclass(
                config,
                str(config.aggregated_path),
                variant=config.model_variant,
                use_pca=config.use_pca,
            )
            logger.info(
                'Multiclass CE training completed in %.4fs',
                time.perf_counter() - start,
            )
        except NotImplementedError as exc:
            logger.warning(
                "Multiclass CE training not supported for variant '%s'; skipping: %s",
                config.model_variant.value,
                exc,
            )
```

with:

```python
    if config.model_type == ModelType.MULTI:
        logger.info(
            "CE multiclass artifacts missing for '%s'; training now...",
            dataset_name,
        )
        start = time.perf_counter()
        train_ce_multiclass(
            config,
            str(config.aggregated_path),
            variant=config.model_variant,
            use_pca=config.use_pca,
        )
        logger.info(
            'Multiclass CE training completed in %.4fs',
            time.perf_counter() - start,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 6: Commit**

```bash
git add src/firce/runtime/bootstrap.py tests/test_bootstrap.py
git commit -m "fix: train multiclass models for every variant in ensure_model_artifacts"
```

---

### Task 3: Lint, format, and final regression

**Files:**
- Modify: `src/firce/runtime/bootstrap.py`, `tests/test_bootstrap.py` (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any formatting issues**

Run:
```bash
uv run ruff check src/firce/runtime/bootstrap.py src/firce/conformalEval/utils.py src/firce/runtime/sim_types.py src/fire/simulations.py tests/test_bootstrap.py
uv run ruff format --check src/firce/runtime/bootstrap.py src/firce/conformalEval/utils.py src/firce/runtime/sim_types.py src/fire/simulations.py tests/test_bootstrap.py
```

Expect `ruff check` to flag `TC001`/`TC002` (typing-only imports) across `bootstrap.py`, `sim_types.py`, and `fire/simulations.py` — adding `from __future__ import annotations` makes ruff's type-checking-import rule (already enabled in `.ruff.toml`) want *every* type-only import moved into the `TYPE_CHECKING` block, not just xgboost's. Run `uv run ruff check --fix src/firce/runtime/bootstrap.py src/firce/conformalEval/utils.py src/firce/runtime/sim_types.py src/fire/simulations.py tests/test_bootstrap.py` to apply it (safe — ruff correctly leaves runtime-used imports like `MLP_CE` alone since it's actually instantiated, not just type-annotated), then re-run the full suite and `import firce.runtime.bootstrap` check to confirm nothing broke.

If formatting differs after the fix, run `uv run ruff format src/firce/runtime/bootstrap.py src/firce/conformalEval/utils.py src/firce/runtime/sim_types.py src/fire/simulations.py tests/test_bootstrap.py` and re-run the full suite again.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format bootstrap.py changes"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** Issue #88 asks to (1) remove the dead `except NotImplementedError` swallow-catch, (2) add the feedforward-multiclass training call currently skipped by the `model_variant != FEEDFORWARD` guard. Both are the entire diff in Task 2. The lazy-xgboost-annotations fix (Task 1) is a necessary prerequisite discovered while scoping this plan, mirroring the same fix already applied to `ce_model_training.py` in #87.

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code.

**Explicitly out of scope (belongs to other issues):** wiring into `retraining.py` is #89. CLI/config validation for variant combinations is #92. No other files are touched here.
