# Fix _prepare_chunk UNSW Cleaning (#121) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix #121 — `firce/runtime/inference.py::_prepare_chunk` hardcoded `clean_data(chunk, False)` regardless of `runtime.config.is_unsw`, so live streaming chunks for UNSW runs never got the raw-NetFlow-v3-to-short-name renaming or derived-feature computation that `clean_data(is_unsw=True)` provides. Manually confirmed during #108's validation and again during this plan's design: streaming a raw-format UNSW chunk through `process_chunk` produced `Row N is empty` warnings and degenerate (all-identical) predictions before the fix; after changing the hardcoded `False` to `runtime.config.is_unsw`, the same chunk produces correctly varied, non-degenerate predictions with no empty-row warnings.

**Architecture:** One-line fix in `_prepare_chunk`. `load_training_frame` (the one-time aggregated-data load at simulation startup) already correctly passes `config.is_unsw` through to `clean_data` — `_prepare_chunk` (called on every live streaming chunk) is the only place this was missed, and it already has UNSW-specific column-dropping logic immediately after the `clean_data` call (`if runtime.config.is_unsw: to_drop = ...`), confirming it was always meant to be UNSW-aware — the `clean_data` call itself was simply never updated to match. This is not multiclass-specific; it affects UNSW binary streaming equally.

**Tech Stack:** Python 3.11+, `pandas`, `pytest` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- Use the same raw NetFlow-v3-format fixture convention validated in #108's PR (`_make_unsw_raw_csv`-style columns) — `clean_data(is_unsw=True)` only performs its renaming when these exact raw column names are present.

---

## File Structure

- Modify: `src/firce/runtime/inference.py` — `_prepare_chunk`'s `clean_data` call.
- Modify: `tests/test_inference.py` — add a test proving UNSW streaming chunks get correctly renamed/derived features.

---

### Task 1: Fix `_prepare_chunk` and add a regression test

**Files:**
- Modify: `src/firce/runtime/inference.py`
- Modify: `tests/test_inference.py`

**Interfaces:**
- Consumes: `_prepare_chunk` (`firce.runtime.inference`), `SimulationRuntime`, `CircularDequeLogger`, `PerformanceStats`.

- [ ] **Step 1: Write a failing test**

Add to `tests/test_inference.py`:

```python
def test_prepare_chunk_unsw_applies_column_renaming(tmp_path, monkeypatch):
    from firce.runtime.inference import _prepare_chunk

    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path, model_type=ModelType.MULTI, is_unsw=True)
    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=200, columns=['MC_Label']),
        scaler=None,
        pca=None,
        model=None,
        label_encoder=None,
        monitor=None,
        train_df=pd.DataFrame(),
    )

    n = 5
    raw_chunk = pd.DataFrame({
        'IPV4_SRC_ADDR': ['10.0.0.1'] * n,
        'IPV4_DST_ADDR': ['10.0.0.2'] * n,
        'L4_SRC_PORT': [1024] * n,
        'L4_DST_PORT': [80] * n,
        'PROTOCOL': [6] * n,
        'FLOW_START_MILLISECONDS': [1_600_000_000_000 + i * 1000 for i in range(n)],
        'FLOW_END_MILLISECONDS': [1_600_000_000_500 + i * 1000 for i in range(n)],
        'FLOW_DURATION_MILLISECONDS': [10.0] * n,
        'IN_PKTS': [5] * n,
        'OUT_PKTS': [5] * n,
        'IN_BYTES': [500.0] * n,
        'OUT_BYTES': [500.0] * n,
        'SRC_TO_DST_IAT_MIN': [1.0] * n,
        'SRC_TO_DST_IAT_MAX': [1.0] * n,
        'SRC_TO_DST_IAT_AVG': [1.0] * n,
        'SRC_TO_DST_IAT_STDDEV': [1.0] * n,
        'DST_TO_SRC_IAT_MIN': [1.0] * n,
        'DST_TO_SRC_IAT_MAX': [1.0] * n,
        'DST_TO_SRC_IAT_AVG': [1.0] * n,
        'DST_TO_SRC_IAT_STDDEV': [1.0] * n,
    })

    clean_chunk, _ = _prepare_chunk(runtime, raw_chunk)

    assert 'tot_fwd_pkts' in clean_chunk.columns
    assert 'tot_bwd_pkts' in clean_chunk.columns
    assert 'fwd_pkt_len_mean' in clean_chunk.columns
    assert 'IN_PKTS' not in clean_chunk.columns
    assert 'IPV4_SRC_ADDR' not in clean_chunk.columns
```

- [ ] **Step 2: Run it to confirm it fails against the current (buggy) code**

Run: `uv run pytest tests/test_inference.py::test_prepare_chunk_unsw_applies_column_renaming -v`
Expected: FAIL — `clean_chunk` still has the raw `IPV4_SRC_ADDR`/`IN_PKTS` column names (or `KeyError`/empty result), since `clean_data(chunk, False)` never performs the renaming.

- [ ] **Step 3: Fix `_prepare_chunk`**

In `src/firce/runtime/inference.py`, change:

```python
    clean_chunk = clean_data(chunk, False)
```

to:

```python
    clean_chunk = clean_data(chunk, runtime.config.is_unsw)
```

- [ ] **Step 4: Run the test again to confirm it passes**

Run: `uv run pytest tests/test_inference.py::test_prepare_chunk_unsw_applies_column_renaming -v`
Expected: PASS

- [ ] **Step 5: Run the full inference test file to confirm no regressions**

Run: `uv run pytest tests/test_inference.py -v`
Expected: all pass — non-UNSW behavior is provably unchanged (`runtime.config.is_unsw` defaults to `False`, same value now explicitly passed as was hardcoded before for every non-UNSW test in this file).

- [ ] **Step 6: Commit**

```bash
git add src/firce/runtime/inference.py tests/test_inference.py
git commit -m "fix: pass is_unsw through to clean_data in _prepare_chunk

_prepare_chunk hardcoded clean_data(chunk, False) regardless of
runtime.config.is_unsw, so live streaming chunks for UNSW runs never
got the raw-NetFlow-v3-to-short-name renaming or derived-feature
computation that load_training_frame already correctly applies to
the one-time aggregated training load. Confirmed via manual smoke
test: streaming a raw-format UNSW chunk produced degenerate,
all-identical predictions before this fix and correctly varied
predictions after."
```

---

### Task 2: Lint, format, and final regression

**Files:**
- `src/firce/runtime/inference.py`, `tests/test_inference.py` (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any issues**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
```

If issues appear, run `uv run ruff check --fix .` and `uv run ruff format .`, then re-run the full suite.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures beyond the pre-existing/expected skips (xgboost/tensorflow/cade not installed in lean group, unrelated MC rolling-logger skips).

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format _prepare_chunk UNSW cleaning fix"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** #121 asks to confirm the intended `flows_path` convention and, if raw format is expected to work, pass `is_unsw` through. Strong internal-consistency evidence (`load_training_frame` already does this for the one-time load; `_prepare_chunk` already has UNSW-specific logic immediately after the `clean_data` call, implying it was always meant to be UNSW-aware) supports treating this as a genuine bug rather than an intentional convention difference — confirmed empirically via manual smoke test showing degenerate predictions before the fix and correct predictions after.

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code, validated by a real manual smoke test during this plan's design.

**Type consistency:** N/A — one-line change, no new interfaces.

**Explicitly out of scope:** Ground-truth extraction for live UNSW streaming (`Attack`→`MC_Label`/`Label`→`BinLabel` derivation for a *streaming* chunk, as opposed to the one-time `aggregated_path` load which already does this via `load_training_frame`) remains unaddressed — a raw-format UNSW streaming chunk still won't have `MC_Label`/`BinLabel` populated for accuracy tracking, only for feature correctness. This wasn't part of #121's stated problem (which was specifically about "predictions/rolling-log entries end up built from effectively empty/garbage feature data") and is a distinct, not-yet-filed gap.
