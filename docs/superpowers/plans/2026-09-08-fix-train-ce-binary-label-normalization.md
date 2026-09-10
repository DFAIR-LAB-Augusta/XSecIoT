# Fix train_ce_binary Label Normalization (#114) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two real bugs found and filed as #114 while writing `train_ce_binary` tests in #104: the non-UNSW `Label`→`BinLabel` path's crude exact-literal-`'Benign'`-only mapping shadows the richer, intended case-insensitive normalization block, and that richer block's `dtype == object` check is itself broken under pandas 3's `infer_string=True` default (CSV-read string columns are no longer literally `object` dtype), causing an outright crash on string-valued `BinLabel` columns.

**Architecture:** `train_ce_binary` (`src/firce/ce_model_training.py:202-537`) contains two label-normalization blocks in sequence. The second one (marked `# Start New Code` / `# End new code` at lines 298-335 — clearly a later addition meant to supersede the first) already does everything correctly: case-insensitive string→int mapping via a `label_map` dict, numeric coercion via `pd.to_numeric(errors='coerce')`, and dropping invalid rows. The first block (lines 271-290) was never simplified after the second was added — it still does its own crude, buggy mapping for the non-UNSW path, unconditionally creating an int `BinLabel` that shadows the second block's more correct logic. The `is_unsw` branch (lines 250-269) already does this correctly — it just copies `Label`→`BinLabel` raw (or renames `Bin_Label`→`BinLabel`) with no value-level mapping, deferring entirely to the second block. The fix makes the non-UNSW branch match that same pattern (raw copy/no-op, defer to block two), and fixes block two's dtype check from `y_series.dtype == object` to `pd.api.types.is_string_dtype(y_series)`, which correctly detects both the legacy `object` dtype and pandas 3's Arrow-backed string dtype (verified directly against the pinned pandas version during this plan's design).

**Tech Stack:** Python 3.11+, `pandas`, `pytest` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- This changes real binary training behavior (intentionally, per #114's explicit ask) — run the full test suite, not just the changed file, to catch any test that implicitly depended on the old crude/buggy behavior.
- Two existing tests in `tests/test_ce_model_training.py` were written specifically to *document* the bugs this plan fixes (`test_train_ce_binary_normalizes_string_labels`'s comment, and `test_train_ce_binary_string_binlabel_crashes_under_pandas3`'s entire premise). Both must be updated to assert the corrected behavior, not just left in place expecting the old broken behavior.

---

## File Structure

- Modify: `src/firce/ce_model_training.py` — simplify the non-UNSW label-normalization branch (lines 271-290), fix the dtype check in the shared normalization block (line 305).
- Modify: `tests/test_ce_model_training.py` — add a new test for the original bug scenario (uppercase labels via `Label` column), update the two tests that documented the old broken behavior.

---

### Task 1: Add a failing test for the original bug scenario

**Files:**
- Modify: `tests/test_ce_model_training.py`

**Interfaces:**
- Consumes: `train_ce_binary` (`firce.ce_model_training`), `sim_config_factory` fixture (from `tests/conftest.py`).

- [ ] **Step 1: Write the test**

Add to `tests/test_ce_model_training.py`, immediately after `test_train_ce_binary_normalizes_string_labels`:

```python
def test_train_ce_binary_normalizes_case_insensitive_string_labels_via_label_column(
    tmp_path, monkeypatch, sim_config_factory
):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    n = 40
    rng = np.random.default_rng(0)
    pd.DataFrame({
        'flow_duration': rng.random(n) * 100,
        'tot_fwd_pkt': rng.integers(1, 50, size=n),
        'Label': ['BENIGN' if i % 2 == 0 else 'ATTACK' for i in range(n)],
    }).to_csv(csv_path, index=False)
    config = sim_config_factory(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)

    outdir = train_ce_binary(config, str(csv_path), PerformanceStats())

    import joblib

    model = joblib.load(outdir / 'dt_model_binary.pkl')
    assert set(model.classes_.tolist()) == {0, 1}
```

- [ ] **Step 2: Run it to confirm it fails with the real bug**

Run: `uv run pytest tests/test_ce_model_training.py::test_train_ce_binary_normalizes_case_insensitive_string_labels_via_label_column -v`
Expected: FAIL — `assert {1} == {0, 1}` (all uppercase `'BENIGN'`/`'ATTACK'` rows get mapped to label `1` by the crude exact-literal-`'Benign'`-only map, confirming the bug described in #114).

- [ ] **Step 3: Commit**

```bash
git add tests/test_ce_model_training.py
git commit -m "test: add failing test for #114's original uppercase-label bug scenario"
```

---

### Task 2: Fix `train_ce_binary`'s label normalization

**Files:**
- Modify: `src/firce/ce_model_training.py`

**Interfaces:**
- No new interfaces — internal fix to `train_ce_binary`'s existing label-normalization logic.

- [ ] **Step 1: Simplify the non-UNSW branch**

In `src/firce/ce_model_training.py`, change:

```python
    else:
        if 'Label' in df.columns:
            df['BinLabel'] = df['Label'].map({'Benign': 0}).fillna(1).astype(int)
        elif 'BinLabel' in df.columns:
            if df['BinLabel'].dtype == object:
                df['BinLabel'] = df['BinLabel'].map({'Benign': 0}).fillna(1)
                logger.debug('')
            non_finite_mask = ~np.isfinite(df['BinLabel'])
            if non_finite_mask.any():
                offending_vals = df.loc[non_finite_mask, 'BinLabel'].head(5).tolist()
                logger.error(
                    f"[train_ce_binary] Non-finite values found in 'BinLabel' before casting to int: {offending_vals}"
                )
                raise ValueError(f"Non-finite values in 'BinLabel': {offending_vals}")

            df['BinLabel'] = df['BinLabel'].astype(int)
        else:
            raise ValueError(
                f"Dataset must contain either 'Label' or 'BinLabel' column.Columns found: {df.columns.tolist()}"
            )
```

to:

```python
    else:
        if 'Label' in df.columns:
            df['BinLabel'] = df['Label']
        elif 'BinLabel' in df.columns:
            logger.debug("'BinLabel' already present; deferring normalization to the shared label-mapping step.")
        else:
            raise ValueError(
                f"Dataset must contain either 'Label' or 'BinLabel' column.Columns found: {df.columns.tolist()}"
            )
```

This mirrors the `is_unsw` branch immediately above it (lines 250-269), which already just copies `Label`→`BinLabel` raw and defers all value-level mapping to the shared block below. All actual label normalization (case-insensitive string mapping, numeric coercion, invalid-row dropping) now happens exactly once, in the shared block, for both `is_unsw` and non-`is_unsw` paths.

- [ ] **Step 2: Fix the shared block's dtype check**

In the same file, change:

```python
    if y_series.dtype == object:
```

to:

```python
    if pd.api.types.is_string_dtype(y_series):
```

`pd.api.types.is_string_dtype` correctly detects both the legacy `object` dtype and pandas 3's Arrow-backed `str` dtype (the default for CSV-read string columns under `pd.options.future.infer_string = True`), unlike the direct `== object` comparison which only matches the legacy dtype.

- [ ] **Step 3: Run the new test to confirm it now passes**

Run: `uv run pytest tests/test_ce_model_training.py::test_train_ce_binary_normalizes_case_insensitive_string_labels_via_label_column -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/firce/ce_model_training.py
git commit -m "fix: correctly normalize case-insensitive binary labels in train_ce_binary

The non-UNSW Label path's crude exact-literal-'Benign'-only map
unconditionally created BinLabel before the richer, case-insensitive
normalization block ('Start New Code') could run, making that block
dead code. That block's own dtype == object check was also broken
under pandas 3's infer_string=True default. Fixed by making the
non-UNSW branch match the is_unsw branch's existing raw-copy
pattern (defer all mapping to the shared block), and by replacing
the dtype check with pd.api.types.is_string_dtype, which correctly
detects pandas 3's Arrow-backed string dtype too."
```

---

### Task 3: Update the two tests that documented the old broken behavior

**Files:**
- Modify: `tests/test_ce_model_training.py`

- [ ] **Step 1: Update `test_train_ce_binary_normalizes_string_labels`'s stale comment**

Change:

```python
def test_train_ce_binary_normalizes_string_labels(tmp_path, monkeypatch, sim_config_factory):
    # The non-UNSW 'Label' path maps only the exact literal 'Benign' to 0 and
    # everything else to 1 (see #114 - the richer case-insensitive map further
    # down the function never actually reaches this path, since 'BinLabel' is
    # already created as an int column by the time that block runs).
```

to:

```python
def test_train_ce_binary_normalizes_string_labels(tmp_path, monkeypatch, sim_config_factory):
```

(just remove the now-inaccurate comment block; the test body and assertion are unchanged and still valid — `'Benign'`/`'Attack'` still map to `{0, 1}`, now via the shared case-insensitive block instead of the old crude one).

- [ ] **Step 2: Replace `test_train_ce_binary_string_binlabel_crashes_under_pandas3`**

Change:

```python
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
```

to:

```python
def test_train_ce_binary_normalizes_string_binlabel_column(tmp_path, monkeypatch, sim_config_factory):
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

    outdir = train_ce_binary(config, str(csv_path), PerformanceStats())

    import joblib

    model = joblib.load(outdir / 'dt_model_binary.pkl')
    assert set(model.classes_.tolist()) == {0, 1}
```

- [ ] **Step 3: Run the full file**

Run: `uv run pytest tests/test_ce_model_training.py -v`
Expected: all pass (previously ~23 tests, now 24 with the new Task 1 test), including both updated tests and the new one.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ce_model_training.py
git commit -m "test: update label-normalization tests to reflect the #114 fix"
```

---

### Task 4: Lint, format, and final regression

**Files:**
- `src/firce/ce_model_training.py`, `tests/test_ce_model_training.py` (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any issues**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
```

If issues appear, run `uv run ruff check --fix .` and `uv run ruff format .`, then re-run the full suite.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures beyond the pre-existing/expected skips (xgboost/tensorflow/cade not installed in lean group, unrelated MC rolling-logger skips). Pay particular attention to any other test in the suite that constructs a binary dataset with a `Label`/`BinLabel` column — this fix changes real behavior, so a test relying on the old crude mapping (e.g., an uppercase label silently becoming `1`) would now fail differently and needs the same kind of fix as Task 3, not a workaround.

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format train_ce_binary label normalization fix"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** #114 asks to "decide the intended behavior... and fix." The intended behavior is clearly the already-written `# Start New Code` block (case-insensitive mapping, numeric coercion, invalid-row dropping) — this plan makes that block the single source of truth for both `is_unsw` and non-`is_unsw` paths, and fixes its own pandas-3 dtype-detection bug so it actually works. Both concrete impacts named in the issue (silently-wrong uppercase labels, outright crash on string `BinLabel`) are covered by dedicated tests.

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code and exact diffs.

**Type consistency:** The new non-UNSW branch matches the existing `is_unsw` branch's pattern exactly (`df['BinLabel'] = df['Label']` raw copy, no-op if `BinLabel` already present, same `ValueError` message if neither exists) — no divergence between the two paths' error handling.

**Explicitly out of scope:** No other `dtype == object` checks exist elsewhere in the codebase needing this same fix (confirmed via `grep` during #114's filing — both occurrences are in this one function, both addressed here). `is_unsw=True` path test coverage for `train_ce_binary` remains a separately-deferred item (from #104 Part 1), not addressed here since this fix doesn't change that branch's behavior.
