# Multiclass CE Model Training (#87) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/firce/ce_model_training.py::train_ce_multiclass` for real — it currently `raise NotImplementedError(...)` before any of its (dead, buggy) body runs. Support all six `ModelVariant`s (dt/knn/rf/svm/xgb/feedforward), proper categorical `MC_Label` handling (including UNSW-style datasets where the label lives in an `Attack` column), macro-averaged metrics, and artifacts persisted with the same naming convention as `train_ce_binary`.

**Architecture:** One cohesive function mirroring `train_ce_binary`'s structure (this codebase's established pattern — one big function per model-type, not a web of small helpers): resolve/validate the multiclass label → drop metadata columns → select numeric features → integer-encode labels via a `LabelEncoder` → scale (+ optional PCA) → train the selected variant → compute macro metrics → persist artifacts. `xgboost` is lazily imported inside its own branch (in both `train_ce_binary` and `train_ce_multiclass`) so the rest of the module — and every other model variant — works without `xgboost` installed; per the user, xgboost isn't used in any published work from this lab, so it doesn't need first-class dependency status.

**Tech Stack:** Python 3.11+, `torch`, `numpy`, `pandas`, `scikit-learn`, `pytest` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- `xgboost` is intentionally excluded from CI's `torch` dependency group (see [[project_xseciot_mc_models_plan]] memory / PR #103) — any test exercising the `xgb` variant must use `pytest.importorskip('xgboost')` so it skips cleanly rather than erroring when xgboost isn't installed.
- `train_ce_multiclass`'s call site in `bootstrap.py` (`train_ce_multiclass(config, str(config.aggregated_path), variant=config.model_variant, use_pca=config.use_pca)`) must keep working unchanged — same parameter names/order, changing only the return type from `None` to `Path` (backward compatible, since the caller discards the return value). Do not modify `bootstrap.py` — that wiring is #88.
- `SimulationConfig.aggregated_path`/`flows_path` fields are validated to exist on disk at construction time — tests must point them at a real (even if unused) file.
- Training artifacts are written to `Path('multi_class_models') / dataset`, a path **relative to the current working directory** (matching `train_ce_binary`'s identical convention) — tests must `monkeypatch.chdir(tmp_path)` first to keep them hermetic.

---

## File Structure

- Modify: `src/firce/ce_model_training.py` — remove the module-level `import xgboost as xgb`; add a local import inside `train_ce_binary`'s XGB branch; replace the entire dead `train_ce_multiclass` body with a real implementation; add `LabelEncoder` and `FeedForwardMulticlass` imports.
- Create: `tests/test_ce_model_training.py` — tests for `train_ce_multiclass` across all variants, UNSW label handling, and error paths.

---

### Task 1: Lazy `xgboost` import

**Files:**
- Modify: `src/firce/ce_model_training.py`

**Interfaces:**
- No public API change — `train_ce_binary`'s behavior when `xgboost` is installed is unchanged.

- [ ] **Step 1: Remove the module-level xgboost import**

In `src/firce/ce_model_training.py`, remove line 35:

```python
import xgboost as xgb
```

from the top-level import block (between `import torch.nn as nn` and the blank line before `from sklearn.decomposition import PCA`).

- [ ] **Step 2: Add a local import inside `train_ce_binary`'s XGB branch**

Find (around line 372-374):

```python
        case ModelVariant.XGB:
            model = xgb.XGBClassifier(objective='binary:logistic', random_state=config.seed)
            model.fit(Xf, y)
```

Replace with:

```python
        case ModelVariant.XGB:
            import xgboost as xgb

            model = xgb.XGBClassifier(objective='binary:logistic', random_state=config.seed)
            model.fit(Xf, y)
```

- [ ] **Step 3: Verify the module still imports and the full suite still passes**

Run: `uv run python -c "import firce.ce_model_training; print('ok')"`
Expected: `ok` (proves the module no longer requires `xgboost` at import time)

Run: `uv run pytest -q`
Expected: all existing tests still pass (no behavior change for anything currently exercised)

- [ ] **Step 4: Commit**

```bash
git add src/firce/ce_model_training.py
git commit -m "refactor: lazily import xgboost so the module works without it installed"
```

---

### Task 2: Implement `train_ce_multiclass`

**Files:**
- Modify: `src/firce/ce_model_training.py`
- Test: `tests/test_ce_model_training.py`

**Interfaces:**
- Consumes: `FeedForwardMulticlass` (from `firce.models.feedforward_multiclass`), `LabelEncoder` (from `sklearn.preprocessing`), `_unsw_clean`, `CE_DROP_COLS`, `SupportsPredict` (all already in this file).
- Produces: `train_ce_multiclass(config: SimulationConfig, flows_csv: str, variant: ModelVariant, use_pca: bool = True) -> Path`. Artifacts written to `multi_class_models/<dataset>/`: `scaler_multi.pkl`, `label_encoder_multi.pkl`, optionally `pca_multi.pkl`, and either `{variant.value}_model_multi.pkl` or `feedforward_model_multi.pt`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ce_model_training.py`:

```python
import numpy as np
import pandas as pd
import pytest
import torch

from firce.ce_model_training import train_ce_multiclass
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig

DEVICE = torch.device('cpu')


def _make_config(tmp_path, **overrides):
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
        device=DEVICE,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


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


def _make_unsw_multiclass_csv(tmp_path, n=60, seed=0, dirname='UNSW_DS'):
    rng = np.random.default_rng(seed)
    ds_dir = tmp_path / dirname
    ds_dir.mkdir(exist_ok=True)
    csv_path = ds_dir / 'flows.csv'
    labels = np.array(['Benign', 'DoS', 'Reconnaissance'])
    idx = rng.integers(0, 3, size=n)
    df = pd.DataFrame({
        'src_ip': ['10.0.0.1'] * n,
        'dst_ip': ['10.0.0.2'] * n,
        'src_port': rng.integers(1024, 65535, size=n),
        'dst_port': rng.integers(1, 1024, size=n),
        'protocol': rng.integers(0, 2, size=n),
        'flow_duration': rng.random(n) * 100,
        'in_pkts': rng.integers(1, 50, size=n),
        'out_pkts': rng.integers(0, 50, size=n),
        'in_bytes': rng.random(n) * 1000,
        'out_bytes': rng.random(n) * 1000,
        'Label': (idx != 0).astype(int),
        'Attack': labels[idx],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.parametrize('variant', [ModelVariant.DT, ModelVariant.KNN, ModelVariant.RF, ModelVariant.SVM])
def test_train_ce_multiclass_classical_variants(tmp_path, monkeypatch, variant):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=variant)

    outdir = train_ce_multiclass(config, str(csv_path), variant=variant, use_pca=False)

    assert outdir.resolve() == (tmp_path / 'multi_class_models' / 'DS').resolve()
    assert (outdir / 'scaler_multi.pkl').exists()
    assert (outdir / 'label_encoder_multi.pkl').exists()
    assert (outdir / f'{variant.value}_model_multi.pkl').exists()
    assert not (outdir / 'pca_multi.pkl').exists()


def test_train_ce_multiclass_with_pca_writes_pca_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)

    outdir = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=True)

    assert (outdir / 'pca_multi.pkl').exists()


def test_train_ce_multiclass_feedforward(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.FEEDFORWARD)

    outdir = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.FEEDFORWARD, use_pca=False)

    ckpt_path = outdir / 'feedforward_model_multi.pt'
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert ckpt['num_classes'] == 3
    assert ckpt['input_dim'] > 0


def test_train_ce_multiclass_xgb(tmp_path, monkeypatch):
    pytest.importorskip('xgboost')
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.XGB)

    outdir = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.XGB, use_pca=False)

    assert (outdir / 'xgb_model_multi.pkl').exists()


def test_train_ce_multiclass_unsw_uses_attack_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_unsw_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT, is_unsw=True)

    outdir = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False)

    assert outdir.resolve() == (tmp_path / 'multi_class_models' / 'UNSW_DS').resolve()
    assert (outdir / 'label_encoder_multi.pkl').exists()
    import joblib

    encoder = joblib.load(outdir / 'label_encoder_multi.pkl')
    assert sorted(encoder.classes_.tolist()) == ['Benign', 'DoS', 'Reconnaissance']


def test_train_ce_multiclass_missing_label_column_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    pd.DataFrame({'flow_duration': [1, 2, 3], 'tot_fwd_pkt': [1, 2, 3]}).to_csv(csv_path, index=False)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)

    with pytest.raises(ValueError, match="must contain an 'MC_Label' column"):
        train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False)


def test_train_ce_multiclass_too_few_classes_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    pd.DataFrame({
        'flow_duration': [1, 2, 3],
        'tot_fwd_pkt': [1, 2, 3],
        'MC_Label': ['Benign', 'Benign', 'Benign'],
    }).to_csv(csv_path, index=False)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)

    with pytest.raises(ValueError, match='at least 2 distinct MC_Label classes'):
        train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ce_model_training.py -v`
Expected: FAIL — every test hits `NotImplementedError: Multiclass CE model training is currently disabled.`

- [ ] **Step 3: Replace the dead `train_ce_multiclass` body**

First, in `src/firce/ce_model_training.py`, update the imports:

Change:
```python
from sklearn.preprocessing import StandardScaler
```
to:
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

Add, after `from firce.models.feedforward_binary import FeedForwardBinary`:
```python
from firce.models.feedforward_multiclass import FeedForwardMulticlass
```

Then replace the entire `train_ce_multiclass` function (from `def train_ce_multiclass(` through its closing `logging.info(...)` line, i.e. everything currently between the function signature and the `if __name__ == '__main__':` guard) with:

```python
def train_ce_multiclass(
    config: SimulationConfig,
    flows_csv: str,
    variant: ModelVariant,
    use_pca: bool = True,
) -> Path:
    """
    Train a multiclass CE model on labeled flow data and save all artifacts.

    Mirrors `train_ce_binary`'s preprocessing (numeric feature selection,
    standard scaling, optional PCA) but resolves a categorical multiclass
    label (`MC_Label`, or `Attack` for UNSW-style datasets) instead of a
    binary one, integer-encodes it via a `LabelEncoder`, and reports
    macro-averaged metrics.

    Args:
        config: Simulation configuration (uses `is_unsw`, `seed`, `device`).
        flows_csv: Path to the CSV file containing labeled multiclass flow data.
        variant: Model architecture to use. One of "dt", "knn", "rf", "svm",
            "xgb", "feedforward".
        use_pca: If True, apply PCA to reduce feature space to 95% explained variance.

    Returns:
        The output directory artifacts were written to.

    Raises:
        ValueError: If the dataset has no usable multiclass label column,
            fewer than 2 distinct classes, or an unsupported variant is given.
    """
    logger.info(f'Training multiclass CE model with {flows_csv} dataset')
    df = clean_data(pd.read_csv(flows_csv), config.is_unsw)
    dataset = Path(flows_csv).parent.name
    outdir = Path('multi_class_models') / dataset
    outdir.mkdir(parents=True, exist_ok=True)

    if config.is_unsw:
        if 'MC_Label' not in df.columns:
            if 'Attack' not in df.columns:
                raise ValueError(
                    f"UNSW multiclass dataset must contain 'Attack' or 'MC_Label'. "
                    f'Columns found: {df.columns.tolist()}'
                )
            logger.info("Using UNSW dataset format: mapping 'Attack' to multiclass 'MC_Label'")
            df['MC_Label'] = df['Attack']
        df = _unsw_clean(df)
    elif 'MC_Label' not in df.columns:
        raise ValueError(
            f"Dataset must contain an 'MC_Label' column for multiclass training. "
            f'Columns found: {df.columns.tolist()}'
        )

    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')], errors='ignore')
    df = df.drop(columns=CE_DROP_COLS, errors='ignore')

    df['MC_Label'] = df['MC_Label'].astype(str).str.strip()
    invalid = df['MC_Label'].isin(['', 'nan', 'None'])
    if invalid.any():
        n_bad = int(invalid.sum())
        logger.warning(f'Dropping {n_bad} rows with missing/invalid MC_Label before training')
        df = df.loc[~invalid].copy()

    if df['MC_Label'].nunique() < 2:
        raise ValueError(
            f"Need at least 2 distinct MC_Label classes to train, found: {df['MC_Label'].unique().tolist()}"
        )

    X = df.select_dtypes(include=[np.number]).drop(columns=['MC_Label'], errors='ignore')
    y_raw = df['MC_Label']

    if X.isna().any().any():
        X = X.fillna(X.mean())

    label_encoder = LabelEncoder().fit(y_raw)
    y = label_encoder.transform(y_raw)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    joblib.dump(scaler, outdir / 'scaler_multi.pkl')
    joblib.dump(label_encoder, outdir / 'label_encoder_multi.pkl')

    if use_pca:
        pca = PCA(n_components=0.95).fit(Xs)
        Xf = pca.transform(Xs)
        joblib.dump(pca, outdir / 'pca_multi.pkl')
    else:
        Xf = Xs

    y_pred: NDArray[np.int_] | None = None

    logger.debug(f"Training multiclass classifier model with variant '{variant.value}'")
    match variant:
        case ModelVariant.DT:
            model = DecisionTreeClassifier(random_state=config.seed)
            model.fit(Xf, y)
        case ModelVariant.KNN:
            model = KNeighborsClassifier()
            model.fit(Xf, y)
        case ModelVariant.RF:
            model = RandomForestClassifier(random_state=config.seed)
            model.fit(Xf, y)
        case ModelVariant.SVM:
            model = SVC(kernel='rbf', probability=True, random_state=config.seed)
            model.fit(Xf, y)
        case ModelVariant.XGB:
            import xgboost as xgb

            model = xgb.XGBClassifier(
                objective='multi:softmax', num_class=len(label_encoder.classes_), random_state=config.seed
            )
            model.fit(Xf, y)
        case ModelVariant.FEEDFORWARD:
            Xf = np.asarray(Xf, dtype=np.float32)
            y_arr = y.astype(np.int64)

            device = config.device
            logger.info(f'[feedforward-multiclass] Using device: {device}')

            torch.manual_seed(config.seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(config.seed)

            X_tensor = torch.from_numpy(Xf)
            Y_tensor = torch.from_numpy(y_arr)

            epochs = 20
            N = X_tensor.shape[0]
            batch_size = 2048 if N >= 8192 else 512
            ds = TensorDataset(X_tensor, Y_tensor)

            model = FeedForwardMulticlass(input_dim=Xf.shape[1], num_classes=len(label_encoder.classes_)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            logger.info('FFN Multiclass Model Training Starting (bs=%d, epochs=%d)', batch_size, epochs)

            model.train()
            for epoch in range(epochs):
                rng = np.random.default_rng(config.seed + epoch)
                idx = rng.permutation(N).tolist()
                subset = Subset(ds, idx)
                loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

                running_loss = 0.0
                for xb, yb in loader:
                    xb = xb.to(device, non_blocking=False)
                    yb = yb.to(device, non_blocking=False).view(-1).long()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item()) * xb.size(0)

                logger.debug('[feedforward-multiclass][epoch %02d] loss=%.6f', epoch + 1, running_loss / N)

            model.eval()
            eval_bs = 4096
            eval_loader = DataLoader(ds, batch_size=eval_bs, shuffle=False, num_workers=0)
            logits_chunks = []
            with torch.no_grad():
                for xb, _ in eval_loader:
                    xb = xb.to(device, dtype=torch.float32, non_blocking=False).contiguous()
                    logits_chunks.append(model(xb).to('cpu').numpy())
            logits_all = np.concatenate(logits_chunks, axis=0)
            y_pred = np.argmax(logits_all, axis=1).astype(np.int_)
        case _:
            raise ValueError(f"Unknown variant '{variant.value}'")

    if variant != ModelVariant.FEEDFORWARD:
        sk = cast('SupportsPredict', model)
        y_pred_np = np.asarray(sk.predict(Xf))
        y_pred = y_pred_np.astype(np.int_, copy=False).reshape(-1)

    if y_pred is None:
        raise RuntimeError(f'Internal error: y_pred not computed for variant {variant.value!r}')

    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, average='macro', zero_division=0))
    rec = float(recall_score(y, y_pred, average='macro', zero_division=0))
    f1 = float(f1_score(y, y_pred, average='macro', zero_division=0))

    logger.info('Multiclass Model Performance on Training Data:')
    logger.info(f'Accuracy:  {acc:.4f}')
    logger.info(f'Macro Precision: {prec:.4f}')
    logger.info(f'Macro Recall:    {rec:.4f}')
    logger.info(f'Macro F1 Score:  {f1:.4f}')
    report = classification_report(
        y,
        y_pred,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_.astype(str),
        digits=4,
    )
    logger.info('\n%s', report)

    if variant == ModelVariant.FEEDFORWARD and isinstance(model, nn.Module):
        torch.save(
            {
                'state_dict': model.state_dict(),
                'input_dim': int(Xf.shape[1]),
                'num_classes': len(label_encoder.classes_),
                'dropout': 0.3,
                'random_state': config.seed,
            },
            outdir / 'feedforward_model_multi.pt',
        )
    else:
        joblib.dump(model, outdir / f'{variant.value}_model_multi.pkl')

    logging.info(f"[ce_models] Trained '{variant.value}' multiclass CE model and wrote artifacts to {outdir}/")
    return outdir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ce_model_training.py -v`
Expected: 10 passed (or 9 passed + 1 skipped if `xgboost` isn't installed in the current environment)

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 6: Commit**

```bash
git add src/firce/ce_model_training.py tests/test_ce_model_training.py
git commit -m "feat: implement train_ce_multiclass for all model variants"
```

---

### Task 3: Lint, format, and final regression

**Files:**
- Modify: `src/firce/ce_model_training.py`, `tests/test_ce_model_training.py` (formatting only, if needed)

- [ ] **Step 1: Run ruff and fix any formatting issues**

Run:
```bash
uv run ruff check src/firce/ce_model_training.py tests/test_ce_model_training.py
uv run ruff format --check src/firce/ce_model_training.py tests/test_ce_model_training.py
```

If formatting differs, run `uv run ruff format src/firce/ce_model_training.py tests/test_ce_model_training.py` and re-run the full suite to confirm nothing broke.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all tests pass

- [ ] **Step 3: Commit if anything changed**

```bash
git add -A
git commit -m "style: ruff format ce_model_training changes"
```

(skip this commit if Step 1 made no changes)

---

## Self-Review

**Spec coverage:** Issue #87 asks to (1) remove the `NotImplementedError` and get dt/knn/rf/svm/xgb working with proper `Label` handling and macro metrics, (2) add a feedforward-multiclass training path, (3) persist artifacts under `multi_class_models/<dataset>/` matching `train_ce_binary`'s naming convention. All three are covered by Task 2. The lazy-xgboost fix (Task 1) is a necessary prerequisite discovered while scoping this plan — without it, any test importing `ce_model_training` breaks CI (xgboost was deliberately excluded from CI's `torch` group to fix a real nccl collision in #103) as soon as it needs to exercise both torch and xgboost paths together.

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code.

**Type consistency:** `train_ce_multiclass`'s signature (`config, flows_csv, variant, use_pca=True`) matches exactly what `bootstrap.py` already calls — verified by reading `bootstrap.py:164-169` in this codebase, not modified by this plan.

**Explicitly out of scope (belongs to other issues):** wiring this into `bootstrap.py` (removing the `except NotImplementedError` swallow-catch and the `model_variant != FEEDFORWARD` guard) is #88. Wiring into `retraining.py` is #89. CLI/config validation for variant combinations is #92. Dedicated CE-evaluator multiclass tests (ICE/CCE/TCE/Approx-CCE) are #93. None of those files are touched here.

**xgboost scope note:** per explicit user direction, xgboost isn't used in any published work from this lab and doesn't warrant engineering investment beyond "don't let it break everything else" — the xgb branch is implemented (for completeness/symmetry with the binary path) but its test is skip-if-missing rather than a hard CI requirement.
