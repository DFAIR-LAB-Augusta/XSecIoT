from __future__ import annotations

import importlib
import sys
import types

from typing import Any

import numpy as np
import pytest


def _ensure_adaptive_sig_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    cce.py imports:
        from src.core.conformalEval.adaptive_significance_controller import AdaptiveSignificanceController

    If that module doesn't exist (repo rename), we inject a tiny stub so the import succeeds.
    """
    mod_name = 'src.core.conformalEval.adaptive_significance_controller'
    if mod_name in sys.modules:
        return

    stub = types.ModuleType(mod_name)

    class AdaptiveSignificanceController:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._thresholds: dict[Any, float] = {}
            self.updated: list[tuple[np.ndarray, np.ndarray]] = []

        def update(self, classes: np.ndarray, p_values: np.ndarray) -> None:
            self.updated.append((np.asarray(classes, dtype=object), np.asarray(p_values, dtype=float)))
            for c in np.unique(classes.astype(object)):
                self._thresholds[c] = float(self._thresholds.get(c, 0.10))

        def get_thresholds(self) -> dict[Any, float]:
            return dict(self._thresholds)

    stub.AdaptiveSignificanceController = AdaptiveSignificanceController  # type: ignore[attr-defined]
    sys.modules[mod_name] = stub


def _import_cce(monkeypatch: pytest.MonkeyPatch):
    """
    cce.py loads TOML at import-time via load_conformal_config(), so make imports robust.

    If the TOML file is missing in local runs, we monkeypatch utils.load_conformal_config()
    to return a minimal config.
    """
    _ensure_adaptive_sig_importable(monkeypatch)

    mod_name = 'src.core.conformalEval.cce'

    sys.modules.pop(mod_name, None)

    import src.core.conformalEval.utils as utils

    fake_cfg = {
        'conformal_eval_config': {
            'folds': 3,
            'significance': 0.10,
            'n_jobs': 1,
        },
        'adaptive_significance': {
            'decay': 0.9,
            'max_alpha': 0.30,
            'min_alpha': 0.10,
            'window_size': 10,
            'alpha_step': 0.05,
            'increase_threshold': 0.60,
            'decrease_threshold': 0.10,
        },
    }
    monkeypatch.setattr(utils, 'load_conformal_config', lambda *a, **k: fake_cfg, raising=True)

    return importlib.import_module(mod_name)


def _toy_binary(n: int = 90, d: int = 6, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.25 * X[:, 1] - 0.1 > 0.0).astype(np.int32)
    return X, y


def _toy_multiclass(n: int = 120, d: int = 6, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    s0 = X[:, 0] - 0.2 * X[:, 1]
    s1 = -X[:, 0] + 0.5 * X[:, 2]
    s2 = 0.3 * X[:, 1] - 0.4 * X[:, 2] + 0.1 * X[:, 3]
    y = np.argmax(np.stack([s0, s1, s2], axis=1), axis=1).astype(np.int32)
    return X, y


def test_get_thresholds_raises_before_calibrate(monkeypatch: pytest.MonkeyPatch) -> None:
    cce_mod = _import_cce(monkeypatch)
    from sklearn.tree import DecisionTreeClassifier

    cce = cce_mod.CrossConformalEvaluator(DecisionTreeClassifier(random_state=0), folds=3, n_jobs=1)
    with pytest.raises(RuntimeError, match='call calibrate'):
        _ = cce.get_thresholds()


def test_predict_p_values_raises_before_calibrate(monkeypatch: pytest.MonkeyPatch) -> None:
    cce_mod = _import_cce(monkeypatch)
    from sklearn.tree import DecisionTreeClassifier

    cce = cce_mod.CrossConformalEvaluator(DecisionTreeClassifier(random_state=0), folds=3, n_jobs=1)
    X, _ = _toy_binary()
    with pytest.raises(RuntimeError, match='must be calibrated'):
        _ = cce.predict_p_values(X[:5])


def test_process_fold_outputs_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    cce_mod = _import_cce(monkeypatch)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier

    X, y = _toy_binary(n=60)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    train_idx, calib_idx = next(iter(skf.split(X, y)))

    cce = cce_mod.CrossConformalEvaluator(DecisionTreeClassifier(random_state=0), folds=3, n_jobs=1)

    fold_scores, y_calib, y_pred_calib, model = cce._process_fold(X, y, train_idx, calib_idx)

    assert hasattr(model, 'predict_proba')
    assert isinstance(y_calib, np.ndarray)
    assert isinstance(y_pred_calib, np.ndarray)
    assert y_calib.shape == y_pred_calib.shape
    assert set(fold_scores.keys()) == set(np.unique(y))
    # scores are in [0, 1]
    all_s = np.array([v for vs in fold_scores.values() for v in vs], dtype=float)
    assert np.all(all_s >= 0.0)
    assert np.all(all_s <= 1.0)


def test_calibrate_binary_sets_thresholds_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    cce_mod = _import_cce(monkeypatch)
    from sklearn.tree import DecisionTreeClassifier

    from src.core.perf_stats import PerformanceStats

    X, y = _toy_binary(n=90)
    perf = PerformanceStats()

    cce = cce_mod.CrossConformalEvaluator(
        DecisionTreeClassifier(random_state=0),
        folds=3,
        significance=0.10,
        random_state=0,
        n_jobs=1,
        significance_controller=None,
    )
    cce.calibrate(X, y, perf_stats=perf)

    assert cce.calibration_scores is not None
    assert cce.thresholds is not None
    assert set(cce.calibration_scores.keys()) == set(np.unique(y))
    assert set(cce.thresholds.keys()) == set(np.unique(y))
    assert len(perf.ce_stats.accuracies) == 1
    assert 0.0 <= perf.ce_stats.accuracies[0] <= 1.0


def test_calibrate_multiclass_predict_p_values(monkeypatch: pytest.MonkeyPatch) -> None:
    cce_mod = _import_cce(monkeypatch)
    from sklearn.tree import DecisionTreeClassifier

    from src.core.perf_stats import PerformanceStats

    X, y = _toy_multiclass(n=120)
    perf = PerformanceStats()

    cce = cce_mod.CrossConformalEvaluator(
        DecisionTreeClassifier(random_state=0),
        folds=3,
        significance=0.10,
        random_state=0,
        n_jobs=1,
    )
    cce.calibrate(X, y, perf_stats=perf)

    out = cce.predict_p_values(X[:10])
    assert set(out.keys()) == {'class', 'p_value'}
    assert out['class'].shape == (10,)
    assert out['p_value'].shape == (10,)
    assert np.all(out['p_value'] > 0.0)
    assert np.all(out['p_value'] <= 1.0)

    thr = cce.get_thresholds()
    assert isinstance(thr, dict)
    assert set(thr.keys()) == set(np.unique(y))


def test_calibrate_uses_significance_controller_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    cce_mod = _import_cce(monkeypatch)
    from sklearn.tree import DecisionTreeClassifier

    from src.core.perf_stats import PerformanceStats

    class DummyASC:
        def __init__(self) -> None:
            self.updates: list[tuple[np.ndarray, np.ndarray]] = []
            self._thr: dict[Any, float] = {}

        def update(self, classes: np.ndarray, p_values: np.ndarray) -> None:
            self.updates.append((np.asarray(classes, dtype=object), np.asarray(p_values, dtype=float)))
            for c in np.unique(classes.astype(object)):
                self._thr[c] = 0.1234

        def get_thresholds(self) -> dict[Any, float]:
            return dict(self._thr)

    X, y = _toy_binary(n=90)
    perf = PerformanceStats()
    asc = DummyASC()

    cce = cce_mod.CrossConformalEvaluator(
        DecisionTreeClassifier(random_state=0),
        folds=3,
        significance=0.10,
        random_state=0,
        n_jobs=1,
        significance_controller=asc,
    )
    cce.calibrate(X, y, perf_stats=perf)

    assert cce.thresholds is not None
    assert set(cce.thresholds.keys()) == set(np.unique(y))
    assert all(v == pytest.approx(0.1234) for v in cce.thresholds.values())
    assert len(asc.updates) >= 1


def test_calibrate_n_jobs_minus_one_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    cce_mod = _import_cce(monkeypatch)
    from sklearn.tree import DecisionTreeClassifier

    from src.core.perf_stats import PerformanceStats

    X, y = _toy_binary(n=60)
    perf = PerformanceStats()

    cce = cce_mod.CrossConformalEvaluator(
        DecisionTreeClassifier(random_state=0),
        folds=2,
        significance=0.10,
        random_state=0,
        n_jobs=-1,
    )
    cce.calibrate(X, y, perf_stats=perf)
    assert cce.thresholds is not None
    assert set(cce.thresholds.keys()) == set(np.unique(y))
