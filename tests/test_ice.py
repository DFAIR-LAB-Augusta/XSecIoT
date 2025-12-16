from __future__ import annotations

import logging
import re

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from sklearn.linear_model import LogisticRegression

from src.core.conformalEval.ice import InductiveConformalEvaluator
from src.core.conformalEval.utils import compute_class_thresholds
from src.core.perf_stats import PerformanceStats


def _toy_binary(n: int = 200, d: int = 6, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.25 * X[:, 1] > 0.0).astype(np.int64)
    if len(np.unique(y)) < 2:
        y[: n // 2] = 0
        y[n // 2 :] = 1
    return X, y


def _toy_multiclass(n: int = 300, d: int = 6, seed: int = 321) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)

    s = X[:, 0] + 0.5 * X[:, 1]
    y = np.zeros(n, dtype=np.int64)
    y[s > 0.8] = 2
    y[(s > -0.2) & (s <= 0.8)] = 1

    uniq = np.unique(y)
    if uniq.size < 3:
        y[:3] = np.array([0, 1, 2], dtype=np.int64)

    return X, y


def _make_model(random_state: int = 0) -> LogisticRegression:
    return LogisticRegression(
        max_iter=200,
        solver='lbfgs',
        multi_class='auto',
        n_jobs=1,
        random_state=random_state,
    )


def test_get_thresholds_raises_before_calibrate() -> None:
    ice = InductiveConformalEvaluator(model=_make_model(), calibration_split=0.2, random_state=0, significance=0.1)

    msg = 'Thresholds not available: call calibrate() first.'
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        ice.get_thresholds()


def test_predict_p_values_raises_before_calibrate() -> None:
    X, _ = _toy_binary()
    ice = InductiveConformalEvaluator(model=_make_model(), calibration_split=0.2, random_state=0, significance=0.1)

    msg = 'ICE must be calibrated before computing p-values.'
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        ice.predict_p_values(X[:5])


def test_init_ignores_unsupported_kwargs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    _ = InductiveConformalEvaluator(
        model=_make_model(),
        calibration_split=0.2,
        random_state=0,
        significance=0.1,
        some_unused_kwarg=True,
        another_one='x',
    )

    assert any('ICE: ignoring unsupported kwargs' in r.message for r in caplog.records)


def test_calibrate_and_predict_binary_happy_path() -> None:
    X, y = _toy_binary()
    stats = PerformanceStats()

    ice = InductiveConformalEvaluator(
        model=_make_model(random_state=0),
        calibration_split=0.25,
        random_state=0,
        significance=0.10,
    )
    ice.calibrate(X, y, perf_stats=stats)

    assert ice.calibration_scores is not None
    assert ice.thresholds is not None
    assert set(ice.thresholds.keys()).issubset(set(np.unique(y)))

    out = ice.predict_p_values(X[:12])
    assert set(out.keys()) == {'class', 'p_value'}
    assert out['class'].shape == (12,)
    assert out['p_value'].shape == (12,)
    assert np.all((out['p_value'] >= 0.0) & (out['p_value'] <= 1.0))

    thr = ice.get_thresholds()
    assert isinstance(thr, dict)
    assert len(thr) > 0

    assert len(stats.ce_stats.accuracies) == 1
    assert len(stats.ce_stats.f1s) == 1


def test_calibrate_multiclass_weighted_metrics_branch() -> None:
    X, y = _toy_multiclass()
    stats = PerformanceStats()

    ice = InductiveConformalEvaluator(
        model=_make_model(random_state=1),
        calibration_split=0.25,
        random_state=1,
        significance=0.10,
    )
    ice.calibrate(X, y, perf_stats=stats)

    assert ice.calibration_scores is not None
    assert ice.thresholds is not None
    assert len(ice.thresholds) >= 2

    out = ice.predict_p_values(X[:9])
    assert out['class'].shape == (9,)
    assert out['p_value'].shape == (9,)
    assert np.all((out['p_value'] >= 0.0) & (out['p_value'] <= 1.0))

    assert len(stats.ce_stats.accuracies) == 1


def test_calibrate_significance_override_changes_thresholds() -> None:
    X, y = _toy_binary()
    stats = PerformanceStats()

    ice = InductiveConformalEvaluator(
        model=_make_model(random_state=2),
        calibration_split=0.25,
        random_state=2,
        significance=0.25,
    )

    override_sig = 0.05
    ice.calibrate(X, y, perf_stats=stats, significance=override_sig)

    assert ice.calibration_scores is not None
    assert ice.thresholds is not None

    expected = compute_class_thresholds(ice.calibration_scores, override_sig)
    assert set(expected.keys()) == set(ice.thresholds.keys())
    for k, v in expected.items():
        assert np.isclose(float(ice.thresholds[k]), float(v), atol=1e-12)


@dataclass
class _DummySigController:
    """
    Minimal controller stub to hit the controller branch.

    Note: ICE passes (fake_preds=y_calib, scores=nonconformity scores) into update().
    """

    last_update: Optional[Tuple[np.ndarray, np.ndarray]] = None
    _thresholds: Optional[Dict[Any, float]] = None

    def update(self, classes: np.ndarray, p_values: np.ndarray) -> None:
        self.last_update = (np.asarray(classes), np.asarray(p_values))
        uniq = np.unique(classes.astype(object))
        self._thresholds = {c: 0.123 for c in uniq}

    def get_thresholds(self) -> Dict[Any, float]:
        assert self._thresholds is not None
        return self._thresholds


def test_calibrate_with_significance_controller_uses_controller_thresholds() -> None:
    X, y = _toy_binary()
    stats = PerformanceStats()
    ctlr = _DummySigController()

    ice = InductiveConformalEvaluator(
        model=_make_model(random_state=3),
        calibration_split=0.25,
        random_state=3,
        significance=0.10,
        significance_controller=ctlr,  # type: ignore
    )
    ice.calibrate(X, y, perf_stats=stats)

    assert ctlr.last_update is not None
    classes, pvals = ctlr.last_update
    assert classes.shape[0] == pvals.shape[0]
    assert ice.thresholds == ctlr.get_thresholds()
