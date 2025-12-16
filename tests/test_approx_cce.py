from __future__ import annotations

import re

import numpy as np
import pytest

from sklearn.tree import DecisionTreeClassifier

from src.core.perf_stats import PerformanceStats


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


@pytest.mark.slow
def test_approx_cce_calibrate_binary(monkeypatch) -> None:
    from src.core.conformalEval.approx_cce import ApproxCrossConformalEvaluator

    X, y = _toy_binary(n=100)

    perf_stats = PerformanceStats()

    model = DecisionTreeClassifier(random_state=0)

    approx_cce = ApproxCrossConformalEvaluator(
        model=model,
        folds=3,
        significance=0.10,
        random_state=0,
        n_jobs=1,
    )

    approx_cce.calibrate(X, y, perf_stats=perf_stats)

    assert approx_cce.calibration_scores is not None
    assert approx_cce.thresholds is not None
    assert set(approx_cce.calibration_scores.keys()) == set(np.unique(y))
    assert set(approx_cce.thresholds.keys()) == set(np.unique(y))

    assert len(perf_stats.ce_stats.accuracies) == 1
    assert 0.0 <= perf_stats.ce_stats.accuracies[0] <= 1.0


@pytest.mark.slow
def test_approx_cce_calibrate_multiclass(monkeypatch) -> None:
    from src.core.conformalEval.approx_cce import ApproxCrossConformalEvaluator

    X, y = _toy_multiclass(n=120)

    perf_stats = PerformanceStats()

    model = DecisionTreeClassifier(random_state=0)

    approx_cce = ApproxCrossConformalEvaluator(
        model=model,
        folds=3,
        significance=0.10,
        random_state=0,
        n_jobs=1,
    )

    approx_cce.calibrate(X, y, perf_stats=perf_stats)

    assert approx_cce.calibration_scores is not None
    assert approx_cce.thresholds is not None
    assert set(approx_cce.calibration_scores.keys()) == set(np.unique(y))
    assert set(approx_cce.thresholds.keys()) == set(np.unique(y))

    assert len(perf_stats.ce_stats.accuracies) == 1
    assert 0.0 <= perf_stats.ce_stats.accuracies[0] <= 1.0


def test_predict_p_values_raises_before_calibrate(monkeypatch) -> None:
    from src.core.conformalEval.approx_cce import ApproxCrossConformalEvaluator

    X, _ = _toy_binary()

    approx_cce = ApproxCrossConformalEvaluator(
        model=DecisionTreeClassifier(random_state=0),
        folds=3,
        significance=0.10,
        random_state=0,
        n_jobs=1,
    )

    with pytest.raises(RuntimeError, match='must be calibrated'):
        approx_cce.predict_p_values(X[:5])


def test_predict_p_values_binary(monkeypatch) -> None:
    from src.core.conformalEval.approx_cce import ApproxCrossConformalEvaluator

    X, y = _toy_binary(n=100)

    perf_stats = PerformanceStats()

    model = DecisionTreeClassifier(random_state=0)

    approx_cce = ApproxCrossConformalEvaluator(
        model=model,
        folds=3,
        significance=0.10,
        random_state=0,
        n_jobs=1,
    )

    approx_cce.calibrate(X, y, perf_stats=perf_stats)

    out = approx_cce.predict_p_values(X[:10])

    assert set(out.keys()) == {'class', 'p_value'}
    assert out['class'].shape == (10,)
    assert out['p_value'].shape == (10,)
    assert np.all(out['p_value'] > 0.0)
    assert np.all(out['p_value'] <= 1.0)


def test_get_thresholds_raises_before_calibrate(monkeypatch) -> None:
    from sklearn.tree import DecisionTreeClassifier

    from src.core.conformalEval.approx_cce import ApproxCrossConformalEvaluator

    approx_cce = ApproxCrossConformalEvaluator(
        model=DecisionTreeClassifier(random_state=0),
        folds=3,
        significance=0.10,
        random_state=0,
        n_jobs=1,
    )

    msg = 'Thresholds not available: call calibrate() first.'
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        approx_cce.get_thresholds()


def test_get_thresholds_after_calibrate(monkeypatch) -> None:
    from src.core.conformalEval.approx_cce import ApproxCrossConformalEvaluator

    X, y = _toy_binary(n=100)

    perf_stats = PerformanceStats()

    model = DecisionTreeClassifier(random_state=0)

    approx_cce = ApproxCrossConformalEvaluator(
        model=model,
        folds=3,
        significance=0.10,
        random_state=0,
        n_jobs=1,
    )

    approx_cce.calibrate(X, y, perf_stats=perf_stats)

    thresholds = approx_cce.get_thresholds()

    assert isinstance(thresholds, dict)
    assert len(thresholds) == len(np.unique(y))
