from __future__ import annotations

import re

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from sklearn.tree import DecisionTreeClassifier

from src.core.config import CEType
from src.core.conformalEval.conformal_evaluators import ConformalEvaluator, ConformalEvaluatorFactory
from src.core.perf_stats import PerformanceStats


def _toy_binary(n: int = 64, d: int = 6, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.25 * X[:, 1] > 0.0).astype(np.int32)
    return X, y


class _DummyASC:
    """
    Minimal stand-in for AdaptiveSignificanceController so these tests
    don't depend on TOML config or the controller internals.
    """

    def __init__(self, thresholds: Dict[Any, float] | None = None) -> None:
        self._thresholds = dict(thresholds or {})
        self.updates: list[tuple[np.ndarray, np.ndarray]] = []

    def update(self, classes: np.ndarray, p_values: np.ndarray) -> None:
        self.updates.append((np.asarray(classes, dtype=object), np.asarray(p_values, dtype=float)))

    def get_thresholds(self) -> Dict[Any, float]:
        return dict(self._thresholds)


def test_factory_create_returns_expected_impl_types() -> None:
    model = DecisionTreeClassifier(random_state=0)

    ice = ConformalEvaluatorFactory.create(CEType.ICE, model=model, folds=3, significance=0.1, n_jobs=1)  # type: ignore
    assert ice.__class__.__name__ in {'InductiveConformalEvaluator'}

    cce = ConformalEvaluatorFactory.create(CEType.CCE, model=model, folds=3, significance=0.1, n_jobs=1)  # type: ignore
    assert cce.__class__.__name__ in {'CrossConformalEvaluator'}

    tce = ConformalEvaluatorFactory.create(CEType.APPROX_TCE, model=model, folds=3, significance=0.1, n_jobs=1)  # type: ignore
    assert tce.__class__.__name__ in {'ApproximateTransductiveConformalEvaluator'}

    approx_cce = ConformalEvaluatorFactory.create(CEType.APPROX_CCE, model=model, folds=3, significance=0.1, n_jobs=1)  # type: ignore
    assert approx_cce.__class__.__name__ in {'ApproxCrossConformalEvaluator'}


def test_factory_create_unknown_raises() -> None:
    model = DecisionTreeClassifier(random_state=0)
    with pytest.raises(ValueError, match=re.escape('Unknown evaluator type: totally_fake')):
        ConformalEvaluatorFactory.create('totally_fake', model=model)  # type: ignore[arg-type]


@pytest.mark.skipif(
    True,
    reason='Need to rewrite conformal_evaluators.',
)
def test_conformal_evaluator_detect_drift_requires_calibration() -> None:
    model = DecisionTreeClassifier(random_state=0)
    ce = ConformalEvaluator(evaluator_type=CEType.ICE, model=model)
    X, _ = _toy_binary()
    with pytest.raises(RuntimeError, match=re.escape('ConformalEvaluator must be calibrated before drift detection.')):
        ce.detect_drift(X[:1])


@pytest.mark.skipif(
    True,
    reason='Need to rewrite conformal_evaluators.',
)
def test_conformal_evaluator_calibrate_sets_thresholds_smoke() -> None:
    X, y = _toy_binary(n=80, d=6, seed=7)
    model = DecisionTreeClassifier(random_state=0)

    ce = ConformalEvaluator(
        evaluator_type=CEType.ICE,
        model=model,
        calibration_split=0.25,
        random_state=0,
        significance=0.10,
    )
    ce.calibrate(X, y, perf_stats=PerformanceStats(), calibration_split=0.25, random_state=0, significance=0.1)
    assert ce.thresholds is not None
    assert set(ce.thresholds.keys()).issubset(set(np.unique(y).tolist()))
    assert all(isinstance(v, float) for v in ce.thresholds.values())


@pytest.mark.skipif(
    True,
    reason='Need to rewrite conformal_evaluators.',
)
def test_detect_drift_case1_dict_format(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DecisionTreeClassifier(random_state=0)
    ce = ConformalEvaluator(evaluator_type=CEType.ICE, model=model)

    ce.thresholds = {1: 0.50}
    ce.evaluator.model.n_features_in_ = 999  # type: ignore[attr-defined]

    def _fake_predict_p_values(_: np.ndarray) -> Dict[str, np.ndarray]:
        return {'class': np.array([1]), 'p_value': np.array([0.40], dtype=float)}

    monkeypatch.setattr(ce.evaluator, 'predict_p_values', _fake_predict_p_values)
    drifted = ce.detect_drift(np.zeros((1, 6), dtype=np.float32))
    assert drifted.shape == (1,)
    assert drifted[0] is True


@pytest.mark.skipif(
    True,
    reason='Need to rewrite conformal_evaluators.',
)
def test_detect_drift_case2_dataframe_format(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DecisionTreeClassifier(random_state=0)
    ce = ConformalEvaluator(evaluator_type=CEType.ICE, model=model)

    ce.thresholds = {0: 0.10}

    def _fake_predict_p_values(_: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({'class': [0], 'p_value': [0.25]})

    monkeypatch.setattr(ce.evaluator, 'predict_p_values', _fake_predict_p_values)
    drifted = ce.detect_drift(np.zeros((1, 6), dtype=np.float32))
    assert drifted.shape == (1,)
    assert drifted[0] is False


@pytest.mark.skipif(
    True,
    reason='Need to rewrite conformal_evaluators.',
)
def test_detect_drift_case3_ndarray_binary_format(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DecisionTreeClassifier(random_state=0)
    ce = ConformalEvaluator(evaluator_type=CEType.ICE, model=model)

    ce.thresholds = {0: 0.30}

    monkeypatch.setattr(ce.evaluator, 'predict_p_values', lambda X: np.array([0.01]))

    out = ce.detect_drift(np.zeros((1, 2)))
    assert out.shape == (1,)
    assert out[0] is True


@pytest.mark.skipif(
    True,
    reason='Need to rewrite conformal_evaluators.',
)
def test_detect_drift_case4_scalar_format(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DecisionTreeClassifier(random_state=0)
    ce = ConformalEvaluator(evaluator_type=CEType.ICE, model=model)
    ce.thresholds = {0: 0.20}

    def _fake_predict_p_values(_: np.ndarray) -> float:
        return 0.10

    monkeypatch.setattr(ce.evaluator, 'predict_p_values', _fake_predict_p_values)
    drifted = ce.detect_drift(np.zeros((1, 6), dtype=np.float32))
    assert drifted.shape == (1,)
    assert drifted[0] is True


@pytest.mark.skipif(
    True,
    reason='Need to rewrite conformal_evaluators.',
)
def test_detect_drift_uses_significance_controller_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _DummyASC(thresholds={1: 0.90})
    ce = ConformalEvaluator(
        ce_type=CEType.ICE,
        model=DecisionTreeClassifier(random_state=0),
        folds=3,
        significance=0.1,
        n_jobs=1,
        significance_controller=controller,  # type: ignore[arg-type]
    )

    ce.thresholds = {1: 0.50}

    def _fake_predict_p_values(_: np.ndarray) -> Dict[str, np.ndarray]:
        return {'class': np.array([1]), 'p_value': np.array([0.60], dtype=float)}

    monkeypatch.setattr(ce.evaluator, 'predict_p_values', _fake_predict_p_values)
    drifted = ce.detect_drift(np.zeros((1, 6), dtype=np.float32))

    assert drifted.shape == (1,)
    assert drifted[0] is True
    assert len(controller.updates) == 1
    upd_classes, upd_pvals = controller.updates[0]
    assert upd_classes.tolist() == [1]
    assert np.allclose(upd_pvals, np.array([0.60], dtype=float))


@pytest.mark.skipif(
    True,
    reason='Need to rewrite conformal_evaluators.',
)
def test_detect_drift_unexpected_format_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DecisionTreeClassifier(random_state=0)
    ce = ConformalEvaluator(evaluator_type=CEType.ICE, model=model)
    ce.thresholds = {0: 0.20}

    def _fake_predict_p_values(_: np.ndarray) -> Any:
        return {'nope': 123}

    monkeypatch.setattr(ce.evaluator, 'predict_p_values', _fake_predict_p_values)

    with pytest.raises(ValueError, match=re.escape('Unexpected p-values format returned from evaluator.')):
        ce.detect_drift(np.zeros((1, 6), dtype=np.float32))
