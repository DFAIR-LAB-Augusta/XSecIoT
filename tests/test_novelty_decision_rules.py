import numpy as np
import pytest

from sklearn.tree import DecisionTreeClassifier

from firce.conformalEval.ice import InductiveConformalEvaluator
from firce.novelty.decision_rules import compute_all_class_p_values
from firce.utils.perf_stats import PerformanceStats


def _make_multiclass_data(n=90, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float64)
    y = np.select(
        [X[:, 0] > 0.5, X[:, 1] > 0.5],
        ['PortScan', 'XMasAttack'],
        default='Benign',
    )
    return X, y


def _make_calibrated_ice(seed=0):
    X, y = _make_multiclass_data(seed=seed)
    ice = InductiveConformalEvaluator(DecisionTreeClassifier(random_state=0), calibration_split=0.3, random_state=0)
    ice.calibrate(X, y, PerformanceStats())
    return ice, X, y


def test_compute_all_class_p_values_covers_every_known_class():
    ice, X, y = _make_calibrated_ice()

    result = compute_all_class_p_values(ice.model, ice.calibration_scores, X[:10])

    assert set(result.keys()) == set(np.unique(y).tolist())
    for cls, p_values in result.items():
        assert p_values.shape == (10,)
        assert np.all((p_values >= 0.0) & (p_values <= 1.0))


def test_compute_all_class_p_values_matches_predicted_class_p_value():
    ice, X, y = _make_calibrated_ice()

    all_class_result = compute_all_class_p_values(ice.model, ice.calibration_scores, X[:10])
    predicted_result = ice.predict_p_values(X[:10])

    for i, cls in enumerate(predicted_result['class']):
        assert all_class_result[cls][i] == pytest.approx(predicted_result['p_value'][i])
