import numpy as np
import pytest

from sklearn.tree import DecisionTreeClassifier

from firce.conformalEval.approx_cce import ApproxCrossConformalEvaluator
from firce.conformalEval.cce import CrossConformalEvaluator
from firce.conformalEval.conformal_evaluators import ConformalEvaluator
from firce.conformalEval.ice import InductiveConformalEvaluator
from firce.conformalEval.tce import ApproximateTransductiveConformalEvaluator
from firce.drift_monitor.conformal_monitor import ConformalDriftMonitor
from firce.utils.config import CEType
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


EVALUATOR_FACTORIES = {
    'ice': lambda: InductiveConformalEvaluator(
        DecisionTreeClassifier(random_state=0), calibration_split=0.3, random_state=0
    ),
    'cce': lambda: CrossConformalEvaluator(DecisionTreeClassifier(random_state=0), folds=3, random_state=0),
    'approx_tce': lambda: ApproximateTransductiveConformalEvaluator(DecisionTreeClassifier(random_state=0)),
    'approx_cce': lambda: ApproxCrossConformalEvaluator(
        DecisionTreeClassifier(random_state=0), folds=3, random_state=0
    ),
}


@pytest.mark.parametrize('evaluator_name', list(EVALUATOR_FACTORIES))
def test_calibrate_produces_per_class_thresholds(evaluator_name):
    X, y = _make_multiclass_data()
    evaluator = EVALUATOR_FACTORIES[evaluator_name]()

    evaluator.calibrate(X, y, PerformanceStats())
    thresholds = evaluator.get_thresholds()

    assert set(thresholds.keys()) == set(np.unique(y).tolist())
    for thresh in thresholds.values():
        assert 0.0 <= thresh <= 1.0


@pytest.mark.parametrize('evaluator_name', list(EVALUATOR_FACTORIES))
def test_predict_p_values_are_valid_and_match_known_classes(evaluator_name):
    X, y = _make_multiclass_data()
    evaluator = EVALUATOR_FACTORIES[evaluator_name]()
    evaluator.calibrate(X, y, PerformanceStats())

    result = evaluator.predict_p_values(X[:10])

    assert set(result.keys()) == {'class', 'p_value'}
    assert len(result['class']) == 10
    assert len(result['p_value']) == 10
    assert set(np.asarray(result['class']).tolist()) <= set(np.unique(y).tolist())
    assert np.all((np.asarray(result['p_value']) >= 0.0) & (np.asarray(result['p_value']) <= 1.0))


@pytest.mark.parametrize(
    'ce_type,extra_kwargs',
    [
        (CEType.ICE, {'calibration_split': 0.3, 'random_state': 0}),
        (CEType.CCE, {'folds': 3, 'random_state': 0}),
        (CEType.APPROX_TCE, {}),
        (CEType.APPROX_CCE, {'folds': 3, 'random_state': 0}),
    ],
)
def test_conformal_evaluator_wrapper_multiclass_detect_drift(ce_type, extra_kwargs):
    X, y = _make_multiclass_data()
    ce = ConformalEvaluator(ce_type, DecisionTreeClassifier(random_state=0), **extra_kwargs)
    ce.calibrate(X, y, PerformanceStats())

    single_row = X[[0]]
    drift_flags = ce.detect_drift(single_row)

    assert drift_flags.shape == (1,)
    assert drift_flags.dtype == bool


@pytest.mark.parametrize(
    'ce_type,extra_kwargs',
    [
        (CEType.ICE, {'calibration_split': 0.3, 'random_state': 0}),
        (CEType.CCE, {'folds': 3, 'random_state': 0}),
        (CEType.APPROX_TCE, {}),
        (CEType.APPROX_CCE, {'folds': 3, 'random_state': 0}),
    ],
)
def test_conformal_drift_monitor_multiclass_fit_and_detect(ce_type, extra_kwargs):
    X, y = _make_multiclass_data()
    monitor = ConformalDriftMonitor(ce_type, DecisionTreeClassifier(random_state=0), **extra_kwargs)

    monitor.fit(X, y, PerformanceStats())
    result = monitor.detect(X[[0]])

    assert result.row_flags.shape == (1,)
    assert result.row_flags.dtype == bool
    assert result.chunk_drift == bool(result.row_flags.any())
    assert result.metadata['chunk_size'] == 1
