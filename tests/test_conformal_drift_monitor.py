import numpy as np

from sklearn.tree import DecisionTreeClassifier

from firce.drift_monitor.conformal_monitor import ConformalDriftMonitor
from firce.utils.config import CEType
from firce.utils.perf_stats import PerformanceStats


def test_conformal_drift_monitor_exposes_model_and_calibration_scores():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = np.select([X[:, 0] > 0.5], ['Attack'], default='Benign')

    monitor = ConformalDriftMonitor(
        CEType.ICE, DecisionTreeClassifier(random_state=0), calibration_split=0.3, random_state=0
    )
    monitor.fit(X, y, PerformanceStats())

    assert hasattr(monitor.model, 'predict_proba')
    assert set(monitor.calibration_scores.keys()) == {'Benign', 'Attack'}
