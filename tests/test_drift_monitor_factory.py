import pytest

from sklearn.tree import DecisionTreeClassifier

from firce.drift_monitor.conformal_monitor import ConformalDriftMonitor
from firce.drift_monitor.factory import build_monitor
from firce.utils.config import CEType, ModelType, ModelVariant, MonitorType, SimulationConfig


def _make_config(tmp_path, **overrides):
    dummy = tmp_path / 'dummy.csv'
    dummy.write_text('a\n1\n')
    defaults = dict(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=dummy,
        flows_path=dummy,
        is_unsw=False,
        seed=0,
        monitor_type=MonitorType.NONE,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_build_monitor_returns_none_when_disabled(tmp_path):
    config = _make_config(tmp_path, monitor_type=MonitorType.NONE)

    monitor = build_monitor(config, model=DecisionTreeClassifier())

    assert monitor is None


def test_build_monitor_returns_conformal_drift_monitor_for_ce(tmp_path):
    config = _make_config(tmp_path, monitor_type=MonitorType.CE, ce_type=CEType.ICE)

    monitor = build_monitor(config, model=DecisionTreeClassifier(random_state=0))

    assert isinstance(monitor, ConformalDriftMonitor)


def test_build_monitor_dispatches_cade(tmp_path):
    pytest.importorskip('cade')
    config = _make_config(
        tmp_path,
        monitor_type=MonitorType.CADE,
        monitor_kwargs={'dims': [4, 2]},
    )

    from firce.drift_monitor.cade_monitor import CadeDriftMonitor

    monitor = build_monitor(config, model=None)

    assert isinstance(monitor, CadeDriftMonitor)
