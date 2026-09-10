import numpy as np
import pandas as pd
import pytest
import torch

from firce.runtime.retraining import retrain_runtime
from firce.runtime.sim_types import SimulationRuntime
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig
from firce.utils.perf_stats import PerformanceStats

DEVICE = torch.device('cpu')


class _StubMonitor:
    """Duck-typed drift monitor stub — decouples this test from CE/CADE's
    unverified multiclass internals (#93/#94), so it only exercises
    retrain_runtime's own model_type branching and artifact loading."""

    def __init__(self):
        self.fit_calls: list = []

    def fit(self, X, y, perf_stats):
        self.fit_calls.append((X, y))


ROLLING_COLUMNS = [
    'device_id',
    'session_id',
    'src_ip',
    'dst_ip',
    'src_port',
    'dst_port',
    'protocol',
    'timestamp',
    'flow_duration',
    'tot_fwd_pkt',
    'tot_bwd_pkts',
    'totlen_fwd_pkts',
    'totlen_bwd_pkts',
    'MC_Label',
]


def _make_multiclass_rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(['Benign', 'PortScan', 'XMasAttack'])
    idx = rng.integers(0, 3, size=n)
    return pd.DataFrame({
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


BINARY_ROLLING_COLUMNS = [
    'device_id',
    'session_id',
    'src_ip',
    'dst_ip',
    'src_port',
    'dst_port',
    'protocol',
    'timestamp',
    'flow_duration',
    'tot_fwd_pkt',
    'tot_bwd_pkts',
    'totlen_fwd_pkts',
    'totlen_bwd_pkts',
    'BinLabel',
]


def _make_binary_rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 2, size=n)
    return pd.DataFrame({
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
        'BinLabel': idx,
    })


def _make_config(tmp_path, **overrides):
    dummy = tmp_path / 'dummy.csv'
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


def _make_runtime(tmp_path, model_variant=ModelVariant.DT):
    config = _make_config(tmp_path, model_variant=model_variant)
    rolling = CircularDequeLogger(None, max_rows=200, columns=ROLLING_COLUMNS)
    for row in _make_multiclass_rows().itertuples(index=False, name=None):
        rolling.append(list(row))

    return SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=rolling,
        scaler=None,
        pca=None,
        model=None,
        monitor=_StubMonitor(),
        train_df=pd.DataFrame(),
    )


@pytest.mark.parametrize('model_variant', [ModelVariant.DT, ModelVariant.KNN, ModelVariant.RF, ModelVariant.SVM])
def test_retrain_runtime_multiclass_updates_model_and_scaler(tmp_path, monkeypatch, model_variant):
    monkeypatch.chdir(tmp_path)
    runtime = _make_runtime(tmp_path, model_variant=model_variant)

    retrain_runtime(runtime)

    assert runtime.scaler is not None
    assert runtime.model is not None
    assert len(runtime.monitor.fit_calls) == 1
    _, y_fit = runtime.monitor.fit_calls[0]
    assert set(y_fit.tolist()) <= {'Benign', 'PortScan', 'XMasAttack'}


def test_retrain_runtime_raises_when_monitor_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = _make_runtime(tmp_path)
    runtime.monitor = None

    with pytest.raises(RuntimeError, match='Monitor is disabled'):
        retrain_runtime(runtime)


def test_retrain_runtime_rejects_single_class_retraining_data(tmp_path, monkeypatch):
    # train_ce_multiclass itself rejects <2 distinct MC_Label classes before
    # retrain_runtime ever reaches _fit_monitor_on_retrained_data's own
    # single-class skip-refit branch - that guard fires first in practice.
    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path)
    rolling = CircularDequeLogger(None, max_rows=200, columns=ROLLING_COLUMNS)
    single_class_rows = _make_multiclass_rows().copy()
    single_class_rows['MC_Label'] = 'Benign'
    for row in single_class_rows.itertuples(index=False, name=None):
        rolling.append(list(row))

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=rolling,
        scaler=None,
        pca=None,
        model=None,
        monitor=_StubMonitor(),
        train_df=pd.DataFrame(),
    )

    with pytest.raises(ValueError, match='at least 2 distinct MC_Label classes'):
        retrain_runtime(runtime)


def test_retrain_runtime_binary_updates_model_and_scaler(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)
    rolling = CircularDequeLogger(None, max_rows=200, columns=BINARY_ROLLING_COLUMNS)
    for row in _make_binary_rows().itertuples(index=False, name=None):
        rolling.append(list(row))

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=rolling,
        scaler=None,
        pca=None,
        model=None,
        monitor=_StubMonitor(),
        train_df=pd.DataFrame(),
    )

    retrain_runtime(runtime)

    assert runtime.scaler is not None
    assert runtime.model is not None
    assert len(runtime.monitor.fit_calls) == 1
    _, y_fit = runtime.monitor.fit_calls[0]
    assert set(y_fit.tolist()) <= {0, 1}


def test_retrain_runtime_multiclass_feedforward(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = _make_runtime(tmp_path, model_variant=ModelVariant.FEEDFORWARD)

    retrain_runtime(runtime)

    assert runtime.model is not None
    input_dim = runtime.model.trunk[0].in_features
    out = runtime.model(torch.randn(2, input_dim))
    assert out.shape == (2, 3)
