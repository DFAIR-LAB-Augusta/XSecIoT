import numpy as np
import pandas as pd
import pytest
import torch

from firce.runtime.bootstrap import ensure_model_artifacts
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig
from firce.utils.perf_stats import PerformanceStats

DEVICE = torch.device('cpu')


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


def _make_binary_csv(tmp_path, n=60, seed=0, dirname='DS'):
    rng = np.random.default_rng(seed)
    ds_dir = tmp_path / dirname
    ds_dir.mkdir(exist_ok=True)
    csv_path = ds_dir / 'flows.csv'
    labels = np.array(['Benign', 'Attack'])
    idx = rng.integers(0, 2, size=n)
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
        'Label': labels[idx],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_config(csv_path, **overrides):
    defaults = dict(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=csv_path,
        flows_path=csv_path,
        is_unsw=False,
        seed=0,
        device=DEVICE,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_ensure_model_artifacts_multiclass_trains(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(csv_path, model_variant=ModelVariant.DT)

    ensure_model_artifacts(config, PerformanceStats())

    assert (tmp_path / 'multi_class_models' / 'DS' / 'dt_model_multi.pkl').exists()


def test_ensure_model_artifacts_multiclass_feedforward_now_trains(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(csv_path, model_variant=ModelVariant.FEEDFORWARD)

    ensure_model_artifacts(config, PerformanceStats())

    assert (tmp_path / 'multi_class_models' / 'DS' / 'feedforward_model_multi.pt').exists()


def test_ensure_model_artifacts_multiclass_propagates_real_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    pd.DataFrame({
        'flow_duration': [1, 2, 3],
        'tot_fwd_pkt': [1, 2, 3],
        'MC_Label': ['Benign', 'Benign', 'Benign'],
    }).to_csv(csv_path, index=False)
    config = _make_config(csv_path, model_variant=ModelVariant.DT)

    with pytest.raises(ValueError, match='at least 2 distinct MC_Label classes'):
        ensure_model_artifacts(config, PerformanceStats())


def test_ensure_model_artifacts_binary_trains(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_binary_csv(tmp_path)
    config = _make_config(csv_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)

    ensure_model_artifacts(config, PerformanceStats())

    assert (tmp_path / 'binary_models' / 'DS' / 'dt_model_binary.pkl').exists()


def test_ensure_model_artifacts_binary_propagates_real_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    pd.DataFrame({'flow_duration': [1, 2, 3], 'tot_fwd_pkt': [1, 2, 3]}).to_csv(csv_path, index=False)
    config = _make_config(csv_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)

    with pytest.raises(ValueError, match="must contain either 'Label' or 'BinLabel'"):
        ensure_model_artifacts(config, PerformanceStats())
