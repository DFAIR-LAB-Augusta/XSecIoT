import numpy as np
import pandas as pd
import pytest
import torch

from firce.runtime.bootstrap import get_rolling_columns, initialize_simulation_runtime
from firce.utils.config import CEType, ModelType, ModelVariant, MonitorType, SimulationConfig

DEVICE = torch.device('cpu')


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


def test_get_rolling_columns_multiclass_includes_mc_label(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.MULTI)
    columns = get_rolling_columns(config)

    assert 'MC_Label' in columns
    assert 'BinLabel' not in columns


def test_get_rolling_columns_binary_still_includes_bin_label(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT)
    columns = get_rolling_columns(config)

    assert 'BinLabel' in columns
    assert 'MC_Label' not in columns


def test_get_rolling_columns_unsw_multiclass_includes_mc_label(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.MULTI, is_unsw=True)
    columns = get_rolling_columns(config)

    assert 'MC_Label' in columns
    assert 'BinLabel' not in columns
    assert len(columns) == 21


def test_get_rolling_columns_unsw_binary_still_includes_bin_label(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.BINARY, model_variant=ModelVariant.DT, is_unsw=True)
    columns = get_rolling_columns(config)

    assert 'BinLabel' in columns
    assert 'MC_Label' not in columns
    assert len(columns) == 21


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


@pytest.mark.parametrize(
    'model_variant', [ModelVariant.DT, ModelVariant.KNN, ModelVariant.RF, ModelVariant.SVM, ModelVariant.FEEDFORWARD]
)
def test_initialize_simulation_runtime_multiclass_all_variants(tmp_path, monkeypatch, model_variant):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(
        tmp_path,
        model_variant=model_variant,
        aggregated_path=csv_path,
        flows_path=csv_path,
        monitor_type=MonitorType.NONE,
    )

    runtime = initialize_simulation_runtime(config)

    assert runtime.model is not None
    assert runtime.scaler is not None
    assert runtime.label_encoder is not None
    assert 'MC_Label' in runtime.rolling.columns


def test_initialize_simulation_runtime_multiclass_fits_ce_monitor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(
        tmp_path,
        model_variant=ModelVariant.DT,
        aggregated_path=csv_path,
        flows_path=csv_path,
        monitor_type=MonitorType.CE,
        ce_type=CEType.ICE,
    )

    runtime = initialize_simulation_runtime(config)

    assert runtime.monitor is not None
    thresholds = runtime.monitor._evaluator.thresholds
    assert set(thresholds.keys()) == {'Benign', 'PortScan', 'XMasAttack'}


def test_initialize_simulation_runtime_binary_fits_ce_monitor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_binary_csv(tmp_path)
    config = _make_config(
        tmp_path,
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        aggregated_path=csv_path,
        flows_path=csv_path,
        monitor_type=MonitorType.CE,
        ce_type=CEType.ICE,
    )

    runtime = initialize_simulation_runtime(config)

    assert runtime.monitor is not None
    thresholds = runtime.monitor._evaluator.thresholds
    assert set(thresholds.keys()) == {0, 1}
    assert runtime.label_encoder is None


def _make_unsw_raw_csv(tmp_path, n=60, seed=0, dirname='UNSW_DS'):
    rng = np.random.default_rng(seed)
    ds_dir = tmp_path / dirname
    ds_dir.mkdir(exist_ok=True)
    csv_path = ds_dir / 'flows.csv'
    labels = np.array(['Benign', 'DoS', 'Reconnaissance'])
    idx = rng.integers(0, 3, size=n)
    base_ms = 1_600_000_000_000
    df = pd.DataFrame({
        'IPV4_SRC_ADDR': ['10.0.0.1'] * n,
        'IPV4_DST_ADDR': ['10.0.0.2'] * n,
        'L4_SRC_PORT': rng.integers(1024, 65535, size=n),
        'L4_DST_PORT': rng.integers(1, 1024, size=n),
        'PROTOCOL': rng.integers(0, 2, size=n),
        'FLOW_START_MILLISECONDS': base_ms + np.arange(n) * 1000,
        'FLOW_END_MILLISECONDS': base_ms + np.arange(n) * 1000 + 500,
        'FLOW_DURATION_MILLISECONDS': rng.random(n) * 100,
        'IN_PKTS': rng.integers(1, 50, size=n),
        'OUT_PKTS': rng.integers(1, 50, size=n),
        'IN_BYTES': rng.random(n) * 1000,
        'OUT_BYTES': rng.random(n) * 1000,
        'SRC_TO_DST_IAT_MIN': rng.random(n) * 10,
        'SRC_TO_DST_IAT_MAX': rng.random(n) * 10,
        'SRC_TO_DST_IAT_AVG': rng.random(n) * 10,
        'SRC_TO_DST_IAT_STDDEV': rng.random(n) * 10,
        'DST_TO_SRC_IAT_MIN': rng.random(n) * 10,
        'DST_TO_SRC_IAT_MAX': rng.random(n) * 10,
        'DST_TO_SRC_IAT_AVG': rng.random(n) * 10,
        'DST_TO_SRC_IAT_STDDEV': rng.random(n) * 10,
        'Label': (idx != 0).astype(int),
        'Attack': labels[idx],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def test_initialize_simulation_runtime_unsw_multiclass_seeds_mc_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_unsw_raw_csv(tmp_path)
    config = _make_config(
        tmp_path,
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        aggregated_path=csv_path,
        flows_path=csv_path,
        is_unsw=True,
        monitor_type=MonitorType.NONE,
        use_circular_logger=True,
    )

    runtime = initialize_simulation_runtime(config)

    assert 'MC_Label' in runtime.rolling.columns
    assert 'BinLabel' not in runtime.rolling.columns
    seeded = runtime.rolling.to_dataframe()
    assert len(seeded) == 60
    assert set(seeded['MC_Label'].unique()) == {'Benign', 'DoS', 'Reconnaissance'}
    assert runtime.label_encoder is not None
    assert sorted(runtime.label_encoder.classes_.tolist()) == ['Benign', 'DoS', 'Reconnaissance']


def test_initialize_simulation_runtime_unsw_binary_still_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_unsw_raw_csv(tmp_path)
    config = _make_config(
        tmp_path,
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        aggregated_path=csv_path,
        flows_path=csv_path,
        is_unsw=True,
        monitor_type=MonitorType.NONE,
        use_circular_logger=True,
    )

    runtime = initialize_simulation_runtime(config)

    assert 'BinLabel' in runtime.rolling.columns
    assert 'MC_Label' not in runtime.rolling.columns
    seeded = runtime.rolling.to_dataframe()
    assert len(seeded) == 60
    assert set(seeded['BinLabel'].unique()) <= {0, 1}
