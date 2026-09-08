import numpy as np
import pandas as pd
import torch

from firce.runtime.bootstrap import get_rolling_columns
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


import pytest

from firce.runtime.bootstrap import initialize_simulation_runtime


def test_initialize_simulation_runtime_rejects_unsw_multiclass(tmp_path):
    config = _make_config(tmp_path, model_type=ModelType.MULTI, is_unsw=True)

    with pytest.raises(ValueError, match='UNSW \\+ multiclass'):
        initialize_simulation_runtime(config)


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
