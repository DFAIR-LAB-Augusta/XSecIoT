import numpy as np
import pandas as pd
import torch

from fire.simulations import load_simulation_objects
from firce.ce_model_training import train_ce_multiclass
from firce.models.feedforward_multiclass import FeedForwardMulticlass
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig

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


def test_load_simulation_objects_multiclass_dt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)
    train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False)

    scaler, pca, model = load_simulation_objects(str(csv_path), 'multi', 'dt', use_pca=False)

    assert scaler is not None
    assert pca is None
    assert hasattr(model, 'predict')


def test_load_simulation_objects_multiclass_feedforward(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.FEEDFORWARD)
    train_ce_multiclass(config, str(csv_path), variant=ModelVariant.FEEDFORWARD, use_pca=False)

    scaler, pca, model = load_simulation_objects(str(csv_path), 'multi', 'feedforward', use_pca=False)

    assert isinstance(model, FeedForwardMulticlass)
    out = model(torch.randn(2, model.trunk[0].in_features))
    assert out.shape == (2, 3)
