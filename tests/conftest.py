import numpy as np
import pandas as pd
import pytest
import torch

from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig


@pytest.fixture
def device() -> torch.device:
    return torch.device('cpu')


@pytest.fixture
def sim_config_factory(device):
    def _make(tmp_path, **overrides):
        dummy = tmp_path / 'dummy.csv'
        if not dummy.exists():
            dummy.write_text('a\n1\n')
        defaults = dict(
            model_type=ModelType.MULTI,
            model_variant=ModelVariant.DT,
            ce_type=CEType.NONE,
            aggregated_path=dummy,
            flows_path=dummy,
            is_unsw=False,
            seed=0,
            device=device,
        )
        defaults.update(overrides)
        return SimulationConfig(**defaults)

    return _make


@pytest.fixture
def binary_flow_csv_factory():
    def _make(tmp_path, n=60, seed=0, dirname='DS'):
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

    return _make


@pytest.fixture
def multiclass_flow_csv_factory():
    def _make(tmp_path, n=60, seed=0, dirname='DS'):
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

    return _make
