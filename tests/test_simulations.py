import sys

import numpy as np
import pandas as pd
import torch

from firce.ce_model_training import train_ce_binary
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig
from firce.utils.perf_stats import PerformanceStats
from fire.simulations import (
    _get_dataset_name,
    _parse_args,
    continuous_simulation,
    parallel_simulation,
    preprocess_chunk,
    sequential_simulation,
)

DEVICE = torch.device('cpu')


def _train_binary_fixture(tmp_path, csv_path):
    config = SimulationConfig(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=csv_path,
        flows_path=csv_path,
        is_unsw=False,
        seed=0,
        device=DEVICE,
        use_pca=True,
    )
    train_ce_binary(config, str(csv_path), PerformanceStats())


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'agg.csv'])
    args = _parse_args()

    assert args.aggregated_file == 'agg.csv'
    assert args.mode == 'sequential'
    assert args.model_type == 'binary'
    assert args.model_variant == 'dt'
    assert args.chunk_size == 1000
    assert not args.unsw


def test_parse_args_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['prog', 'agg.csv', '--mode', 'parallel', '--model_type', 'multi', '--model_variant', 'rf', '--unsw'],
    )
    args = _parse_args()

    assert args.mode == 'parallel'
    assert args.model_type == 'multi'
    assert args.model_variant == 'rf'
    assert args.unsw


def test_get_dataset_name():
    assert _get_dataset_name('/foo/BAR/agg.csv') == 'BAR'


def test_preprocess_chunk_drops_columns_and_fills_na():
    df = pd.DataFrame({'a': [1, np.nan], 'b': [np.nan, 2], 'drop': [9, 9]})

    cleaned = preprocess_chunk(df, ['drop'])

    assert 'drop' not in cleaned.columns
    assert not cleaned.isna().any().any()


def _make_binary_sim_csv(csv_path, n=30, seed=0, with_end_time=False):
    rng = np.random.default_rng(seed)
    labels = np.array(['Benign', 'Attack'])
    idx = rng.integers(0, 2, size=n)
    data = {
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
    }
    if with_end_time:
        data['end_time_x'] = pd.date_range('2020-01-01', periods=n, freq='s').astype(str)
    pd.DataFrame(data).to_csv(csv_path, index=False)


def test_sequential_simulation_binary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    n = 30
    _make_binary_sim_csv(csv_path, n=n)

    _train_binary_fixture(tmp_path, csv_path)

    preds = sequential_simulation(str(csv_path), model_type='binary', model_variant='dt', chunk_size=5, delay=0)

    assert len(preds) == n
    assert set(preds) <= {'Benign', 'Attack'}


def test_continuous_simulation_binary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    n = 30
    _make_binary_sim_csv(csv_path, n=n, with_end_time=True)

    _train_binary_fixture(tmp_path, csv_path)

    true_labels, preds = continuous_simulation(
        str(csv_path), model_type='binary', model_variant='dt', chunk_size=5, window_duration=300, delay=0
    )

    # window_duration=300s never trims this 30-row/30-second fixture, so each
    # of the 6 chunks re-predicts the entire accumulated window (5, 10, 15,
    # 20, 25, 30 rows) rather than just new rows - this is the function's
    # actual designed behavior, not a bug.
    expected_total = sum(range(5, n + 1, 5))
    assert len(true_labels) == expected_total
    assert len(preds) == expected_total
    assert set(preds) <= {'Benign', 'Attack'}


def test_parallel_simulation_binary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    n = 30
    _make_binary_sim_csv(csv_path, n=n)

    _train_binary_fixture(tmp_path, csv_path)

    preds = parallel_simulation(str(csv_path), model_type='binary', model_variant='dt', chunk_size=10, num_processes=1)

    assert len(preds) == n
    assert set(preds) <= {'Benign', 'Attack'}
