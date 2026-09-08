import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from firce.ce_model_training import train_ce_binary, train_ce_multiclass
from firce.runtime.constants import DROP_COLS, PRED_THRESHOLD
from firce.runtime.inference import _prepare_chunk, _record_prediction_outcome, predict_row
from firce.runtime.sim_types import SimulationRuntime
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig
from firce.utils.perf_stats import PerformanceStats

DEVICE = torch.device('cpu')

FLOW_COLUMNS = [
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


BINARY_FLOW_COLUMNS = [
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


def _train_and_build_runtime(tmp_path, monkeypatch, model_variant=ModelVariant.DT):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    _make_multiclass_rows().to_csv(csv_path, index=False)

    config = _make_config(tmp_path, model_variant=model_variant)

    model_dir = train_ce_multiclass(config, str(csv_path), variant=model_variant, use_pca=False)
    scaler = joblib.load(model_dir / 'scaler_multi.pkl')
    label_encoder = joblib.load(model_dir / 'label_encoder_multi.pkl')

    if model_variant == ModelVariant.FEEDFORWARD:
        from firce.models.feedforward_multiclass import FeedForwardMulticlass

        ckpt = torch.load(model_dir / 'feedforward_model_multi.pt', map_location='cpu')
        model = FeedForwardMulticlass(input_dim=int(ckpt['input_dim']), num_classes=int(ckpt['num_classes']))
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model.eval()
    else:
        model = joblib.load(model_dir / f'{model_variant.value}_model_multi.pkl')

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=200, columns=FLOW_COLUMNS),
        scaler=scaler,
        pca=None,
        model=model,
        label_encoder=label_encoder,
        monitor=None,
        train_df=pd.DataFrame(),
    )
    return runtime, csv_path


def _train_and_build_binary_runtime(tmp_path, monkeypatch, model_variant=ModelVariant.DT):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    _make_binary_rows().to_csv(csv_path, index=False)

    config = _make_config(tmp_path, model_type=ModelType.BINARY, model_variant=model_variant)

    model_dir = train_ce_binary(config, str(csv_path), PerformanceStats())
    scaler = joblib.load(model_dir / 'scaler_binary.pkl')

    if model_variant == ModelVariant.FEEDFORWARD:
        from firce.models.feedforward_binary import FeedForwardBinary

        ckpt = torch.load(model_dir / 'feedforward_model_binary.pt', map_location='cpu')
        model = FeedForwardBinary(input_dim=int(ckpt['input_dim']))
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model.eval()
    else:
        model = joblib.load(model_dir / f'{model_variant.value}_model_binary.pkl')

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=200, columns=BINARY_FLOW_COLUMNS),
        scaler=scaler,
        pca=None,
        model=model,
        label_encoder=None,
        monitor=None,
        train_df=pd.DataFrame(),
    )
    return runtime, csv_path


def test_prepare_chunk_extracts_mc_label_ground_truth(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)

    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)

    assert ground_truth is not None
    assert set(ground_truth.tolist()) <= {'Benign', 'PortScan', 'XMasAttack'}
    assert 'MC_Label' not in clean_chunk.columns


@pytest.mark.parametrize('model_variant', [ModelVariant.DT, ModelVariant.FEEDFORWARD])
def test_predict_row_multiclass_returns_valid_class_index(tmp_path, monkeypatch, model_variant):
    runtime, csv_path = _train_and_build_runtime(tmp_path, monkeypatch, model_variant=model_variant)
    chunk = pd.read_csv(csv_path)
    clean_chunk, _ = _prepare_chunk(runtime, chunk)
    row = clean_chunk.iloc[[0]]

    prediction = predict_row(row, DROP_COLS, runtime.scaler, runtime.pca, runtime.config, runtime.model, PRED_THRESHOLD)

    assert prediction in range(len(runtime.label_encoder.classes_))


def test_record_prediction_outcome_multiclass_tracks_correctness(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)
    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)
    raw_row = clean_chunk.iloc[0]
    row_to_log = clean_chunk.iloc[[0]].copy()

    true_label = ground_truth.iloc[0]
    true_idx = int(runtime.label_encoder.transform([true_label])[0])

    _record_prediction_outcome(
        runtime=runtime,
        row_index=0,
        raw_row=raw_row,
        row_to_log=row_to_log,
        prediction=true_idx,
        ground_truth=ground_truth,
    )

    assert row_to_log['MC_Label'].iloc[0] == true_label
    assert runtime.perf_stats.correct_log == [True]


def test_record_prediction_outcome_multiclass_tracks_incorrect(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)
    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)
    raw_row = clean_chunk.iloc[0]
    row_to_log = clean_chunk.iloc[[0]].copy()

    true_label = ground_truth.iloc[0]
    true_idx = int(runtime.label_encoder.transform([true_label])[0])
    wrong_idx = (true_idx + 1) % len(runtime.label_encoder.classes_)

    _record_prediction_outcome(
        runtime=runtime,
        row_index=0,
        raw_row=raw_row,
        row_to_log=row_to_log,
        prediction=wrong_idx,
        ground_truth=ground_truth,
    )

    assert row_to_log['MC_Label'].iloc[0] == runtime.label_encoder.classes_[wrong_idx]
    assert runtime.perf_stats.correct_log == [False]


def test_prepare_chunk_extracts_binlabel_ground_truth(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_binary_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)

    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)

    assert ground_truth is not None
    assert set(ground_truth.tolist()) <= {0, 1}
    assert 'BinLabel' not in clean_chunk.columns


@pytest.mark.parametrize('model_variant', [ModelVariant.DT, ModelVariant.FEEDFORWARD])
def test_predict_row_binary_returns_valid_class(tmp_path, monkeypatch, model_variant):
    runtime, csv_path = _train_and_build_binary_runtime(tmp_path, monkeypatch, model_variant=model_variant)
    chunk = pd.read_csv(csv_path)
    clean_chunk, _ = _prepare_chunk(runtime, chunk)
    row = clean_chunk.iloc[[0]]

    prediction = predict_row(row, DROP_COLS, runtime.scaler, runtime.pca, runtime.config, runtime.model, PRED_THRESHOLD)

    assert prediction in (0, 1)


def test_record_prediction_outcome_binary_tracks_correctness(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_binary_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)
    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)
    raw_row = clean_chunk.iloc[0]
    row_to_log = clean_chunk.iloc[[0]].copy()

    true_value = int(ground_truth.iloc[0])

    _record_prediction_outcome(
        runtime=runtime,
        row_index=0,
        raw_row=raw_row,
        row_to_log=row_to_log,
        prediction=true_value,
        ground_truth=ground_truth,
    )

    assert row_to_log['BinLabel'].iloc[0] == true_value
    assert runtime.perf_stats.correct_log == [True]


def test_record_prediction_outcome_binary_tracks_incorrect(tmp_path, monkeypatch):
    runtime, csv_path = _train_and_build_binary_runtime(tmp_path, monkeypatch)
    chunk = pd.read_csv(csv_path)
    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)
    raw_row = clean_chunk.iloc[0]
    row_to_log = clean_chunk.iloc[[0]].copy()

    true_value = int(ground_truth.iloc[0])
    wrong_value = 1 - true_value

    _record_prediction_outcome(
        runtime=runtime,
        row_index=0,
        raw_row=raw_row,
        row_to_log=row_to_log,
        prediction=wrong_value,
        ground_truth=ground_truth,
    )

    assert row_to_log['BinLabel'].iloc[0] == wrong_value
    assert runtime.perf_stats.correct_log == [False]
