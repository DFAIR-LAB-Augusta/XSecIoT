import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from firce.ce_model_training import train_ce_binary, train_ce_multiclass
from firce.drift_monitor.conformal_monitor import ConformalDriftMonitor
from firce.runtime.constants import DROP_COLS, PRED_THRESHOLD
from firce.runtime.inference import (
    _generate_novelty_reports,
    _prepare_chunk,
    _record_prediction_outcome,
    _score_chunk_novelty,
    predict_row,
)
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


def test_append_unsw_row_multiclass_writes_mc_label(tmp_path, monkeypatch):
    from firce.runtime.constants import get_unsw_rolling_columns
    from firce.runtime.inference import _append_unsw_row

    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path, model_type=ModelType.MULTI, is_unsw=True)
    columns = get_unsw_rolling_columns(ModelType.MULTI)
    rolling = CircularDequeLogger(None, max_rows=200, columns=columns)
    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=rolling,
        scaler=None,
        pca=None,
        model=None,
        label_encoder=None,
        monitor=None,
        train_df=pd.DataFrame(),
    )

    row_values = {col: 1.0 for col in columns if col != 'MC_Label'}
    row_values['MC_Label'] = 'DoS'
    row_to_log = pd.DataFrame([row_values])

    _append_unsw_row(runtime, row_to_log)

    appended = rolling.to_dataframe()
    assert len(appended) == 1
    assert appended['MC_Label'].iloc[0] == 'DoS'


def test_append_unsw_row_binary_still_coerces_label(tmp_path, monkeypatch):
    from firce.runtime.constants import get_unsw_rolling_columns
    from firce.runtime.inference import _append_unsw_row

    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path, model_type=ModelType.BINARY, is_unsw=True)
    columns = get_unsw_rolling_columns(ModelType.BINARY)
    rolling = CircularDequeLogger(None, max_rows=200, columns=columns)
    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=rolling,
        scaler=None,
        pca=None,
        model=None,
        label_encoder=None,
        monitor=None,
        train_df=pd.DataFrame(),
    )

    row_values = {col: 1.0 for col in columns if col != 'BinLabel'}
    row_values['BinLabel'] = 'Attack'
    row_to_log = pd.DataFrame([row_values])

    _append_unsw_row(runtime, row_to_log)

    appended = rolling.to_dataframe()
    assert len(appended) == 1
    assert appended['BinLabel'].iloc[0] == 1


def test_prepare_chunk_unsw_applies_column_renaming(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path, model_type=ModelType.MULTI, is_unsw=True)
    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=200, columns=['MC_Label']),
        scaler=None,
        pca=None,
        model=None,
        label_encoder=None,
        monitor=None,
        train_df=pd.DataFrame(),
    )

    n = 5
    raw_chunk = pd.DataFrame({
        'IPV4_SRC_ADDR': ['10.0.0.1'] * n,
        'IPV4_DST_ADDR': ['10.0.0.2'] * n,
        'L4_SRC_PORT': [1024] * n,
        'L4_DST_PORT': [80] * n,
        'PROTOCOL': [6] * n,
        'FLOW_START_MILLISECONDS': [1_600_000_000_000 + i * 1000 for i in range(n)],
        'FLOW_END_MILLISECONDS': [1_600_000_000_500 + i * 1000 for i in range(n)],
        'FLOW_DURATION_MILLISECONDS': [10.0] * n,
        'IN_PKTS': [5] * n,
        'OUT_PKTS': [5] * n,
        'IN_BYTES': [500.0] * n,
        'OUT_BYTES': [500.0] * n,
        'SRC_TO_DST_IAT_MIN': [1.0] * n,
        'SRC_TO_DST_IAT_MAX': [1.0] * n,
        'SRC_TO_DST_IAT_AVG': [1.0] * n,
        'SRC_TO_DST_IAT_STDDEV': [1.0] * n,
        'DST_TO_SRC_IAT_MIN': [1.0] * n,
        'DST_TO_SRC_IAT_MAX': [1.0] * n,
        'DST_TO_SRC_IAT_AVG': [1.0] * n,
        'DST_TO_SRC_IAT_STDDEV': [1.0] * n,
    })

    clean_chunk, _ = _prepare_chunk(runtime, raw_chunk)

    assert 'tot_fwd_pkts' in clean_chunk.columns
    assert 'tot_bwd_pkts' in clean_chunk.columns
    assert 'fwd_pkt_len_mean' in clean_chunk.columns
    assert 'IN_PKTS' not in clean_chunk.columns
    assert 'IPV4_SRC_ADDR' not in clean_chunk.columns


def _make_novelty_test_runtime(tmp_path, novelty_enabled=True, **config_overrides):
    dummy = tmp_path / 'dummy.csv'
    dummy.write_text('a\n1\n')
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({
        'flow_duration': rng.normal(size=n),
        'tot_fwd_pkt': rng.normal(size=n),
    })
    y = np.select([X['flow_duration'] > 0.5], ['Attack'], default='Benign')

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    monitor = ConformalDriftMonitor(
        CEType.ICE, DecisionTreeClassifier(random_state=0), calibration_split=0.3, random_state=0
    )
    monitor.fit(X_scaled, y, PerformanceStats())

    config = SimulationConfig(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.ICE,
        aggregated_path=dummy,
        flows_path=dummy,
        device=DEVICE,
        novelty_enabled=novelty_enabled,
        **config_overrides,
    )

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=500, columns=['flow_duration', 'tot_fwd_pkt', 'BinLabel']),
        scaler=scaler,
        pca=None,
        model=monitor.model,
        label_encoder=None,
        monitor=monitor,
        train_df=pd.DataFrame(),
    )
    clean_chunk = X.copy()
    return runtime, clean_chunk


def test_score_chunk_novelty_returns_none_when_disabled(tmp_path):
    runtime, clean_chunk = _make_novelty_test_runtime(tmp_path, novelty_enabled=False)

    result = _score_chunk_novelty(runtime, clean_chunk)

    assert result is None


def test_score_chunk_novelty_returns_flags_and_features_when_enabled(tmp_path):
    runtime, clean_chunk = _make_novelty_test_runtime(tmp_path, novelty_enabled=True)

    result = _score_chunk_novelty(runtime, clean_chunk)

    assert result is not None
    novelty_flags, x_monitor = result
    assert novelty_flags.shape == (len(clean_chunk),)
    assert novelty_flags.dtype == bool
    assert x_monitor.shape[0] == len(clean_chunk)


def test_generate_novelty_reports_populates_runtime_with_explanations_only_by_default(tmp_path):
    # novelty_llm_backend_type stays None (default) - explanation-only path.
    runtime, clean_chunk = _make_novelty_test_runtime(
        tmp_path, novelty_enabled=True, novelty_tau=0.99, novelty_alpha=0.99
    )
    novelty_flags, x_monitor = _score_chunk_novelty(runtime, clean_chunk)

    _generate_novelty_reports(runtime, clean_chunk, x_monitor, novelty_flags)

    if novelty_flags.any():
        assert len(runtime.novelty_reports) > 0
        record = runtime.novelty_reports[0]
        assert 'explanation' in record
        assert record['llm_report'] is None  # no LLM backend configured
