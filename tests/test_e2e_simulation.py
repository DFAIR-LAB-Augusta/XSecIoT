import shutil

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from firce.ce_model_training import train_ce_binary, train_ce_multiclass
from firce.conformalEval.utils import clone_model
from firce.drift_monitor.base import DriftDetectionResult
from firce.drift_monitor.conformal_monitor import ConformalDriftMonitor
from firce.pipelines.simulation_pipeline import run_simulation_pipeline
from firce.runtime.bootstrap import get_rolling_columns
from firce.runtime.constants import FULL_DROP_COLS
from firce.runtime.inference import process_chunk
from firce.runtime.sim_types import SimulationRuntime
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.config import CEType, ModelType, ModelVariant, MonitorType, SimulationConfig
from firce.utils.perf_stats import PerformanceStats
from fire.simulations import preprocess_chunk

DEVICE = torch.device('cpu')
FIXTURES = Path(__file__).parent / 'fixtures'


def test_run_simulation_pipeline_end_to_end_binary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'CETrain_e2e'
    ds_dir.mkdir()
    train_csv = ds_dir / 'train.csv'
    stream_csv = ds_dir / 'stream.csv'
    shutil.copy(FIXTURES / 'ce_flows_e2e_train.csv', train_csv)
    shutil.copy(FIXTURES / 'ce_flows_e2e_stream.csv', stream_csv)

    config = SimulationConfig(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.ICE,
        aggregated_path=train_csv,
        flows_path=stream_csv,
        is_unsw=False,
        seed=0,
        device=DEVICE,
        monitor_type=MonitorType.CE,
        chunk_size=50,
        use_circular_logger=True,
        use_pca=False,
    )

    run_simulation_pipeline(config)

    model_dir = tmp_path / 'binary_models' / 'CETrain_e2e'
    assert (model_dir / 'dt_model_binary.pkl').exists()
    assert (model_dir / 'scaler_binary.pkl').exists()

    plot_dir = tmp_path / 'logging' / 'chunk_size_50' / 'DFAIR'
    assert (plot_dir / 'dt_ice_binary_0_0_accuracy_plot.png').exists()


def test_run_simulation_pipeline_end_to_end_multiclass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'CETrain_mc_e2e'
    ds_dir.mkdir()
    train_csv = ds_dir / 'train.csv'
    stream_csv = ds_dir / 'stream.csv'
    shutil.copy(FIXTURES / 'ce_flows_e2e_mc_train.csv', train_csv)
    shutil.copy(FIXTURES / 'ce_flows_e2e_mc_stream.csv', stream_csv)

    config = SimulationConfig(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        ce_type=CEType.ICE,
        aggregated_path=train_csv,
        flows_path=stream_csv,
        is_unsw=False,
        seed=0,
        device=DEVICE,
        monitor_type=MonitorType.CE,
        chunk_size=50,
        use_circular_logger=True,
        use_pca=False,
    )

    run_simulation_pipeline(config)

    model_dir = tmp_path / 'multi_class_models' / 'CETrain_mc_e2e'
    assert (model_dir / 'dt_model_multi.pkl').exists()
    assert (model_dir / 'scaler_multi.pkl').exists()
    assert (model_dir / 'label_encoder_multi.pkl').exists()

    plot_dir = tmp_path / 'logging' / 'chunk_size_50' / 'DFAIR'
    assert (plot_dir / 'dt_ice_multi_0_0_accuracy_plot.png').exists()


class _AlwaysDriftMonitor:
    """Duck-typed drift monitor stub that always reports drift - decouples
    this test from real CE calibration internals (already covered by #93),
    so it only exercises process_chunk's own drift -> retrain wiring."""

    def __init__(self):
        self.fit_calls: list = []

    def fit(self, X, y, perf_stats):
        self.fit_calls.append((X, y))

    def detect(self, X):
        return DriftDetectionResult(row_flags=np.ones(len(X), dtype=bool), chunk_drift=True)


def test_process_chunk_triggers_retrain_on_detected_drift(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    train_csv = ds_dir / 'train.csv'
    shutil.copy(FIXTURES / 'ce_flows_e2e_train.csv', train_csv)

    config = SimulationConfig(
        model_type=ModelType.BINARY,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=train_csv,
        flows_path=train_csv,
        is_unsw=False,
        seed=0,
        device=DEVICE,
    )

    model_dir = train_ce_binary(config, str(train_csv), PerformanceStats())
    scaler = joblib.load(model_dir / 'scaler_binary.pkl')
    model = joblib.load(model_dir / 'dt_model_binary.pkl')

    retrain_calls = []
    monkeypatch.setattr(
        'firce.runtime.inference.retrain_runtime',
        lambda runtime: retrain_calls.append(runtime),
    )

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=500, columns=get_rolling_columns(config)),
        scaler=scaler,
        pca=None,
        model=model,
        label_encoder=None,
        monitor=_AlwaysDriftMonitor(),
        train_df=pd.DataFrame(),
    )

    chunk = pd.read_csv(FIXTURES / 'ce_flows_e2e_stream.csv').head(20)
    process_chunk(runtime, chunk, chunk_num=0)

    assert len(retrain_calls) == 1
    assert retrain_calls[0] is runtime
    assert runtime.perf_stats.drift_detected_indices == [0]


def test_process_chunk_triggers_retrain_on_detected_drift_multiclass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    train_csv = ds_dir / 'train.csv'
    shutil.copy(FIXTURES / 'ce_flows_e2e_mc_train.csv', train_csv)

    config = SimulationConfig(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=train_csv,
        flows_path=train_csv,
        is_unsw=False,
        seed=0,
        device=DEVICE,
    )

    model_dir = train_ce_multiclass(config, str(train_csv), variant=ModelVariant.DT, use_pca=False)
    scaler = joblib.load(model_dir / 'scaler_multi.pkl')
    model = joblib.load(model_dir / 'dt_model_multi.pkl')
    label_encoder = joblib.load(model_dir / 'label_encoder_multi.pkl')

    retrain_calls = []
    monkeypatch.setattr(
        'firce.runtime.inference.retrain_runtime',
        lambda runtime: retrain_calls.append(runtime),
    )

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=500, columns=get_rolling_columns(config)),
        scaler=scaler,
        pca=None,
        model=model,
        label_encoder=label_encoder,
        monitor=_AlwaysDriftMonitor(),
        train_df=pd.DataFrame(),
    )

    chunk = pd.read_csv(FIXTURES / 'ce_flows_e2e_mc_stream.csv').head(20)
    process_chunk(runtime, chunk, chunk_num=0)

    assert len(retrain_calls) == 1
    assert retrain_calls[0] is runtime
    assert runtime.perf_stats.drift_detected_indices == [0]


def test_process_chunk_generates_novelty_report_for_genuinely_held_out_class(tmp_path, monkeypatch):
    # RandomForestClassifier (ModelVariant.RF), not DT: confirmed via direct
    # execution in #101 that a plain decision tree is too discrete/overconfident
    # for a genuine open-world demonstration - RandomForest gives real signal.
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()

    full_train_df = pd.read_csv(FIXTURES / 'ce_flows_e2e_mc_train.csv')
    # train_ce_multiclass drops any 'Unnamed: *' column internally (a stray
    # index artifact in this fixture CSV); build_runtime_monitor's feature
    # prep (mirrored below for the manually-built monitor) does not, so
    # strip it up front to keep both paths' feature sets consistent.
    full_train_df = full_train_df.loc[:, ~full_train_df.columns.str.startswith('Unnamed')]
    held_out_class = sorted(full_train_df['MC_Label'].unique())[-1]
    known_only_df = full_train_df[full_train_df['MC_Label'] != held_out_class]
    train_csv = ds_dir / 'train.csv'
    known_only_df.to_csv(train_csv, index=False)

    config = SimulationConfig(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.RF,
        ce_type=CEType.ICE,
        aggregated_path=train_csv,
        flows_path=train_csv,
        is_unsw=False,
        seed=0,
        device=DEVICE,
        novelty_enabled=True,
        # tau=0.97: confirmed via direct execution that RF on this real captured
        # fixture data confidently (but wrongly) classifies the held-out class
        # with a flat 0.96 max-softmax, while the conformal p-value signal
        # correctly identifies low support (~0.05-0.1, well under alpha=0.5) -
        # a permissive tau is needed for the primary AND-combined rule to
        # actually flag it, matching #97's own documented finding that the
        # confidence signal alone is often unreliable and needs pairing with
        # a permissive-enough threshold to let the conformal signal through.
        novelty_tau=0.97,
        novelty_alpha=0.5,
        novelty_selective_mode='unknown_only',
    )

    model_dir = train_ce_multiclass(config, str(train_csv), variant=ModelVariant.RF, use_pca=False)
    scaler = joblib.load(model_dir / 'scaler_multi.pkl')
    model = joblib.load(model_dir / 'rf_model_multi.pkl')
    label_encoder = joblib.load(model_dir / 'label_encoder_multi.pkl')

    # clone_model (not the same `model` object `runtime.model` will use): ICE.calibrate()
    # refits its model in place - passing the original object directly would
    # silently overwrite runtime.model's int-encoded training with this
    # monitor's own separate calibration fit (the exact #125 bug pattern).
    monitor = ConformalDriftMonitor(CEType.ICE, clone_model(model), calibration_split=0.3, random_state=0)
    # Mirror firce.runtime.bootstrap.build_runtime_monitor's real feature prep
    # exactly, so the monitor's calibration matches what runtime.scaler expects.
    x_train = preprocess_chunk(known_only_df.copy(), FULL_DROP_COLS).select_dtypes(include=['number'])
    monitor.fit(scaler.transform(x_train), known_only_df['MC_Label'].to_numpy(), PerformanceStats())

    runtime = SimulationRuntime(
        config=config,
        perf_stats=PerformanceStats(),
        sig_controller=None,
        rolling=CircularDequeLogger(None, max_rows=500, columns=get_rolling_columns(config)),
        scaler=scaler,
        pca=None,
        model=model,
        label_encoder=label_encoder,
        monitor=monitor,
        train_df=known_only_df,
    )

    # Stream the FULL (unfiltered) fixture, which includes the held-out class.
    full_stream_df = pd.read_csv(FIXTURES / 'ce_flows_e2e_mc_stream.csv')
    held_out_stream_rows = full_stream_df[full_stream_df['MC_Label'] == held_out_class]
    assert len(held_out_stream_rows) > 0, 'fixture must actually contain the held-out class in the stream file'

    process_chunk(runtime, held_out_stream_rows, chunk_num=0)

    assert len(runtime.novelty_reports) > 0
    record = runtime.novelty_reports[0]
    assert 'contributions' in record['explanation']
    assert record['llm_report'] is None  # no LLM backend configured for this test
