import shutil

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from firce.ce_model_training import train_ce_binary
from firce.drift_monitor.base import DriftDetectionResult
from firce.pipelines.simulation_pipeline import run_simulation_pipeline
from firce.runtime.bootstrap import get_rolling_columns
from firce.runtime.inference import process_chunk
from firce.runtime.sim_types import SimulationRuntime
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.config import CEType, ModelType, ModelVariant, MonitorType, SimulationConfig
from firce.utils.perf_stats import PerformanceStats

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
