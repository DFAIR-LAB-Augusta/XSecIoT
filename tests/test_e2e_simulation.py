import shutil

from pathlib import Path

import torch

from firce.pipelines.simulation_pipeline import run_simulation_pipeline
from firce.utils.config import CEType, ModelType, ModelVariant, MonitorType, SimulationConfig

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
