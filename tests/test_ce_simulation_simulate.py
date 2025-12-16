import os

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

from src.core import ce_simulation as sim


class DummyLogger:
    def __init__(self, *args, **kwargs):
        self.rows = []
        self.columns = kwargs.get('columns', [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def append(self, row):
        self.rows.append(row)

    def flush(self):
        return None


@pytest.mark.slow
def test_simulate_calls_sim_loop_and_log_results(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sim, 'CircularDequeLogger', DummyLogger)
    monkeypatch.setattr(sim, 'RollingCSV', DummyLogger)

    monkeypatch.setattr(sim, '_ensure_models_exist', lambda *a, **k: None)
    monkeypatch.setattr(sim, 'load_simulation_objects', lambda *a, **k: (object(), None, object()))

    calls = {'sim_loop': 0, 'log_results': 0}

    def fake_sim_loop(*args, **kwargs):
        calls['sim_loop'] += 1

    def fake_log_results(*args, **kwargs):
        calls['log_results'] += 1

    monkeypatch.setattr(sim, '_sim_loop', fake_sim_loop)
    monkeypatch.setattr(sim, '_log_results', fake_log_results)

    agg = tmp_path / 'agg.csv'
    flows = tmp_path / 'flows.csv'

    df_train = pd.DataFrame([
        {'f1': 1.0, 'f2': 2.0, 'Label': 'Benign'},
        {'f1': 3.0, 'f2': 4.0, 'Label': 'Attack'},
    ])

    chunks = [
        pd.DataFrame([{'f1': 1.0, 'f2': 2.0, 'BinLabel': 0}]),
        pd.DataFrame([{'f1': 3.0, 'f2': 4.0, 'BinLabel': 1}]),
    ]

    def fake_read_csv(path, *args, **kwargs):
        if Path(path) == agg:
            return df_train
        if Path(path) == flows and 'chunksize' in kwargs:
            return iter(chunks)
        raise AssertionError(f'Unexpected read_csv call: {path}, kwargs={kwargs}')

    monkeypatch.setattr(sim.pd, 'read_csv', fake_read_csv)

    config = SimpleNamespace(
        aggregated_path=agg,
        flows_path=flows,
        log_path=tmp_path / 'ce_log.csv.gz',
        chunk_size=1,
        max_rows=10,
        use_pca=False,
        log_to_file=False,
        model_variant=sim.ModelVariant.KNN,
        model_type=sim.ModelType.BINARY,
        ce_type=sim.CEType.NONE,
        debug=False,
        use_circular_logger=True,
        use_ASC=False,
        use_svm=False,
        use_adaptive_chunking=False,
        use_mlp=False,
        use_cuml=False,
        is_unsw=False,
        device='cpu',
        adaptive_chunk_config=None,
    )

    sim._simulate(config)  # type: ignore

    assert calls['sim_loop'] == 2
    assert calls['log_results'] == 1
