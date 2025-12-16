import os

from types import SimpleNamespace

import numpy as np
import pandas as pd

os.environ.setdefault('MPLBACKEND', 'Agg')

from src.core import ce_simulation as sim


class DummyRolling:
    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(row)

    def flush(self):
        return None


class DummyScaler:
    def __init__(self, cols):
        self._cols = list(cols)

    def get_feature_names_out(self):
        return np.array(self._cols)

    def transform(self, X):
        if hasattr(X, 'to_numpy'):
            return X.to_numpy()
        return np.asarray(X)


class DummyCE:
    def __init__(self, drift: bool):
        self._drift = drift

    def detect_drift(self, X):
        return np.array([self._drift], dtype=bool)


def test_sim_loop_appends_and_tracks_correct_log(monkeypatch) -> None:
    monkeypatch.setattr(sim, 'clean_data', lambda df, _: df)
    monkeypatch.setattr(sim, 'preprocess_chunk', lambda df, _: df)

    preds = iter([0, 1])
    monkeypatch.setattr(sim, '_predict_row', lambda *args, **kwargs: next(preds))

    config = SimpleNamespace(
        model_type=sim.ModelType.BINARY,
        is_unsw=False,
        use_mlp=False,
        use_pca=False,
    )

    chunk = pd.DataFrame([
        {'f1': 1.0, 'f2': 2.0, 'BinLabel': 0},
        {'f1': 3.0, 'f2': 4.0, 'BinLabel': 0},
    ])

    rolling = DummyRolling()
    perf = sim.PerformanceStats()

    sim._sim_loop(
        config=config,  # type: ignore
        rolling=rolling,  # type: ignore
        scaler=DummyScaler(cols=['f1', 'f2']),  # type: ignore
        pca=None,
        model=object(),  # type: ignore
        ce=None,  # CE disabled path
        chunk=chunk,
        perf_stats=perf,
        sig_controller=None,
        chunkNum=7,
    )

    assert len(rolling.rows) == 2
    assert len(perf.correct_log) == 2
    assert bool(perf.correct_log[0]) is True
    assert bool(perf.correct_log[1]) is False


def test_sim_loop_calls_retrain_on_drift(monkeypatch) -> None:
    monkeypatch.setattr(sim, 'clean_data', lambda df, _: df)
    monkeypatch.setattr(sim, 'preprocess_chunk', lambda df, _: df)
    monkeypatch.setattr(sim, '_predict_row', lambda *args, **kwargs: 0)

    called = {'retrain': 0}

    def fake_retrain(config, scaler, pca, model, ce, rolling, perf_stats, sig_controller):
        called['retrain'] += 1
        return scaler, pca, model, ce

    monkeypatch.setattr(sim, '_retrain', fake_retrain)

    config = SimpleNamespace(
        model_type=sim.ModelType.BINARY,
        is_unsw=False,
        use_mlp=False,
        use_pca=False,
    )
    chunk = pd.DataFrame([{'f1': 1.0, 'f2': 2.0, 'BinLabel': 0}])

    rolling = DummyRolling()
    perf = sim.PerformanceStats()

    sim._sim_loop(
        config=config,  # type: ignore
        rolling=rolling,  # type: ignore
        scaler=DummyScaler(cols=['f1', 'f2']),  # type: ignore
        pca=None,
        model=object(),  # type: ignore
        ce=DummyCE(drift=True),  # type: ignore
        chunk=chunk,
        perf_stats=perf,
        sig_controller=None,
        chunkNum=3,
    )

    assert called['retrain'] == 1
    assert perf.drift_detected_indices == [3]
