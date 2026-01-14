import os

from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

os.environ.setdefault('MPLBACKEND', 'Agg')

from src.core import ce_simulation as sim


class DummyCE:
    def __init__(self):
        self.calibrated = 0

    def calibrate(self, X, y, perf_stats):
        self.calibrated += 1


class DummyCircular:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_dataframe(self):
        return self._df


def test_retrain_raises_if_ce_none() -> None:
    config = SimpleNamespace(model_type=sim.ModelType.BINARY)
    with pytest.raises(RuntimeError):
        sim._retrain(
            config=config,  # type: ignore
            scaler=StandardScaler(),
            pca=None,
            model=object(),
            ce=None,  # type: ignore
            rolling=object(),  # type: ignore
            perf_stats=sim.PerformanceStats(),
            sig_controller=None,
        )


def test_retrain_skips_calibrate_if_single_class(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sim, 'CircularDequeLogger', DummyCircular)

    monkeypatch.setattr(sim, 'clean_data', lambda df, _: df)
    monkeypatch.setattr(sim, 'preprocess_chunk', lambda df, drop: df.drop(columns=drop, errors='ignore'))
    monkeypatch.setattr(sim, 'FULL_DROP_COLS', ['BinLabel'])

    df_log = pd.DataFrame([
        {'f1': 1.0, 'f2': 2.0, 'BinLabel': 0},
        {'f1': 3.0, 'f2': 4.0, 'BinLabel': 0},
    ])
    rolling = DummyCircular(df_log)

    model_dir = tmp_path / 'model_artifacts'
    model_dir.mkdir(parents=True, exist_ok=True)

    def fake_train_ce_binary(*args, **kwargs):
        X = df_log[['f1', 'f2']]
        y = df_log['BinLabel']

        scaler = StandardScaler().fit(X)
        joblib.dump(scaler, model_dir / 'scaler_binary.pkl')

        clf = DecisionTreeClassifier(random_state=0).fit(X, y)
        joblib.dump(clf, model_dir / 'dt_model_binary.pkl')

        return model_dir

    monkeypatch.setattr(sim, 'train_ce_binary', fake_train_ce_binary)

    config = SimpleNamespace(
        model_type=sim.ModelType.BINARY,
        model_variant=sim.ModelVariant.DT,
        use_pca=False,
        use_mlp=False,
        device='cpu',
        log_path=tmp_path / 'ce_log.csv.gz',
        max_rows=10000,
        is_unsw=False,
    )

    ce = DummyCE()
    perf = sim.PerformanceStats()

    scaler_out, pca_out, model_out, ce_out = sim._retrain(
        config=config,  # type: ignore
        scaler=StandardScaler(),
        pca=None,
        model=object(),
        ce=ce,  # type: ignore
        rolling=rolling,  # type: ignore
        perf_stats=perf,
        sig_controller=None,
    )

    assert scaler_out is not None
    assert pca_out is None
    assert model_out is not None
    assert ce_out.calibrated == 0  # type: ignore
