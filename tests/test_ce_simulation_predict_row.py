import os

from types import SimpleNamespace

import numpy as np
import pandas as pd

os.environ.setdefault('MPLBACKEND', 'Agg')

from src.core import ce_simulation as sim


class DummyScaler:
    def __init__(self, cols):
        self._cols = list(cols)

    def get_feature_names_out(self):
        return np.array(self._cols)

    def transform(self, X):
        # accept DataFrame or ndarray
        if hasattr(X, 'to_numpy'):
            return X.to_numpy()
        return np.asarray(X)


class DummyModel:
    def __init__(self, p1: float):
        self.p1 = float(p1)

    def predict_proba(self, X):
        n = len(X)
        return np.tile([1.0 - self.p1, self.p1], (n, 1))


def test_predict_row_threshold_works_and_ignores_non_numeric() -> None:
    df = pd.DataFrame([
        {
            'f1': 1.0,
            'f2': 2.0,
            'src_ip': '1.2.3.4',
        }
    ])
    drop_cols = ['src_ip']

    scaler = DummyScaler(cols=['f1', 'f2'])
    config = SimpleNamespace(use_pca=False, model_variant=sim.ModelVariant.KNN, device='cpu')

    pred_hi = sim._predict_row(
        row=df,
        drop_cols=drop_cols,
        scaler=scaler,  # type: ignore
        pca=None,
        config=config,  # type: ignore
        model=DummyModel(p1=0.8),  # type: ignore
        threshold=0.5,
    )
    assert pred_hi == 1

    pred_lo = sim._predict_row(
        row=df,
        drop_cols=drop_cols,
        scaler=scaler,  # type: ignore
        pca=None,
        config=config,  # type: ignore
        model=DummyModel(p1=0.2),  # type: ignore
        threshold=0.5,
    )
    assert pred_lo == 0
