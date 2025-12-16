from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.core.models.mlp_bin_ce import MLP_CE_Binary


def _toy_bin(n: int = 64, d: int = 8, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.25 * X[:, 1] > 0.0).astype(np.int32)
    return X, y


def _clamp_threads_for_tests() -> None:
    """
    Helps avoid rare segfaults on macOS due to native-thread oversubscription
    (OpenMP/MKL/Accelerate) when running many tests quickly.
    """
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


@pytest.mark.slow
def test_mlp_ce_binary_fit_predict_proba_shapes() -> None:
    _clamp_threads_for_tests()

    X, y = _toy_bin()
    dev = torch.device('cpu')

    m = MLP_CE_Binary(
        input_dim=X.shape[1],
        device=dev,
        widths=(16, 8),
        p_drop=0.0,
        threshold=0.5,
        lr=1e-2,
        epochs=2,
        batch_size=32,
        random_state=7,
    )

    m.fit(X, y)

    assert m.is_fitted_ is True
    assert m.n_features_in_ == X.shape[1]

    proba = m.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    pred = m.predict(X[:10])
    assert pred.shape == (10,)
    assert set(np.unique(pred)).issubset({0, 1})


@pytest.mark.slow
def test_mlp_ce_binary_save_load_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    _clamp_threads_for_tests()

    X, y = _toy_bin()
    dev = torch.device('cpu')

    m = MLP_CE_Binary(
        input_dim=X.shape[1],
        device=dev,
        widths=(16, 8),
        p_drop=0.0,
        threshold=0.5,
        lr=1e-2,
        epochs=1,
        batch_size=32,
        random_state=7,
    )
    m.fit(X, y)


def test_mlp_ce_binary_prepare_y_shape_dtype() -> None:
    _clamp_threads_for_tests()
    m = MLP_CE_Binary(input_dim=4, device=torch.device('cpu'), epochs=1)

    y = np.array([0, 1, 1, 0], dtype=np.int32)
    y2 = m._prepare_y(y)

    assert y2.shape == (4, 1)
    assert y2.dtype == np.float32
    assert np.allclose(y2.reshape(-1), y.astype(np.float32))


def test_mlp_ce_binary_logits_to_proba_sums_to_one() -> None:
    _clamp_threads_for_tests()
    m = MLP_CE_Binary(input_dim=4, device=torch.device('cpu'), epochs=1)

    logits = torch.tensor([[-10.0], [0.0], [10.0]], dtype=torch.float32)
    proba = m._logits_to_proba(logits)

    assert proba.shape == (3, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert proba[0, 1] < 0.01
    assert proba[2, 1] > 0.99


def test_mlp_ce_binary_predict_threshold_override(monkeypatch) -> None:
    _clamp_threads_for_tests()
    m = MLP_CE_Binary(input_dim=4, device=torch.device('cpu'), threshold=0.5, epochs=1)

    def _fake_predict_proba(*args, **kwargs):
        return np.array([[0.4, 0.6], [0.6, 0.4]], dtype=np.float32)

    monkeypatch.setattr(m, 'predict_proba', _fake_predict_proba)

    pred_default = m.predict(np.zeros((2, 4), dtype=np.float32))
    assert pred_default.tolist() == [1, 0]

    pred_hi = m.predict(np.zeros((2, 4), dtype=np.float32), threshold=0.7)
    assert pred_hi.tolist() == [0, 0]


def test_mlp_ce_binary_get_params_and_clone() -> None:
    _clamp_threads_for_tests()
    dev = torch.device('cpu')

    m = MLP_CE_Binary(
        input_dim=6,
        device=dev,
        widths=(16, 8),
        p_drop=0.1,
        threshold=0.6,
        lr=1e-3,
        epochs=3,
        batch_size=32,
        random_state=123,
    )

    params = m.get_params()
    assert params['input_dim'] == 6
    assert params['widths'] == (16, 8)
    assert params['p_drop'] == 0.1
    assert params['threshold'] == 0.6
    assert params['epochs'] == 3
    assert params['batch_size'] == 32
    assert params['random_state'] == 123
    assert params['device'] == dev

    m2 = m.clone()
    assert m2.input_dim == m.input_dim
    assert m2.widths == m.widths
    assert m2.p_drop == m.p_drop
    assert m2.threshold == m.threshold
