from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.core.models.mlp_mc_ce import MLP_CE_Multiclass


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


def _toy_mc(
    n: int = 90,
    d: int = 10,
    classes: tuple[str, ...] = ('Benign', 'TCP_SYN_Flood', 'UDP_Flood'),
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y_idx = rng.integers(0, len(classes), size=n)
    y = np.array([classes[i] for i in y_idx], dtype=object)
    return X, y


def test_init_rejects_num_classes_lt_2() -> None:
    _clamp_threads_for_tests()
    with pytest.raises(ValueError):
        MLP_CE_Multiclass(
            input_dim=4,
            num_classes=1,
            device=torch.device('cpu'),
        )


def test_ensure_mapping_infers_classes_and_builds_mapping() -> None:
    _clamp_threads_for_tests()
    X, y = _toy_mc(classes=('A', 'B', 'C'))
    m = MLP_CE_Multiclass(
        input_dim=X.shape[1],
        num_classes=3,
        device=torch.device('cpu'),
        classes=None,
        epochs=1,
    )

    assert m.classes_ is None
    m._ensure_mapping(y)

    assert m.classes_ is not None
    assert set(m.classes_.tolist()) == {'A', 'B', 'C'}
    assert m._class_to_index is not None
    assert set(m._class_to_index.keys()) == {'A', 'B', 'C'}
    assert set(m._class_to_index.values()) == {0, 1, 2}


def test_ensure_mapping_mismatch_raises() -> None:
    _clamp_threads_for_tests()
    X, y = _toy_mc(classes=('A', 'B', 'C'))
    m = MLP_CE_Multiclass(
        input_dim=X.shape[1],
        num_classes=4,
        device=torch.device('cpu'),
        classes=None,
        epochs=1,
    )

    with pytest.raises(ValueError, match='class count mismatch'):
        m._ensure_mapping(y)


def test_prepare_y_string_labels_to_int64_indices() -> None:
    _clamp_threads_for_tests()
    X, y = _toy_mc(classes=('Benign', 'TCP_SYN_Flood', 'UDP_Flood'))
    classes = np.array(['Benign', 'TCP_SYN_Flood', 'UDP_Flood'], dtype=object)

    m = MLP_CE_Multiclass(
        input_dim=X.shape[1],
        num_classes=3,
        device=torch.device('cpu'),
        classes=classes,
        epochs=1,
    )

    y_idx = m._prepare_y(y)
    assert y_idx.dtype == np.int64
    assert y_idx.shape == (y.shape[0],)
    assert int(y_idx.min()) >= 0
    assert int(y_idx.max()) <= 2


def test_prepare_y_unknown_label_raises() -> None:
    _clamp_threads_for_tests()
    X, y = _toy_mc(classes=('A', 'B', 'C'))
    classes = np.array(['A', 'B', 'C'], dtype=object)

    m = MLP_CE_Multiclass(
        input_dim=X.shape[1],
        num_classes=3,
        device=torch.device('cpu'),
        classes=classes,
        epochs=1,
    )

    y_bad = y.copy()
    y_bad[0] = 'NOT_A_CLASS'

    with pytest.raises(ValueError, match='Unknown class label'):
        m._prepare_y(y_bad)


def test_criterion_is_cross_entropy_loss() -> None:
    _clamp_threads_for_tests()
    m = MLP_CE_Multiclass(input_dim=4, num_classes=3, device=torch.device('cpu'), epochs=1)
    crit = m._criterion()
    assert isinstance(crit, torch.nn.CrossEntropyLoss)


def test_logits_to_proba_softmax_rows_sum_to_one() -> None:
    _clamp_threads_for_tests()
    m = MLP_CE_Multiclass(input_dim=4, num_classes=3, device=torch.device('cpu'), epochs=1)

    logits = torch.tensor(
        [[10.0, 0.0, -10.0], [0.0, 0.0, 0.0], [-5.0, 1.0, 3.0]],
        dtype=torch.float32,
    )
    probs = m._logits_to_proba(logits)
    assert probs.shape == (3, 3)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert probs[0, 0] > 0.99


def test_predict_return_indices_true(monkeypatch) -> None:
    _clamp_threads_for_tests()
    m = MLP_CE_Multiclass(input_dim=4, num_classes=3, device=torch.device('cpu'), epochs=1)

    def _fake_predict_proba(*args, **kwargs) -> np.ndarray:
        return np.array(
            [
                [0.1, 0.7, 0.2],  # -> 1
                [0.9, 0.05, 0.05],  # -> 0
                [0.2, 0.1, 0.7],  # -> 2
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(m, 'predict_proba', _fake_predict_proba)

    out = m.predict(np.zeros((3, 4), dtype=np.float32), return_indices=True)
    assert out.dtype.kind in ('i', 'u')
    assert out.tolist() == [1, 0, 2]


def test_predict_returns_labels_when_classes_present(monkeypatch) -> None:
    _clamp_threads_for_tests()
    m = MLP_CE_Multiclass(
        input_dim=4,
        num_classes=3,
        device=torch.device('cpu'),
        classes=np.array(['Benign', 'TCP', 'UDP'], dtype=object),
        epochs=1,
    )

    def _fake_predict_proba(*args, **kwargs) -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0], [0.2, 0.1, 0.7]], dtype=np.float32)

    monkeypatch.setattr(m, 'predict_proba', _fake_predict_proba)

    out = m.predict(np.zeros((2, 4), dtype=np.float32), return_indices=False)
    assert out.shape == (2,)
    assert out.tolist() == ['TCP', 'UDP']


def test_predict_returns_indices_if_classes_none(monkeypatch) -> None:
    _clamp_threads_for_tests()
    m = MLP_CE_Multiclass(input_dim=4, num_classes=3, device=torch.device('cpu'), classes=None, epochs=1)

    def _fake_predict_proba(*args, **kwargs) -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(m, 'predict_proba', _fake_predict_proba)

    out = m.predict(np.zeros((1, 4), dtype=np.float32), return_indices=False)
    assert out.shape == (1,)
    assert out.dtype.kind in ('i', 'u')
    assert out.tolist() == [1]


def test_get_params_and_clone() -> None:
    _clamp_threads_for_tests()
    dev = torch.device('cpu')
    classes = np.array(['A', 'B', 'C'], dtype=object)

    m = MLP_CE_Multiclass(
        input_dim=6,
        num_classes=3,
        device=dev,
        classes=classes,
        widths=(16, 8),
        p_drop=0.1,
        lr=1e-3,
        epochs=3,
        batch_size=32,
        random_state=123,
    )

    params = m.get_params()
    assert params['input_dim'] == 6
    assert params['num_classes'] == 3
    assert params['widths'] == (16, 8)
    assert params['p_drop'] == 0.1
    assert params['epochs'] == 3
    assert params['batch_size'] == 32
    assert params['random_state'] == 123
    assert params['device'] == dev

    # classes should be a copy
    assert params['classes'] is not None
    assert params['classes'].tolist() == ['A', 'B', 'C']
    assert params['classes'] is not m.classes_

    m2 = m.clone()
    assert m2.input_dim == m.input_dim
    assert m2.num_classes == m.num_classes
    assert m2.widths == m.widths
    assert m2.p_drop == m.p_drop
    assert m2.lr == m.lr
    assert m2.epochs == m.epochs
    assert m2.batch_size == m.batch_size
    assert m2.random_state == m.random_state
    assert m2.classes_ is not None
    assert m2.classes_.tolist() == m.classes_.tolist()  # type: ignore


@pytest.mark.slow
def test_mlp_ce_multiclass_fit_predict_proba_shapes() -> None:
    """
    Slow-ish because it runs a short training loop.
    Keep epochs tiny so it's still reasonable locally.
    """
    _clamp_threads_for_tests()

    X, y = _toy_mc(n=120, d=12, classes=('Benign', 'TCP_SYN_Flood', 'UDP_Flood'), seed=999)
    dev = torch.device('cpu')

    m = MLP_CE_Multiclass(
        input_dim=X.shape[1],
        num_classes=3,
        device=dev,
        classes=np.array(['Benign', 'TCP_SYN_Flood', 'UDP_Flood'], dtype=object),
        widths=(32, 16),
        p_drop=0.0,
        lr=5e-3,
        epochs=2,
        batch_size=32,
        random_state=7,
    )

    m.fit(X, y)

    proba = m.predict_proba(X[:10])
    assert proba.shape == (10, 3)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    pred_idx = m.predict(X[:10], return_indices=True)
    assert pred_idx.shape == (10,)
    assert int(pred_idx.min()) >= 0
    assert int(pred_idx.max()) <= 2

    pred_lbl = m.predict(X[:10], return_indices=False)
    assert pred_lbl.shape == (10,)
    assert set(np.unique(pred_lbl)).issubset({'Benign', 'TCP_SYN_Flood', 'UDP_Flood'})
