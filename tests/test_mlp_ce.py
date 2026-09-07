import numpy as np
import pytest
import torch

from firce.models.mlp_ce import MLP_CE

DEVICE = torch.device('cpu')


def _make_binary_data(n=40, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_mlp_ce_binary_fit_predict_proba_shape():
    X, y = _make_binary_data()
    model = MLP_CE(input_dim=X.shape[1], device=DEVICE, epochs=3)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert model.classes_.tolist() == [0, 1]
    assert model.is_fitted_ is True
    assert model.n_features_in_ == X.shape[1]


def test_mlp_ce_binary_predict_matches_threshold():
    X, y = _make_binary_data()
    model = MLP_CE(input_dim=X.shape[1], device=DEVICE, epochs=3, threshold=0.5)
    model.fit(X, y)

    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    assert preds.tolist() == (proba > 0.5).astype(int).tolist()


def test_mlp_ce_binary_save_load_roundtrip(tmp_path):
    X, y = _make_binary_data()
    model = MLP_CE(input_dim=X.shape[1], device=DEVICE, epochs=3)
    model.fit(X, y)
    proba_before = model.predict_proba(X)

    ckpt_path = tmp_path / 'model.pt'
    model.save(str(ckpt_path))
    loaded = MLP_CE.load(str(ckpt_path), device=DEVICE)

    assert loaded.classes_.tolist() == model.classes_.tolist()
    assert loaded.n_features_in_ == model.n_features_in_
    proba_after = loaded.predict_proba(X)
    assert np.allclose(proba_before, proba_after, atol=1e-5)


def test_mlp_ce_binary_get_params_set_params_clone():
    model = MLP_CE(input_dim=4, device=DEVICE, epochs=3, threshold=0.7)
    params = model.get_params()
    assert params['threshold'] == 0.7
    assert params['input_dim'] == 4

    clone = model.clone()
    assert clone.get_params()['threshold'] == 0.7
    assert clone.is_fitted_ is False
    assert clone.classes_ is None

    model.set_params(threshold=0.3)
    assert model.threshold == 0.3
