import numpy as np
import pytest
import torch

from firce.models.mlp_ce import MLP_CE
from firce.models.mlp_ce_multiclass import MLPCEMulticlass

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


def _make_multiclass_data(n=60, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    labels = np.array(['Benign', 'PortScan', 'XMasAttack'])
    idx = np.argmax(np.stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]], axis=1), axis=1)
    y = labels[idx]
    return X, y, labels


def test_mlp_ce_multiclass_fit_predict_proba_shape():
    X, y, labels = _make_multiclass_data()
    model = MLPCEMulticlass(input_dim=X.shape[1], classes=labels, device=DEVICE, epochs=5)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert sorted(model.classes_.tolist()) == sorted(labels.tolist())
    assert model.is_fitted_ is True


def test_mlp_ce_multiclass_predict_returns_known_labels():
    X, y, labels = _make_multiclass_data()
    model = MLPCEMulticlass(input_dim=X.shape[1], classes=labels, device=DEVICE, epochs=5)
    model.fit(X, y)

    preds = model.predict(X)
    assert set(preds.tolist()) <= set(labels.tolist())
    assert preds.shape == (X.shape[0],)


def test_mlp_ce_multiclass_requires_at_least_two_classes():
    with pytest.raises(ValueError, match='at least 2 distinct labels'):
        MLPCEMulticlass(input_dim=4, classes=['OnlyOne'], device=DEVICE)


def test_mlp_ce_multiclass_unknown_label_in_fit_raises():
    X, y, labels = _make_multiclass_data()
    model = MLPCEMulticlass(input_dim=X.shape[1], classes=labels[:2], device=DEVICE, epochs=2)
    with pytest.raises(ValueError, match='not found in known classes'):
        model.fit(X, y)


def test_mlp_ce_multiclass_save_load_roundtrip(tmp_path):
    X, y, labels = _make_multiclass_data()
    model = MLPCEMulticlass(input_dim=X.shape[1], classes=labels, device=DEVICE, epochs=5)
    model.fit(X, y)
    proba_before = model.predict_proba(X)

    ckpt_path = tmp_path / 'mc_model.pt'
    model.save(str(ckpt_path))
    loaded = MLPCEMulticlass.load(str(ckpt_path), device=DEVICE)

    assert sorted(loaded.classes_.tolist()) == sorted(model.classes_.tolist())
    proba_after = loaded.predict_proba(X)
    assert np.allclose(proba_before, proba_after, atol=1e-5)


def test_mlp_ce_multiclass_get_params_set_params_clone():
    X, y, labels = _make_multiclass_data()
    model = MLPCEMulticlass(input_dim=X.shape[1], classes=labels, device=DEVICE, epochs=3)
    model.fit(X, y)

    params = model.get_params()
    assert params['input_dim'] == X.shape[1]
    assert sorted(params['classes']) == sorted(labels.tolist())

    clone = model.clone()
    assert sorted(clone.classes_.tolist()) == sorted(labels.tolist())
    assert clone.is_fitted_ is False

    model.set_params(epochs=7)
    assert model.epochs == 7
