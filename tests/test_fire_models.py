from __future__ import annotations

import sys

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.FIRE import models as fire_models


def _write_min_agg_csv(path: Path, *, multiclass: bool = True) -> None:
    df = pd.DataFrame({
        'f1': [1.0, 2.0, np.nan, 4.0],
        'f2': [0.1, 0.2, 0.3, 0.4],
        'Label': ['Benign', 'TCPAttack', 'Benign', 'UDPAttack']
        if multiclass
        else ['Benign', 'Attack', 'Benign', 'Attack'],
        'src_ip': ['x'] * 4,
        'dst_ip': ['y'] * 4,
        'start_time': ['t'] * 4,
        'end_time_x': ['t'] * 4,
        'end_time_y': ['t'] * 4,
        'time_diff': [1] * 4,
        'time_diff_seconds': [1] * 4,
        'Attack': [0, 1, 0, 2],
        'start_time_x': ['t'] * 4,
        'start_time_y': ['t'] * 4,
    })
    df.to_csv(path, index=False)


def _touch(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b'')


class _DummySkModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._fit_called = False

    def fit(self, X: Any, y: Any) -> '_DummySkModel':
        self._fit_called = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = len(X)
        return np.zeros(n, dtype=int)

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        return np.tile(np.array([[1.0, 0.0]]), (n, 1))


class _DummyXGBTrainModel:
    def predict(self, dtest: Any) -> np.ndarray:
        n = getattr(dtest, 'n', 1)
        return np.zeros(n, dtype=np.float32)


class _DummyDMatrix:
    def __init__(self, X: Any, label: Any = None, feature_names: Any = None) -> None:
        self.n = len(X)


class _DummyXGBClassifier(_DummySkModel):
    pass


class _DummyKerasSequential:
    def __init__(self, layers: Any = None) -> None:
        self.layers = layers

    def compile(self, *args: Any, **kwargs: Any) -> None:
        return None

    def fit(self, *args: Any, **kwargs: Any) -> None:
        return None

    def evaluate(self, X: Any, y: Any, verbose: int = 0) -> tuple[float, float]:
        return 0.0, 1.0

    def predict(self, X: Any) -> np.ndarray:
        n = len(X)
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class _DummyLabelEncoder:
    def __init__(self) -> None:
        self.classes_: np.ndarray = np.array([], dtype=object)

    def fit_transform(self, y: Any) -> np.ndarray:
        y = np.asarray(y, dtype=object)
        self.classes_ = np.unique(y)
        mapping = {c: i for i, c in enumerate(self.classes_)}
        return np.array([mapping[v] for v in y], dtype=int)

    def inverse_transform(self, y_idx: Any) -> np.ndarray:
        y_idx = np.asarray(y_idx, dtype=int)
        return self.classes_[y_idx]


def _dummy_to_categorical(y: Any) -> np.ndarray:
    y = np.asarray(y, dtype=int).reshape(-1)
    k = int(y.max()) + 1 if y.size else 0
    out = np.zeros((y.size, k), dtype=np.float32)
    if y.size:
        out[np.arange(y.size), y] = 1.0
    return out


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'argv', ['prog', 'data.csv'])
    args = fire_models._parse_args()
    assert args.aggregated_file == 'data.csv'
    assert not args.unsw
    assert not args.pca
    assert not args.shap
    assert not args.lime


@pytest.mark.slow
def test_explain_with_lime_writes_html(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyExplanation:
        def save_to_file(self, path: str) -> None:
            _touch(path)

    class _DummyExplainer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def explain_instance(self, x: Any, predict_proba: Any) -> _DummyExplanation:
            return _DummyExplanation()

    monkeypatch.setattr(fire_models, 'LimeTabularExplainer', _DummyExplainer)

    class DummyModel:
        def predict_proba(self, x: Any) -> np.ndarray:
            return np.zeros((len(x), 2), dtype=np.float32)

    X_train = np.zeros((5, 3), dtype=np.float32)
    X_test = np.zeros((1, 3), dtype=np.float32)

    outdir = tmp_path / 'lime_out'
    outdir.mkdir()

    fire_models._explain_with_lime(
        DummyModel(),
        X_train,
        X_test,
        feature_names=['a', 'b', 'c'],
        class_names=['c0', 'c1'],
        outputPath=str(outdir),
        output_prefix='lpref',
    )

    assert (outdir / 'lpref_instance.html').exists()


@pytest.mark.slow
def test_explain_with_shap_tree_and_list_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyExplainer:
        def shap_values(self, x: np.ndarray) -> list[np.ndarray]:
            return [np.zeros_like(x), np.zeros_like(x)]

    monkeypatch.setattr(fire_models.shap, 'TreeExplainer', lambda m: _DummyExplainer())
    monkeypatch.setattr(fire_models.shap, 'summary_plot', lambda *a, **k: None)

    monkeypatch.setattr(fire_models.plt, 'savefig', lambda p: _touch(str(p)))
    monkeypatch.setattr(fire_models.plt, 'figure', lambda *a, **k: None)
    monkeypatch.setattr(fire_models.plt, 'tight_layout', lambda *a, **k: None)
    monkeypatch.setattr(fire_models.plt, 'close', lambda *a, **k: None)

    class DummyModel:
        pass

    X_sample = np.zeros((4, 2), dtype=np.float32)
    outdir = tmp_path / 'shap_out'
    outdir.mkdir()

    fire_models._explain_with_shap(
        DummyModel(),
        X_sample,
        outputPath=str(outdir),
        feature_names=['f0', 'f1'],
        model_type='tree',
        output_prefix='spref',
    )

    assert (outdir / 'spref_class0summary.png').exists()
    assert (outdir / 'spref_class1summary.png').exists()


def test_run_feature_engineering_saves_files_and_fills_nan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds_dir = tmp_path / 'datasetA'
    ds_dir.mkdir()
    agg = ds_dir / 'agg.csv'
    _write_min_agg_csv(agg, multiclass=True)

    monkeypatch.chdir(tmp_path)

    scaler, pca, X_pca = fire_models.run_feature_engineering(str(agg))
    assert hasattr(scaler, 'transform')
    assert hasattr(pca, 'transform')
    assert isinstance(X_pca, np.ndarray)
    assert X_pca.shape[0] == 4

    fe = tmp_path / 'feature_engineering' / 'datasetA'
    assert (fe / 'scaler.pkl').exists()
    assert (fe / 'pca.pkl').exists()


def test_run_binary_classification_fast_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fire_models, 'cross_val_score', lambda *a, **k: np.array([0.5, 0.6]))
    monkeypatch.setattr(fire_models, 'train_test_split', lambda X, y, **k: (X[:2], X[2:], y[:2], y[2:]))

    monkeypatch.setattr(fire_models, 'RandomForestClassifier', _DummySkModel)
    monkeypatch.setattr(fire_models, 'KNeighborsClassifier', _DummySkModel)
    monkeypatch.setattr(fire_models, 'DecisionTreeClassifier', _DummySkModel)
    monkeypatch.setattr(fire_models, 'SVC', _DummySkModel)

    monkeypatch.setattr(fire_models, 'confusion_matrix', lambda *a, **k: np.zeros((2, 2), dtype=int))
    monkeypatch.setattr(fire_models, 'classification_report', lambda *a, **k: 'ok')
    monkeypatch.setattr(fire_models, 'accuracy_score', lambda *a, **k: 1.0)

    monkeypatch.setattr(fire_models.xgb, 'DMatrix', _DummyDMatrix)
    monkeypatch.setattr(fire_models.xgb, 'train', lambda *a, **k: _DummyXGBTrainModel())
    monkeypatch.setattr(fire_models.xgb, 'XGBClassifier', _DummyXGBClassifier)

    monkeypatch.setattr(fire_models, 'Sequential', _DummyKerasSequential)

    monkeypatch.setattr(fire_models, '_explain_with_shap', lambda *a, **k: None)
    monkeypatch.setattr(fire_models, '_explain_with_lime', lambda *a, **k: None)

    monkeypatch.setattr(fire_models.joblib, 'dump', lambda obj, path: _touch(str(path)))

    ds_dir = tmp_path / 'datasetB'
    ds_dir.mkdir()
    agg = ds_dir / 'agg.csv'
    _write_min_agg_csv(agg, multiclass=True)

    monkeypatch.chdir(tmp_path)

    # Non-PCA branch
    fire_models.run_binary_classification(str(agg), isUNSW=False, isPCA=False)
    outdir = tmp_path / 'binary_models' / 'datasetB'
    assert outdir.exists()
    assert (outdir / 'rf_model_binary.pkl').exists()
    assert (outdir / 'scaler_binary.pkl').exists()

    # PCA branch
    fire_models.run_binary_classification(str(agg), isUNSW=False, isPCA=True)
    assert (outdir / 'pca_binary.pkl').exists()


def test_run_multiclass_classification_fast_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fire_models, 'cross_val_score', lambda *a, **k: np.array([0.6, 0.7]))
    monkeypatch.setattr(fire_models, 'train_test_split', lambda X, y, **k: (X[:2], X[2:], y[:2], y[2:]))

    monkeypatch.setattr(fire_models, 'RandomForestClassifier', _DummySkModel)
    monkeypatch.setattr(fire_models, 'KNeighborsClassifier', _DummySkModel)
    monkeypatch.setattr(fire_models, 'DecisionTreeClassifier', _DummySkModel)
    monkeypatch.setattr(fire_models, 'SVC', _DummySkModel)

    monkeypatch.setattr(fire_models, 'confusion_matrix', lambda *a, **k: np.zeros((3, 3), dtype=int))
    monkeypatch.setattr(fire_models, 'classification_report', lambda *a, **k: 'ok')

    monkeypatch.setattr(fire_models, 'LabelEncoder', _DummyLabelEncoder)
    monkeypatch.setattr(fire_models, 'to_categorical', _dummy_to_categorical)
    monkeypatch.setattr(fire_models, 'Sequential', _DummyKerasSequential)
    monkeypatch.setattr(fire_models.xgb, 'XGBClassifier', _DummyXGBClassifier)

    monkeypatch.setattr(fire_models, '_explain_with_shap', lambda *a, **k: None)
    monkeypatch.setattr(fire_models, '_explain_with_lime', lambda *a, **k: None)
    monkeypatch.setattr(fire_models.joblib, 'dump', lambda obj, path: _touch(str(path)))

    ds_dir = tmp_path / 'datasetC'
    ds_dir.mkdir()
    agg = ds_dir / 'agg.csv'
    _write_min_agg_csv(agg, multiclass=True)

    monkeypatch.chdir(tmp_path)

    fire_models.run_multiclass_classification(str(agg), isUNSW=False, isPCA=False)
    outdir = tmp_path / 'multi_class_models' / 'datasetC'
    assert outdir.exists()
    assert (outdir / 'random_forest_multi.pkl').exists()
    assert (outdir / 'scaler_multi.pkl').exists()

    fire_models.run_multiclass_classification(str(agg), isUNSW=False, isPCA=True)
    assert (outdir / 'pca_multi.pkl').exists()
