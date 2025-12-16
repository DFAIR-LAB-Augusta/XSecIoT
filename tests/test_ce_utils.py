from __future__ import annotations

import os

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from src.core.conformalEval.utils import (
    clone_model,
    compute_class_thresholds,
    compute_nonconformity_scores,
    compute_p_values,
    load_conformal_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def _clamp_threads_for_tests() -> None:
    """
    Keep native thread pools from going nuts (OpenMP/MKL/etc).
    Helpful if xgboost is linked with OpenMP on macOS.
    """
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


def test_clone_model_prefers_custom_clone() -> None:
    class Custom:
        def __init__(self, x: int) -> None:
            self.x = x

        def clone(self) -> 'Custom':
            return Custom(self.x + 1)

    m = Custom(10)
    c = clone_model(m)
    assert isinstance(c, Custom)
    assert c is not m
    assert c.x == 11


def test_clone_model_sk_clone_path() -> None:
    pytest.importorskip('sklearn')
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=0.25, max_iter=50, solver='liblinear')
    c = clone_model(m)

    assert c is not m
    assert isinstance(c, LogisticRegression)
    assert c.get_params()['C'] == pytest.approx(0.25)
    assert c.get_params()['max_iter'] == 50
    assert c.get_params()['solver'] == 'liblinear'


def test_clone_model_get_params_ctor_signature_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Force sklearn.clone to fail so we exercise the get_params + ctor signature path.
    """
    sklearn = pytest.importorskip('sklearn')
    import sklearn.base  # noqa: F811

    def _boom(_: Any) -> Any:
        raise RuntimeError('force sklearn.clone failure')

    monkeypatch.setattr(sklearn.base, 'clone', _boom, raising=True)

    class WeirdCtor:
        def __init__(self, a: int, b: int = 2) -> None:
            self.a = a
            self.b = b

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            return {'a': 7, 'b': 9, 'extra': 123}

    m = WeirdCtor(a=1, b=2)
    c = clone_model(m)

    assert isinstance(c, WeirdCtor)
    assert c is not m
    assert c.a == 7
    assert c.b == 9
    assert not hasattr(c, 'extra')


def test_clone_model_deepcopy_fallback() -> None:
    class Plain:
        def __init__(self) -> None:
            self.items = [1, 2, 3]

    m = Plain()
    c = clone_model(m)

    assert isinstance(c, Plain)
    assert c is not m
    assert c.items == [1, 2, 3]
    c.items.append(4)
    assert m.items == [1, 2, 3]


def test_clone_model_xgbclassifier_special_case_smoke() -> None:
    """
    This should be cheap (no fit), but can pull OpenMP libs on macOS; clamp threads.
    """
    _clamp_threads_for_tests()

    pytest.importorskip('xgboost')
    from xgboost import XGBClassifier

    m = XGBClassifier(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.2,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=123,
        eval_metric='logloss',
    )
    c = clone_model(m)

    assert isinstance(c, XGBClassifier)
    assert c is not m

    mp = m.get_params()
    cp = c.get_params()
    for k in ('n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'random_state'):
        assert cp.get(k) == mp.get(k)


def test_compute_nonconformity_scores_basic() -> None:
    class_list = np.array(['Benign', 'UDP', 'TCP'], dtype=object)

    probas = np.array(
        [
            [0.7, 0.2, 0.1],  # true Benign => score = 0.3
            [0.1, 0.8, 0.1],  # true UDP    => score = 0.2
            [0.2, 0.3, 0.5],  # true TCP    => score = 0.5
        ],
        dtype=np.float64,
    )
    true_labels = np.array(['Benign', 'UDP', 'TCP'], dtype=object)

    scores = compute_nonconformity_scores(probas, true_labels, class_list)
    assert scores.shape == (3,)
    assert np.allclose(scores, np.array([0.3, 0.2, 0.5]))


def test_compute_p_values_basic() -> None:
    scores = np.array([0.2, 0.6, 0.1], dtype=np.float64)
    preds = np.array(['A', 'B', 'A'], dtype=object)

    calibration_scores = {
        'A': np.array([0.0, 0.1, 0.2, 0.9], dtype=np.float64),
        'B': np.array([0.5, 0.7], dtype=np.float64),
    }

    p = compute_p_values(scores, preds, calibration_scores)
    assert p.shape == scores.shape

    # For A score=0.2: calib >= 0.2 are [0.2,0.9] => 2; p=(2+1)/(4+1)=0.6
    # For B score=0.6: calib >= 0.6 are [0.7]     => 1; p=(1+1)/(2+1)=0.666...
    # For A score=0.1: calib >= 0.1 are [0.1,0.2,0.9] => 3; p=(3+1)/5=0.8
    assert p[0] == pytest.approx(0.6)
    assert p[1] == pytest.approx(2 / 3)
    assert p[2] == pytest.approx(0.8)


def test_compute_class_thresholds_quantiles() -> None:
    calibration_scores = {
        'A': np.array([0.0, 0.5, 1.0], dtype=np.float64),
        'B': np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64),
    }
    thr = compute_class_thresholds(calibration_scores, significance=0.25)
    assert set(thr.keys()) == {'A', 'B'}

    # threshold = quantile(scores, 1 - significance) = q(0.75)
    assert thr['A'] == pytest.approx(float(np.quantile(calibration_scores['A'], 0.75)))
    assert thr['B'] == pytest.approx(float(np.quantile(calibration_scores['B'], 0.75)))


def test_load_conformal_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_conformal_config(tmp_path / 'nope.toml')


def test_load_conformal_config_reads_toml(tmp_path: Path) -> None:
    p = tmp_path / 'conformal_config.toml'
    p.write_text(
        """
[main]
significance = 0.05
folds = 5

[model]
variant = "rf"
""".strip(),
        encoding='utf-8',
    )

    cfg = load_conformal_config(p)
    assert isinstance(cfg, dict)
    assert cfg['main']['significance'] == pytest.approx(0.05)
    assert cfg['main']['folds'] == 5
    assert cfg['model']['variant'] == 'rf'
