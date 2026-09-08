import inspect
import sys

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from fire.models import (
    _explain_with_lime,
    _explain_with_shap,
    _parse_args,
    run_binary_classification,
    run_multiclass_classification,
)


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'data.csv'])
    args = _parse_args()

    assert args.aggregated_file == 'data.csv'
    assert not args.unsw
    assert not args.pca
    assert not args.shap
    assert not args.lime


def test_parse_args_xai_flags(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'data.csv', '--shap', '--lime', '--pca'])
    args = _parse_args()

    assert args.shap
    assert args.lime
    assert args.pca


def test_run_binary_classification_does_not_collide_with_firce_artifact_dir():
    # fire.models.run_binary_classification (legacy Keras-based) used to write
    # to the same 'binary_models/<dataset>' directory firce.ce_model_training's
    # train_ce_binary uses, with overlapping filenames (scaler_binary.pkl,
    # rf_model_binary.pkl, etc.) - see xseciot issue #126. Namespaced away
    # instead of removed, since fire.simulations still expects to be able to
    # read the un-suffixed 'binary_models' path (firce's own convention).
    source = inspect.getsource(run_binary_classification)
    assert "os.path.join(os.getcwd(), 'binary_models_legacy_fire', dataset_name)" in source
    assert "os.path.join(os.getcwd(), 'binary_models', dataset_name)" not in source


def test_run_multiclass_classification_does_not_collide_with_firce_artifact_dir():
    source = inspect.getsource(run_multiclass_classification)
    assert "os.path.join(os.getcwd(), 'multi_class_models_legacy_fire', dataset_name)" in source
    assert "os.path.join(os.getcwd(), 'multi_class_models', dataset_name)" not in source


def _make_xai_fixture():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(40, 4))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(5, 4))
    model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    return model, X_train, X_test


def test_explain_with_lime_writes_instance_html(tmp_path):
    model, X_train, X_test = _make_xai_fixture()

    _explain_with_lime(
        model,
        X_train,
        X_test,
        feature_names=['f0', 'f1', 'f2', 'f3'],
        class_names=['Benign', 'Attack'],
        outputPath=str(tmp_path),
        output_prefix='lime_test',
    )

    assert (tmp_path / 'lime_test_instance.html').exists()


def test_explain_with_shap_writes_summary_png(tmp_path):
    model, X_train, _ = _make_xai_fixture()

    _explain_with_shap(
        model,
        X_train[:10],
        outputPath=str(tmp_path),
        feature_names=['f0', 'f1', 'f2', 'f3'],
        model_type='tree',
        output_prefix='shap_test',
    )

    assert (tmp_path / 'shap_test_summary.png').exists()
