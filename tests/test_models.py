import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("src"))

from src.FIRE.models import _explain_with_lime, _explain_with_shap, _parse_args, run_feature_engineering


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "data.csv"])
    args = _parse_args()
    assert args.aggregated_file == "data.csv"
    assert not args.unsw
    assert not args.pca
    assert not args.shap
    assert not args.lime


def test_explain_with_lime(tmp_path):
    class DummyModel:
        def predict_proba(self, x): return np.zeros((len(x), 2))
    X_train = np.zeros((5, 3))
    X_test = np.zeros((1, 3))
    outdir = tmp_path / "lime_out"
    outdir.mkdir()
    _explain_with_lime(
        DummyModel(),
        X_train,
        X_test,
        feature_names=["a", "b", "c"],
        class_names=["c0", "c1"],
        outputPath=str(outdir),
        output_prefix="lpref"
    )
    assert (outdir / "lpref_instance.html").exists()


def test_explain_with_shap(tmp_path, monkeypatch):
    import shap
    # stub the TreeExplainer and plotting

    class DummyExplainer:
        def __init__(self, model): pass
        def shap_values(self, x): return np.zeros((x.shape[0], x.shape[1]))
    monkeypatch.setattr(shap, "TreeExplainer", lambda m: DummyExplainer(m))
    monkeypatch.setattr(shap, "KernelExplainer", lambda f, d: DummyExplainer(None))
    monkeypatch.setattr(shap, "summary_plot", lambda *a, **k: None)

    class DummyModel: pass  # noqa: E701
    X_sample = np.zeros((4, 2))
    outdir = tmp_path / "shap_out"
    outdir.mkdir()
    _explain_with_shap(
        DummyModel(),
        X_sample,
        outputPath=str(outdir),
        feature_names=["f0", "f1"],
        model_type="tree",
        output_prefix="spref"
    )
    # summary.png should be created
    assert (outdir / "spref_summary.png").exists()


def test_run_feature_engineering(tmp_path, monkeypatch):
    # create a minimal aggregated CSV
    df = pd.DataFrame({
        "f1": [1, 2, 3],
        "f2": [4, 5, 6],
        "Label": ["A", "B", "A"],
        "BinLabel": [0, 1, 0],
        "src_ip": ["x"] * 3, "dst_ip": ["y"] * 3,
        "start_time": ["t"] * 3,
        "end_time_x": ["t"] * 3, "end_time_y": ["t"] * 3,
        "time_diff": [1] * 3, "time_diff_seconds": [1] * 3,
        "Attack": [0, 1, 0],
        "start_time_x": ["t"] * 3, "start_time_y": ["t"] * 3
    })
    agg = tmp_path / "agg.csv"
    df.to_csv(agg, index=False)
    monkeypatch.chdir(tmp_path)

    scaler, pca, X_pca = run_feature_engineering(str(agg))
    assert hasattr(scaler, "transform")
    assert hasattr(pca, "transform")
    assert isinstance(X_pca, np.ndarray)
    assert X_pca.shape[0] == 3

    # files on disk?
    ds = agg.parent.name
    fe = tmp_path / "feature_engineering" / ds
    assert (fe / "scaler.pkl").exists()
    assert (fe / "pca.pkl").exists()
