import sys

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# ensure src/ on path
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from src.FIRE.simulations import (  # noqa: E402
    _get_dataset_name,
    _parse_args,
    continuous_simulation,
    parallel_simulation,
    preprocess_chunk,
    process_chunk,
    sequential_simulation,
)


class DummyModel:
    def predict(self, X):
        # always return 0
        return np.zeros(X.shape[0], dtype=int)


@pytest.fixture
def setup_sim_dir(tmp_path):
    """
    Create a mini dataset + pre-trained scaler/pca/model pickles so that
    the simulations functions can load them.
    """
    # 1) dataset folder + CSV
    ds_dir = tmp_path / "DATASET"
    ds_dir.mkdir()
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "f2": [4.0, 5.0, 6.0],
        "Label": ["Benign", "Attack", "Benign"],
        "end_time_x": ["2021-01-01 00:00:00"] * 3
    })
    agg = ds_dir / "aggregated_data.csv"
    df.to_csv(agg, index=False)

    # 2) train scaler + pca on the two numeric cols
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(df[["f1", "f2"]])
    pca = PCA(n_components=1).fit(scaler.transform(df[["f1", "f2"]]))

    # 3) dump pickles in the expected structure
    bm = tmp_path / "binary_models" / "DATASET"
    mm = tmp_path / "multi_class_models" / "DATASET"
    bm.mkdir(parents=True)
    mm.mkdir(parents=True)

    joblib.dump(scaler, bm / "scaler_binary.pkl")
    joblib.dump(pca, bm / "pca_binary.pkl")
    joblib.dump(DummyModel(), bm / "dt_model_binary.pkl")
    joblib.dump(scaler, mm / "scaler_multi.pkl")
    joblib.dump(pca, mm / "pca_multi.pkl")
    joblib.dump(DummyModel(), mm / "decision_tree_multi.pkl")

    return agg


def test_parse_args_sim():
    sys.argv = ["prog", "agg.csv", "--mode", "parallel", "--model_type", "multi",
                "--model_variant", "rf", "--unsw"]
    args = _parse_args()
    assert args.aggregated_file == "agg.csv"
    assert args.mode == "parallel"
    assert args.model_type == "multi"
    assert args.model_variant == "rf"
    assert args.unsw


def test_preprocess_and_get_name():
    df = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, 2], "drop": [9, 9]})
    cleaned = preprocess_chunk(df, ["drop"])
    assert "drop" not in cleaned.columns
    assert not cleaned.isna().any().any()

    # get_dataset_name
    assert _get_dataset_name("/foo/BAR/agg.csv") == "BAR"


def test_process_chunk_preds(setup_sim_dir, tmp_path, monkeypatch):
    agg = setup_sim_dir
    # switch cwd so _load picks up tmp_path/binary_models etc.
    monkeypatch.chdir(tmp_path)

    chunk = pd.read_csv(agg, nrows=2)
    drop_cols = ['Label', 'BinLabel', 'src_ip', 'dst_ip', 'start_time',
                 'end_time_x', 'end_time_y', 'time_diff', 'time_diff_seconds', 'Attack']

    from src.FIRE.simulations import load_simulation_objects
    scaler, pca, model = load_simulation_objects(str(agg), "binary", "dt")

    preds = process_chunk(
        chunk, drop_cols,
        scaler, pca, model,
        model_variant="dt", model_type="binary", threshold=0.5
    )
    # DummyModel.predict -> zeros -> 'Benign'
    assert preds == ["Benign", "Benign"]


def test_sequential_simulation(setup_sim_dir, tmp_path, monkeypatch):
    agg = setup_sim_dir
    monkeypatch.chdir(tmp_path)
    preds = sequential_simulation(
        str(agg), model_type="binary", model_variant="dt",
        chunk_size=2, delay=0, threshold=0.5, isUNSW=False
    )
    # 3 rows => 3 preds
    assert isinstance(preds, list)
    assert len(preds) == 3


def test_continuous_simulation(setup_sim_dir, tmp_path, monkeypatch):
    agg = setup_sim_dir
    monkeypatch.chdir(tmp_path)
    true_labels, preds = continuous_simulation(
        str(agg), model_type="binary", model_variant="dt",
        chunk_size=3, window_duration=3600, delay=0,
        threshold=0.5, isUNSW=False
    )
    assert len(true_labels) == 3
    assert len(preds) == 3


def test_parallel_simulation(setup_sim_dir, tmp_path, monkeypatch):
    agg = setup_sim_dir
    monkeypatch.chdir(tmp_path)
    preds = parallel_simulation(
        str(agg), model_type="binary", model_variant="dt",
        chunk_size=2, num_processes=2, threshold=0.5, isUNSW=False
    )
    assert len(preds) == 3
