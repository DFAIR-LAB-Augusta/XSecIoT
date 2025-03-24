# src.FIRE.simulations

import argparse
import logging
import multiprocessing as mp
import os
import time

from functools import partial
from typing import List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.core.models.feedforward_binary import FeedForwardBinary
from src.core.models.torch_device import pick_device

logger = logging.getLogger(__name__)
np.random.seed(42)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simulation pipeline for FIRE process (Step 3)"
    )
    parser.add_argument("aggregated_file", type=str,
                        help="Path to aggregated_data.csv")
    parser.add_argument("--mode", type=str, default="sequential",
                        choices=["sequential", "continuous", "parallel"],
                        help="Simulation mode")
    parser.add_argument("--model_type", type=str, default="binary",
                        choices=["binary", "multi"],
                        help="Model type")
    parser.add_argument("--model_variant", type=str, default="dt",
                        help="Model variant: dt, knn, rf, svm, feedforward, xgb")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Chunk size")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between chunks (s)")
    parser.add_argument("--window_duration", type=int, default=300,
                        help="Window duration (s)")
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Processes for parallel mode")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for probabilistic models")
    parser.add_argument("--unsw", action="store_true",
                        help="UNSW dataset mode")
    return parser.parse_args()


def _get_dataset_name(file_path: str) -> str:
    return os.path.basename(os.path.dirname(file_path))


def preprocess_chunk(chunk: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    """
    Preprocess a chunk of flow data by dropping specified columns and filling missing values.

    This function removes columns listed in `drop_cols` (if present), and replaces any
    remaining NaN values in numeric columns with the column-wise mean.

    Args:
        chunk (pd.DataFrame): A DataFrame containing raw flow records.
        drop_cols (List[str]): List of column names to drop during preprocessing.

    Returns:
        pd.DataFrame: Cleaned DataFrame with selected numeric columns and no missing values.
    """
    X = chunk.drop(columns=drop_cols, errors='ignore')
    if X.isna().any().any():
        X = X.fillna(X.mean())
    return X


def load_simulation_objects(
    aggregated_file: str,
    model_type: str,
    model_variant: str,
    use_pca: bool = True
) -> Tuple[StandardScaler, Optional[PCA], ClassifierMixin | xgb.Booster | FeedForwardBinary]:
    """
    Load trained CE model, scaler, and (optionally) PCA from disk.

    Args:
        aggregated_file (str): Path to the aggregated training file to derive dataset name.
        model_type (str): Either 'binary' or 'multi'.
        model_variant (str): Model type (e.g., 'dt', 'knn', 'rf', etc.).
        use_pca (bool): If True, load PCA; otherwise, skip.

    Returns:
        Tuple containing:
            - StandardScaler object
            - PCA object (or None if not used)
            - Trained model object
    """
    dataset_name = _get_dataset_name(aggregated_file)
    if model_type == 'binary':
        base = os.path.join(os.getcwd(), "binary_models", dataset_name)
        scaler_file = os.path.join(base, "scaler_binary.pkl")
        pca_file = os.path.join(base, "pca_binary.pkl")
        if model_variant != 'feedforward':
            model_file = os.path.join(
                base, f"{model_variant}_model_binary.pkl")
        else:
            model_file = os.path.join(base, "feedforward_model_binary.pt")
    else:
        base = os.path.join(os.getcwd(), "multi_class_models", dataset_name)
        scaler_file = os.path.join(base, "scaler_multi.pkl")
        pca_file = os.path.join(base, "pca_multi.pkl")
        mapping = {
            'dt': 'decision_tree_multi.pkl',
            'rf': 'random_forest_multi.pkl',
            'feedforward': 'feedforward_multi.pt',
            'knn': 'knearest_multi.pkl',
            'svm': 'svm_multi.pkl',
            'xgb': 'xgboost_multi.pkl'
        }
        if model_variant not in mapping:
            raise ValueError(
                f"Unsupported multi-class variant: {model_variant}")
        model_file = os.path.join(base, mapping[model_variant])

    scaler = joblib.load(scaler_file)
    pca = joblib.load(pca_file) if use_pca else None
    if model_variant != 'feedforward':
        model = joblib.load(model_file)
        return scaler, pca, model

    device = pick_device()

    logger.debug(f"Loading Torch checkpoint from {model_file} to CPU")
    ckpt = torch.load(model_file, map_location="cpu")
    input_dim = int(ckpt.get("input_dim"))
    p_drop = float(ckpt.get("dropout", 0.3))
    state_dict = ckpt["state_dict"]

    logger.debug(
        f"Rebuilding FeedForwardBinary(input_dim={input_dim}, p_drop={p_drop})")
    torch_model = FeedForwardBinary(input_dim=input_dim, p_drop=p_drop)
    missing, unexpected = torch_model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.debug(f"Missing keys when loading state_dict: {missing}")
    if unexpected:
        logger.debug(f"Unexpected keys when loading state_dict: {unexpected}")

    torch_model.to(device)
    torch_model.eval()
    logger.debug(f"Torch model ready on device={device}")
    return scaler, pca, torch_model


def process_chunk(
    chunk: pd.DataFrame,
    drop_cols: List[str],
    scaler: StandardScaler,
    pca: Optional[PCA],
    model: ClassifierMixin | xgb.Booster | FeedForwardBinary,
    model_variant: str,
    model_type: str,
    threshold: float,
    use_pca: bool = True
) -> List[int | str]:
    """
    Preprocess and predict on a data chunk using the CE model pipeline.

    This function drops unnecessary columns, aligns features with the training
    scaler, scales the data, optionally applies PCA, and returns model predictions.

    Args:
        chunk (pd.DataFrame): Data chunk containing raw CE flow records.
        drop_cols (List[str]): Columns to drop during preprocessing.
        scaler (StandardScaler): Pre-fitted scaler from training.
        pca (Optional[PCA]): Pre-fitted PCA model, or None if not used.
        model (ClassifierMixin | xgb.Booster): Trained model.
        model_variant (str): Name of the model variant used.
        model_type (str): 'binary' or 'multi'.
        threshold (float): Threshold for binarizing feedforward predictions.
        use_pca (bool): Whether to apply PCA before prediction.

    Returns:
        LList[int | str]: List of predicted labels or class indices.
    """
    # 1) preprocess & select numeric
    X_df = preprocess_chunk(
        chunk, drop_cols).select_dtypes(include=[np.number])

    # 2) align columns to scaler.feature_names_in_
    if hasattr(scaler, "feature_names_in_"):
        feat = list(scaler.feature_names_in_)
        means = getattr(scaler, "mean_", None)
        arrs = []
        for idx, name in enumerate(feat):
            if name in X_df.columns:
                arrs.append(X_df[name].to_numpy())
            else:
                # fill with training mean if available, else zeros
                fill = means[idx] if means is not None else 0.0
                arrs.append(np.full(len(X_df), fill))
        X_arr = np.stack(arrs, axis=1)
    else:
        X_arr = X_df.to_numpy()

    # 3) scale & PCA
    X_s = scaler.transform(X_arr)
    X_p = pca.transform(X_s) if use_pca and pca is not None else X_s

    # 4) predict
    if model_variant.startswith('xgb') and isinstance(model, xgb.Booster):
        fnames = [f"f_{i}" for i in range(X_p.shape[1])]
        dtest = xgb.DMatrix(X_p, feature_names=fnames)
        preds = model.predict(dtest)
    else:
        preds = model.predict(X_p)  # type: ignore
        if model_variant == 'feedforward':
            preds = (preds > threshold).astype(int)
        if model_type == 'binary':
            preds = ['Attack' if int(p) == 1 else 'Benign' for p in preds]

    return list(preds)


# ------------------------------
# Simulation Functions
# ------------------------------

def sequential_simulation(
    aggregated_file: str,
    model_type: Literal['binary', 'multi'] = 'binary',
    model_variant: str = 'dt',
    chunk_size: int = 1000,
    delay: float = 1.0,
    threshold: float = 0.5,
    isUNSW: bool = False
) -> List[int | str]:
    drop_cols = [
        'Label', 'BinLabel', 'src_ip', 'dst_ip',
        'start_time', 'end_time_x', 'end_time_y',
        'time_diff', 'time_diff_seconds', 'Attack'
    ]
    if isUNSW:
        drop_cols += ['start_time_x', 'start_time_y']

    scaler, pca, model = load_simulation_objects(
        aggregated_file, model_type, model_variant
    )
    all_preds = []
    t0 = time.time()

    for chunk in pd.read_csv(aggregated_file, chunksize=chunk_size):
        print("\nProcessing new data chunk…")
        t_chunk = time.time()

        preds = process_chunk(
            chunk, drop_cols, scaler, pca, model,  # type: ignore
            model_variant, model_type, threshold
        )
        for i, p in enumerate(preds, 1):
            print(f"Data point {i}: {p}")
        all_preds.extend(preds)

        latency = time.time() - t_chunk
        print(f"Latency: {latency:.4f}s")
        time.sleep(delay)

    print(f"Total sequential time: {time.time() - t0:.4f}s")
    return all_preds


def continuous_simulation(
    aggregated_file: str,
    model_type: Literal['binary', 'multi'] = 'binary',
    model_variant: str = 'dt',
    chunk_size: int = 1000,
    window_duration: int = 300,
    delay: float = 1.0,
    threshold: float = 0.5,
    isUNSW: bool = False
) -> Tuple[List[int | str], List[int | str]]:
    drop_cols = [
        'Label', 'BinLabel', 'src_ip', 'dst_ip',
        'start_time', 'end_time_x', 'end_time_y',
        'time_diff', 'time_diff_seconds', 'Attack'
    ]
    if isUNSW:
        drop_cols += ['start_time_x', 'start_time_y']

    scaler, pca, model = load_simulation_objects(
        aggregated_file, model_type, model_variant
    )
    window_df = pd.DataFrame()
    all_true, all_preds, latencies = [], [], []
    t0 = time.time()

    for chunk in pd.read_csv(aggregated_file, chunksize=chunk_size):
        print("\nProcessing new data chunk…")
        t_chunk = time.time()

        # update window
        window_df = pd.concat([window_df, chunk])
        window_df['end_time_x'] = pd.to_datetime(
            window_df['end_time_x'], errors='coerce'
        )
        latest = window_df['end_time_x'].max()
        window_df = window_df[
            window_df['end_time_x'] >= latest -
            pd.Timedelta(seconds=window_duration)
        ]

        # ensure binary label
        if model_type == 'binary':
            if 'BinLabel' not in window_df:
                window_df['BinLabel'] = window_df['Label'].map(
                    {'Benign': 0, 'Attack': 1})
            else:
                na = window_df['BinLabel'].isna()
                if na.any():
                    window_df.loc[na, 'BinLabel'] = window_df.loc[na, 'Label'].map(
                        {'Benign': 0, 'Attack': 1})

        print("Window snapshot:\n", window_df.head())

        # preprocess & align
        X_df = window_df.drop(columns=drop_cols, errors='ignore').select_dtypes(
            include=[np.number])
        if X_df.isna().any().any():
            X_df = X_df.fillna(X_df.mean()).dropna()
        if X_df.empty:
            print("No numeric data—skipping chunk.")
            continue

        y_true = window_df['BinLabel'] if model_type == 'binary' else window_df['Label']
        # align + scale + pca + predict
        preds = process_chunk(
            window_df, drop_cols, scaler, pca, model,  # type: ignore
            model_variant, model_type, threshold
        )

        for i, p in enumerate(preds, 1):
            print(f"Data point {i}: {p}")
        all_true.extend(y_true.tolist())
        all_preds.extend(preds)

        latency = time.time() - t_chunk
        latencies.append(latency)
        print(f"Chunk latency: {latency:.4f}s")
        time.sleep(delay)

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"Avg latency: {avg_lat:.4f}s, total: {time.time() - t0:.4f}s")
    return all_true, all_preds


def parallel_simulation(
    aggregated_file: str,
    model_type: Literal['binary', 'multi'] = 'binary',
    model_variant: str = 'dt',
    chunk_size: int = 1000,
    num_processes: int = 4,
    threshold: float = 0.5,
    isUNSW: bool = False
) -> List[int | str]:
    drop_cols = [
        'Label', 'BinLabel', 'src_ip', 'dst_ip',
        'start_time', 'end_time_x', 'end_time_y',
        'time_diff', 'time_diff_seconds', 'Attack'
    ]
    if isUNSW:
        drop_cols += ['start_time_x', 'start_time_y']

    scaler, pca, model = load_simulation_objects(
        aggregated_file, model_type, model_variant
    )
    data = pd.read_csv(aggregated_file)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    print(f"Processing {len(chunks)} chunks in parallel…")

    worker = partial(
        process_chunk,
        drop_cols=drop_cols,
        scaler=scaler,
        pca=pca,  # type: ignore
        model=model,
        model_variant=model_variant,
        model_type=model_type,
        threshold=threshold
    )

    t0 = time.time()
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(worker, chunks)
    print(f"Parallel total time: {time.time() - t0:.4f}s")

    # flatten
    return [pred for chunk_preds in results for pred in chunk_preds]


if __name__ == '__main__':
    args = _parse_args()
    if args.mode == "sequential":
        preds = sequential_simulation(
            args.aggregated_file,
            model_type=args.model_type,
            model_variant=args.model_variant,
            chunk_size=args.chunk_size,
            delay=args.delay,
            threshold=args.threshold,
            isUNSW=args.unsw
        )
        print("Sequential predictions:", preds)
    elif args.mode == "continuous":
        true_labels, preds = continuous_simulation(
            args.aggregated_file,
            model_type=args.model_type,
            model_variant=args.model_variant,
            chunk_size=args.chunk_size,
            window_duration=args.window_duration,
            delay=args.delay,
            threshold=args.threshold,
            isUNSW=args.unsw
        )
        print("Continuous predictions:", preds)
    else:  # parallel
        preds = parallel_simulation(
            args.aggregated_file,
            model_type=args.model_type,
            model_variant=args.model_variant,
            chunk_size=args.chunk_size,
            num_processes=args.num_processes,
            threshold=args.threshold,
            isUNSW=args.unsw
        )
        print("Parallel predictions:", preds)
