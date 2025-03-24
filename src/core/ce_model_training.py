# src/core/ce_model_training
"""
CE Model Training Utilities

This module contains functions for training binary and multiclass
conformal evaluation (CE) models using a variety of classifiers,
including Decision Trees, KNN, Random Forests, SVMs, XGBoost, and
Keras-based feedforward networks.

Training includes preprocessing steps such as numeric feature selection,
standard scaling, and optional PCA dimensionality reduction.

Each model is evaluated on training data, and metrics including accuracy,
precision, recall, and F1 score are logged.

Artifacts are saved to:
    - binary_models/<dataset_name>/ for binary models
    - multi_class_models/<dataset_name>/ for multiclass models
"""
import glob
import logging
import shutil

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Protocol, cast

import joblib
import numpy as np
import pandas as pd
import shortuuid
import torch
import torch.mps
import torch.nn as nn
import xgboost as xgb

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.core.config import ModelVariant, SimulationConfig
from src.core.models.feedforward_binary import FeedForwardBinary
from src.core.perf_stats import PerformanceStats
from src.FIRE.preprocessing import clean_data

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

CE_DROP_COLS: List[str] = [
    "device_id", "session_id", "src_ip", "dst_ip",
    "src_port", "dst_port", "protocol", "timestamp", "Label"
]
FINAL_LOG_COLUMNS: List[str] = [
    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'timestamp', 'flow_duration',
    'flow_byts_s', 'flow_pkts_s', 'fwd_pkts_s', 'bwd_pkts_s', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts',
    'totlen_bwd_pkts', 'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std',
    'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'pkt_len_max', 'pkt_len_min',
    'pkt_len_mean', 'pkt_len_std', 'pkt_len_var', 'fwd_header_len', 'bwd_header_len', 'fwd_seg_size_min',
    'fwd_act_data_pkts', 'flow_iat_mean', 'flow_iat_max', 'flow_iat_min', 'flow_iat_std', 'fwd_iat_tot',
    'fwd_iat_max', 'fwd_iat_min', 'fwd_iat_mean', 'fwd_iat_std', 'bwd_iat_tot', 'bwd_iat_max', 'bwd_iat_min',
    'bwd_iat_mean', 'bwd_iat_std', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
    'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt',
    'down_up_ratio', 'pkt_size_avg', 'init_fwd_win_byts', 'init_bwd_win_byts', 'active_max', 'active_min',
    'active_mean', 'active_std', 'idle_max', 'idle_min', 'idle_mean', 'idle_std', 'fwd_byts_b_avg',
    'fwd_pkts_b_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_blk_rate_avg', 'fwd_seg_size_avg',
    'bwd_seg_size_avg', 'cwr_flag_count', 'subflow_fwd_pkts', 'subflow_bwd_pkts', 'subflow_fwd_byts',
    'subflow_bwd_byts', 'BinLabel'
]
UNSW_DROP_COLS: List[str] = [
    'end_time', 'NUM_PKTS_UP_TO_128_BYTES', 'RETRANSMITTED_OUT_PKTS', 'TCP_FLAGS', 'MAX_IP_PKT_LEN', 'TCP_WIN_MAX_IN',
    'RETRANSMITTED_OUT_BYTES', 'DNS_QUERY_TYPE', 'SRC_TO_DST_AVG_THROUGHPUT', 'RETRANSMITTED_IN_BYTES',
    'FTP_COMMAND_RET_CODE', 'DST_TO_SRC_AVG_THROUGHPUT', 'DURATION_IN', 'NUM_PKTS_128_TO_256_BYTES',
    'DST_TO_SRC_SECOND_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'SHORTEST_FLOW_PKT', 'SRC_TO_DST_SECOND_BYTES',
    'DNS_TTL_ANSWER', 'TCP_WIN_MAX_OUT', 'CLIENT_TCP_FLAGS', 'NUM_PKTS_256_TO_512_BYTES', 'DURATION_OUT',
    'ICMP_IPV4_TYPE', 'MIN_TTL', 'RETRANSMITTED_IN_PKTS', 'LONGEST_FLOW_PKT', 'SERVER_TCP_FLAGS', 'L7_PROTO',
    'NUM_PKTS_512_TO_1024_BYTES', 'DNS_QUERY_ID', 'ICMP_TYPE', 'Attack', 'MIN_IP_PKT_LEN', 'MAX_TTL', 'Label'
]

RANDOM_STATE: int = 42


class SupportsPredict(Protocol):
    def predict(self, X: Any) -> Any:
        raise NotImplementedError


def _unsw_clean(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Removing UNSW-specific columns from the dataset")
    return df.drop(columns=UNSW_DROP_COLS, errors="ignore")


def train_ce_binary(
    config: SimulationConfig,
    flow_path: str,
    perf_stats: PerformanceStats,
    df_log: pd.DataFrame | None = None
) -> Path:
    """
    Train a binary CE model on labeled flow data and save all artifacts.

    This function preprocesses the flow CSV file, selects numeric features,
    encodes the binary label from the "Label" column, and applies scaling.
    PCA is optionally applied to reduce feature dimensionality.
    The trained model and pipeline components (scaler, PCA) are saved.
    After training, model performance on the training data is logged, 
    including accuracy, precision, recall, and F1 score.

    Args:
        flows (str): Path to the CSV file containing flow data with labels.
        variant (ModelVariant): Which model architecture to use. One of:
            - "dt" (Decision Tree)
            - "knn" (K-Nearest Neighbors)
            - "rf" (Random Forest)
            - "svm" (Support Vector Machine)
            - "xgb" (XGBoost)
            - "feedforward" (Keras dense neural net)
        use_pca (bool): If True, apply PCA to reduce feature space to 95% explained variance.

    Raises:
        ValueError: If an unsupported model variant is passed.
    """
    if df_log is None:
        logger.info(f"Training with {flow_path} dataset")
        logger.debug(f"Dataset type for UNSW is set to {config.is_unsw = }")
        df = clean_data(pd.read_csv(flow_path), config.is_unsw)
        dataset = Path(flow_path).parent.name
        outdir = Path("binary_models") / dataset
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        df = df_log
        pattern = f"binary_models/Model_{config.model_variant.value}_Retraining_*"
        for path in glob.glob(pattern):
            if Path(path).is_dir():
                logger.info(f"Removing old retraining directory: {path}")
                shutil.rmtree(path)
        outdir = Path("binary_models") / f"Model_{config.model_variant.value}_Retraining_{shortuuid.ShortUUID().random(length=8)}"  # noqa: E501
        outdir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Output directory for binary model retraining artifacts: {outdir}")

    if config.is_unsw:
        if "BinLabel" not in df.columns:
            if "Label" in df.columns:
                logger.info(
                    "Using UNSW dataset format: mapping 'Label' to binary 'BinLabel'")
                df["BinLabel"] = df["Label"]
            elif "Bin_Label" in df.columns:
                logger.info(
                    "Normalizing column name: renaming 'Bin_Label' -> 'BinLabel'")
                df.rename(columns={"Bin_Label": "BinLabel"}, inplace=True)
            else:
                logger.warning(
                    "Skipping BinLabel creation: neither 'BinLabel' nor 'Label' present. "
                    f"Columns sample: {list(df.columns)[:10]}{' ...' if df.shape[1] > 10 else ''}"
                )
        else:
            logger.debug(
                "'BinLabel' already present; skipping mapping from 'Label'.")
        df = _unsw_clean(df)
        extra_features = set(df.columns) - set(FINAL_LOG_COLUMNS)
        logger.debug(
            f"UNSW features extra ontop of mandatory: {extra_features}")
        if len(extra_features) > 0:
            raise RuntimeError(
                "Diagnose this for retraining to work properly.")

    else:
        if "Label" in df.columns:
            df["BinLabel"] = df["Label"].map(
                {"Benign": 0}).fillna(1).astype(int)
        elif "BinLabel" in df.columns:
            if df["BinLabel"].dtype == object:
                df["BinLabel"] = df["BinLabel"].map({"Benign": 0}).fillna(1)
                logger.debug("")
            non_finite_mask = ~np.isfinite(df["BinLabel"])
            if non_finite_mask.any():
                offending_vals = df.loc[non_finite_mask,
                                        "BinLabel"].head(5).tolist()
                logger.error(
                    f"[train_ce_binary] Non-finite values found in 'BinLabel' before casting to int: {offending_vals}"
                )
                raise ValueError(
                    f"Non-finite values in 'BinLabel': {offending_vals}")

            df["BinLabel"] = df["BinLabel"].astype(int)
        else:
            raise ValueError("Dataset must contain either 'Label' or 'BinLabel' column."
                             f"Columns found: {df.columns.tolist()}"
                             )

    df = df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.drop(columns=CE_DROP_COLS, errors="ignore")

    if config.is_unsw:
        logger.debug(f"UNSW training features: {df.columns.tolist() = }")

    # Start New Code
    label_col = "BinLabel" if "BinLabel" in df.columns else (
        "Label" if "Label" in df.columns else None
    )
    if label_col is None:
        raise ValueError(
            "Dataset must contain either 'Label' or 'BinLabel'. "
            f"Columns found: {df.columns.tolist()}"
        )

    y_series = df[label_col]

    if y_series.dtype == object:
        label_map = {
            "BENIGN": 0, "Benign": 0, "benign": 0, "NORMAL": 0, "Normal": 0, "normal": 0, "0": 0,
            "ATTACK": 1, "Attack": 1, "attack": 1, "MALICIOUS": 1, "Malicious": 1, "malicious": 1, "1": 1,
        }
        y_series = y_series.map(label_map)

    y_series = pd.to_numeric(y_series, errors="coerce")

    invalid = ~np.isfinite(y_series)
    if invalid.any():
        n_bad = int(invalid.sum())
        examples = y_series[invalid].head(5).tolist()
        logger.warning(
            f"Dropping {n_bad} rows with invalid labels before training; {examples = }")
        df = df.loc[~invalid].copy()
        y_series = y_series.loc[~invalid]

    df["BinLabel"] = y_series.astype(int)
    # End new code

    X = df.select_dtypes(include=[np.number]).drop(
        columns=["BinLabel"], errors="ignore")

    y = df["BinLabel"]
    logger.debug(f"Label column 'BinLabel' {y.unique() = }")

    if X.isna().any().any():
        X = X.fillna(X.mean())

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    joblib.dump(scaler, outdir / "scaler_binary.pkl")

    if config.use_pca:
        pca = PCA(n_components=0.95).fit(Xs)
        Xf = pca.transform(Xs)
        joblib.dump(pca, outdir / "pca_binary.pkl")
    else:
        Xf = Xs

    y_pred: NDArray[np.int_] | None = None

    logger.debug(
        f"Training binary classifier model with variant '{config.model_variant.value}'")
    match config.model_variant:
        case ModelVariant.DT:
            model = DecisionTreeClassifier(random_state=RANDOM_STATE)
            model.fit(Xf, y)
        case ModelVariant.KNN:
            model = KNeighborsClassifier()
            model.fit(Xf, y)
        case ModelVariant.RF:
            model = RandomForestClassifier(random_state=RANDOM_STATE)
            model.fit(Xf, y)
        case ModelVariant.SVM:
            model = SVC(kernel="rbf", probability=True,
                        random_state=RANDOM_STATE)
            model.fit(Xf, y)
        case ModelVariant.XGB:
            model = xgb.XGBClassifier(
                objective="binary:logistic", random_state=RANDOM_STATE)
            model.fit(Xf, y)
        case ModelVariant.FEEDFORWARD:
            Xf = np.asarray(Xf, dtype=np.float32)
            y_arr = y.to_numpy().astype(np.float32)

            device = config.device
            logger.info(f"[feedforward] Using device: {device}", )

            torch.manual_seed(RANDOM_STATE)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(RANDOM_STATE)

            X_tensor = torch.from_numpy(Xf)
            Y_tensor = torch.from_numpy(y_arr)
            logger.debug(f"[feedforward] Converted data to tensors: X_tensor.shape={tuple(X_tensor.shape)}, Y_tensor.shape={tuple(Y_tensor.shape)}")  # noqa: E501

            epochs = 20
            N = X_tensor.shape[0]
            batch_size = 2048 if N >= 8192 else 512
            ds = TensorDataset(X_tensor, Y_tensor)
            logger.debug("Dataset ready: N=%d, batch_size=%d", N, batch_size)

            model = FeedForwardBinary(input_dim=Xf.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCEWithLogitsLoss()
            logger.info(
                "FFN Model Training Starting (bs=%d, epochs=%d)", batch_size, epochs)

            model.train()
            logger.debug("FFN Model Train begin")
            for epoch in range(epochs):
                logger.debug("Epoch %d start", epoch)
                rng = np.random.default_rng(RANDOM_STATE + epoch)
                idx = rng.permutation(N).tolist()
                subset = Subset(ds, idx)
                loader = DataLoader(
                    subset, batch_size=batch_size, shuffle=False, num_workers=0)
                logger.debug("Epoch %d loader ready: batches=%d",
                             epoch, len(loader))
                it = iter(loader)
                xb0, yb0 = next(it)
                logger.debug("Epoch %d first batch shapes: %s %s",
                             epoch, tuple(xb0.shape), tuple(yb0.shape))

                running_loss = 0.0
                for b, (xb, yb) in enumerate(loader):
                    logger.debug("Epoch %d batch %d start: xb=%s yb=%s",
                                 epoch, b, tuple(xb.shape), tuple(yb.shape))
                    xb = xb.to(device, non_blocking=False)
                    yb = yb.to(device, non_blocking=False).view(-1).float()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item()) * xb.size(0)

                logger.debug("[feedforward][epoch %02d] loss=%.6f",
                             epoch + 1, running_loss / N)

            try:
                if device.type == "mps":
                    torch.mps.synchronize()
                    logger.debug("MPS synchronized after training")
            except Exception as e:
                logger.debug(f"MPS sync skipped: {e}")

            logger.debug("Eval start")
            model.eval()

            eval_bs = 4096
            eval_loader = DataLoader(
                ds, batch_size=eval_bs, shuffle=False, num_workers=0)
            logger.debug(
                f"Eval loader ready: batches={len(eval_loader)}, eval_bs={eval_bs}")

            probs_chunks = []

            if device.type == "mps":
                logger.debug("Eval on MPS (per-batch sigmoid on MPS)")
                with torch.no_grad():
                    for i, (xb, _) in enumerate(eval_loader):
                        logger.debug(
                            f"Eval batch {i} start: xb_device={xb.device}, xb_shape={tuple(xb.shape)}")
                        xb = xb.to(device, non_blocking=False)
                        logits_i = model(xb)                            # (B,)
                        logger.debug(
                            f"Eval batch {i} logits shape={tuple(logits_i.shape)}")
                        probs_i = torch.sigmoid(logits_i).to(
                            "cpu")
                        logger.debug(
                            f"Eval batch {i} probs shape={tuple(probs_i.shape)}")
                        probs_chunks.append(probs_i.numpy())
            else:
                logger.debug("Eval on CPU (per-batch sigmoid on CPU)")
                try:
                    torch.set_num_threads(1)
                    logger.debug("CPU threads set to 1")
                except Exception as e:
                    logger.debug(f"set_num_threads failed: {e}")
                with torch.no_grad():
                    for i, (xb, _) in enumerate(eval_loader):
                        logger.debug(
                            f"Eval batch {i} start: xb_device={xb.device}, xb_shape={tuple(xb.shape)}")
                        xb = xb.to(dtype=torch.float32).contiguous()
                        logger.debug(
                            f"Eval batch {i} after cast/contig: xb_device={xb.device}, xb_dtype={xb.dtype}, xb_is_contig={xb.is_contiguous()}")  # noqa: E501
                        logits_i = model(xb)                            # (B,)
                        logger.debug(
                            f"Eval batch {i} logits shape={tuple(logits_i.shape)}")
                        probs_i = torch.sigmoid(logits_i)               # (B,)
                        logger.debug(
                            f"Eval batch {i} probs shape={tuple(probs_i.shape)}")
                        probs_chunks.append(probs_i.cpu().numpy())

            y_pred_prob = np.concatenate(probs_chunks, axis=0).reshape(-1)
            logger.debug(
                f"Eval concat complete: y_pred_prob shape={y_pred_prob.shape}")

            y_pred = (y_pred_prob > 0.5).astype(int)

        case _:
            raise ValueError(f"Unknown variant '{config.model_variant.value}'")

    if config.model_variant == ModelVariant.FEEDFORWARD:
        pass
    else:
        sk = cast("SupportsPredict", model)
        y_pred_np = np.asarray(sk.predict(Xf))
        y_pred = y_pred_np.astype(np.int_, copy=False).reshape(-1)

    if y_pred is None:
        raise RuntimeError("Internal error: y_pred not computed for variant "
                           f"{config.model_variant.value!r}")

    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))

    logger.info("Model Performance on Training Data:")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    report = classification_report(y, y_pred, digits=4)
    logger.info("\n%s", report)

    perf_stats.log_classifier_metrics(acc, prec, rec, f1)

    if config.model_variant == ModelVariant.FEEDFORWARD and isinstance(model, nn.Module):
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": int(Xf.shape[1]),
                "dropout": 0.3,
                "random_state": RANDOM_STATE,
            },
            outdir / "feedforward_model_binary.pt",
        )
    else:
        joblib.dump(model, outdir /
                    f"{config.model_variant.value}_model_binary.pkl")

    logging.info(
        f"[ce_models] Trained '{config.model_variant.value}' binary CE model and wrote artifacts to {outdir}/")
    return outdir


def train_ce_multiclass(
    flows_csv: str,
    variant: ModelVariant,
    use_pca: bool = True
) -> None:
    """
    Train a multiclass CE model and persist preprocessing artifacts and model.

    This function drops unnecessary columns, scales numeric features, and
    optionally reduces dimensionality with PCA. It then trains a multiclass
    classifier using one of the supported variants. Artifacts are stored
    under `multi_class_models/<dataset_name>/`.

    After training, macro-averaged metrics (accuracy, precision, recall, F1) are logged on the training data.

    Args:
        flows_csv (str): Path to the CSV file containing labeled CE flow data.
        variant (ModelVariant): Model type to use. Supported options:
            - "dt", "knn", "rf", "svm", "xgb"
            (Note: "feedforward" is unsupported for multiclass classification.)
        use_pca (bool): If True, apply PCA for dimensionality reduction.

    Raises:
        ValueError: If an unsupported variant is given.
        NotImplementedError: If "feedforward" is selected.
    """
    raise NotImplementedError(
        "Multiclass CE model training is currently disabled.")

    if variant == "feedforward":
        raise NotImplementedError(
            "Feedforward is not supported for multiclass CE models.")
    df = pd.read_csv(flows_csv)
    dataset = Path(flows_csv).parent.name
    outdir = Path("multi_class_models") / dataset
    outdir.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=CE_DROP_COLS, errors="ignore")
    y = df["Label"]

    if X.isna().any().any():
        X = X.fillna(X.mean())

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    joblib.dump(scaler, outdir / "scaler_multi.pkl")

    if use_pca:
        pca = PCA(n_components=0.95).fit(Xs)
        Xf = pca.transform(Xs)
        joblib.dump(pca, outdir / "pca_multi.pkl")
    else:
        Xf = Xs

    if variant == "dt":
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    elif variant == "knn":
        model = KNeighborsClassifier()
    elif variant == "rf":
        model = RandomForestClassifier(random_state=RANDOM_STATE)
    elif variant == "svm":
        model = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    elif variant == "xgb":
        model = xgb.XGBClassifier(
            objective="multi:softmax", random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown multiclass variant '{variant}'")

    model.fit(Xf, y)

    y_pred = model.predict(Xf)
    logger.info("Multiclass Model Performance on Training Data:")
    logger.info(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    logger.info(
        f"Macro Precision: {precision_score(y, y_pred, average='macro', zero_division=0):.4f}")
    logger.info(
        f"Macro Recall:    {recall_score(y, y_pred, average='macro', zero_division=0):.4f}")
    logger.info(
        f"Macro F1 Score:  {f1_score(y, y_pred, average='macro', zero_division=0):.4f}")
    report = classification_report(y, y_pred, digits=4)
    logger.info("\n%s", report)

    fname_map = {
        "dt": "decision_tree_multi.pkl",
        "knn": "knearest_multi.pkl",
        "rf": "random_forest_multi.pkl",
        "svm": "svm_multi.pkl",
        "xgb": "xgboost_multi.pkl",
    }
    joblib.dump(model, outdir / fname_map[variant])

    logging.info(
        f"[ce_models] Trained '{variant}' multiclass CE model and wrote artifacts to {outdir}/")


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
        "Use the `train_ce_binary` or `train_ce_multiclass` functions in your application."
    )
