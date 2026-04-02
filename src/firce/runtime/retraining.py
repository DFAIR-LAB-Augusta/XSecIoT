import logging
import time
import warnings

from typing import Any

import joblib
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from firce.ce_model_training import train_ce_binary
from firce.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController
from firce.drift_monitor.base import DriftMonitor
from firce.models.feedforward_binary import FeedForwardBinary
from firce.runtime.constants import FULL_DROP_COLS
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.config import ModelType, ModelVariant, MonitorType, SimulationConfig
from firce.utils.perf_stats import PerformanceStats
from firce.utils.rolling_csv import RollingCSV
from fire.preprocessing import clean_data
from fire.simulations import preprocess_chunk
import logging
import warnings

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch

from firce.ce_model_training import train_ce_binary
from firce.runtime.bootstrap import SimulationRuntime
from firce.runtime.constants import FULL_DROP_COLS
from firce.models.feedforward_binary import FeedForwardBinary
from firce.utils.circular_logger import CircularDequeLogger
from fire.preprocessing import clean_data
from fire.simulations import preprocess_chunk
logger = logging.getLogger(__name__)


def retrain(
    config: SimulationConfig,
    scaler: StandardScaler,
    pca: PCA | None,
    model: Any,
    monitor: DriftMonitor | None,
    rolling: RollingCSV | CircularDequeLogger,
    perf_stats: PerformanceStats,
    _sig_controller: AdaptiveSignificanceController | None = None,
) -> tuple[StandardScaler, PCA | None, Any, DriftMonitor | None]:
    """
    Retrain model and CE using the latest samples from the rolling log file.
    This overwrites the existing trained model artifacts.

    Args:
        config (SimulationConfig): Simulation configuration.
        scaler (StandardScaler): Current scaler (to be replaced).
        pca (Optional[PCA]): Current PCA object (to be replaced).
        model (Any): Current model (to be replaced).
        ce (ConformalEvaluator): Current CE object (to be replaced).

    Returns:
        Tuple of updated (scaler, pca, model, ce).
    """
    if monitor is None:
        raise RuntimeError('CE is disabled; retraining should not have been triggered.')

    start = time.perf_counter()
    if isinstance(rolling, CircularDequeLogger):
        df_log = rolling.to_dataframe().tail(config.max_rows)
        logging.debug('Retraining model using last %d rows of the in-memory circular log', len(df_log))
    else:
        df_log = pd.read_csv(config.log_path, compression='gzip').tail(config.max_rows)

    vals = df_log['BinLabel']

    logger.debug(f'[pre-clean] BinLabel dtype={vals.dtype}, n_rows={len(vals)}')
    logger.debug(f'[pre-clean] BinLabel nunique(excl NaN)={vals.nunique(dropna=True)}, n_nan={int(vals.isna().sum())}')
    uniques = pd.unique(vals)
    logger.debug(f'[pre-clean] BinLabel unique values (raw): {list(uniques)}')
    logger.debug('Retraining model using last %d rows of the rolling log', len(df_log))

    if config.is_unsw:
        ce_columns = [
            'totlen_bwd_pkts',
            'tot_bwd_pkts',
            'totlen_fwd_pkts',
            'tot_fwd_pkts',
            'flow_duration',
            'fwd_iat_min',
            'fwd_iat_max',
            'fwd_iat_mean',
            'fwd_iat_std',
            'bwd_iat_min',
            'bwd_iat_max',
            'bwd_iat_mean',
            'bwd_iat_std',
            'fwd_pkt_len_mean',
            'bwd_pkt_len_mean',
            'pkt_len_mean',
            'flow_iat_mean',
            'down_up_ratio',
            'fwd_iat_tot',
            'bwd_iat_tot',
        ]
        to_drop = set(df_log.columns) - set(ce_columns) - set(['Label', 'BinLabel'])
        df_log = df_log.drop(columns=to_drop)

    model_dir = train_ce_binary(config, config.log_path.as_posix(), perf_stats, df_log)

    scaler = joblib.load(model_dir / 'scaler_binary.pkl')

    if config.use_pca:
        pca = joblib.load(model_dir / 'pca_binary.pkl')
    else:
        pca = None

    if config.model_variant == ModelVariant.FEEDFORWARD:
        logger.debug(f'Loading Torch feedforward model from {model_dir / "feedforward_model_binary.pt"}')
        ckpt = torch.load(model_dir / 'feedforward_model_binary.pt', map_location='cpu')

        input_dim = int(ckpt.get('input_dim'))
        p_drop = float(ckpt.get('dropout', 0.3))
        state_dict = ckpt['state_dict']

        logger.debug(f'Rebuilding FeedForwardBinary(input_dim={input_dim}, p_drop={p_drop}) on device={config.device}')

        model = FeedForwardBinary(input_dim=input_dim, p_drop=p_drop)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug(f'Missing keys while loading state_dict: {missing}')
        if unexpected:
            logger.debug(f'Unexpected keys while loading state_dict: {unexpected}')

        model.to(config.device)
        model.eval()
        logger.debug('Torch feedforward model loaded and set to eval()')
    else:
        model = joblib.load(model_dir / f'{config.model_variant.value}_model_binary.pkl')
        logger.debug(f'Loaded sklearn model from {model_dir / f"{config.model_variant.value}_model_binary.pkl"}')

    clean = clean_data(df_log, config.is_unsw)
    X_df = preprocess_chunk(clean, FULL_DROP_COLS).select_dtypes(include=['number'])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='X does not have valid feature names, but StandardScaler was fitted with feature names'
        )
        Xs = scaler.transform(X_df)
    if config.use_pca and pca is not None:
        Xp = pca.transform(Xs)
        logger.debug('PCA applied to retraining data before CE calibration.')
        logger.debug(f'Input to PCA (Xs) has shape: {Xs.shape}')
        logger.debug(f'Output from PCA (Xp) has shape: {Xp.shape}')
        logger.debug(f'PCA was fit with {pca.n_components_} components')
    else:
        Xp = Xs
    y = clean['BinLabel'] if config.model_type == ModelType.BINARY else clean['Label']

    if y.nunique() < 2:
        logger.warning('Only one class (%s) found in retrain data — skipping retrain.', y.unique())
        return scaler, pca, model, monitor
    elif len(y.unique()) > 2:
        logger.warning(f'More than 2 unique values in y: {y.unique()}')

    logger.debug(f'[pre-clean] BinLabel dtype={y.dtype}, n_rows={len(y)}')
    logger.debug(f'[pre-clean] BinLabel nunique(excl NaN)={y.nunique(dropna=True)}, n_nan={int(y.isna().sum())}')

    uniques = pd.unique(y)
    logger.debug(f'[pre-clean] BinLabel unique values (raw): {list(uniques)}')

    if monitor is not None:
        X_monitor = (
            pca.transform(Xs) if (config.monitor_type == MonitorType.CE and config.use_pca and pca is not None) else Xs
        )
        monitor.fit(X_monitor, y.to_numpy(), perf_stats)

    logger.debug('Retraining complete in %.4fs', time.perf_counter() - start)
    return scaler, pca, model, monitor

logger = logging.getLogger(__name__)


def retrain_runtime(runtime: SimulationRuntime) -> None:
    """
    Retrain model artifacts from the latest rolling log and update runtime in place.

    Args:
        runtime: Mutable simulation runtime.

    Raises:
        RuntimeError: If retraining was triggered without an active monitor.
    """
    if runtime.monitor is None:
        raise RuntimeError("Monitor is disabled; retraining should not be triggered.")

    df_log = _load_retraining_frame(runtime)

    if runtime.config.is_unsw:
        df_log = _prune_unsw_retraining_frame(df_log)

    model_dir = train_ce_binary(
        runtime.config,
        runtime.config.log_path.as_posix(),
        runtime.perf_stats,
        df_log,
    )

    scaler, pca, model = _load_retrained_artifacts(runtime, model_dir)
    runtime.scaler = scaler
    runtime.pca = pca
    runtime.model = model

    _fit_monitor_on_retrained_data(runtime, df_log)


def _load_retraining_frame(runtime: SimulationRuntime) -> pd.DataFrame:
    """
    Load retraining data from the rolling logger.

    Args:
        runtime: Mutable simulation runtime.

    Returns:
        Retraining dataframe.
    """
    if isinstance(runtime.rolling, CircularDequeLogger):
        df_log = runtime.rolling.to_dataframe().tail(runtime.config.max_rows)
        logger.debug(
            "Retraining model using last %d rows from in-memory circular log",
            len(df_log),
        )
    else:
        df_log = pd.read_csv(runtime.config.log_path, compression="gzip").tail(
            runtime.config.max_rows
        )
        logger.debug(
            "Retraining model using last %d rows from disk log",
            len(df_log),
        )

    values = df_log["BinLabel"]
    logger.debug("[pre-clean] BinLabel dtype=%s, n_rows=%d", values.dtype, len(values))
    logger.debug(
        "[pre-clean] BinLabel nunique(excl NaN)=%d, n_nan=%d",
        values.nunique(dropna=True),
        int(values.isna().sum()),
    )
    logger.debug(
        "[pre-clean] BinLabel unique values (raw): %s",
        list(pd.unique(values)),
    )
    return df_log


def _prune_unsw_retraining_frame(df_log: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce UNSW retraining dataframe to CE-compatible columns.

    Args:
        df_log: Retraining dataframe.

    Returns:
        Pruned dataframe.
    """
    ce_columns = [
        "totlen_bwd_pkts",
        "tot_bwd_pkts",
        "totlen_fwd_pkts",
        "tot_fwd_pkts",
        "flow_duration",
        "fwd_iat_min",
        "fwd_iat_max",
        "fwd_iat_mean",
        "fwd_iat_std",
        "bwd_iat_min",
        "bwd_iat_max",
        "bwd_iat_mean",
        "bwd_iat_std",
        "fwd_pkt_len_mean",
        "bwd_pkt_len_mean",
        "pkt_len_mean",
        "flow_iat_mean",
        "down_up_ratio",
        "fwd_iat_tot",
        "bwd_iat_tot",
    ]
    to_drop = set(df_log.columns) - set(ce_columns) - {"Label", "BinLabel"}
    return df_log.drop(columns=list(to_drop))


def _load_retrained_artifacts(
    runtime: SimulationRuntime,
    model_dir: Path,
) -> tuple[Any, Any, Any]:
    """
    Load retrained scaler, PCA, and model artifacts.

    Args:
        runtime: Mutable simulation runtime.
        model_dir: Directory containing trained artifacts.

    Returns:
        Tuple of scaler, pca, and model.
    """
    scaler = joblib.load(model_dir / "scaler_binary.pkl")
    pca = joblib.load(model_dir / "pca_binary.pkl") if runtime.config.use_pca else None

    if runtime.config.model_variant.value == "feedforward":
        logger.debug(
            "Loading Torch feedforward model from %s",
            model_dir / "feedforward_model_binary.pt",
        )
        checkpoint = torch.load(
            model_dir / "feedforward_model_binary.pt",
            map_location="cpu",
        )
        input_dim = int(checkpoint.get("input_dim"))
        p_drop = float(checkpoint.get("dropout", 0.3))
        state_dict = checkpoint["state_dict"]

        model = FeedForwardBinary(input_dim=input_dim, p_drop=p_drop)
        model.load_state_dict(state_dict, strict=False)
        model.to(runtime.config.device)
        model.eval()
    else:
        model = joblib.load(
            model_dir / f"{runtime.config.model_variant.value}_model_binary.pkl"
        )

    return scaler, pca, model


def _fit_monitor_on_retrained_data(
    runtime: SimulationRuntime,
    df_log: pd.DataFrame,
) -> None:
    """
    Refit the drift monitor on retrained rolling data.

    Args:
        runtime: Mutable simulation runtime.
        df_log: Retraining dataframe.
    """
    clean_df = clean_data(df_log, runtime.config.is_unsw)
    x_df = preprocess_chunk(clean_df, FULL_DROP_COLS).select_dtypes(include=["number"])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, but StandardScaler "
                "was fitted with feature names"
            ),
        )
        x_scaled = runtime.scaler.transform(x_df)

    if runtime.config.use_pca and runtime.pca is not None:
        x_monitor = runtime.pca.transform(x_scaled)
    else:
        x_monitor = x_scaled

    y = clean_df["BinLabel"]

    if y.nunique() < 2:
        logger.warning(
            "Only one class (%s) found in retrain data; skipping monitor refit.",
            y.unique(),
        )
        return
    if runtime.monitor is not None:
        runtime.monitor.fit(x_monitor, y.to_numpy(), runtime.perf_stats)