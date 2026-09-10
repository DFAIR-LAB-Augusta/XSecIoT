import logging
import warnings

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch

from firce.ce_model_training import train_ce_binary, train_ce_multiclass
from firce.models.feedforward_binary import FeedForwardBinary
from firce.models.feedforward_multiclass import FeedForwardMulticlass
from firce.runtime.bootstrap import SimulationRuntime
from firce.runtime.constants import FULL_DROP_COLS, _label_column
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.config import ModelType
from fire.preprocessing import clean_data
from fire.simulations import preprocess_chunk

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
        raise RuntimeError('Monitor is disabled; retraining should not be triggered.')

    df_log = _load_retraining_frame(runtime)

    if runtime.config.is_unsw:
        df_log = _prune_unsw_retraining_frame(df_log)

    if runtime.config.model_type == ModelType.BINARY:
        model_dir = train_ce_binary(
            runtime.config,
            runtime.config.log_path.as_posix(),
            runtime.perf_stats,
            df_log,
        )
    else:
        model_dir = train_ce_multiclass(
            runtime.config,
            runtime.config.log_path.as_posix(),
            variant=runtime.config.model_variant,
            use_pca=runtime.config.use_pca,
            df_log=df_log,
        )

    scaler, pca, model, label_encoder = _load_retrained_artifacts(runtime, model_dir)
    runtime.scaler = scaler
    runtime.pca = pca
    runtime.model = model
    runtime.label_encoder = label_encoder

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
            'Retraining model using last %d rows from in-memory circular log',
            len(df_log),
        )
    else:
        df_log = pd.read_csv(runtime.config.log_path, compression='gzip').tail(runtime.config.max_rows)
        logger.debug(
            'Retraining model using last %d rows from disk log',
            len(df_log),
        )

    label_col = _label_column(runtime.config.model_type)
    values = df_log[label_col]
    logger.debug('[pre-clean] %s dtype=%s, n_rows=%d', label_col, values.dtype, len(values))
    logger.debug(
        '[pre-clean] %s nunique(excl NaN)=%d, n_nan=%d',
        label_col,
        values.nunique(dropna=True),
        int(values.isna().sum()),
    )
    logger.debug(
        '[pre-clean] %s unique values (raw): %s',
        label_col,
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
    to_drop = set(df_log.columns) - set(ce_columns) - {'Label', 'BinLabel', 'MC_Label', 'Attack'}
    return df_log.drop(columns=list(to_drop))


def _load_retrained_artifacts(
    runtime: SimulationRuntime,
    model_dir: Path,
) -> tuple[Any, Any, Any, Any]:
    """
    Load retrained scaler, PCA, model, and label encoder artifacts.

    Args:
        runtime: Mutable simulation runtime.
        model_dir: Directory containing trained artifacts.

    Returns:
        Tuple of scaler, pca, model, and label encoder (None for binary).
    """
    suffix = 'binary' if runtime.config.model_type == ModelType.BINARY else 'multi'
    scaler = joblib.load(model_dir / f'scaler_{suffix}.pkl')
    pca = joblib.load(model_dir / f'pca_{suffix}.pkl') if runtime.config.use_pca else None
    label_encoder = (
        joblib.load(model_dir / 'label_encoder_multi.pkl') if runtime.config.model_type == ModelType.MULTI else None
    )

    if runtime.config.model_variant.value == 'feedforward':
        ckpt_path = model_dir / f'feedforward_model_{suffix}.pt'
        logger.debug('Loading Torch feedforward model from %s', ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        input_dim = int(checkpoint.get('input_dim'))
        p_drop = float(checkpoint.get('dropout', 0.3))
        state_dict = checkpoint['state_dict']

        if runtime.config.model_type == ModelType.BINARY:
            model = FeedForwardBinary(input_dim=input_dim, p_drop=p_drop)
        else:
            num_classes = int(checkpoint['num_classes'])
            model = FeedForwardMulticlass(input_dim=input_dim, num_classes=num_classes, p_drop=p_drop)
        model.load_state_dict(state_dict, strict=False)
        model.to(runtime.config.device)
        model.eval()
    else:
        model = joblib.load(model_dir / f'{runtime.config.model_variant.value}_model_{suffix}.pkl')

    return scaler, pca, model, label_encoder


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
    x_df = preprocess_chunk(clean_df, FULL_DROP_COLS).select_dtypes(include=['number'])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=('X does not have valid feature names, but StandardScaler was fitted with feature names'),
        )
        x_scaled = runtime.scaler.transform(x_df)

    if runtime.config.use_pca and runtime.pca is not None:
        x_monitor = runtime.pca.transform(x_scaled)
    else:
        x_monitor = x_scaled

    label_col = _label_column(runtime.config.model_type)
    y = clean_df[label_col]

    if y.nunique() < 2:
        logger.warning(
            'Only one class (%s) found in retrain data; skipping monitor refit.',
            y.unique(),
        )
        return
    if runtime.monitor is not None:
        runtime.monitor.fit(x_monitor, y.to_numpy(), runtime.perf_stats)
