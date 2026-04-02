import logging
import time
import warnings

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from firce.models.feedforward_binary import FeedForwardBinary
from firce.runtime.constants import (
    DROP_COLS,
    FULL_DROP_COLS,
    PRED_THRESHOLD,
    ROLLING_COLS,
)
from firce.runtime.retraining import retrain_runtime
from firce.runtime.sim_types import SimulationRuntime
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.config import ModelType, ModelVariant, MonitorType, SimulationConfig
from fire.preprocessing import clean_data
from fire.simulations import preprocess_chunk

logger = logging.getLogger(__name__)


def predict_row(
    row: pd.DataFrame,
    drop_cols: list[str],
    scaler: StandardScaler,
    pca: PCA | None,
    config: SimulationConfig,
    model: ClassifierMixin | xgb.Booster | FeedForwardBinary,
    threshold: float,
) -> int:
    """
    Predict on a single preprocessed row using the CE model pipeline.

    Args:
        row (pd.Series): One row of raw CE flow data.
        drop_cols (List[str]): Columns to drop before prediction.
        scaler (StandardScaler): Pre-fitted scaler.
        pca (Optional[PCA]): Pre-fitted PCA model (optional).
        model (ClassifierMixin | xgb.Booster): Trained model.
        threshold (float): Threshold for binarizing predictions (for FNN).

    Returns:
        int | str: Predicted class label or name.

    Raises:
        TypeError: If the model type is unsupported for prediction.
    """
    row_df = row.drop(columns=drop_cols, errors='ignore')
    row_df = row_df.select_dtypes(include=[np.number])

    if hasattr(scaler, 'feature_names_in_'):
        feat = list(scaler.feature_names_in_)
        means = getattr(scaler, 'mean_', None)
        arr = []
        for idx, name in enumerate(feat):
            if name in row_df.columns:
                arr.append(row_df[name].item())
            else:
                fill = means[idx] if means is not None else 0.0
                arr.append(fill)
        X_row = np.array(arr).reshape(1, -1)
    else:
        X_row = row_df.to_numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='X does not have valid feature names, but StandardScaler was fitted with feature names'
        )
        X_s = scaler.transform(X_row)
    X_p = pca.transform(X_s) if config.use_pca and pca is not None else X_s

    if config.model_variant == ModelVariant.XGB and isinstance(model, xgb.Booster):
        fnames = [f'f_{i}' for i in range(X_p.shape[1])]
        dtest = xgb.DMatrix(X_p, feature_names=fnames)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='X does not have valid feature names, but StandardScaler was fitted with feature names',
            )
            pred = model.predict(dtest)[0]
        return int(pred)

    if config.model_variant == ModelVariant.FEEDFORWARD and isinstance(model, FeedForwardBinary):
        dev = config.device
        xt = torch.from_numpy(np.asarray(X_p, dtype=np.float32)).to(dev)
        model.eval()
        with torch.no_grad():
            logits = model(xt)
            prob = torch.sigmoid(logits).flatten().item()
        logger.debug(f'[predict_row][ff] prob={prob:.6f}, thr={threshold}')
        return int(prob > threshold)

    if hasattr(model, 'predict'):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='X does not have valid feature names, but StandardScaler was fitted with feature names',
            )
            pred = model.predict(X_p)  # type: ignore
        out = int(np.asarray(pred).reshape(-1)[0])
        logger.debug(f'[predict_row][sk] pred={out}')
        return out

    raise TypeError(f'Unsupported model type for prediction: {type(model)}')


logger = logging.getLogger(__name__)


def process_chunk(
    runtime: SimulationRuntime,
    chunk: pd.DataFrame,
    chunk_num: int,
) -> None:
    """
    Process a single simulation chunk.

    Args:
        runtime: Mutable simulation runtime.
        chunk: Flow chunk to process.
        chunk_num: Chunk index.
    """
    start_iter = time.perf_counter()

    clean_chunk, ground_truth = _prepare_chunk(runtime, chunk)
    _process_chunk_rows(runtime, clean_chunk, ground_truth)

    drift_detected = _detect_chunk_drift(runtime, clean_chunk)
    if drift_detected:
        _handle_detected_drift(runtime, chunk_num)

    runtime.perf_stats.iteration_times.append(time.perf_counter() - start_iter)


def _prepare_chunk(
    runtime: SimulationRuntime,
    chunk: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Clean a chunk and extract optional ground truth.

    Args:
        runtime: Mutable simulation runtime.
        chunk: Raw chunk dataframe.

    Returns:
        Tuple of cleaned chunk and optional ground-truth series.
    """
    logger.debug('Chunk initially has %d columns', len(chunk.columns))
    clean_chunk = clean_data(chunk, False)
    logger.debug('Chunk has %d columns post-cleaning', len(clean_chunk.columns))

    ground_truth = clean_chunk['BinLabel'].reset_index(drop=True) if 'BinLabel' in clean_chunk.columns else None

    if 'BinLabel' in clean_chunk.columns:
        clean_chunk = clean_chunk.drop(columns=['BinLabel'])

    if runtime.config.is_unsw:
        to_drop = clean_chunk.columns.difference(ROLLING_COLS[:-1])
        clean_chunk = clean_chunk.drop(columns=to_drop)
        logger.debug('Chunk has %d columns post-column drop', len(clean_chunk.columns))

    logger.debug('Chunk has rows %d', clean_chunk.shape[0])
    return clean_chunk, ground_truth


def _process_chunk_rows(
    runtime: SimulationRuntime,
    clean_chunk: pd.DataFrame,
    ground_truth: pd.Series | None,
) -> None:
    """
    Process all rows in a cleaned chunk.

    Args:
        runtime: Mutable simulation runtime.
        clean_chunk: Cleaned chunk dataframe.
        ground_truth: Optional true binary labels.
    """
    for row_index in range(clean_chunk.shape[0]):
        start = time.perf_counter()
        raw_row = clean_chunk.iloc[row_index]

        if raw_row.isnull().all():
            logger.warning('Row %d is empty.', row_index)

        x_row = _build_row_features(raw_row)

        if x_row.empty:
            logger.warning('Row %d is empty after preprocessing.', row_index)

        row_to_log = x_row.copy()
        prediction = predict_row(
            x_row,
            DROP_COLS,
            runtime.scaler,
            runtime.pca,
            runtime.config,
            runtime.model,
            PRED_THRESHOLD,
        )

        if prediction not in [0, 1]:
            logger.error('Row %d prediction: %r', row_index, prediction)

        logger.debug('Classified row in %.4fs', time.perf_counter() - start)

        _record_prediction_outcome(
            runtime=runtime,
            row_index=row_index,
            raw_row=raw_row,
            row_to_log=row_to_log,
            prediction=prediction,
            ground_truth=ground_truth,
        )
        _append_row_to_rolling_log(runtime, row_to_log)


def _build_row_features(raw_row: pd.Series) -> pd.DataFrame:
    """
    Convert a raw row into numeric model-ready features.

    Args:
        raw_row: Raw flow row.

    Returns:
        Single-row feature dataframe.
    """
    return preprocess_chunk(
        pd.DataFrame([raw_row]),
        FULL_DROP_COLS,
    ).select_dtypes(include=['number'])


def _record_prediction_outcome(
    runtime: SimulationRuntime,
    row_index: int,
    raw_row: pd.Series,
    row_to_log: pd.DataFrame,
    prediction: int,
    ground_truth: pd.Series | None,
) -> None:
    """
    Record prediction output and update metrics.

    Args:
        runtime: Mutable simulation runtime.
        row_index: Row index within the chunk.
        raw_row: Raw row before preprocessing.
        row_to_log: Row to be logged.
        prediction: Predicted label.
        ground_truth: Optional ground-truth labels.
    """
    if runtime.config.model_type == ModelType.BINARY:
        row_to_log['BinLabel'] = prediction
        logger.debug('Row %d prediction: %r', row_index, prediction)

        if ground_truth is not None:
            true_value = ground_truth.iloc[row_index]
            is_correct = prediction == true_value
            runtime.perf_stats.correct_log.append(is_correct)

            logger.debug(
                '[Index %d] Predicted=%s, Actual=%s',
                row_index,
                prediction,
                true_value,
            )
            if not is_correct:
                logger.info(
                    '[Incorrect] Predicted=%s, Actual=%s',
                    prediction,
                    true_value,
                )
                logger.debug('Row %d details: %s', row_index, raw_row.to_json())
    else:
        row_to_log['Label'] = prediction


def _append_row_to_rolling_log(
    runtime: SimulationRuntime,
    row_to_log: pd.DataFrame,
) -> None:
    """
    Append a predicted row to the rolling logger.

    Args:
        runtime: Mutable simulation runtime.
        row_to_log: Row to append.
    """
    if runtime.config.is_unsw:
        _append_unsw_row(runtime, row_to_log)
        return

    runtime.rolling.append(row_to_log.iloc[0].to_list())


def _append_unsw_row(
    runtime: SimulationRuntime,
    row_to_log: pd.DataFrame,
) -> None:
    """
    Append a UNSW row using the constrained logger schema.

    Args:
        runtime: Mutable simulation runtime.
        row_to_log: Row to append.

    Raises:
        ValueError: If BinLabel is invalid.
        AssertionError: If logger schema does not match expected schema.
    """
    allowed = ROLLING_COLS
    logger_obj = runtime.rolling

    if (
        isinstance(logger_obj, CircularDequeLogger)
        and hasattr(logger_obj, 'columns')
        and logger_obj.columns is not None
    ):
        assert list(logger_obj.columns) == allowed, (
            f'[rolling] Logger schema mismatch: logger has {len(logger_obj.columns)} cols, allowed has {len(allowed)}'
        )

    series = row_to_log.iloc[0]

    extras = [column for column in series.index if column not in allowed]
    kept = [column for column in allowed if column in series.index]
    missing = [column for column in allowed if column not in series.index]

    if extras and not runtime.config.use_mlp:
        logger.warning(
            '[rolling] dropping extras: %s%s',
            extras[:10],
            ' ...' if len(extras) > 10 else '',
        )
        logger.info(
            '[rolling] cols before=%d, kept=%d, dropped=%d, missing=%d',
            len(series.index),
            len(kept),
            len(extras),
            len(missing),
        )

    pruned = series.reindex(index=allowed)
    pruned['BinLabel'] = _coerce_binary_label(pruned['BinLabel'])

    assert len(pruned) == len(allowed), f'[rolling] row width mismatch: {len(pruned)} vs expected {len(allowed)}'
    logger_obj.append(pruned.tolist())


def _coerce_binary_label(value: object) -> int:
    """
    Coerce a raw binary label into integer 0 or 1.

    Args:
        value: Raw label value.

    Returns:
        Integer binary label.

    Raises:
        ValueError: If label cannot be coerced to 0 or 1.
    """
    if isinstance(value, str):
        label_map = {
            'BENIGN': 0,
            'Benign': 0,
            'benign': 0,
            'NORMAL': 0,
            'Normal': 0,
            'normal': 0,
            '0': 0,
            'ATTACK': 1,
            'Attack': 1,
            'attack': 1,
            'MALICIOUS': 1,
            'Malicious': 1,
            'malicious': 1,
            '1': 1,
        }
        value = label_map.get(value, value)

    numeric = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    if pd.isna(numeric) or numeric not in (0, 1):
        raise ValueError(f'[rolling] Bad BinLabel before append: raw={value!r} -> coerced={numeric}')
    return int(numeric)


def _detect_chunk_drift(
    runtime: SimulationRuntime,
    clean_chunk: pd.DataFrame,
) -> bool:
    """
    Detect drift for a processed chunk.

    Args:
        runtime: Mutable simulation runtime.
        clean_chunk: Cleaned chunk dataframe.

    Returns:
        True if drift was detected, else False.
    """
    if runtime.monitor is None:
        logger.warning('Drift monitor is disabled; skipping drift detection and retraining.')
        return False

    start = time.perf_counter()
    x_monitor = _prepare_monitor_chunk_features(runtime, clean_chunk)
    drift_result = runtime.monitor.detect(x_monitor)
    runtime.perf_stats.drift_times.append(time.perf_counter() - start)
    return bool(drift_result is not None and drift_result.chunk_drift)


def _prepare_monitor_chunk_features(
    runtime: SimulationRuntime,
    clean_chunk: pd.DataFrame,
) -> np.ndarray:
    """
    Prepare chunk features for drift monitoring.

    Args:
        runtime: Mutable simulation runtime.
        clean_chunk: Cleaned chunk dataframe.

    Returns:
        Monitor-ready feature matrix.
    """
    if runtime.config.is_unsw:
        x_chunk = preprocess_chunk(clean_chunk, FULL_DROP_COLS)
    else:
        x_chunk = preprocess_chunk(clean_chunk, FULL_DROP_COLS).select_dtypes(include=['number'])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=('X does not have valid feature names, but StandardScaler was fitted with feature names'),
        )
        expected = (
            list(runtime.scaler.get_feature_names_out())
            if hasattr(runtime.scaler, 'get_feature_names_out')
            else list(runtime.scaler.feature_names_in_)
        )

        extra = [column for column in x_chunk.columns if column not in expected]
        if extra:
            logger.debug(
                'Dropping unseen features: %s%s',
                extra[:10],
                ' ...' if len(extra) > 10 else '',
            )
            x_chunk = x_chunk.drop(columns=extra)

        x_chunk = x_chunk.reindex(columns=expected)
        x_scaled = runtime.scaler.transform(x_chunk)

    if runtime.config.monitor_type == MonitorType.CE and runtime.config.use_pca and runtime.pca is not None:
        return runtime.pca.transform(x_scaled)
    return x_scaled


def _handle_detected_drift(
    runtime: SimulationRuntime,
    chunk_num: int,
) -> None:
    """
    Handle a detected drift event by retraining runtime artifacts.

    Args:
        runtime: Mutable simulation runtime.
        chunk_num: Current chunk index.
    """
    runtime.perf_stats.log_drift(chunk_num)
    logger.info('Drift detected in the chunk. Retraining model and recalibrating CE...')
    retrain_runtime(runtime)
