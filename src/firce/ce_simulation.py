# src/core/ce_simulation
"""
CE Simulation Pipeline

This module implements the full conformal evaluation (CE) simulation loop
for detecting distribution drift and adapting ML-based intrusion detection
models in streaming IoT network data.

Key Features:
- Seeds a rolling log from historical training data.
- Loads and calibrates a conformal evaluator (ICE, CCE, or Approx-TCE).
- Streams new flow data in chunks and classifies each instance.
- Optionally detects drift using CE and retrains the model pipeline if enabled.
- Logs predictions, drift flags, and optionally writes logs to disk.
- Saves and logs drift-triggering data chunks for debugging.
- Evaluates and logs model performance metrics (accuracy, precision, recall, F1) during training and retraining.

The simulation can be configured via CLI arguments and supports multiple model
variants (e.g., KNN, SVM, Random Forest, XGBoost, Feedforward NN) and CE strategies.
"""

import inspect
import logging
import time
import warnings

from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from pydantic import ValidationError
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from firce.adaptive_chunking import AdaptiveChunkController
from firce.ce_model_training import _unsw_clean, train_ce_binary, train_ce_multiclass
from firce.config import CEType, ModelType, ModelVariant, MonitorType, SimulationConfig
from firce.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController
from firce.conformalEval.approx_cce import ApproxCrossConformalEvaluator
from firce.conformalEval.cce import CrossConformalEvaluator
from firce.conformalEval.ice import InductiveConformalEvaluator
from firce.conformalEval.tce import ApproximateTransductiveConformalEvaluator
from firce.drift_monitor.base import DriftMonitor
from firce.drift_monitor.factory import build_monitor
from firce.models.feedforward_binary import FeedForwardBinary
from firce.models.mlp_ce import MLP_CE
from firce.perf_stats import PerformanceStats
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.grapher import graph_results
from firce.utils.logger import configure_sim_logging
from firce.utils.parser import parse_sim_args
from firce.utils.rolling_csv import RollingCSV
from FIRE.preprocessing import clean_data
from FIRE.simulations import (
    load_simulation_objects,
    preprocess_chunk,
)

PRED_THRESHOLD: float = 0.5
DROP_COLS: List[str] = [
    'Label',
    'src_ip',
    'dst_ip',
    'start_time',
    'end_time_x',
    'end_time_y',
    'time_diff',
    'time_diff_seconds',
    'Attack',
]
CE_EXTRA_DROP: List[str] = ['device_id', 'session_id', 'src_port', 'dst_port', 'protocol', 'timestamp', 'BinLabel']
FULL_DROP_COLS: List[str] = DROP_COLS + CE_EXTRA_DROP
FINAL_LOG_COLUMNS: List[str] = [
    'src_ip',
    'dst_ip',
    'src_port',
    'dst_port',
    'protocol',
    'timestamp',
    'flow_duration',
    'flow_byts_s',
    'flow_pkts_s',
    'fwd_pkts_s',
    'bwd_pkts_s',
    'tot_fwd_pkts',
    'tot_bwd_pkts',
    'totlen_fwd_pkts',
    'totlen_bwd_pkts',
    'fwd_pkt_len_max',
    'fwd_pkt_len_min',
    'fwd_pkt_len_mean',
    'fwd_pkt_len_std',
    'bwd_pkt_len_max',
    'bwd_pkt_len_min',
    'bwd_pkt_len_mean',
    'bwd_pkt_len_std',
    'pkt_len_max',
    'pkt_len_min',
    'pkt_len_mean',
    'pkt_len_std',
    'pkt_len_var',
    'fwd_header_len',
    'bwd_header_len',
    'fwd_seg_size_min',
    'fwd_act_data_pkts',
    'flow_iat_mean',
    'flow_iat_max',
    'flow_iat_min',
    'flow_iat_std',
    'fwd_iat_tot',
    'fwd_iat_max',
    'fwd_iat_min',
    'fwd_iat_mean',
    'fwd_iat_std',
    'bwd_iat_tot',
    'bwd_iat_max',
    'bwd_iat_min',
    'bwd_iat_mean',
    'bwd_iat_std',
    'fwd_psh_flags',
    'bwd_psh_flags',
    'fwd_urg_flags',
    'bwd_urg_flags',
    'fin_flag_cnt',
    'syn_flag_cnt',
    'rst_flag_cnt',
    'psh_flag_cnt',
    'ack_flag_cnt',
    'urg_flag_cnt',
    'ece_flag_cnt',
    'down_up_ratio',
    'pkt_size_avg',
    'init_fwd_win_byts',
    'init_bwd_win_byts',
    'active_max',
    'active_min',
    'active_mean',
    'active_std',
    'idle_max',
    'idle_min',
    'idle_mean',
    'idle_std',
    'fwd_byts_b_avg',
    'fwd_pkts_b_avg',
    'bwd_byts_b_avg',
    'bwd_pkts_b_avg',
    'fwd_blk_rate_avg',
    'bwd_blk_rate_avg',
    'fwd_seg_size_avg',
    'bwd_seg_size_avg',
    'cwr_flag_count',
    'subflow_fwd_pkts',
    'subflow_bwd_pkts',
    'subflow_fwd_byts',
    'subflow_bwd_byts',
    'BinLabel',
]

UNSW_DROP_COLUMNS: List[str] = [
    'bwd_pkt_len_max',
    'idle_max',
    'bwd_iat_tot',
    'rst_flag_cnt',
    'ece_flag_cnt',
    'fwd_seg_size_avg',
    'subflow_fwd_byts',
    'bwd_pkts_s',
    'bwd_psh_flags',
    'subflow_fwd_pkts',
    'fwd_iat_min',
    'tot_fwd_pkts',
    'timestamp',
    'pkt_len_min',
    'idle_std',
    'flow_iat_mean',
    'bwd_pkts_b_avg',
    'flow_iat_std',
    'flow_byts_s',
    'fwd_pkts_b_avg',
    'bwd_pkt_len_std',
    'bwd_blk_rate_avg',
    'subflow_bwd_byts',
    'active_mean',
    'pkt_len_std',
    'pkt_size_avg',
    'fwd_psh_flags',
    'totlen_fwd_pkts',
    'dst_port',
    'totlen_bwd_pkts',
    'fin_flag_cnt',
    'fwd_iat_tot',
    'src_ip',
    'active_std',
    'flow_iat_min',
    'psh_flag_cnt',
    'bwd_iat_max',
    'fwd_iat_std',
    'flow_pkts_s',
    'fwd_blk_rate_avg',
    'pkt_len_var',
    'protocol',
    'idle_mean',
    'bwd_header_len',
    'active_max',
    'fwd_pkt_len_mean',
    'fwd_pkts_s',
    'bwd_pkt_len_mean',
    'active_min',
    'fwd_seg_size_min',
    'fwd_pkt_len_min',
    'cwr_flag_count',
    'fwd_act_data_pkts',
    'bwd_seg_size_avg',
    'bwd_urg_flags',
    'bwd_iat_min',
    'urg_flag_cnt',
    'fwd_pkt_len_max',
    'flow_iat_max',
    'fwd_pkt_len_std',
    'syn_flag_cnt',
    'subflow_bwd_pkts',
    'bwd_byts_b_avg',
    'idle_min',
    'bwd_pkt_len_min',
    'init_fwd_win_byts',
    'bwd_iat_mean',
    'bwd_iat_std',
    'init_bwd_win_byts',
    'src_port',
    'tot_bwd_pkts',
    'down_up_ratio',
    'flow_duration',
    'fwd_byts_b_avg',
    'ack_flag_cnt',
    'fwd_urg_flags',
    'fwd_header_len',
    'dst_ip',
    'pkt_len_mean',
    'fwd_iat_mean',
    'fwd_iat_max',
    'pkt_len_max',
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def _simulate(
    config: SimulationConfig,
) -> None:
    """
    Run a streaming conformal evaluation (CE) simulation on incoming data flows.

    This method seeds the rolling log from historical training data, loads the model
    and CE evaluator, and processes new flows in sequential chunks. For each chunk,
    the method applies classification and conformal prediction, detects drift, and
    if drift is detected, triggers retraining and recalibration using the updated log.

    Accuracy and drift statistics are computed and plotted over time.

    Args:
        config (SimulationConfig):
            Simulation configuration object with model, CE, PCA, logging, and file paths.
    """
    overall = time.perf_counter()
    sig_controller = AdaptiveSignificanceController() if config.use_ASC else None
    perf_stats = PerformanceStats()

    df_train = pd.read_csv(config.aggregated_path)
    df_train = df_train.drop(columns=['device_id', 'session_id'], errors='ignore')
    if config.model_type == ModelType.BINARY and 'BinLabel' not in df_train.columns and 'Label' in df_train.columns:
        if config.is_unsw:
            df_train['BinLabel'] = df_train['Label']
        else:
            df_train['BinLabel'] = df_train['Label'].map({'Benign': 0}).fillna(1).astype(int)
    df_train = df_train.drop(columns='Label', errors='ignore')
    df_train = df_train.drop(columns='Unnamed: 0', errors='ignore')

    if config.is_unsw:
        df_train = _unsw_clean(clean_data(df_train, config.is_unsw))
        extra_features = set(df_train.columns) - set(FINAL_LOG_COLUMNS)
        logger.debug(f'UNSW features extra ontop of mandatory: {extra_features}')
        if len(extra_features) > 0:
            raise RuntimeError('Diagnose this for retraining to work properly. ')

    _ensure_models_exist(config, perf_stats)

    t_seed = time.perf_counter()
    logger.info('Seeding log from aggregated data...')

    if config.use_circular_logger:
        logger.info('Using in-memory CircularDequeLogger.')
        LoggerCls = CircularDequeLogger
        log_dir = None
    else:
        logger.info('Using disk-based RollingCSV.')
        LoggerCls = RollingCSV
        log_dir = str(config.log_path)

    DROP_BEFORE_SEED = ['timestamp', 'dst_port', 'dst_ip', 'protocol', 'src_ip', 'src_port']
    seed_df = df_train.tail(config.max_rows).copy().drop(columns=DROP_BEFORE_SEED, errors='ignore')
    rolling_cols = [c for c in FINAL_LOG_COLUMNS if c not in DROP_BEFORE_SEED]

    if config.is_unsw:
        rolling_cols = [
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
            'BinLabel',
        ]

    with LoggerCls(log_dir, max_rows=config.max_rows, columns=rolling_cols) as rolling:
        seed_df = seed_df.reindex(columns=rolling_cols)
        vals = seed_df['BinLabel']
        logger.debug(f'[pre-clean] BinLabel dtype={vals.dtype}, n_rows={len(vals)}')
        logger.debug(
            f'[pre-clean] BinLabel nunique(excl NaN)={vals.nunique(dropna=True)}, n_nan={int(vals.isna().sum())}'
        )
        uniques = pd.unique(vals)
        logger.debug(f'[pre-clean] BinLabel unique values (raw): {list(uniques)}')

        for rec in seed_df.tail(config.max_rows).itertuples(index=False, name=None):
            rolling.append(list(rec))

        rolling.flush()
        logger.info(f'Seeded {min(len(df_train), config.max_rows)} rows in {time.perf_counter() - t_seed:.4f}s')
        logger.info(f'Rolling log initialized with columns: {rolling.columns}')

        if isinstance(rolling, CircularDequeLogger) and config.use_mlp and config.is_unsw:
            df_log = rolling.to_dataframe().tail(config.max_rows)
            logger.debug(f'Unique rolling log cols: {df_log.columns}')
            vals = df_log['BinLabel']

            logger.debug(f'[pre-clean] BinLabel dtype={vals.dtype}, n_rows={len(vals)}')
            logger.debug(
                f'[pre-clean] BinLabel nunique(excl NaN)={vals.nunique(dropna=True)}, n_nan={int(vals.isna().sum())}'
            )

        scaler, pca, model = load_simulation_objects(
            str(config.aggregated_path), config.model_type.value, config.model_variant.value, config.use_pca
        )

        clean_tr = df_train.copy()

        monitor: DriftMonitor | None = None

        if config.monitor_type != MonitorType.NONE:
            ce_kwargs = _filter_ce_kwargs(config) if config.monitor_type == MonitorType.CE else {}
            Xtr = preprocess_chunk(clean_tr, FULL_DROP_COLS).select_dtypes(include=['number'])
            logger.info(f'Monitor features: {Xtr.columns}')

            Xs = scaler.transform(Xtr)
            X_monitor = (
                pca.transform(Xs)
                if (config.monitor_type == MonitorType.CE and config.use_pca and pca is not None)
                else Xs
            )

            ytr = clean_tr['BinLabel'] if config.model_type == ModelType.BINARY else clean_tr['Label']

            input_dim = X_monitor.shape[1]

            if config.monitor_type == MonitorType.CE:
                if config.use_svm:
                    if config.max_rows >= 100_000:
                        monitor_model = SVC(
                            probability=True,
                            kernel='linear',
                            verbose=False,
                            random_state=config.seed,
                            shrinking=True,
                        )
                    else:
                        monitor_model = SVC(
                            probability=True,
                            kernel='linear',
                            verbose=False,
                            random_state=config.seed,
                            shrinking=False,
                        )
                elif config.use_mlp:
                    monitor_model = MLP_CE(
                        input_dim=input_dim,
                        widths=tuple(ce_kwargs.get('widths', (256, 128, 64))),
                        p_drop=float(ce_kwargs.get('dropout', 0.2)),
                        threshold=float(ce_kwargs.get('threshold', 0.5)),
                        lr=float(ce_kwargs.get('lr', 1e-3)),
                        epochs=int(ce_kwargs.get('epochs', 20)),
                        batch_size=ce_kwargs.get('batch_size', None),
                        random_state=config.seed,
                        device=config.device,
                    )
                    ce_kwargs.setdefault('n_jobs', 1)
                else:
                    monitor_model = model

                monitor = build_monitor(
                    config=config,
                    model=monitor_model,
                    significance_controller=sig_controller,
                )
            else:
                monitor = build_monitor(
                    config=config,
                    model=None,
                    significance_controller=None,
                )

            tci = time.perf_counter()
            if monitor is not None:
                monitor.fit(X_monitor, ytr.to_numpy(), perf_stats)
            logger.info(
                'Initial monitor fit in %.4fs',
                time.perf_counter() - tci,
            )
        else:
            logger.info('No drift monitor enabled; skipping monitor fit')

        if config.use_adaptive_chunking:
            chunker = AdaptiveChunkController(config.adaptive_chunk_config)
            logger.info('[AdaptiveChunking] Enabled. Initial chunk size: %d', chunker.get_chunk_size())

            flow_data = pd.read_csv(config.flows_path, iterator=True, chunksize=1_000_000)
            current_df = pd.DataFrame()

            chunk_index = 0
            for big_batch in flow_data:
                current_df = pd.concat([current_df, big_batch], ignore_index=True)

                while len(current_df) >= chunker.get_chunk_size():
                    chunk_size = chunker.get_chunk_size()
                    chunk, current_df = current_df.iloc[:chunk_size], current_df.iloc[chunk_size:]

                    _sim_loop(
                        config,
                        rolling,
                        scaler,
                        pca,
                        model,
                        monitor if config.ce_type != CEType.NONE else None,
                        chunk,
                        perf_stats,
                        sig_controller,
                        chunk_index,
                    )

                    drift_occurred = (
                        len(perf_stats.drift_detected_indices) > 0
                        and perf_stats.drift_detected_indices[-1] == chunk_index
                    )
                    chunker.update(drift_occurred, perf_stats)
                    chunk_index += 1
        else:
            for chunkNum, chunk in enumerate(pd.read_csv(config.flows_path, chunksize=config.chunk_size)):
                _sim_loop(config, rolling, scaler, pca, model, monitor, chunk, perf_stats, sig_controller, chunkNum)

    graph_results(config, overall, perf_stats)


def _ensure_models_exist(config: SimulationConfig, perf_stats: PerformanceStats) -> None:
    """
    Train and save CE models if artifacts are missing on disk.

    This function always trains the classifier pipeline (binary or multiclass),
    regardless of CE usage, to ensure the model is available for inference.

    Args:
        config (SimulationConfig): Configuration containing model type and variant.
    """
    ds = config.aggregated_path.parent.name
    if config.model_type == ModelType.BINARY:
        t0 = time.perf_counter()
        train_ce_binary(config, str(config.aggregated_path), perf_stats)
        logger.info('Binary CE training completed in %.4fs', time.perf_counter() - t0)

    if config.model_variant != ModelVariant.FEEDFORWARD and config.model_type == ModelType.MULTI:
        logger.info(f"CE-multiclass artifacts missing for '{ds}'; training now…")
        t0 = time.perf_counter()
        try:
            train_ce_multiclass(
                config, str(config.aggregated_path), variant=config.model_variant, use_pca=config.use_pca
            )
            logger.info(
                f'Multiclass CE training completed in {time.perf_counter() - t0:.4f}s',
            )
        except NotImplementedError as e:
            logger.warning(
                f"Multiclass CE training not supported for variant '{config.model_variant.value}'; "
                f'skipping training: {e}',
            )


def _filter_ce_kwargs(config: SimulationConfig) -> dict[str, Any]:
    """
    Extract only valid keyword arguments for the given CE type's constructor.
    If ce_type is "none", returns an empty dictionary.

    This filters the `config.ce_kwargs` dictionary to retain only those parameters
    accepted by the selected Conformal Evaluator's constructor (`__init__`).

    Args:
        config (SimulationConfig): Current simulation configuration object.

    Returns:
        dict[str, Any]: Valid CE constructor keyword arguments.

    Raises:
        RuntimeError: If CE is disabled ('none') and this function is incorrectly called.
    """
    if config.ce_type == CEType.NONE:
        raise RuntimeError("CE is disabled (ce_type='none'); no CE kwargs should be requested.")

    impl_map = {
        'ice': InductiveConformalEvaluator,
        'cce': CrossConformalEvaluator,
        'approx_tce': ApproximateTransductiveConformalEvaluator,
        'approx_cce': ApproxCrossConformalEvaluator,
    }
    impl_cls = impl_map[config.ce_type.value]
    sig = inspect.signature(impl_cls.__init__)
    return {k: v for k, v in config.ce_kwargs.items() if k in sig.parameters}


def _sim_loop(
    config: SimulationConfig,
    rolling: RollingCSV | CircularDequeLogger,
    scaler: StandardScaler,
    pca: Optional[PCA],
    model: ClassifierMixin | xgb.Booster | FeedForwardBinary,
    monitor: DriftMonitor | None,
    chunk: pd.DataFrame,
    perf_stats: PerformanceStats,
    sig_controller: Optional[AdaptiveSignificanceController] = None,
    chunkNum: int = 0,
) -> None:
    """
    Process a single chunk of flows in the streaming CE simulation loop.

    For each row in the chunk, this function:
      - Cleans and preprocesses the data.
      - Applies classification and stores predictions.
      - Optionally evaluates prediction correctness.
      - Appends processed records to the rolling log.
      - Applies CE-based drift detection if enabled.
      - Triggers model retraining and CE recalibration on drift detection.

    Args:
        config (SimulationConfig): Full simulation configuration object.
        rolling (RollingCSV | CircularDequeLogger): Logger for maintaining a rolling flow log.
        scaler (StandardScaler): Scaler used to normalize input features.
        pca (Optional[PCA]): PCA transformer applied during training, if used.
        model (ClassifierMixin | xgb.Booster): Trained classifier model.
        ce (Optional[ConformalEvaluator]): Conformal evaluator for drift detection.
        chunk (pd.DataFrame): Chunk of flow records to simulate.
        perf_stats (PerformanceStats): Object tracking prediction and drift statistics.
        sig_controller (Optional[AdaptiveSignificanceController], optional):
            Adaptive significance threshold controller. Defaults to None.
        chunkNum (int, optional): Index of the current chunk. Used in logging. Defaults to 0.

    Raises:
        ValueError: If the length of a logged row does not match the logger’s expected columns.
    """
    start_iter = time.perf_counter()

    logging.debug(f'Chunk initially has {len(chunk.columns)} columns')
    clean_ch = clean_data(chunk, False)
    logging.debug(f'Chunk has {len(clean_ch.columns)} columns post-cleaning')

    ground_truth = clean_ch['BinLabel'].reset_index(drop=True) if 'BinLabel' in clean_ch.columns else None
    if 'BinLabel' in clean_ch.columns:
        clean_ch = clean_ch.drop(columns=['BinLabel'])

    if config.is_unsw:
        to_drop = clean_ch.columns.difference(FINAL_LOG_COLUMNS)
        clean_ch = clean_ch.drop(columns=to_drop)
        logging.debug(f'Chunk has {len(clean_ch.columns)} columns post-column drop')

    logging.debug(f'Chunk has rows {clean_ch.shape[0]}')

    for i in range(clean_ch.shape[0]):
        tc = time.perf_counter()
        raw_row = clean_ch.iloc[i]

        if raw_row.isnull().all():
            logging.warning(f'Row {i} is empty.')

        X_row = preprocess_chunk(pd.DataFrame([raw_row]), FULL_DROP_COLS).select_dtypes(include=['number'])

        if X_row.empty:
            logging.warning(f'Row {i} is empty after preprocessing.')

        row_to_log = X_row.copy()
        pred_raw = _predict_row(X_row, DROP_COLS, scaler, pca, config, model, PRED_THRESHOLD)
        if pred_raw not in [0, 1]:
            logging.error(f'Row {i} prediction: {pred_raw!r}')
        logger.debug(f'Classified row in {time.perf_counter() - tc:.4f}s')

        if config.model_type == ModelType.BINARY:
            label_col = 'BinLabel'
            logging.debug(f'Row {i} prediction: {pred_raw!r}')

            row_to_log[label_col] = pred_raw

            if ground_truth is not None:
                true_val = ground_truth.iloc[i]
                is_correct = pred_raw == true_val
                perf_stats.correct_log.append(is_correct)

                logger.debug(f'[Index {i}] Predicted={pred_raw}, Actual={true_val}')
                if not is_correct:
                    logger.info(f'[Incorrect] Predicted={pred_raw}, Actual={true_val}')
                    logger.debug(f'Row {i} details: {raw_row.to_json()}')
        else:
            label_col = 'Label'
            row_to_log[label_col] = pred_raw

        if config.is_unsw:
            rolling_cols = [
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
                'BinLabel',
            ]
            allowed = rolling_cols

            if isinstance(rolling, CircularDequeLogger) and hasattr(rolling, 'columns') and rolling.columns is not None:
                assert list(rolling.columns) == allowed, (
                    f'[rolling] Logger schema mismatch: logger has {len(rolling.columns)} cols, allowed has {len(allowed)}'
                )

            s = row_to_log.iloc[0]

            extras = [c for c in s.index if c not in allowed]
            kept = [c for c in allowed if c in s.index]
            missing = [c for c in allowed if c not in s.index]

            if extras and not config.use_mlp:
                logger.warning(f'[rolling] dropping extras: {extras[:10]}{" ..." if len(extras) > 10 else ""}')

                logger.info(
                    f'[rolling] cols before={len(s.index)}, kept={len(kept)}, dropped={len(extras)}, missing={len(missing)}'
                )

            s_pruned = s.reindex(index=allowed)

            raw_bl = s_pruned['BinLabel']
            if isinstance(raw_bl, str):
                map_dict = {
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
                raw_bl = map_dict.get(raw_bl, raw_bl)

            bl_num = pd.to_numeric(pd.Series([raw_bl]), errors='coerce').iloc[0]

            if pd.isna(bl_num) or bl_num not in (0, 1):
                preview = {k: s_pruned[k] for k in allowed[:5]}
                raise ValueError(
                    f'[rolling] Bad BinLabel before append: raw={s_pruned["BinLabel"]} -> coerced={bl_num} '
                    f'(allowed only 0/1). First-5 fields preview={preview}'
                )

            s_pruned['BinLabel'] = int(bl_num)

            assert len(s_pruned) == len(allowed), (
                f'[rolling] row width mismatch: {len(s_pruned)} vs expected {len(allowed)}'
            )
            rolling.append(s_pruned.tolist())
        else:
            rolling.append(row_to_log.iloc[0].to_list())

    if monitor is None:
        logger.warning('Drift monitor is disabled; skipping drift detection and retraining.')
        drift_result = None
    else:
        start_drift = time.perf_counter()

        if config.is_unsw:
            X_chunk = preprocess_chunk(clean_ch, FULL_DROP_COLS)
        else:
            X_chunk = preprocess_chunk(clean_ch, FULL_DROP_COLS).select_dtypes(include=['number'])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='X does not have valid feature names, but StandardScaler was fitted with feature names',
            )
            expected = (
                list(scaler.get_feature_names_out())
                if hasattr(scaler, 'get_feature_names_out')
                else list(scaler.feature_names_in_)
            )

            extra = [c for c in X_chunk.columns if c not in expected]
            if extra:
                logger.debug(f'Dropping unseen features: {extra[:10]}{" ..." if len(extra) > 10 else ""}')
                X_chunk = X_chunk.drop(columns=extra)

            X_chunk = X_chunk.reindex(columns=expected)
            Xs = scaler.transform(X_chunk)

        X_monitor = (
            pca.transform(Xs) if (config.monitor_type == MonitorType.CE and config.use_pca and pca is not None) else Xs
        )

        drift_result = monitor.detect(X_monitor)
        perf_stats.drift_times.append(time.perf_counter() - start_drift)
    if drift_result is not None and drift_result.chunk_drift:
        perf_stats.log_drift(chunkNum)
        logger.info('Drift detected in the chunk. Retraining model and recalibrating CE...')
        scaler, pca, model, monitor = _retrain(config, scaler, pca, model, monitor, rolling, perf_stats, sig_controller)

    elapsed_iter = time.perf_counter() - start_iter
    perf_stats.iteration_times.append(elapsed_iter)


def _predict_row(
    row: pd.DataFrame,
    drop_cols: List[str],
    scaler: StandardScaler,
    pca: Optional[PCA],
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


def _retrain(
    config: SimulationConfig,
    scaler: StandardScaler,
    pca: Optional[PCA],
    model: Any,
    monitor: DriftMonitor | None,
    rolling: RollingCSV | CircularDequeLogger,
    perf_stats: PerformanceStats,
    sig_controller: Optional[AdaptiveSignificanceController] = None,
) -> Tuple[StandardScaler, Optional[PCA], Any, DriftMonitor | None]:
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


def main() -> None:
    """
    Parse CLI arguments, initialize configuration, and launch the CE simulation.
    """
    args = parse_sim_args()
    try:
        config = SimulationConfig(
            model_type=args.modelType,
            model_variant=args.modelVariant,
            ce_type=args.ceType,
            aggregated_path=args.aggregated_file,
            flows_path=args.flows_file,
            ce_kwargs={'folds': 5, 'significance': 0.05, 'random_state': args.seed},
            chunk_size=args.chunk_size,
            use_pca=args.use_pca,
            use_ASC=args.useASC,
            use_circular_logger=args.useCircularLogger,
            debug=args.debug,
            log_to_file=args.log2File,
            max_rows=args.max_rows,
            use_svm=args.useSVM,
            use_adaptive_chunking=args.useAC,
            is_unsw=args.unsw,
            use_mlp=args.useMLP,
            seed=args.seed,
            runNum=args.runNum,
            monitor_type=args.monitorType,
            monitor_kwargs=(
                {
                    'dims': args.cadeDims,
                    'margin': args.cadeMargin,
                    'mad_threshold': args.cadeMadThreshold,
                    'min_drift_ratio': args.cadeMinDriftRatio,
                    'min_drift_count': args.cadeMinDriftCount,
                    'batch_size': args.cadeBatchSize,
                    'epochs': args.cadeEpochs,
                    'lr': args.cadeLr,
                    'cae_lambda_1': args.cadeLambda1,
                    'similar_ratio': args.cadeSimilarRatio,
                    'display_interval': args.cadeDisplayInterval,
                    'force_retrain': args.cadeForceRetrain,
                    'weights_path': args.cadeWeightsPath,
                    'device': args.cadeDevice,
                }
                if args.monitorType == MonitorType.CADE
                else {}
            ),
        )
    except ValidationError as e:
        logging.error(e)
        raise

    logger.info(f'Simulation configuration: {config}')
    configure_sim_logging(config)
    logger.info(f'Simulation configuration: {config}')
    _simulate(config)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception('Fatal error during CE simulation: %s', str(e))
        raise


# TODO
"""
- Add print out for detection of malicous data -> Done
- Use labels to show if working -> Done
- Process data row by row for CE
- Label dataset to for precise simulation -> Done
- Use toml for config -> Done 
- Update CE classes to use utils functions -> Done

- Test new updates -> IP

- Update READMEs
"""

# FIXME
"""
- UNSW/CIC dataset support
- Update README with new features
- Multiclass CE w/ XAI
- PCA Support for CE
- FFN & KNN support for CE
"""
