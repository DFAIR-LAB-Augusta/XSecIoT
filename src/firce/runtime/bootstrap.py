import logging
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from firce.ce_model_training import _unsw_clean, train_ce_binary, train_ce_multiclass
from firce.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController
from firce.drift_monitor.base import DriftMonitor
from firce.drift_monitor.factory import build_monitor
from firce.models.feedforward_binary import FeedForwardBinary
from firce.models.mlp_ce import MLP_CE
from firce.runtime.constants import FINAL_LOG_COLUMNS, FULL_DROP_COLS, ROLLING_COLS
from firce.runtime.monitoring import filter_ce_kwargs
from firce.utils.circular_logger import CircularDequeLogger
from firce.utils.config import ModelType, ModelVariant, MonitorType, SimulationConfig
from firce.utils.perf_stats import PerformanceStats
from firce.utils.rolling_csv import RollingCSV
from fire.preprocessing import clean_data
from fire.simulations import load_simulation_objects, preprocess_chunk

logger = logging.getLogger(__name__)


@dataclass
class SimulationRuntime:
    """Mutable runtime state for a simulation run."""

    config: SimulationConfig
    perf_stats: PerformanceStats
    sig_controller: AdaptiveSignificanceController | None
    rolling: RollingCSV | CircularDequeLogger
    scaler: StandardScaler
    pca: PCA | None
    model: ClassifierMixin | xgb.Booster | FeedForwardBinary
    monitor: DriftMonitor | None
    train_df: pd.DataFrame


def initialize_simulation_runtime(config: SimulationConfig) -> SimulationRuntime:
    """
    Build and return a fully initialized simulation runtime.

    Args:
        config: Simulation configuration.

    Returns:
        Fully initialized runtime state.
    """
    sig_controller = create_sig_controller(config)
    perf_stats = create_perf_stats()
    train_df = load_training_frame(config)

    ensure_model_artifacts(config, perf_stats)

    rolling = create_rolling_logger(config)
    seed_rolling_logger(config, rolling, train_df)

    scaler, pca, model = load_runtime_artifacts(config)
    monitor = build_runtime_monitor(
        config=config,
        train_df=train_df,
        scaler=scaler,
        pca=pca,
        model=model,
        sig_controller=sig_controller,
        perf_stats=perf_stats,
    )

    return SimulationRuntime(
        config=config,
        perf_stats=perf_stats,
        sig_controller=sig_controller,
        rolling=rolling,
        scaler=scaler,
        pca=pca,
        model=model,
        monitor=monitor,
        train_df=train_df,
    )


def create_sig_controller(
    config: SimulationConfig,
) -> AdaptiveSignificanceController | None:
    """
    Create the adaptive significance controller if enabled.

    Args:
        config: Simulation configuration.

    Returns:
        Adaptive significance controller or None.
    """
    return AdaptiveSignificanceController() if config.use_ASC else None


def create_perf_stats() -> PerformanceStats:
    """
    Create a new performance statistics tracker.

    Returns:
        Performance statistics tracker.
    """
    return PerformanceStats()


def load_training_frame(config: SimulationConfig) -> pd.DataFrame:
    """
    Load and normalize the aggregated training dataframe.

    Args:
        config: Simulation configuration.

    Returns:
        Cleaned training dataframe.

    Raises:
        RuntimeError: If UNSW columns do not match expected constraints.
    """
    df_train = pd.read_csv(config.aggregated_path)
    df_train = df_train.drop(columns=["device_id", "session_id"], errors="ignore")

    if (
        config.model_type == ModelType.BINARY
        and "BinLabel" not in df_train.columns
        and "Label" in df_train.columns
    ):
        if config.is_unsw:
            df_train["BinLabel"] = df_train["Label"]
        else:
            df_train["BinLabel"] = (
                df_train["Label"].map({"Benign": 0}).fillna(1).astype(int)
            )

    df_train = df_train.drop(columns="Label", errors="ignore")
    df_train = df_train.drop(columns="Unnamed: 0", errors="ignore")

    if config.is_unsw:
        df_train = _unsw_clean(clean_data(df_train, config.is_unsw))
        extra_features = set(df_train.columns) - set(FINAL_LOG_COLUMNS)
        logger.debug("UNSW extra features beyond mandatory set: %s", extra_features)
        if extra_features:
            raise RuntimeError(
                "Unexpected UNSW features found. Diagnose before retraining."
            )

    return df_train


def ensure_model_artifacts(
    config: SimulationConfig,
    perf_stats: PerformanceStats,
) -> None:
    """
    Ensure simulation model artifacts exist by training them if needed.

    Args:
        config: Simulation configuration.
        perf_stats: Performance statistics tracker.
    """
    dataset_name = config.aggregated_path.parent.name

    if config.model_type == ModelType.BINARY:
        start = time.perf_counter()
        train_ce_binary(config, str(config.aggregated_path), perf_stats)
        logger.info(
            "Binary CE training completed in %.4fs",
            time.perf_counter() - start,
        )

    if (
        config.model_variant != ModelVariant.FEEDFORWARD
        and config.model_type == ModelType.MULTI
    ):
        logger.info(
            "CE multiclass artifacts missing for '%s'; training now...",
            dataset_name,
        )
        start = time.perf_counter()
        try:
            train_ce_multiclass(
                config,
                str(config.aggregated_path),
                variant=config.model_variant,
                use_pca=config.use_pca,
            )
            logger.info(
                "Multiclass CE training completed in %.4fs",
                time.perf_counter() - start,
            )
        except NotImplementedError as exc:
            logger.warning(
                "Multiclass CE training not supported for variant '%s'; skipping: %s",
                config.model_variant.value,
                exc,
            )


def create_rolling_logger(
    config: SimulationConfig,
) -> RollingCSV | CircularDequeLogger:
    """
    Create the rolling logger implementation.

    Args:
        config: Simulation configuration.

    Returns:
        Rolling logger instance.
    """
    columns = get_rolling_columns(config)

    if config.use_circular_logger:
        logger.info("Using in-memory CircularDequeLogger.")
        return CircularDequeLogger(None, max_rows=config.max_rows, columns=columns)

    logger.info("Using disk-based RollingCSV.")
    return RollingCSV(str(config.log_path), max_rows=config.max_rows, columns=columns)


def get_seed_drop_columns() -> list[str]:
    """
    Get columns dropped before seeding the rolling logger.

    Returns:
        Columns excluded from seeded rolling history.
    """
    return ["timestamp", "dst_port", "dst_ip", "protocol", "src_ip", "src_port"]


def get_rolling_columns(config: SimulationConfig) -> list[str]:
    """
    Get the rolling logger schema for the given configuration.

    Args:
        config: Simulation configuration.

    Returns:
        Rolling logger column list.
    """
    if config.is_unsw:
        return ROLLING_COLS.copy()

    drop_before_seed = set(get_seed_drop_columns())
    return [col for col in FINAL_LOG_COLUMNS if col not in drop_before_seed]


def build_seed_frame(
    config: SimulationConfig,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the dataframe used to seed the rolling logger.

    Args:
        config: Simulation configuration.
        train_df: Cleaned training dataframe.

    Returns:
        Seed dataframe aligned to rolling schema.
    """
    drop_before_seed = get_seed_drop_columns()
    rolling_cols = get_rolling_columns(config)

    seed_df = (
        train_df.tail(config.max_rows)
        .copy()
        .drop(columns=drop_before_seed, errors="ignore")
    )
    return seed_df.reindex(columns=rolling_cols)


def seed_rolling_logger(
    config: SimulationConfig,
    rolling: RollingCSV | CircularDequeLogger,
    train_df: pd.DataFrame,
) -> None:
    """
    Seed the rolling logger from historical training data.

    Args:
        config: Simulation configuration.
        rolling: Rolling logger instance.
        train_df: Cleaned training dataframe.
    """
    start = time.perf_counter()
    logger.info("Seeding log from aggregated data...")

    seed_df = build_seed_frame(config, train_df)

    if "BinLabel" in seed_df.columns:
        values = seed_df["BinLabel"]
        logger.debug(
            "[pre-clean] BinLabel dtype=%s, n_rows=%d",
            values.dtype,
            len(values),
        )
        logger.debug(
            "[pre-clean] BinLabel nunique(excl NaN)=%d, n_nan=%d",
            values.nunique(dropna=True),
            int(values.isna().sum()),
        )
        logger.debug(
            "[pre-clean] BinLabel unique values (raw): %s",
            list(pd.unique(values)),
        )

    for record in seed_df.tail(config.max_rows).itertuples(index=False, name=None):
        rolling.append(list(record))

    rolling.flush()
    logger.info(
        "Seeded %d rows in %.4fs",
        min(len(train_df), config.max_rows),
        time.perf_counter() - start,
    )
    logger.info("Rolling log initialized with columns: %s", rolling.columns)

    if isinstance(rolling, CircularDequeLogger) and config.use_mlp and config.is_unsw:
        df_log = rolling.to_dataframe().tail(config.max_rows)
        logger.debug("Unique rolling log cols: %s", df_log.columns)
        values = df_log["BinLabel"]
        logger.debug(
            "[pre-clean] BinLabel dtype=%s, n_rows=%d",
            values.dtype,
            len(values),
        )
        logger.debug(
            "[pre-clean] BinLabel nunique(excl NaN)=%d, n_nan=%d",
            values.nunique(dropna=True),
            int(values.isna().sum()),
        )


def load_runtime_artifacts(
    config: SimulationConfig,
) -> tuple[
    StandardScaler,
    PCA | None,
    ClassifierMixin | xgb.Booster | FeedForwardBinary,
]:
    """
    Load scaler, PCA, and model artifacts for simulation.

    Args:
        config: Simulation configuration.

    Returns:
        Tuple of scaler, PCA transformer, and trained model.
    """
    return load_simulation_objects(
        str(config.aggregated_path),
        config.model_type.value,
        config.model_variant.value,
        config.use_pca,
    )


def build_runtime_monitor(
    config: SimulationConfig,
    train_df: pd.DataFrame,
    scaler: StandardScaler,
    pca: PCA | None,
    model: ClassifierMixin | xgb.Booster | FeedForwardBinary,
    sig_controller: AdaptiveSignificanceController | None,
    perf_stats: PerformanceStats,
) -> DriftMonitor | None:
    """
    Build and fit the initial drift monitor.

    Args:
        config: Simulation configuration.
        train_df: Training dataframe used for initial fitting.
        scaler: Fitted scaler.
        pca: Optional PCA transformer.
        model: Trained classifier model.
        sig_controller: Adaptive significance controller.
        perf_stats: Performance statistics tracker.

    Returns:
        Initialized drift monitor, or None if disabled.
    """
    if config.monitor_type == MonitorType.NONE:
        logger.info("No drift monitor enabled; skipping monitor fit")
        return None

    ce_kwargs = filter_ce_kwargs(config) if config.monitor_type == MonitorType.CE else {}

    x_train = preprocess_chunk(train_df.copy(), FULL_DROP_COLS).select_dtypes(
        include=["number"]
    )
    logger.info("Monitor features: %s", x_train.columns)

    x_scaled = scaler.transform(x_train)
    x_monitor = (
        pca.transform(x_scaled)
        if (
            config.monitor_type == MonitorType.CE
            and config.use_pca
            and pca is not None
        )
        else x_scaled
    )

    y_train = (
        train_df["BinLabel"]
        if config.model_type == ModelType.BINARY
        else train_df["Label"]
    )

    monitor_model = _build_monitor_model(
        config=config,
        model=model,
        input_dim=x_monitor.shape[1],
        ce_kwargs=ce_kwargs,
    )

    if config.monitor_type == MonitorType.CE:
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

    start = time.perf_counter()
    monitor.fit(x_monitor, y_train.to_numpy(), perf_stats) if monitor is not None else None
    logger.info("Initial monitor fit in %.4fs", time.perf_counter() - start)
    return monitor


def _build_monitor_model(
    config: SimulationConfig,
    model: ClassifierMixin | xgb.Booster | FeedForwardBinary,
    input_dim: int,
    ce_kwargs: dict[str, Any],
) -> ClassifierMixin | xgb.Booster | FeedForwardBinary | MLP_CE:
    """
    Build the model used internally by the monitor.

    Args:
        config: Simulation configuration.
        model: Main classifier model.
        input_dim: Monitor input dimension.
        ce_kwargs: Filtered conformal evaluator keyword arguments.

    Returns:
        Model instance used by the drift monitor.
    """
    if config.monitor_type != MonitorType.CE:
        return model

    if config.use_svm:
        shrinking = config.max_rows >= 100_000
        return SVC(
            probability=True,
            kernel="linear",
            verbose=False,
            random_state=config.seed,
            shrinking=shrinking,
        )

    if config.use_mlp:
        ce_kwargs.setdefault("n_jobs", 1)
        return MLP_CE(
            input_dim=input_dim,
            widths=tuple(ce_kwargs.get("widths", (256, 128, 64))),
            p_drop=float(ce_kwargs.get("dropout", 0.2)),
            threshold=float(ce_kwargs.get("threshold", 0.5)),
            lr=float(ce_kwargs.get("lr", 1e-3)),
            epochs=int(ce_kwargs.get("epochs", 20)),
            batch_size=ce_kwargs.get("batch_size", None),
            random_state=config.seed,
            device=config.device,
        )

    return model

if __name__ == '__main__':
    raise NotImplementedError('This module is not intended to be run directly. ')
