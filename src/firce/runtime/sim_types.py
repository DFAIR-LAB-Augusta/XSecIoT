from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import xgboost as xgb

    from sklearn.base import ClassifierMixin
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from firce.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController
    from firce.drift_monitor.base import DriftMonitor
    from firce.models.feedforward_binary import FeedForwardBinary
    from firce.utils.circular_logger import CircularDequeLogger
    from firce.utils.config import SimulationConfig
    from firce.utils.perf_stats import PerformanceStats
    from firce.utils.rolling_csv import RollingCSV


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
    label_encoder: Any | None = None
    llm_backend: Any | None = None
    novelty_reports: list = field(default_factory=list)
