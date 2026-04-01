from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.conformalEval.conformal_evaluators import ConformalEvaluator
from src.core.drift_monitor.base import DriftDetectionResult
from src.core.perf_stats import PerformanceStats

if TYPE_CHECKING:
    from src.core.config import CEType
    from src.core.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController


class ConformalDriftMonitor:
    """Adapter that exposes FIRCE CE backends through the DriftMonitor interface."""

    def __init__(
        self,
        ce_type: CEType,
        model: Any,
        significance_controller: AdaptiveSignificanceController | None = None,
        **ce_kwargs: Any,
    ) -> None:
        self._evaluator = ConformalEvaluator(
            ce_type,
            model,
            significance_controller=significance_controller,
            **ce_kwargs,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        perf_stats: PerformanceStats | None = None,
    ) -> None:
        if perf_stats is None:
            perf_stats = PerformanceStats()
        self._evaluator.calibrate(X_train, y_train, perf_stats)

    def detect(self, X: np.ndarray) -> DriftDetectionResult:
        flags = np.asarray(self._evaluator.detect_drift(X), dtype=bool).reshape(-1)
        return DriftDetectionResult(
            row_flags=flags,
            chunk_drift=bool(flags.any()),
            scores=None,
            metadata={
                'drift_count': int(flags.sum()),
                'chunk_size': int(len(flags)),
            },
        )
