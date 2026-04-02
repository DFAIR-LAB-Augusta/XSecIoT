from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cade.runtime import CadeRuntimeDetector

from firce.drift_monitor.base import DriftDetectionResult

from .cade_config import CadeMonitorConfig

if TYPE_CHECKING:
    from firce.utils.config import SimulationConfig
    from firce.utils.perf_stats import PerformanceStats


class CadeDriftMonitor:
    def __init__(self, config: SimulationConfig) -> None:
        cade_cfg = CadeMonitorConfig(**config.monitor_kwargs)

        self._detector = CadeRuntimeDetector(
            dims=cade_cfg.dims,
            margin=cade_cfg.margin,
            mad_threshold=cade_cfg.mad_threshold,
            min_drift_ratio=cade_cfg.min_drift_ratio,
            min_drift_count=cade_cfg.min_drift_count,
            batch_size=cade_cfg.batch_size,
            epochs=cade_cfg.epochs,
            lr=cade_cfg.lr,
            cae_lambda_1=cade_cfg.cae_lambda_1,
            similar_ratio=cade_cfg.similar_ratio,
            display_interval=cade_cfg.display_interval,
            force_retrain=cade_cfg.force_retrain,
            weights_path=cade_cfg.weights_path,
            device=cade_cfg.device,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        perf_stats: PerformanceStats | None = None,
    ) -> None:
        self._detector.fit(X_train, y_train)

    def detect(self, X: np.ndarray) -> DriftDetectionResult:
        out = self._detector.detect(X)
        row_flags = np.asarray(out.row_flags, dtype=bool).reshape(-1)
        scores = np.asarray(out.scores, dtype=float).reshape(-1)

        return DriftDetectionResult(
            row_flags=row_flags,
            chunk_drift=bool(row_flags.any()),
            scores=scores,
            metadata={
                'drift_count': int(row_flags.sum()),
                'chunk_size': int(len(row_flags)),
                'drift_ratio': float(row_flags.mean()) if len(row_flags) else 0.0,
            },
        )
