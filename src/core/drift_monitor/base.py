from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np

    from src.core.perf_stats import PerformanceStats


@dataclass(slots=True)
class DriftDetectionResult:
    """Standardized drift detection output for all monitor backends."""

    row_flags: np.ndarray
    chunk_drift: bool
    scores: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DriftMonitor(Protocol):
    """Protocol implemented by all FIRCE drift monitor backends."""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        perf_stats: PerformanceStats | None = None,
    ) -> None:
        """Fit or recalibrate the monitor on training data."""
        ...

    def detect(self, X: np.ndarray) -> DriftDetectionResult:
        """Detect drift on a chunk of samples."""
        ...
