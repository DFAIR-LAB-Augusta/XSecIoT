# src/core/perf_stats
import logging

from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    """
    Stores training metrics across calibrations or classifier retrains.

    Attributes:
        accuracies (List[float]): List of accuracy scores.
        precisions (List[float]): List of precision scores.
        recalls (List[float]): List of recall scores.
        f1s (List[float]): List of F1 scores.
    """
    accuracies: List[float] = field(default_factory=list)
    precisions: List[float] = field(default_factory=list)
    recalls: List[float] = field(default_factory=list)
    f1s: List[float] = field(default_factory=list)

    def log_metrics(self, acc: float, prec: float, rec: float, f1: float) -> None:
        self.accuracies.append(acc)
        self.precisions.append(prec)
        self.recalls.append(rec)
        self.f1s.append(f1)
    
    def summarize_metrics(self) -> None:
        if not self.accuracies:
            logger.warning("No CE training metrics available.")
            return

        logger.info(f"[==OVERALL SIM STATS==] Calibrations: {len(self.accuracies)}")
        logger.info(f"[==OVERALL SIM STATS==] Avg Accuracy: {np.mean(self.accuracies):.4f}")
        logger.info(f"[==OVERALL SIM STATS==] Avg Precision: {np.mean(self.precisions):.4f}")
        logger.info(f"[==OVERALL SIM STATS==] Avg Recall: {np.mean(self.recalls):.4f}")
        logger.info(f"[==OVERALL SIM STATS==] Avg F1 Score: {np.mean(self.f1s):.4f}")
        logger.info(f"[==OVERALL SIM STATS==] Std Accuracy: {np.std(self.accuracies):.4f}")


@dataclass
class PerformanceStats:
    """
    Stores performance metrics collected during CE simulation.

    Attributes:
        iteration_times (List[float]): Duration (in seconds) of each call to _sim_loop,
            typically corresponding to each processed chunk of flow data.
        drift_times (List[float]): Duration (in seconds) for each CE drift detection check,
            measured per row during simulation.
        correct_log (List[bool]): Boolean list indicating whether each prediction
            during simulation matched the ground truth label.
        drift_detected_indices (List[int]): Indices (chunk numbers) where drift was detected.
        drift_intervals (List[int]): Intervals (in chunks) between successive drift detections.
        
    """
    iteration_times: List[float] = field(default_factory=list)
    drift_times: List[float] = field(default_factory=list)
    correct_log: List[bool] = field(default_factory=list)

    drift_detected_indices: List[int] = field(default_factory=list)
    drift_intervals: List[int] = field(default_factory=list)

    ce_stats: ModelStats = field(default_factory=ModelStats)
    classifier_stats: ModelStats = field(default_factory=ModelStats)

    chunk_sizes: List[int] = field(default_factory=list)

    def log_drift(self, current_index: int):
        if self.drift_detected_indices:
            prev_index = self.drift_detected_indices[-1]
            self.drift_intervals.append(current_index - prev_index)
        self.drift_detected_indices.append(current_index)

    def log_ce_metrics(self, acc: float, prec: float, rec: float, f1: float) -> None:
        self.ce_stats.log_metrics(acc, prec, rec, f1)
    
    def log_classifier_metrics(self, acc: float, prec: float, rec: float, f1: float) -> None:
        self.classifier_stats.log_metrics(acc, prec, rec, f1)

    def summarize_ce_metrics(self) -> None:
        logger.info("=== CE Model Calibration Summary ===")
        self.ce_stats.summarize_metrics()
    
    def summarize_classifier_metrics(self) -> None:
        logger.info("=== Classifier Model Performance Summary ===")
        self.classifier_stats.summarize_metrics()


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )