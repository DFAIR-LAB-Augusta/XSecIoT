# src/core/conformalEval/adaptive_sig_ctlr
import logging

from collections import deque
from typing import Any, Deque, Dict

import numpy as np

from src.core.conformalEval.utils import load_conformal_config

logger = logging.getLogger(__name__)
STATIC_VALS: Dict = load_conformal_config()
DECAY: float = STATIC_VALS["adaptive_significance"]["decay"]
MAX_ALPHA: float = STATIC_VALS["adaptive_significance"]["max_alpha"]
MIN_ALPHA: float = STATIC_VALS["adaptive_significance"]["min_alpha"]
WINDOW_SIZE: int = STATIC_VALS["adaptive_significance"]["window_size"]
ALPHA_STEP: float = STATIC_VALS["adaptive_significance"]["alpha_step"]
INCREASE_THRESHOLD: float = STATIC_VALS["adaptive_significance"]["increase_threshold"]
DECREASE_THRESHOLD: float = STATIC_VALS["adaptive_significance"]["decrease_threshold"]


class AdaptiveSignificanceController:
    """
    Dynamically adjusts significance thresholds based on recent p-values.

    Attributes:
        window_size (int): Number of recent p-values to maintain per class.
        min_alpha (float): Minimum allowed significance level.
        max_alpha (float): Maximum allowed significance level.
        decay (float): Weighting factor for exponential smoothing (0 < decay < 1).
        alpha_step (float): Increment/decrement amount for adaptive thresholds.
        increase_threshold (float): Drift rate threshold to increase alpha.
        decrease_threshold (float): Drift rate threshold to decrease alpha.
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        min_alpha: float = MIN_ALPHA,
        max_alpha: float = MAX_ALPHA,
        decay: float = DECAY,
        alpha_step: float = ALPHA_STEP,
        increase_threshold: float = MAX_ALPHA,
        decrease_threshold: float = MIN_ALPHA
    ):
        self.window_size = window_size
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.decay = decay
        self.alpha_step = alpha_step
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold

        self.pvalue_history: Dict[Any, Deque[float]] = {}
        self.adaptive_thresholds: Dict[Any, float] = {}
        logger.info("Initializing Adaptive Significance Controller")

    def update(self, classes: np.ndarray, p_values: np.ndarray) -> None:
        """
        Update p-value history and recompute thresholds.

        Args:
            classes (np.ndarray): Predicted class labels for the samples.
            p_values (np.ndarray): Corresponding p-values.
        """
        for cls, pval in zip(classes, p_values):
            if cls not in self.pvalue_history:
                self.pvalue_history[cls] = deque(maxlen=self.window_size)
                self.adaptive_thresholds[cls] = self.min_alpha 

            self.pvalue_history[cls].append(pval)

        self._recompute_thresholds()

    def _recompute_thresholds(self) -> None:
        """
        Recompute thresholds using recent drift rate to adapt significance levels.
        """
        for cls, history in self.pvalue_history.items():
            
            current_alpha = self.adaptive_thresholds.get(cls, self.min_alpha)
            history_arr = np.array(history)
            logger.debug(f"[Class {cls}] p-value history: {history_arr}")
            if len(history_arr) == 0:
                continue

            drift_rate = np.mean(history_arr < current_alpha)
            updated_alpha = current_alpha
            logger.debug(f"[Class {cls}] Drift rate: {drift_rate:.4f} — alpha: {current_alpha:.4f}")

            if drift_rate > self.increase_threshold:
                updated_alpha = min(current_alpha + self.alpha_step, self.max_alpha)
                logger.debug(
                    f"[Class {cls}] High drift rate ({drift_rate:.2f}) — increasing alpha to {updated_alpha:.4f}"
                    )
            elif drift_rate < self.decrease_threshold:
                updated_alpha = max(current_alpha - self.alpha_step, self.min_alpha)
                logger.debug(
                    f"[Class {cls}] Low drift rate ({drift_rate:.2f}) — decreasing alpha to {updated_alpha:.4f}"
                )
            else:
                logger.debug(
                    f"[Class {cls}] Stable drift rate ({drift_rate:.2f}) — keeping alpha at {updated_alpha:.4f}"
                )

            self.adaptive_thresholds[cls] = updated_alpha

        summary = {str(cls): round(thresh, 4) for cls, thresh in self.adaptive_thresholds.items()}
        logger.info(f"[AdaptiveSignificanceController] Updated thresholds: {summary}")
        
    def get_thresholds(self) -> Dict[Any, float]:
        """
        Return the current set of adaptive thresholds.

        Returns:
            Dict[Any, float]: Per-class significance thresholds.
        """
        return self.adaptive_thresholds


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )