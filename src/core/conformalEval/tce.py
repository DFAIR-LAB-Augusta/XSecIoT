# src.core.conformalEval.tce
"""
Approximate Transductive Conformal Evaluator (TCE)

This module provides an implementation of an approximate version of
transductive conformal evaluation for concept drift detection. It uses
the full training set for both fitting and calibration, approximating
transductive inference while retaining scalability.

Classes:
    - ApproximateTransductiveConformalEvaluator: Implements calibration and p-value estimation using
      the full training set as pseudo-calibration data.
"""
# src.core.conformalEval.approx_cce
import logging

from typing import Any, Dict, Optional

import numpy as np

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from src.core.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController
from src.core.conformalEval.utils import compute_p_values, load_conformal_config
from src.core.perf_stats import PerformanceStats

logger = logging.getLogger(__name__)

STATIC_VALS: Dict = load_conformal_config()
SIGNIFICANCE = STATIC_VALS["conformal_eval_config"]["significance"]


class ApproximateTransductiveConformalEvaluator:
    """
    Approximate Transductive Conformal Evaluator (Approx-TCE).

    This evaluator approximates the transductive conformal approach by
    training on the entire dataset and reusing it for calibration.
    It avoids true transductive inference by simplifying the calibration step
    to a single pass over the training data.

    Attributes:
        model (Any): ML model with fit() and predict_proba() methods.
        significance (float): Quantile level used for computing thresholds.
        calibration_scores (Optional[Dict[Any, np.ndarray]]): Class-wise calibration score arrays.
        thresholds (Optional[Dict[Any, float]]): Class-wise threshold values.
    """

    def __init__(
        self,
        model: Any,
        significance: float = SIGNIFICANCE,
        significance_controller: Optional[AdaptiveSignificanceController] = None,
        **kwargs,
    ):
        """
        Initialize the Approx-TCE evaluator.

        Args:
            model (Any): A scikit-learn-compatible model supporting `fit()` and `predict_proba()`.
            significance (float): Significance level for thresholding (default: 0.05).
        """
        if kwargs:
            logger.debug(f"ICE: ignoring unsupported kwargs: {sorted(kwargs)}")
        self.model = model
        self.significance = significance
        self.significance_controller = significance_controller
        self.calibration_scores: Optional[Dict[Any, np.ndarray]] = None
        self.thresholds: Optional[Dict[Any, float]] = None

    def calibrate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        perf_stats: PerformanceStats
    ) -> None:
        """
        Train model on all available data and compute calibration thresholds.

        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features).
            y (np.ndarray): Corresponding class labels (n_samples,).

        This method fits the model and computes per-class nonconformity scores,
        defined as 1 - P(true label), for all samples. Then, it derives per-class
        thresholds based on the specified significance level.
        """
        self.model.fit(X, y)

        probas = self.model.predict_proba(X)
        preds = self.model.predict(X)

        acc = float(accuracy_score(y, preds))
        prec = float(precision_score(y, preds, average='binary' if len(
            np.unique(y)) == 2 else 'weighted'))
        rec = float(recall_score(y, preds, average='binary' if len(
            np.unique(y)) == 2 else 'weighted'))
        f1 = float(f1_score(y, preds, average='binary' if len(
            np.unique(y)) == 2 else 'weighted'))

        logger.info("Model Performance on Training Data:")
        logger.info("Accuracy:  %.4f", acc)
        logger.info("Precision: %.4f", prec)
        logger.info("Recall:    %.4f", rec)
        logger.info("F1 Score:  %.4f", f1)

        report = classification_report(y, preds, digits=4)
        logger.info("\n%s", report)

        if perf_stats is not None:
            perf_stats.log_ce_metrics(acc, prec, rec, f1)
        else:
            logger.warning(
                "PerformanceStats instance not provided; metrics will not be logged.")

        true_label_indices = np.array([
            np.where(self.model.classes_ == y_i)[0][0]
            for y_i in y
        ])
        scores = 1.0 - probas[np.arange(len(y)), true_label_indices]

        self.calibration_scores = {
            cls: scores[y == cls]
            for cls in np.unique(y)
        }

        if self.significance_controller:
            preds = self.model.predict(X)
            self.significance_controller.update(preds, scores)
            self.thresholds = self.significance_controller.get_thresholds()
        else:
            self.thresholds = {
                cls: float(np.quantile(scores_cls, 1 - self.significance))
                for cls, scores_cls in self.calibration_scores.items()
            }

    def predict_p_values(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Estimate p-values for a batch of test samples.

        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features).

        Returns:
            Dict[str, np.ndarray]: Dictionary with:
                - "class": predicted class labels.
                - "p_value": p-value for each sample based on calibration scores.

        Raises:
            RuntimeError: If calibration has not been performed.
        """
        if self.calibration_scores is None:
            raise RuntimeError(
                "Approx-TCE must be calibrated before computing p-values."
            )

        probas = self.model.predict_proba(X)
        preds = self.model.predict(X)

        new_scores = 1.0 - np.array([
            probas[i, np.where(self.model.classes_ == preds[i])[0][0]]
            for i in range(len(preds))
        ])

        p_values = compute_p_values(new_scores, preds, self.calibration_scores)

        return {"class": preds, "p_value": p_values}

    def get_thresholds(self) -> Dict[Any, float]:
        """
        Retrieve class-specific thresholds derived during calibration.

        Returns:
            Dict[Any, float]: A mapping from class labels to threshold values.

        Raises:
            RuntimeError: If `calibrate()` has not been called yet.
        """
        if self.thresholds is None:
            raise RuntimeError(
                "Thresholds not available: call calibrate() first."
            )
        return self.thresholds


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )
