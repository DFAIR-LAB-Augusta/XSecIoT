# src.core.conformalEval.ice
"""
Inductive Conformal Evaluator (ICE) Module

This module implements the Inductive Conformal Evaluation (ICE) method for
statistical calibration and concept drift detection. ICE separates the training
dataset into a proper training set and a calibration set. After training, it
uses the calibration set to compute nonconformity scores and per-class thresholds.

Classes:
    - InductiveConformalEvaluator: Implements ICE calibration and drift detection logic.

Typical usage:
    ice = InductiveConformalEvaluator(model)
    ice.calibrate(X_train, y_train)
    result = ice.predict_p_values(X_test)
    thresholds = ice.get_thresholds()
"""
import logging

from typing import Any, Dict, Optional

import numpy as np

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.core.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController
from src.core.conformalEval.utils import (
    compute_class_thresholds,
    compute_nonconformity_scores,
    compute_p_values,
    load_conformal_config,
)
from src.core.perf_stats import PerformanceStats

STATIC_VALS: Dict = load_conformal_config()
SIGNIFICANCE = STATIC_VALS["conformal_eval_config"]["significance"]
CALIBRATION_SPLIT = STATIC_VALS["conformal_eval_config"]["calibration_split"]

logger = logging.getLogger(__name__)


class InductiveConformalEvaluator:
    """
    Inductive Conformal Evaluator (ICE).

    Performs calibration by splitting the training data into a proper training
    and calibration set. After training the model, it computes nonconformity scores
    and derives per-class thresholds for evaluating new samples.
    """

    def __init__(
        self,
        model: Any,
        calibration_split: float = CALIBRATION_SPLIT,
        random_state: Optional[int] = None,
        significance: float = SIGNIFICANCE,
        significance_controller: Optional[AdaptiveSignificanceController] = None,
        **kwargs,
    ):
        """
        Initialize the ICE evaluator.

        Args:
            model (Any): A scikit-learn compatible model implementing `fit()` and `predict_proba()`.
            calibration_split (float): Proportion of data to reserve for calibration.
            random_state (Optional[int]): Seed for reproducible train-test split.
        """
        if kwargs:
            logger.debug(f"ICE: ignoring unsupported kwargs: {sorted(kwargs)}")
        self.model = model
        self.calibration_split = calibration_split
        self.random_state = random_state
        self.significance = significance
        self.significance_controller = significance_controller
        self.calibration_scores: Optional[Dict[Any, np.ndarray]] = None
        self.thresholds: Optional[Dict[Any, float]] = None

    def calibrate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        perf_stats: PerformanceStats,
        significance: Optional[float] = None
    ) -> None:
        """
        Split data into training and calibration sets, then compute thresholds.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Class labels.
            significance (Optional[float]): Override significance level (used if no controller).
        """
        sig = significance if significance is not None else self.significance

        X_train, X_calib, y_train, y_calib = train_test_split(
            X,
            y,
            test_size=self.calibration_split,
            random_state=self.random_state,
            stratify=y
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X)
        probas = self.model.predict_proba(X_calib)

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

        scores = compute_nonconformity_scores(
            probas, y_calib, self.model.classes_)

        self.calibration_scores = {
            cls: scores[y_calib == cls]
            for cls in np.unique(y_calib)
        }

        if self.significance_controller:
            fake_preds = y_calib  # assume predicted correctly
            self.significance_controller.update(fake_preds, scores)
            self.thresholds = self.significance_controller.get_thresholds()
        else:
            self.thresholds = compute_class_thresholds(
                self.calibration_scores, sig)

    def predict_p_values(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Estimate p-values for a batch of test samples.

        For each sample, computes a nonconformity score and estimates its
        p-value as the proportion of calibration scores greater than or equal
        to the sample's score.

        Args:
            X (np.ndarray): Feature matrix for test samples (n_samples, n_features).

        Returns:
            Dict[str, np.ndarray]: Dictionary with keys:
                - "class": predicted class labels.
                - "p_value": corresponding p-values for each sample.
        """
        if self.calibration_scores is None:
            raise RuntimeError(
                "ICE must be calibrated before computing p-values.")

        probas = self.model.predict_proba(X)
        preds = self.model.predict(X)
        scores = 1.0 - np.array([
            probas[i, np.where(self.model.classes_ == preds[i])[0][0]]
            for i in range(len(preds))
        ])

        p_values = compute_p_values(scores, preds, self.calibration_scores)

        return {"class": preds, "p_value": p_values}

    def get_thresholds(self) -> Dict[Any, float]:
        """
        Retrieve the per-class thresholds computed during calibration.

        Returns:
            Dict[Any, float]: Mapping of class labels to threshold values.

        Raises:
            RuntimeError: If calibration has not been performed.
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
