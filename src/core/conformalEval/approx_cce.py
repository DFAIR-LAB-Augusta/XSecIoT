# src.core.conformalEval.approx_cce
"""
Approximate Cross Conformal Evaluation (Approx-CCE) Module

This module provides a fast and memory-efficient implementation of Approx-CCE,
a variant of Cross Conformal Evaluation (CCE) tailored for high-throughput,
low-latency applications such as streaming network traffic analysis.

Approx-CCE uses a single model trained on the full dataset and performs
calibration using fold-partitioned nonconformity scores, significantly
reducing computation compared to full CCE.

Classes:
    - ApproxCrossConformalEvaluator: Implements fast calibration, threshold
      computation, and p-value prediction using a single shared model.

Example:
    >>> approx_cce = ApproxCrossConformalEvaluator(model, folds=5, significance=0.05)
    >>> approx_cce.calibrate(X_train, y_train)
    >>> result = approx_cce.predict_p_values(X_test)
    >>> thresholds = approx_cce.get_thresholds()
"""
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import numpy as np

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from src.core.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController
from src.core.conformalEval.utils import (
    clone_model,
    compute_class_thresholds,
    compute_nonconformity_scores,
    compute_p_values,
    load_conformal_config,
)
from src.core.perf_stats import PerformanceStats

STATIC_VALS: Dict = load_conformal_config()
FOLDS = STATIC_VALS["conformal_eval_config"]["folds"]
SIGNIFICANCE = STATIC_VALS["conformal_eval_config"]["significance"]
N_JOBS = STATIC_VALS["conformal_eval_config"]["n_jobs"]
CALIBRATION_SPLIT = STATIC_VALS["conformal_eval_config"]["calibration_split"]

logger = logging.getLogger(__name__)


class ApproxCrossConformalEvaluator:
    """
    Approximate Cross Conformal Evaluator (Approx-CCE) with threaded calibration.

    A fast alternative to full CCE using a single shared model, with fold-based
    calibration accelerated via parallel threads. Suitable for high-throughput scenarios.
    """

    def __init__(
        self,
        model: Any,
        folds: int = FOLDS,
        significance: float = SIGNIFICANCE,
        random_state: Optional[int] = None,
        n_jobs: int = N_JOBS,
        significance_controller: Optional[AdaptiveSignificanceController] = None
    ):
        """
        Initialize Approx-CCE evaluator.

        Args:
            model (Any): A classifier implementing `fit()` and `predict_proba()`.
            folds (int): Number of partitions to estimate calibration scores.
            significance (float): Rejection threshold significance level (e.g., 0.05).
            random_state (Optional[int]): Seed for reproducibility.
            n_jobs (int): Number of threads to use; -1 = all available.
        """
        self.model = model
        self.folds = folds
        self.significance = significance
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.significance_controller = significance_controller
        self.calibration_scores: Optional[Dict[Any, np.ndarray]] = None
        self.thresholds: Optional[Dict[Any, float]] = None
        self.fitted_model = None

    def _process_fold(self, X: np.ndarray, y: np.ndarray, calib_idx: np.ndarray) -> Dict[Any, list]:
        """
        Compute nonconformity scores for one calibration fold.
        Assumes self.fitted_model is thread-safe for concurrent predict_proba()

        Args:
            X (np.ndarray): Full feature matrix.
            y (np.ndarray): Full label array.
            calib_idx (np.ndarray): Indices for this calibration fold.

        Returns:
            Dict[Any, list]: Per-class nonconformity scores for the fold.
        """
        if self.fitted_model is None:
            raise RuntimeError(
                "Model must be fitted before calibration fold processing.")

        model = self.fitted_model
        X_calib, y_calib = X[calib_idx], y[calib_idx]
        probas = model.predict_proba(X_calib)
        scores = compute_nonconformity_scores(probas, y_calib, model.classes_)

        fold_scores: Dict[Any, list] = {cls: [] for cls in np.unique(y)}
        for i, cls in enumerate(y_calib):
            fold_scores[cls].append(scores[i])

        return fold_scores

    def calibrate(self, X: np.ndarray, y: np.ndarray, perf_stats: PerformanceStats) -> None:
        """
        Calibrate Approx-CCE using parallelized fold scoring.
        """

        model_ = clone_model(self.model)
        model_.fit(X, y)
        self.fitted_model = model_
        preds = model_.predict(X)
        probas = model_.predict_proba(X)

        acc = float(accuracy_score(y, preds))
        prec = float(precision_score(y, preds, average='binary' if len(
            np.unique(y)) == 2 else 'weighted'))
        rec = float(recall_score(y, preds, average='binary' if len(
            np.unique(y)) == 2 else 'weighted'))
        f1 = float(f1_score(y, preds, average='binary' if len(
            np.unique(y)) == 2 else 'weighted'))

        logger.info("CE Model Performance on Training Data:")
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

        skf = StratifiedKFold(n_splits=self.folds,
                              shuffle=True, random_state=self.random_state)
        fold_indices = [calib_idx for _, calib_idx in skf.split(X, y)]

        all_scores: Dict[Any, list] = {cls: [] for cls in np.unique(y)}

        with ThreadPoolExecutor(max_workers=self.folds) as executor:
            futures = [
                executor.submit(self._process_fold, X, y, calib_idx)
                for calib_idx in fold_indices
            ]
            for f in as_completed(futures):
                result = f.result()
                for cls, scores in result.items():
                    all_scores[cls].extend(scores)

        self.calibration_scores = {cls: np.array(
            scores) for cls, scores in all_scores.items()}
        probas = self.fitted_model.predict_proba(X)
        preds = self.fitted_model.predict(X)
        scores = 1.0 - np.array([
            probas[i, np.where(self.fitted_model.classes_ == preds[i])[0][0]]
            for i in range(len(preds))
        ])

        if self.significance_controller:
            self.significance_controller.update(preds, scores)
            self.thresholds = self.significance_controller.get_thresholds()
        else:
            self.thresholds = compute_class_thresholds(
                self.calibration_scores, self.significance)

    def predict_p_values(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict class labels and compute p-values by comparing scores to
        calibrated per-class thresholds.

        Args:
            X (np.ndarray): Test samples.

        Returns:
            Dict[str, np.ndarray]: Keys: 'class', 'p_value'
        """
        if self.calibration_scores is None or self.fitted_model is None:
            raise RuntimeError("Model must be calibrated before prediction.")
        model = self.fitted_model

        preds = model.predict(X)
        classes = model.classes_
        probas = model.predict_proba(X)

        scores = 1.0 - np.array([
            probas[i, np.where(classes == preds[i])[0][0]]
            for i in range(len(preds))
        ])

        p_values = compute_p_values(scores, preds, self.calibration_scores)
        return {"class": preds, "p_value": p_values}

    def get_thresholds(self) -> Dict[Any, float]:
        """
        Get per-class significance thresholds.

        Returns:
            Dict[Any, float]: Class label → threshold mapping.
        """
        if self.thresholds is None:
            raise RuntimeError(
                "Thresholds not available: call calibrate() first.")
        return self.thresholds


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )
