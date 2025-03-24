# src.core.conformalEval.cce
"""
Cross Conformal Evaluator (CCE) Module

This module provides a parallel implementation of Cross Conformal Evaluation (CCE)
for uncertainty quantification and drift detection. CCE performs k-fold calibration
using a classifier's nonconformity scores and computes per-class thresholds to
detect distributional deviation in incoming test samples.

Supported models include any scikit-learn-compatible estimators that implement
`fit()` and `predict_proba()`, including KNN, SVM, Decision Trees, Random Forests,
and XGBoost classifiers.

Typical usage:
    >>> cce = CrossConformalEvaluator(model, folds=5, significance=0.05)
    >>> cce.calibrate(X_train, y_train)
    >>> result = cce.predict_p_values(X_test)
    >>> thresholds = cce.get_thresholds()
"""
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import numpy as np

from scipy.stats import mode
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from src.core.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController
from src.core.conformalEval.utils import (
    clone_model,
    compute_class_thresholds,
    compute_p_values,
    load_conformal_config,
)
from src.core.perf_stats import PerformanceStats

logger = logging.getLogger(__name__)
STATIC_VALS: Dict = load_conformal_config()
FOLDS = STATIC_VALS["conformal_eval_config"]["folds"]
SIGNIFICANCE = STATIC_VALS["conformal_eval_config"]["significance"]
N_JOBS = STATIC_VALS["conformal_eval_config"]["n_jobs"]


class CrossConformalEvaluator:
    """
    Implements the Cross Conformal Evaluation (CCE) framework with parallelized
    k-fold calibration for efficient nonconformity score estimation.

    Attributes:
        model (Any): A scikit-learn or XGBoost-compatible classifier.
        folds (int): Number of stratified folds for calibration.
        significance (float): Significance level used to compute rejection thresholds.
        random_state (Optional[int]): Seed for reproducibility in data splitting.
        n_jobs (int): Number of threads for parallel processing (-1 uses all cores).
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
        Initialize the CrossConformalEvaluator.

        Args:
            model (Any): A classifier implementing `fit()` and `predict_proba()`.
            folds (int): Number of cross-validation folds (default is 5).
            significance (float): Significance level in [0, 1] for thresholding.
            random_state (Optional[int]): Seed for StratifiedKFold shuffling.
            n_jobs (int): Number of threads to use in parallel; -1 for all cores.
        """
        self.model = model
        self.folds = folds
        self.significance = significance
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.significance_controller = significance_controller
        self.calibration_scores: Optional[Dict[Any, np.ndarray]] = None
        self.thresholds: Optional[Dict[Any, float]] = None
        logger.info(
            f"significance_controller is {'set' if significance_controller else 'None'}")

    def _process_fold(self, X, y, train_idx, calib_idx):
        """
        Train model on train fold, compute nonconformity scores on calibration fold,
        and return trained model.

        Returns:
            Tuple[
                Dict[Any, list],  # Nonconformity scores by class
                np.ndarray,       # True labels for calibration
                np.ndarray,       # Predicted labels on calibration
                Any               # Trained model
            ]
        """
        X_train, X_calib = X[train_idx], X[calib_idx]
        y_train, y_calib = y[train_idx], y[calib_idx]

        model_ = clone_model(self.model)
        model_.fit(X_train, y_train)

        probas = model_.predict_proba(X_calib)
        classes = model_.classes_
        scores = 1.0 - np.array([
            probas[i, np.where(classes == y_calib[i])[0][0]]
            for i in range(len(y_calib))
        ])
        fold_scores: Dict[Any, list] = {cls: [] for cls in np.unique(y)}
        for cls in np.unique(y_calib):
            cls_scores = scores[y_calib == cls]
            fold_scores[cls].extend(cls_scores.tolist())

        return fold_scores, y_calib, model_.predict(X_calib), model_

    def calibrate(self, X: np.ndarray, y: np.ndarray, perf_stats: PerformanceStats) -> None:
        """
        Run k-fold calibration, store per-fold models, compute thresholds from pooled NCMs.
        """
        self.fold_models = []

        all_true = []
        all_pred = []
        all_scores: Dict[Any, list] = {cls: [] for cls in np.unique(y)}
        skf = StratifiedKFold(n_splits=self.folds,
                              shuffle=True, random_state=self.random_state)

        fold_args = list(skf.split(X, y))
        with ThreadPoolExecutor(max_workers=self.folds if self.n_jobs == -1 else self.n_jobs) as executor:
            futures = [
                executor.submit(self._process_fold, X, y, train_idx, calib_idx)
                for train_idx, calib_idx in fold_args
            ]
            for f in as_completed(futures):
                result, y_true_fold, y_pred_fold, model_ = f.result()
                all_true.extend(y_true_fold)
                all_pred.extend(y_pred_fold)
                self.fold_models.append(model_)
                for cls, scores in result.items():
                    all_scores[cls].extend(scores)

        self.calibration_scores = {cls: np.array(
            scores) for cls, scores in all_scores.items()}
        thresholds = compute_class_thresholds(
            self.calibration_scores, self.significance)

        if self.significance_controller:
            logger.info(
                "Initializing AdaptiveSignificanceController thresholds")
            for cls, scores in self.calibration_scores.items():
                fake_preds = np.array([cls] * len(scores))
                self.significance_controller.update(fake_preds, scores)
            self.thresholds = self.significance_controller.get_thresholds()
        else:
            self.thresholds = thresholds

        logger.info("CCE calibration completed successfully.")
        acc = float(accuracy_score(all_true, all_pred))
        prec = float(precision_score(all_true, all_pred,
                     average='binary' if len(np.unique(y)) == 2 else 'weighted'))
        rec = float(recall_score(all_true, all_pred, average='binary' if len(
            np.unique(y)) == 2 else 'weighted'))
        f1 = float(f1_score(all_true, all_pred, average='binary' if len(
            np.unique(y)) == 2 else 'weighted'))

        logger.info("[CCE] Model Performance Across Calibration Folds:")
        logger.info("Accuracy:  %.4f", acc)
        logger.info("Precision: %.4f", prec)
        logger.info("Recall:    %.4f", rec)
        logger.info("F1 Score:  %.4f", f1)
        logger.info("\n%s", classification_report(
            all_true, all_pred, digits=4))

        if perf_stats:
            perf_stats.log_ce_metrics(acc, prec, rec, f1)

    def predict_p_values(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute p-values using predictions from all fold-trained models.
        This mimics the original CCE approach where multiple models contribute
        to a single p-value estimation via score averaging.

        Args:
            X (np.ndarray): Test feature matrix of shape (n_samples, n_features).

        Returns:
            Dict[str, np.ndarray]: Dictionary with keys:
                - "class": final predicted labels (majority vote across models)
                - "p_value": per-sample p-values from averaged nonconformity scores

        Raises:
            RuntimeError: If calibration was not performed or models are missing.
        """
        if self.calibration_scores is None or not self.fold_models:
            raise RuntimeError(
                "CCE must be calibrated before computing p-values.")

        all_preds = []
        all_scores = []

        for model in self.fold_models:
            probas = model.predict_proba(X)
            preds = model.predict(X)
            all_preds.append(preds)

            scores = 1.0 - np.array([
                probas[i, np.where(model.classes_ == preds[i])[0][0]]
                for i in range(len(preds))
            ])
            all_scores.append(scores)

        all_preds = np.stack(all_preds, axis=0)
        all_scores = np.stack(all_scores, axis=0)

        final_preds, _ = mode(all_preds, axis=0, keepdims=False)

        avg_scores = np.mean(all_scores, axis=0)

        p_values = compute_p_values(
            avg_scores, final_preds, self.calibration_scores)

        return {"class": final_preds, "p_value": p_values}

    def get_thresholds(self) -> Dict[Any, float]:
        """
        Retrieve per-class rejection thresholds computed during calibration.

        Returns:
            Dict[Any, float]: Mapping from class label to threshold.

        Raises:
            RuntimeError: If called before thresholds have been computed.
        """
        if self.thresholds is None:
            raise RuntimeError(
                "Thresholds not available: call calibrate() first.")
        return self.thresholds


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )
