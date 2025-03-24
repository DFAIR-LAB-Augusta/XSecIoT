# src.core.conformalEval.conformal_evaluators
"""
Unified Conformal Evaluation Interface

This module provides a wrapper (`ConformalEvaluator`) and a factory (`ConformalEvaluatorFactory`)
to support interchangeable usage of various conformal evaluation strategies for
drift detection in machine learning pipelines.

Supported CE Backends:
- Inductive Conformal Evaluation (ICE)
- Cross Conformal Evaluation (CCE)
- Approximate Transductive Conformal Evaluation (Approx-TCE)

Key Capabilities:
- Instantiate appropriate CE implementation via a factory pattern.
- Calibrate CE models on labeled data with confidence thresholds.
- Detect distribution drift based on p-value thresholds.
- Handle diverse p-value output formats robustly (scalar, array, dict, DataFrame).

This abstraction simplifies downstream CE integration and enables dynamic evaluator
selection with consistent calibration and drift detection APIs.
"""
import logging
import time

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.core.config import CEType
from src.core.conformalEval.adaptive_sig_ctlr import AdaptiveSignificanceController
from src.core.conformalEval.approx_cce import ApproxCrossConformalEvaluator as _ApproxCCEImpl
from src.core.conformalEval.cce import CrossConformalEvaluator as _CCEImpl
from src.core.conformalEval.ice import InductiveConformalEvaluator as _ICEImpl
from src.core.conformalEval.tce import ApproximateTransductiveConformalEvaluator as _ApproxTCEImpl
from src.core.perf_stats import PerformanceStats

logger = logging.getLogger(__name__)


class ConformalEvaluatorFactory:
    """
    Factory class for creating conformal evaluator instances.

    Supports dynamic instantiation of Inductive, Cross, and Approximate Transductive
    conformal evaluation strategies based on the specified evaluator type.
    """
    @staticmethod
    def create(
        evaluator_type: CEType,
        model: str,
        significance_controller: Optional[AdaptiveSignificanceController] = None,
        **kwargs
    ) -> _ICEImpl | _CCEImpl | _ApproxTCEImpl | _ApproxCCEImpl:
        """
        Instantiate the appropriate conformal evaluator based on type.

        Args:
            evaluator_type (str): One of 'ice', 'cce', or 'approx_tce'.
            model (Any): Trained ML model to wrap in conformal evaluation.
            **kwargs: Additional keyword arguments passed to the evaluator.

        Returns:
            Any: Instance of the selected conformal evaluator.
        """
        if evaluator_type == CEType.ICE:
            return _ICEImpl(model=model, significance_controller=significance_controller, **kwargs)
        elif evaluator_type == CEType.CCE:
            return _CCEImpl(model=model, significance_controller=significance_controller, **kwargs)
        elif evaluator_type == CEType.APPROX_TCE:
            return _ApproxTCEImpl(model=model, significance_controller=significance_controller, ** kwargs)
        elif evaluator_type == CEType.APPROX_CCE:
            return _ApproxCCEImpl(model=model, significance_controller=significance_controller, ** kwargs)
        else:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")


class ConformalEvaluator:
    """
    Unified wrapper for conformal evaluation using various methods.

    Handles calibration and drift detection using a specified CE strategy
    (ICE, CCE, or Approximate TCE). This abstraction allows switching between
    CE types without changing calling logic elsewhere.
    """

    def __init__(
        self,
        evaluator_type: CEType,
        model: Any,
        significance_controller: Optional[AdaptiveSignificanceController] = None,
        **kwargs
    ):
        """
        Initialize a ConformalEvaluator with the desired backend strategy.

        Args:
            evaluator_type (str): One of 'ice', 'cce', or 'approx_tce'.
            model (Any): Pretrained classifier to use for conformal evaluation.
            **kwargs: Any other CE-specific keyword arguments.
        """
        self.evaluator = ConformalEvaluatorFactory.create(
            evaluator_type, model, significance_controller, **kwargs)
        self.thresholds: Optional[Dict[Any, float]] = None
        self.significance_controller = significance_controller

    def calibrate(self, X_train: np.ndarray, y_train: np.ndarray, perf_stats: PerformanceStats, **calib_kwargs):
        """
        Calibrate the underlying CE model on a training set.

        Sets per-class p-value thresholds for later drift detection.

        Args:
            X_train (np.ndarray): Input features of shape (n_samples, n_features).
            y_train (np.ndarray): Corresponding ground truth labels.
            **calib_kwargs: Optional calibration parameters like number of folds.
        """
        logger.info("Starting CE calibration with %d samples...", len(X_train))
        t0 = time.perf_counter()
        self.evaluator.calibrate(X_train, y_train, perf_stats, **calib_kwargs)
        self.thresholds = self.evaluator.get_thresholds()
        elapsed = time.perf_counter() - t0
        logger.info("CE calibration completed in %.4fs", elapsed)
        logger.debug("Extracted thresholds: %s", self.thresholds)

    def detect_drift(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate a single sample and detect drift based on conformal thresholds.

        Args:
            X (np.ndarray): Input feature matrix of shape (1, n_features).

        Returns:
            np.ndarray: Boolean array of shape (1,) where True indicates drift.

        Raises:
            RuntimeError: If CE has not been calibrated yet.
            ValueError: If the returned p-value format is not supported.
        """
        if self.thresholds is None:
            raise RuntimeError(
                "Conformal evaluator must be calibrated before detection.")
        logger.debug("Starting drift detection.")
        t0 = time.perf_counter()
        model_input_shape = X.shape
        model_expected_shape = getattr(
            self.evaluator.model, "n_features_in_", None)
        if model_expected_shape is not None and model_input_shape[1] != model_expected_shape:
            logger.debug(
                f"Model expects {self.evaluator.model.n_features_in_} features, received {X.shape[1]} features."
            )
        else:
            logger.debug(
                "Model input shape is valid: %s compared to expected %s",
                model_input_shape,
                model_expected_shape
            )

        p_values = self.evaluator.predict_p_values(X)

        elapsed = time.perf_counter() - t0
        logger.debug("Drift detection completed in %.4fs", elapsed)
        logger.debug("Received p_values of type %s", type(p_values))

        # Case 1: Structured dictionary
        if isinstance(p_values, dict) and 'class' in p_values and 'p_value' in p_values:
            logger.debug(
                "Case 1: Detected structured p_values dictionary format.")
            cls = p_values['class'][0]
            pval = p_values['p_value'][0]

            if self.significance_controller:
                self.significance_controller.update(
                    np.array([cls]), np.array([pval]))
                thresh = self.significance_controller.get_thresholds().get(
                    cls, self.thresholds.get(cls))
            else:
                thresh = self.thresholds.get(cls)

            drifted = pval < thresh
            logger.debug(
                "Drift check (class: %s): p-value=%.6f vs threshold=%.6f", cls, pval, thresh)
            if drifted:
                logger.info("Drift detected for class %s", cls)
            return np.array([drifted])

        # Case 2: DataFrame with one row
        if isinstance(p_values, pd.DataFrame):
            logger.debug(
                "Case 2: Detected structured p_values in DataFrame with one row format.")
            cls = p_values.iloc[0]['class']
            pval = p_values.iloc[0]['p_value']

            if self.significance_controller:
                self.significance_controller.update(
                    np.array([cls]), np.array([pval]))
                thresh = self.significance_controller.get_thresholds().get(
                    cls, self.thresholds.get(cls))
            else:
                thresh = self.thresholds.get(cls)

            drifted = pval < thresh
            logger.debug(
                "Drift check (class: %s): p-value=%.6f vs threshold=%.6f", cls, pval, thresh)
            if drifted:
                logger.info("Drift detected for class %s", cls)
            return np.array([drifted])

        # Case 3: 1D p-value array (binary only)
        if isinstance(p_values, np.ndarray) and p_values.ndim == 1:
            logger.debug(
                "Case 3: Detected structured p_values in 1D p-value array (binary only) format.")
            pval = p_values
            if self.significance_controller:
                pred_class = list(self.thresholds.keys())[0]  # fallback
                self.significance_controller.update(
                    np.array([pred_class]), np.array([pval]))
                thresh = list(
                    self.significance_controller.get_thresholds().values())[0]
            else:
                thresh = list(self.thresholds.values())[0]

            drifted = pval < thresh
            logger.debug(
                "Drift check (binary): p-value=%.6f vs threshold=%.6f", pval, thresh)
            if drifted:
                logger.info("Drift detected (binary)")
            return np.array([drifted])

        # Case 4: Scalar p-value
        if isinstance(p_values, (float, int)):
            logger.debug(
                "Case 4: Detected structured p_values in Scalar p-value format.")
            pval = float(p_values)
            if self.significance_controller:
                pred_class = list(self.thresholds.keys())[0]  # fallback
                self.significance_controller.update(
                    np.array([pred_class]), np.array([pval]))
                thresh = list(
                    self.significance_controller.get_thresholds().values())[0]
            else:
                thresh = list(self.thresholds.values())[0]

            drifted = pval < thresh
            logger.debug(
                "Drift check (scalar): p-value=%.6f vs threshold=%.6f", pval, thresh)
            if drifted:
                logger.info("Drift detected (scalar)")
            return np.array([drifted])

        raise ValueError(f"Unexpected p_values format: {type(p_values)}")


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )
