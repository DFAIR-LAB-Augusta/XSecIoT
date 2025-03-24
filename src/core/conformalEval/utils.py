# src/core/conformalEval/utils
import copy
import inspect
import logging

from pathlib import Path
from typing import Any

import numpy as np
import toml

from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def clone_model(model: Any) -> Any:
    """
    Return a fresh, unfitted copy of ``model``.

    The cloning strategy is tolerant of non-scikit-learn estimators and follows
    this precedence:
      1) Use ``model.clone()`` if available (preferred for custom models).
      2) Special-case ``xgboost.XGBClassifier`` via its constructor + ``get_params()``.
      3) Use ``sklearn.base.clone`` when compatible.
      4) Reconstruct via ``model.get_params()`` and the model's constructor signature.
      5) Fall back to ``copy.deepcopy``.

    Args:
        model: The estimator or model instance to clone.

    Returns:
        A new, unfitted instance of the same model (or a functionally equivalent one).

    Raises:
        ValueError: If cloning fails and a deep copy cannot be created.
    """
    # 1) Custom clone method
    if hasattr(model, "clone") and callable(getattr(model, "clone")):
        try:
            return model.clone()
        except Exception as e:
            logger.debug(f"[clone_model] model.clone() failed: {e!r}")

    # 2) Special handling for XGBClassifier
    if XGBClassifier is not None and isinstance(model, XGBClassifier):
        try:
            params = model.get_params()
            return XGBClassifier(**params)
        except Exception as e:
            logger.debug(f"[clone_model] XGBClassifier reinit failed: {e!r}")

    # 3) scikit-learn clone (when available/compatible)
    try:
        from sklearn.base import clone as sk_clone  # type: ignore
        return sk_clone(model)
    except Exception as e:
        logger.debug(f"[clone_model] sklearn.clone failed: {e!r}")

    # 4) Reconstruct via get_params + constructor signature
    if hasattr(model, "get_params") and callable(getattr(model, "get_params")):
        try:
            params = model.get_params(deep=True)
            sig = inspect.signature(model.__class__)
            ctor_kwargs = {k: v for k,
                           v in params.items() if k in sig.parameters}
            return model.__class__(**ctor_kwargs)
        except Exception as e:
            logger.debug(f"[clone_model] get_params reinit failed: {e!r}")

    # 5) Last resort: deepcopy
    try:
        return copy.deepcopy(model)
    except Exception as e:
        logger.debug(f"[clone_model] deepcopy failed: {e!r}")
        raise ValueError(
            f"Unsupported model type for cloning: {type(model)!r}") from e


def compute_nonconformity_scores(probas, true_labels, class_list):
    """
    Compute nonconformity scores as 1 - P(true_label) for each sample.

    Args:
        probas (np.ndarray): Predicted probabilities (n_samples, n_classes)
        true_labels (np.ndarray): Array of true class labels
        class_list (np.ndarray): Ordered list of class labels (from model.classes_)

    Returns:
        np.ndarray: Array of nonconformity scores
    """
    label_indices = np.array([
        np.where(class_list == label)[0][0] for label in true_labels
    ])
    return 1.0 - probas[np.arange(len(probas)), label_indices]


def compute_p_values(scores, preds, calibration_scores):
    """
    Compute p-values by comparing prediction nonconformity scores to calibration scores.

    Args:
        scores (np.ndarray): Nonconformity scores for test predictions
        preds (np.ndarray): Predicted labels
        calibration_scores (Dict[Any, np.ndarray]): Calibration score distribution per class

    Returns:
        np.ndarray: Array of p-values
    """
    p_values = np.empty_like(scores)
    for i, cls in enumerate(preds):
        calib = calibration_scores[cls]
        p_values[i] = (np.sum(calib >= scores[i]) + 1) / (len(calib) + 1)
    return p_values


def compute_class_thresholds(calibration_scores, significance):
    """
    Compute per-class rejection thresholds from calibration scores.

    Args:
        calibration_scores (Dict[Any, np.ndarray]): Per-class calibration scores
        significance (float): Rejection significance level (e.g., 0.05)

    Returns:
        Dict[Any, float]: Per-class thresholds
    """
    return {
        cls: float(np.quantile(scores, 1 - significance))
        for cls, scores in calibration_scores.items()
    }


def load_conformal_config(path: Path = Path("src/core/conformalEval/conformal_config.toml")) -> dict:
    """
    Load conformal evaluator configuration from TOML file.

    Args:
        path (Path): Path to the TOML config file.

    Returns:
        dict[str, Any]: Dictionary of loaded configuration parameters.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing TOML config at {path}")

    return toml.load(path)


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )
