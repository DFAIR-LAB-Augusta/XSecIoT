# firce.novelty.explain
"""
Local XAI explainability layer for flagged unknown/emerging behaviors
(dissertation proposal Section 4.4.3).

Produces structured, per-event feature-attribution explanations - not plots -
since the consumer is #99 (LLM-assisted reporting), which needs machine-
readable feature/contribution/direction data for a single flagged event at
inference time, not a post-hoc batch review (contrast with
fire.models._explain_with_shap/_explain_with_lime, which write PNG/HTML
plots after a full training run).

Every classifier variant in this codebase (sklearn dt/knn/rf/svm/xgb, and the
PyTorch-backed MLPCEBase/MLPCEMulticlass wrappers) exposes a uniform
predict_proba(X: np.ndarray) -> np.ndarray, so both explainers here are
written generically against that interface.
"""

import logging

from typing import Any, Dict, Optional

import numpy as np
import shap

from lime.lime_tabular import LimeTabularExplainer

logger = logging.getLogger(__name__)

_VALID_MODES = ('sampled', 'unknown_only', 'windowed')


def select_events_to_explain(
    flags: np.ndarray,
    mode: str,
    sample_rate: float = 0.1,
    window_size: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Decide which event indices should receive a generated explanation, under
    a latency budget (proposal Section 4.4.3, "selective generation").

    Args:
        flags: Boolean novelty-flag array (as returned by firce.novelty.decision_rules.is_novel),
            one entry per event.
        mode: One of 'sampled', 'unknown_only', 'windowed'.
        sample_rate: Fraction of all events to explain, only used when mode='sampled'.
        window_size: Size of each consecutive block, only used when mode='windowed'.
        rng: Optional numpy random Generator for reproducible sampling (mode='sampled' only).

    Returns:
        Sorted array of integer indices into `flags` selected for explanation.

    Raises:
        ValueError: If `mode` is not one of the supported values.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f'Unknown selective-generation mode {mode!r}; expected one of {_VALID_MODES}')

    n = len(flags)

    if mode == 'unknown_only':
        return np.flatnonzero(flags)

    if mode == 'sampled':
        generator = rng if rng is not None else np.random.default_rng()
        n_selected = int(round(sample_rate * n))
        return np.sort(generator.choice(n, size=n_selected, replace=False))

    # mode == 'windowed'
    selected = []
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        window_flags = flags[start:end]
        flagged_in_window = np.flatnonzero(window_flags)
        if len(flagged_in_window) > 0:
            selected.append(start + int(flagged_in_window[0]))
        else:
            selected.append(start)
    return np.array(selected, dtype=int)


def explain_with_shap(
    model: Any,
    X_background: np.ndarray,
    x_instance: np.ndarray,
    feature_names: list,
    model_type: str = 'kernel',
) -> Dict[str, Any]:
    """
    Generate a structured SHAP feature-attribution explanation for one instance.

    Attributes the model's predicted (top) class - the most direct answer to
    "why does the model only weakly support its top guess," which is the
    question behind "why flagged as unknown."

    Args:
        model: A fitted classifier exposing predict_proba() and classes_.
        X_background: Background/reference data for the explainer (calibration
            or training data recommended). For model_type='kernel' keep this small
            (e.g. <=50 rows) - KernelExplainer cost scales with background size.
        x_instance: A single feature vector, shape (n_features,).
        feature_names: Names for each feature, in the same order as x_instance.
        model_type: 'tree' for tree-based models (fast TreeExplainer path) or
            'kernel' for any other model (generic, slower KernelExplainer path).

    Returns:
        Dict with keys:
            - 'predicted_class': the model's top predicted class for x_instance.
            - 'contributions': dict mapping each feature name to its SHAP value
              (float) for the predicted class.
    """
    x_row = np.asarray(x_instance).reshape(1, -1)
    predicted_class = model.predict(x_row)[0]
    class_index = int(np.where(model.classes_ == predicted_class)[0][0])

    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_row)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_background)
        shap_values = explainer.shap_values(x_row)

    if isinstance(shap_values, list):
        instance_values = np.asarray(shap_values[class_index])[0]
    else:
        values = np.asarray(shap_values)
        instance_values = values[0, :, class_index] if values.ndim == 3 else values[0]

    contributions = {name: float(value) for name, value in zip(feature_names, instance_values)}
    return {'predicted_class': predicted_class, 'contributions': contributions}


def explain_with_lime(
    model: Any,
    X_train: np.ndarray,
    x_instance: np.ndarray,
    feature_names: list,
    class_names: list,
) -> Dict[str, Any]:
    """
    Generate a structured LIME feature-weight explanation for one instance.

    Args:
        model: A fitted classifier exposing predict_proba() and classes_.
        X_train: Training data used to build LIME's local perturbation neighborhood.
        x_instance: A single feature vector, shape (n_features,).
        feature_names: Names for each feature, in the same order as x_instance.
        class_names: Human-readable names for each class, in classes_ order.

    Returns:
        Dict with keys:
            - 'predicted_class': the model's top predicted class for x_instance.
            - 'contributions': dict mapping feature description (LIME's own
              rule-based label, e.g. "f0 > 0.50") to its local weight (float).
              LIME only returns weights for the features it selects as most
              relevant to the local neighborhood, so this may be a subset of
              feature_names.
    """
    x_row = np.asarray(x_instance).reshape(1, -1)
    predicted_class = model.predict(x_row)[0]

    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train), feature_names=feature_names, class_names=class_names, mode='classification'
    )
    explanation = explainer.explain_instance(np.asarray(x_instance), model.predict_proba)

    contributions = {label: float(weight) for label, weight in explanation.as_list()}
    return {'predicted_class': predicted_class, 'contributions': contributions}
