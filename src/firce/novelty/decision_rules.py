# firce.novelty.decision_rules
"""
Novelty/"unknown" detection decision rules (dissertation proposal Section 4.4.2).

Combines a model-confidence signal (softmax-derived) with a conformal-reliability
signal (per-class p-values) to flag samples inconsistent with the current known
label space. Also implements the alternative confidence criteria the proposal
calls out for prototyping/comparison: margin, entropy, temperature-scaled
confidence, and conformal prediction-set size.

None of the thresholds here (tau, alpha, temperature) are hardcoded defaults to
rely on in production - the proposal is explicit that they are selected
empirically per-deployment on a held-out time period.
"""

from typing import Any, Dict

import numpy as np

from firce.conformalEval.utils import compute_nonconformity_scores, compute_p_values


def compute_all_class_p_values(
    model: Any, calibration_scores: Dict[Any, np.ndarray], X: np.ndarray
) -> Dict[Any, np.ndarray]:
    """
    Compute a conformal p-value for every known class, for every sample in X.

    Unlike InductiveConformalEvaluator.predict_p_values / CrossConformalEvaluator.predict_p_values
    (which only return the p-value for the single *predicted* class per sample), this
    returns a full per-class p-value vector - required by the novelty decision rule's
    "for all classes c, p_c < alpha" condition.

    Args:
        model: A fitted classifier exposing predict_proba() and classes_ (e.g. the
            .model attribute of a calibrated InductiveConformalEvaluator/CrossConformalEvaluator).
        calibration_scores: Per-class calibration nonconformity score arrays, as produced
            by evaluator.calibrate() (the evaluator's .calibration_scores attribute).
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        Dict mapping each class label to an array of shape (n_samples,) of p-values in [0, 1].
    """
    probas = model.predict_proba(X)
    n_samples = X.shape[0]
    result: Dict[Any, np.ndarray] = {}
    for cls in calibration_scores:
        hypothetical_labels = np.full(n_samples, cls, dtype=object)
        scores = compute_nonconformity_scores(probas, hypothetical_labels, model.classes_)
        result[cls] = compute_p_values(scores, hypothetical_labels, calibration_scores)
    return result


def max_softmax_confidence(probas: np.ndarray) -> np.ndarray:
    """Model-confidence signal: max predicted probability per sample."""
    return np.max(probas, axis=1)


def is_novel(probas: np.ndarray, all_class_p_values: Dict[Any, np.ndarray], tau: float, alpha: float) -> np.ndarray:
    """
    Primary novelty decision rule (proposal Section 4.4.2):

        (max softmax < tau) AND (for all known classes c, p_c < alpha)

    Both a low model-confidence signal AND low conformal support across every
    known class are required, to reduce false "unknown" alerts from either
    signal alone.

    Args:
        probas: Predicted probability matrix of shape (n_samples, n_classes).
        all_class_p_values: Per-class p-value arrays, as returned by compute_all_class_p_values.
        tau: Confidence threshold - samples with max softmax >= tau are never flagged.
        alpha: Conformal significance threshold - a sample is only flagged if every
            class's p-value is below this.

    Returns:
        Boolean array of shape (n_samples,), True where the sample is flagged as novel/unknown.
    """
    low_confidence = max_softmax_confidence(probas) < tau
    p_value_matrix = np.stack(list(all_class_p_values.values()), axis=1)
    all_classes_rejected = np.all(p_value_matrix < alpha, axis=1)
    return low_confidence & all_classes_rejected


def margin_confidence(probas: np.ndarray) -> np.ndarray:
    """Alternative confidence signal: difference between the top-1 and top-2 predicted probabilities."""
    sorted_probas = np.sort(probas, axis=1)
    return sorted_probas[:, -1] - sorted_probas[:, -2]


def entropy_confidence(probas: np.ndarray) -> np.ndarray:
    """Alternative confidence signal: Shannon entropy of the predicted probability distribution (nats)."""
    safe_probas = np.clip(probas, np.finfo(float).tiny, 1.0)
    return -np.sum(probas * np.log(safe_probas), axis=1)


def temperature_scaled_confidence(probas: np.ndarray, temperature: float) -> np.ndarray:
    """
    Alternative confidence signal: max probability after temperature scaling.

    Approximates standard logit temperature scaling in probability space
    (raise each probability to the power 1/temperature and renormalize),
    since classifiers here are accessed only via predict_proba, not raw logits.
    temperature > 1 flattens the distribution (lower confidence); temperature == 1 is a no-op.
    """
    scaled = np.power(probas, 1.0 / temperature)
    scaled = scaled / np.sum(scaled, axis=1, keepdims=True)
    return np.max(scaled, axis=1)


def conformal_prediction_set_size(all_class_p_values: Dict[Any, np.ndarray], alpha: float) -> np.ndarray:
    """
    Alternative novelty signal: size of the conformal prediction set at significance alpha.

    A class is "non-rejected" (included in the prediction set) if its p-value is
    >= alpha. An empty prediction set (size 0) is itself a strong novelty signal -
    no known class is conformally supported at this significance level.

    Args:
        all_class_p_values: Per-class p-value arrays, as returned by compute_all_class_p_values.
        alpha: Conformal significance threshold.

    Returns:
        Integer array of shape (n_samples,): count of non-rejected classes per sample.
    """
    p_value_matrix = np.stack(list(all_class_p_values.values()), axis=1)
    return np.sum(p_value_matrix >= alpha, axis=1)
