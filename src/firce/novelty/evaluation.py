# firce.novelty.evaluation
"""
Direction 3 evaluation harness (dissertation proposal Sections 4.4.6, 4.6.3).

Three independent evaluation stages, mirroring how multiclass systems fail
in deployment:
  1. evaluate_closed_world - standard multiclass metrics on known classes.
  2. evaluate_open_world_novelty - unknown-detection quality when a class is
     held out from training entirely.
  3. evaluate_explanation_utility - explanation latency and attribution
     stability (#98's SHAP/LIME output).

Analyst-utility proxies (time-to-triage, agreement on suggested labels) are
explicitly out of scope - the proposal itself defers the annotation
protocol needed for those ("TBD once the pipeline design is fixed").
"""

import itertools
import logging

from typing import Any, Dict, List, Optional

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_closed_world(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[list] = None) -> Dict[str, Any]:
    """
    Stage 1: standard closed-world multiclass evaluation (proposal Section 4.4.6, item 1).

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        labels: Ordered label set for the confusion matrix (defaults to sorted unique labels present).

    Returns:
        Dict with keys: accuracy, macro_precision, macro_recall, macro_f1,
        weighted_precision, weighted_recall, weighted_f1, confusion_matrix
        (np.ndarray), labels (the label order used for the confusion matrix).
    """
    if labels is None:
        labels = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))

    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    return {
        'accuracy': accuracy,
        'macro_precision': float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'weighted_precision': float(
            precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
        ),
        'weighted_recall': float(recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels),
        'labels': labels,
    }


def evaluate_open_world_novelty(
    is_actually_unknown: np.ndarray, novelty_flags: np.ndarray, novelty_scores: np.ndarray
) -> Dict[str, Any]:
    """
    Stage 2: open-world novelty-detection evaluation (proposal Section 4.4.6,
    item 2; metrics per Section 4.6.3's "unknown-vs-known AUROC/AUPRC, unknown F1").

    Args:
        is_actually_unknown: Ground truth - True where the sample is genuinely
            from a class not present in training (e.g. a held-out/time-shifted class).
        novelty_flags: The novelty-detection rule's binary decision (e.g.
            firce.novelty.decision_rules.is_novel's output) for each sample.
        novelty_scores: A continuous per-sample novelty score for AUROC/AUPRC
            (higher = more novel) - which signal to use (e.g. 1 - max_softmax,
            a p-value-derived score, or a combination) is an open design
            question left to the caller, not hard-coded here.

    Returns:
        Dict with keys: precision, recall, f1 (of novelty_flags vs.
        is_actually_unknown), auroc, auprc (of novelty_scores vs.
        is_actually_unknown), alert_burden (fraction of all samples flagged,
        regardless of correctness - proposal's "rate of unknown flags").
    """
    return {
        'precision': float(precision_score(is_actually_unknown, novelty_flags, zero_division=0)),
        'recall': float(recall_score(is_actually_unknown, novelty_flags, zero_division=0)),
        'f1': float(f1_score(is_actually_unknown, novelty_flags, zero_division=0)),
        'auroc': float(roc_auc_score(is_actually_unknown, novelty_scores)),
        'auprc': float(average_precision_score(is_actually_unknown, novelty_scores)),
        'alert_burden': float(np.mean(novelty_flags)),
    }


def _top_k_feature_set(contributions: Dict[str, float], top_k: int) -> frozenset:
    ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    return frozenset(name for name, _value in ranked[:top_k])


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def evaluate_explanation_utility(
    explanations: List[dict], latencies_seconds: List[float], top_k: int = 5
) -> Dict[str, Any]:
    """
    Stage 3: explanation and reporting utility evaluation (proposal Section
    4.4.6, item 3; "explanation latency, stability/consistency across similar
    events" per Section 4.6.3's "attribution consistency" metric).

    Analyst-utility proxies (time-to-triage, agreement on suggested labels)
    are explicitly out of scope - the proposal defers their exact rubric.

    Args:
        explanations: List of #98's explain_with_shap/explain_with_lime output
            dicts ({'predicted_class': ..., 'contributions': {feature: value}}).
        latencies_seconds: Per-explanation generation latency, same order/length as explanations.
        top_k: Number of top-|contribution| features considered per explanation
            when measuring attribution consistency.

    Returns:
        Dict with keys: latency_mean_seconds, latency_p50_seconds,
        latency_p95_seconds, latency_max_seconds, attribution_consistency_overall
        (average pairwise Jaccard overlap of top-k feature sets across ALL
        explanation pairs sharing a predicted_class), attribution_consistency_by_class
        (same, broken out per predicted_class - a class with fewer than 2
        explanations is omitted, since consistency is undefined for a single sample).
    """
    latencies = np.asarray(latencies_seconds, dtype=float)

    by_class: Dict[Any, List[frozenset]] = {}
    for exp in explanations:
        by_class.setdefault(exp['predicted_class'], []).append(_top_k_feature_set(exp['contributions'], top_k))

    consistency_by_class: Dict[Any, float] = {}
    all_pair_scores: List[float] = []
    for cls, feature_sets in by_class.items():
        if len(feature_sets) < 2:
            continue
        pair_scores = [_jaccard(a, b) for a, b in itertools.combinations(feature_sets, 2)]
        consistency_by_class[cls] = float(np.mean(pair_scores))
        all_pair_scores.extend(pair_scores)

    return {
        'latency_mean_seconds': float(np.mean(latencies)),
        'latency_p50_seconds': float(np.percentile(latencies, 50)),
        'latency_p95_seconds': float(np.percentile(latencies, 95)),
        'latency_max_seconds': float(np.max(latencies)),
        'attribution_consistency_overall': float(np.mean(all_pair_scores)) if all_pair_scores else None,
        'attribution_consistency_by_class': consistency_by_class,
    }
