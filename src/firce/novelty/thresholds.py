# firce.novelty.thresholds
"""
Threshold selection for the novelty decision rules.

Picks tau and/or alpha on known-class validation data so that a rule's false-alert
rate (the share of known-class samples it flags as novel) matches a target. The
combined rule uses one shared validation quantile for tau and alpha, so its thresholds and
flagged set grow monotonically with the target. The caller supplies the validation split;
nothing here is random.
"""

from typing import Any, Dict

import numpy as np

from firce.novelty.decision_rules import NOVELTY_RULES, apply_novelty_rule, max_softmax_confidence

DEFAULT_FA_TARGETS = (0.01, 0.05, 0.10)


def _quantile_threshold(signal: np.ndarray, target_fa: float) -> float:
    """Largest threshold t such that the rate of signal < t is <= target_fa."""
    k = int(np.floor(target_fa * signal.shape[0] + 1e-9))
    return float(np.sort(signal)[k])


def _select_combined(confidence: np.ndarray, max_p: np.ndarray, target_fa: float) -> tuple[float, float]:
    """
    Shared-quantile thresholds for the combined rule.

    tau and alpha are the k-th order statistics of confidence and max_p for one shared rank k, so the
    false-alert rate (confidence < tau) & (max_p < alpha) is nondecreasing in k. Returns the pair for the
    largest k whose rate is at or below target_fa, found by binary search; k = 0 flags nothing, so a
    solution always exists. Ties resolve to the strict < comparison, which never over-flags.
    """
    n = confidence.shape[0]
    sorted_c = np.sort(confidence)
    sorted_p = np.sort(max_p)

    def rate(k: int) -> float:
        return float(np.mean((confidence < sorted_c[k]) & (max_p < sorted_p[k])))

    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if rate(mid) <= target_fa:
            lo = mid
        else:
            hi = mid - 1
    return float(sorted_c[lo]), float(sorted_p[lo])


def select_thresholds(
    rule: str, probas_val: np.ndarray, all_class_p_values_val: Dict[Any, np.ndarray], target_fa: float
) -> Dict[str, float]:
    """
    Select tau and alpha for a novelty rule on known-class validation data.

    Args:
        rule: One of NOVELTY_RULES.
        probas_val: Validation probability matrix of shape (n_samples, n_classes).
        all_class_p_values_val: Per-class validation p-values, as from compute_all_class_p_values.
        target_fa: Target false-alert rate in (0, 1).

    Returns:
        {'tau', 'alpha', 'achieved_fa'}; the unused threshold of a single-signal rule is
        inert (alpha 0.0 for confidence, tau 1.0 for conformal). achieved_fa is the flag
        rate of apply_novelty_rule on the validation data and never exceeds target_fa.

    Raises:
        ValueError: On an unknown rule, a target outside (0, 1), or empty input.
    """
    if rule not in NOVELTY_RULES:
        raise ValueError(f'Unknown novelty rule {rule!r}; expected one of {NOVELTY_RULES}')
    if not 0.0 < target_fa < 1.0:
        raise ValueError(f'target_fa must be in (0, 1), got {target_fa}')
    if probas_val.shape[0] == 0 or not all_class_p_values_val:
        raise ValueError('Validation data is empty')

    confidence = max_softmax_confidence(probas_val)
    max_p = np.max(np.stack(list(all_class_p_values_val.values()), axis=1), axis=1)
    if rule == 'confidence':
        tau, alpha = _quantile_threshold(confidence, target_fa), 0.0
    elif rule == 'conformal':
        tau, alpha = 1.0, _quantile_threshold(max_p, target_fa)
    else:
        tau, alpha = _select_combined(confidence, max_p, target_fa)
    flags = apply_novelty_rule(rule, probas_val, all_class_p_values_val, tau, alpha)
    return {'tau': tau, 'alpha': alpha, 'achieved_fa': float(np.mean(flags))}
