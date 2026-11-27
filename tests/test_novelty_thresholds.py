import numpy as np
import pytest

from firce.novelty.decision_rules import NOVELTY_RULES, apply_novelty_rule
from firce.novelty.thresholds import DEFAULT_FA_TARGETS, select_thresholds

N_ROWS = 6000
N_CLASSES = 3


def _make_known_class_sample(rng, calibration_scores):
    """Synthetic known-class sample: Dirichlet-like probas and p-values from the calibration scores.

    The p-value for class c is the share of class-c calibration scores (1 - proba_c) at or above the
    row's score, a stand-in for compute_all_class_p_values that avoids fitting a model.
    """
    probas = rng.dirichlet(np.full(N_CLASSES, 0.4), size=N_ROWS)
    p_values = {}
    for c in range(N_CLASSES):
        ref = np.sort(calibration_scores[c])
        scores = 1.0 - probas[:, c]
        p_values[c] = (ref.shape[0] - np.searchsorted(ref, scores, side='left') + 1) / (ref.shape[0] + 1)
    return probas, p_values


@pytest.fixture(scope='module')
def known_class_splits():
    rng = np.random.default_rng(178)
    calibration_scores = {c: 1.0 - rng.beta(2.0 + c, 1.0, size=2000) for c in range(N_CLASSES)}
    return _make_known_class_sample(rng, calibration_scores), _make_known_class_sample(rng, calibration_scores)


@pytest.mark.parametrize(['rule'], [[rule] for rule in NOVELTY_RULES])
@pytest.mark.parametrize(['target'], [[target] for target in DEFAULT_FA_TARGETS])
def test_matched_false_alert_held_back_rate(known_class_splits, rule, target):
    (probas_val, p_val), (probas_test, p_test) = known_class_splits
    selected = select_thresholds(rule, probas_val, p_val, target)
    assert selected['achieved_fa'] <= target
    flags = apply_novelty_rule(rule, probas_test, p_test, selected['tau'], selected['alpha'])
    assert abs(float(np.mean(flags)) - target) <= 0.01


@pytest.mark.parametrize(['rule'], [[rule] for rule in NOVELTY_RULES])
def test_matched_false_alert_deterministic(known_class_splits, rule):
    (probas_val, p_val), _ = known_class_splits
    assert select_thresholds(rule, probas_val, p_val, 0.05) == select_thresholds(rule, probas_val, p_val, 0.05)


def test_matched_false_alert_unused_threshold_is_inert(known_class_splits):
    (probas_val, p_val), _ = known_class_splits
    assert select_thresholds('confidence', probas_val, p_val, 0.05)['alpha'] == 0.0
    assert select_thresholds('conformal', probas_val, p_val, 0.05)['tau'] == 1.0


def test_matched_false_alert_rejects_invalid_input(known_class_splits):
    (probas_val, p_val), _ = known_class_splits
    with pytest.raises(ValueError, match='bogus'):
        select_thresholds('bogus', probas_val, p_val, 0.05)
    for bad_target in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            select_thresholds('confidence', probas_val, p_val, bad_target)
    with pytest.raises(ValueError):
        select_thresholds('confidence', probas_val[:0], {c: v[:0] for c, v in p_val.items()}, 0.05)
