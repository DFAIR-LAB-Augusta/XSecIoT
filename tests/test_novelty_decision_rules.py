import numpy as np
import pytest

from sklearn.tree import DecisionTreeClassifier

from firce.conformalEval.ice import InductiveConformalEvaluator
from firce.novelty.decision_rules import (
    compute_all_class_p_values,
    conformal_prediction_set_size,
    entropy_confidence,
    is_novel,
    margin_confidence,
    temperature_scaled_confidence,
)
from firce.utils.perf_stats import PerformanceStats


def _make_multiclass_data(n=90, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float64)
    y = np.select(
        [X[:, 0] > 0.5, X[:, 1] > 0.5],
        ['PortScan', 'XMasAttack'],
        default='Benign',
    )
    return X, y


def _make_calibrated_ice(seed=0):
    X, y = _make_multiclass_data(seed=seed)
    ice = InductiveConformalEvaluator(DecisionTreeClassifier(random_state=0), calibration_split=0.3, random_state=0)
    ice.calibrate(X, y, PerformanceStats())
    return ice, X, y


def test_compute_all_class_p_values_covers_every_known_class():
    ice, X, y = _make_calibrated_ice()

    result = compute_all_class_p_values(ice.model, ice.calibration_scores, X[:10])

    assert set(result.keys()) == set(np.unique(y).tolist())
    for cls, p_values in result.items():
        assert p_values.shape == (10,)
        assert np.all((p_values >= 0.0) & (p_values <= 1.0))


def test_compute_all_class_p_values_matches_predicted_class_p_value():
    ice, X, y = _make_calibrated_ice()

    all_class_result = compute_all_class_p_values(ice.model, ice.calibration_scores, X[:10])
    predicted_result = ice.predict_p_values(X[:10])

    for i, cls in enumerate(predicted_result['class']):
        assert all_class_result[cls][i] == pytest.approx(predicted_result['p_value'][i])


def test_is_novel_flags_low_confidence_low_conformal_support_samples():
    ice, X, y = _make_calibrated_ice()
    probas = ice.model.predict_proba(X)
    all_class_p_values = compute_all_class_p_values(ice.model, ice.calibration_scores, X)

    # Very permissive thresholds (strictly greater than the maximum possible
    # value of either signal, both of which are bounded in [0, 1]): every
    # sample must satisfy both "< tau" and "all p_c < alpha", so all flagged.
    flags_permissive = is_novel(probas, all_class_p_values, tau=1.0 + 1e-9, alpha=1.0 + 1e-9)
    assert flags_permissive.dtype == bool
    assert flags_permissive.shape == (len(X),)
    assert flags_permissive.all()

    # Very strict thresholds: nothing should be flagged (tau=0 means max_softmax
    # can never be < 0; alpha=0 means no p-value can ever be < 0).
    flags_strict = is_novel(probas, all_class_p_values, tau=0.0, alpha=0.0)
    assert not flags_strict.any()


def test_is_novel_requires_both_conditions_not_just_one():
    # Construct a controlled 2-sample, 2-class case by hand.
    probas = np.array([
        [0.9, 0.1],  # high confidence in class 'a' -> should NOT be flagged even if p-values are low
        [0.5, 0.5],  # low confidence -> flagged only if ALSO all p-values are low
    ])
    all_class_p_values = {
        'a': np.array([0.01, 0.01]),
        'b': np.array([0.01, 0.01]),
    }

    flags = is_novel(probas, all_class_p_values, tau=0.6, alpha=0.05)

    assert not flags[0]  # max_softmax=0.9 >= tau=0.6, condition fails regardless of p-values
    assert flags[1]  # max_softmax=0.5 < tau=0.6 AND all p-values < alpha=0.05


def test_margin_confidence_is_difference_between_top_two_probabilities():
    probas = np.array([
        [0.7, 0.2, 0.1],
        [0.4, 0.35, 0.25],
        [1.0, 0.0, 0.0],
    ])

    margins = margin_confidence(probas)

    assert margins == pytest.approx([0.5, 0.05, 1.0])


def test_entropy_confidence_is_zero_for_certain_prediction_and_positive_for_uniform():
    certain = np.array([[1.0, 0.0, 0.0]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])

    certain_entropy = entropy_confidence(certain)
    uniform_entropy = entropy_confidence(uniform)

    assert certain_entropy[0] == pytest.approx(0.0)
    assert uniform_entropy[0] > certain_entropy[0]
    assert uniform_entropy[0] == pytest.approx(np.log(3))


def test_temperature_scaled_confidence_at_temperature_one_is_unchanged():
    probas = np.array([[0.7, 0.2, 0.1]])

    scaled = temperature_scaled_confidence(probas, temperature=1.0)

    assert scaled[0] == pytest.approx(0.7)


def test_temperature_scaled_confidence_above_one_flattens_distribution():
    probas = np.array([[0.7, 0.2, 0.1]])

    scaled = temperature_scaled_confidence(probas, temperature=5.0)

    assert scaled[0] < 0.7
    assert scaled[0] > 1 / 3


def test_conformal_prediction_set_size_counts_non_rejected_classes():
    all_class_p_values = {
        'a': np.array([0.9, 0.01, 0.2]),
        'b': np.array([0.01, 0.01, 0.2]),
        'c': np.array([0.01, 0.01, 0.01]),
    }

    sizes = conformal_prediction_set_size(all_class_p_values, alpha=0.05)

    # sample 0: only 'a' survives (0.9 >= 0.05) -> size 1
    # sample 1: none survive -> size 0 (a genuine "unknown" signal - empty prediction set)
    # sample 2: 'a' and 'b' survive (0.2 >= 0.05) -> size 2
    assert sizes.tolist() == [1, 0, 2]


def test_conformal_prediction_set_size_on_real_calibrated_evaluator():
    ice, X, y = _make_calibrated_ice()
    all_class_p_values = compute_all_class_p_values(ice.model, ice.calibration_scores, X[:10])

    sizes = conformal_prediction_set_size(all_class_p_values, alpha=0.05)

    assert sizes.shape == (10,)
    assert np.all(sizes >= 0)
    assert np.all(sizes <= len(all_class_p_values))


def test_is_novel_end_to_end_flagged_set_grows_monotonically_as_thresholds_loosen():
    # Real end-to-end sanity check against a genuinely calibrated evaluator (not
    # hand-constructed arrays): loosening tau and/or alpha can only ever add
    # flagged samples, never remove one that was already flagged at stricter
    # thresholds. This must hold for any correct implementation of the rule,
    # regardless of which specific samples happen to be flagged for this
    # particular classifier/dataset combination.
    #
    # (A more literal "does this specific out-of-range point get flagged"
    # sanity check was tried and dropped: this dataset's labeling rule is a
    # simple threshold on X[:,0]/X[:,1] that partitions the *entire* feature
    # space, so there is no "outside all known regions" to construct - every
    # classifier tried (DecisionTree, RandomForest, KNN) confidently
    # extrapolates any far-away point into one of the existing classes, since
    # the same generating rule still cleanly applies at extreme values. The
    # monotonicity invariant below is the correctness property that actually
    # matters and holds regardless of this dataset's geometry.)
    ice, X, y = _make_calibrated_ice()
    probas = ice.model.predict_proba(X)
    all_class_p_values = compute_all_class_p_values(ice.model, ice.calibration_scores, X)

    strict_flags = is_novel(probas, all_class_p_values, tau=0.3, alpha=0.1)
    medium_flags = is_novel(probas, all_class_p_values, tau=0.6, alpha=0.3)
    loose_flags = is_novel(probas, all_class_p_values, tau=0.9, alpha=0.6)

    assert np.all(loose_flags[strict_flags])
    assert np.all(loose_flags[medium_flags])
    assert loose_flags.sum() >= medium_flags.sum() >= strict_flags.sum()
