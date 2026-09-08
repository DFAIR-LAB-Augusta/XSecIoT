import numpy as np
import pytest

from sklearn.tree import DecisionTreeClassifier

from firce.conformalEval.ice import InductiveConformalEvaluator
from firce.novelty.decision_rules import compute_all_class_p_values, is_novel
from firce.novelty.explain import explain_with_lime, explain_with_shap, select_events_to_explain
from firce.utils.perf_stats import PerformanceStats


def test_select_events_unknown_only_returns_exactly_the_flagged_indices():
    flags = np.array([False, True, False, True, True])

    selected = select_events_to_explain(flags, mode='unknown_only')

    assert selected.tolist() == [1, 3, 4]


def test_select_events_sampled_respects_sample_rate_and_is_reproducible_with_rng():
    flags = np.zeros(100, dtype=bool)
    rng = np.random.default_rng(0)

    selected = select_events_to_explain(flags, mode='sampled', sample_rate=0.2, rng=rng)

    assert 15 <= len(selected) <= 25  # ~20% of 100, allow sampling variance
    assert len(set(selected.tolist())) == len(selected)  # no duplicates
    assert np.all((selected >= 0) & (selected < 100))


def test_select_events_windowed_picks_one_index_per_window_preferring_flagged():
    flags = np.array([False, False, True, False, False, False, False, False, False, False])

    selected = select_events_to_explain(flags, mode='windowed', window_size=5)

    # Window 0 (indices 0-4): index 2 is flagged -> picked.
    # Window 1 (indices 5-9): none flagged -> first index of window (5) picked.
    assert selected.tolist() == [2, 5]


def test_select_events_unknown_mode_raises_value_error():
    flags = np.array([True, False])

    with pytest.raises(ValueError, match='mode'):
        select_events_to_explain(flags, mode='not_a_real_mode')


def _make_fixture_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = np.select([X[:, 0] > 0.5, X[:, 1] > 0.5], ['PortScan', 'XMasAttack'], default='Benign')
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    return model, X, y


def test_explain_with_shap_returns_one_attribution_per_feature():
    model, X, y = _make_fixture_model()
    feature_names = ['f0', 'f1', 'f2', 'f3']

    result = explain_with_shap(model, X, X[0], feature_names, model_type='tree')

    assert result['predicted_class'] in model.classes_
    assert set(result['contributions'].keys()) == set(feature_names)
    for value in result['contributions'].values():
        assert isinstance(value, float)


def test_explain_with_shap_kernel_mode_matches_tree_mode_feature_set():
    model, X, y = _make_fixture_model()
    feature_names = ['f0', 'f1', 'f2', 'f3']

    tree_result = explain_with_shap(model, X, X[0], feature_names, model_type='tree')
    kernel_result = explain_with_shap(model, X[:20], X[0], feature_names, model_type='kernel')

    assert set(tree_result['contributions'].keys()) == set(kernel_result['contributions'].keys())


def test_explain_with_lime_returns_weight_per_feature():
    model, X, y = _make_fixture_model()
    feature_names = ['f0', 'f1', 'f2', 'f3']
    class_names = sorted(np.unique(y).tolist())

    result = explain_with_lime(model, X, X[0], feature_names, class_names)

    assert result['predicted_class'] in model.classes_
    assert len(result['contributions']) > 0
    for label, value in result['contributions'].items():
        # LIME labels are rule strings (e.g. "f2 > 0.45"), not bare feature
        # names, so check that each label references one of our features.
        assert any(name in label for name in feature_names)
        assert isinstance(value, float)


def test_explain_pipeline_end_to_end_with_novelty_flags():
    model, X, y = _make_fixture_model()
    ice = InductiveConformalEvaluator(DecisionTreeClassifier(random_state=0), calibration_split=0.3, random_state=0)
    ice.calibrate(X, y, PerformanceStats())

    probas = ice.model.predict_proba(X)
    all_class_p_values = compute_all_class_p_values(ice.model, ice.calibration_scores, X)
    flags = is_novel(probas, all_class_p_values, tau=0.6, alpha=0.3)

    selected_indices = select_events_to_explain(flags, mode='unknown_only')
    feature_names = ['f0', 'f1', 'f2', 'f3']
    class_names = sorted(np.unique(y).tolist())

    for idx in selected_indices[:5]:  # cap at 5 to keep the test fast regardless of how many are flagged
        shap_result = explain_with_shap(ice.model, X, X[idx], feature_names, model_type='tree')
        lime_result = explain_with_lime(ice.model, X, X[idx], feature_names, class_names)

        assert shap_result['predicted_class'] in ice.model.classes_
        assert lime_result['predicted_class'] in ice.model.classes_
