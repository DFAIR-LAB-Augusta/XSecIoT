import time

import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier

from firce.conformalEval.ice import InductiveConformalEvaluator
from firce.novelty.decision_rules import compute_all_class_p_values, is_novel, max_softmax_confidence
from firce.novelty.evaluation import evaluate_closed_world, evaluate_explanation_utility, evaluate_open_world_novelty
from firce.novelty.explain import explain_with_shap
from firce.utils.perf_stats import PerformanceStats


def test_evaluate_closed_world_returns_standard_multiclass_metrics():
    y_true = np.array(['Benign', 'Benign', 'PortScan', 'PortScan', 'XMasAttack', 'XMasAttack'])
    y_pred = np.array(['Benign', 'PortScan', 'PortScan', 'PortScan', 'XMasAttack', 'Benign'])

    result = evaluate_closed_world(y_true, y_pred)

    assert 0.0 <= result['accuracy'] <= 1.0
    assert 0.0 <= result['macro_precision'] <= 1.0
    assert 0.0 <= result['macro_recall'] <= 1.0
    assert 0.0 <= result['macro_f1'] <= 1.0
    assert 0.0 <= result['weighted_precision'] <= 1.0
    assert 0.0 <= result['weighted_recall'] <= 1.0
    assert 0.0 <= result['weighted_f1'] <= 1.0
    assert result['confusion_matrix'].shape == (3, 3)
    assert set(result['labels']) == {'Benign', 'PortScan', 'XMasAttack'}


def test_evaluate_closed_world_perfect_predictions_score_one():
    y_true = np.array(['Benign', 'PortScan', 'XMasAttack'])
    y_pred = np.array(['Benign', 'PortScan', 'XMasAttack'])

    result = evaluate_closed_world(y_true, y_pred)

    assert result['accuracy'] == 1.0
    assert result['macro_f1'] == 1.0


def test_evaluate_open_world_novelty_returns_detection_metrics_and_alert_burden():
    is_actually_unknown = np.array([False, False, False, True, True, True])
    novelty_flags = np.array([False, False, True, True, True, False])  # 1 false positive, 1 false negative
    novelty_scores = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.4])  # higher = more novel

    result = evaluate_open_world_novelty(is_actually_unknown, novelty_flags, novelty_scores)

    assert 0.0 <= result['precision'] <= 1.0
    assert 0.0 <= result['recall'] <= 1.0
    assert 0.0 <= result['f1'] <= 1.0
    assert 0.0 <= result['auroc'] <= 1.0
    assert 0.0 <= result['auprc'] <= 1.0
    assert result['alert_burden'] == novelty_flags.mean()


def test_evaluate_open_world_novelty_perfect_detection_scores_one():
    is_actually_unknown = np.array([False, False, True, True])
    novelty_flags = np.array([False, False, True, True])
    novelty_scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = evaluate_open_world_novelty(is_actually_unknown, novelty_flags, novelty_scores)

    assert result['precision'] == 1.0
    assert result['recall'] == 1.0
    assert result['f1'] == 1.0
    assert result['auroc'] == 1.0


def test_evaluate_explanation_utility_returns_latency_stats():
    explanations = [
        {'predicted_class': 'Benign', 'contributions': {'a': 0.5, 'b': 0.1}},
        {'predicted_class': 'Benign', 'contributions': {'a': 0.4, 'b': 0.2}},
    ]
    latencies = [0.01, 0.02]

    result = evaluate_explanation_utility(explanations, latencies)

    assert result['latency_mean_seconds'] == pytest.approx(0.015)
    assert result['latency_p50_seconds'] > 0
    assert result['latency_max_seconds'] == 0.02


def test_evaluate_explanation_utility_perfect_agreement_scores_one():
    explanations = [
        {'predicted_class': 'Benign', 'contributions': {'a': 0.9, 'b': 0.5, 'c': 0.1}},
        {'predicted_class': 'Benign', 'contributions': {'a': 0.8, 'b': 0.4, 'c': 0.2}},
    ]
    latencies = [0.01, 0.01]

    result = evaluate_explanation_utility(explanations, latencies, top_k=2)

    # Both explanations' top-2 features by |contribution| are {'a', 'b'} - full agreement.
    assert result['attribution_consistency_overall'] == 1.0
    assert result['attribution_consistency_by_class']['Benign'] == 1.0


def test_evaluate_explanation_utility_disagreement_scores_less_than_one():
    explanations = [
        {'predicted_class': 'Benign', 'contributions': {'a': 0.9, 'b': 0.1, 'c': 0.05}},
        {'predicted_class': 'Benign', 'contributions': {'a': 0.1, 'b': 0.05, 'c': 0.9}},
    ]
    latencies = [0.01, 0.01]

    result = evaluate_explanation_utility(explanations, latencies, top_k=1)

    # Top-1 feature disagrees entirely ('a' vs 'c') -> Jaccard overlap 0.
    assert result['attribution_consistency_overall'] == 0.0


def test_evaluate_explanation_utility_single_explanation_per_class_has_no_consistency_score():
    explanations = [{'predicted_class': 'Benign', 'contributions': {'a': 0.9}}]
    latencies = [0.01]

    result = evaluate_explanation_utility(explanations, latencies)

    # No pair to compare -> consistency is undefined for this class, not 0/1.
    assert result['attribution_consistency_by_class'] == {}


def _make_open_world_fixture(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float64)
    y = np.select([X[:, 0] > 0.5, X[:, 1] > 0.5], ['PortScan', 'XMasAttack'], default='Benign')
    return X, y


def test_eval_harness_end_to_end_open_world_scenario():
    # RandomForestClassifier (not DecisionTreeClassifier): confirmed via
    # direct execution that a plain decision tree is too discrete/overconfident
    # for a genuine open-world demonstration (held-out-class max-softmax was a
    # flat 1.0, identical to known classes) - RandomForest gives real signal.
    X, y = _make_open_world_fixture()
    held_out_mask = y == 'XMasAttack'
    known_mask = ~held_out_mask

    ice = InductiveConformalEvaluator(RandomForestClassifier(random_state=0), calibration_split=0.3, random_state=0)
    ice.calibrate(X[known_mask], y[known_mask], PerformanceStats())  # XMasAttack never seen in training

    probas = ice.model.predict_proba(X)
    all_class_p_values = compute_all_class_p_values(ice.model, ice.calibration_scores, X)
    novelty_flags = is_novel(probas, all_class_p_values, tau=0.8, alpha=0.5)
    novelty_scores = 1.0 - max_softmax_confidence(probas)

    # Stage 1: closed-world metrics, evaluated only on the known-class subset
    # (the model was never trained to recognize XMasAttack as a class).
    y_pred_known = ice.model.predict(X[known_mask])
    closed_world = evaluate_closed_world(y[known_mask], y_pred_known)
    assert closed_world['accuracy'] > 0.8  # RF should do well on the classes it was actually trained on

    # Stage 2: open-world detection - is_actually_unknown ground truth is
    # exactly held_out_mask, since XMasAttack was excluded from training.
    open_world = evaluate_open_world_novelty(held_out_mask, novelty_flags, novelty_scores)
    assert open_world['recall'] > 0.8  # most held-out-class samples should be flagged
    assert open_world['auroc'] > 0.8

    # Stage 3: explanation utility, generated only for flagged (held-out-class) events.
    flagged_indices = np.flatnonzero(novelty_flags)[:5]  # cap at 5 to keep the test fast
    explanations = []
    latencies = []
    feature_names = ['f0', 'f1', 'f2', 'f3']
    for idx in flagged_indices:
        start = time.perf_counter()
        explanation = explain_with_shap(ice.model, X[known_mask], X[idx], feature_names, model_type='tree')
        latencies.append(time.perf_counter() - start)
        explanations.append(explanation)

    explanation_utility = evaluate_explanation_utility(explanations, latencies, top_k=2)
    assert explanation_utility['latency_mean_seconds'] > 0
