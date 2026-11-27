"""Tests for Study A evaluation (issue #181); synthetic ScoredSplit objects, no model training."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from firce.experiments.scoring import ScoredSplit
from firce.experiments.study_a import (
    build_scores_frame,
    build_sessions,
    evaluate_seed,
    make_run_id,
    summarize,
    write_results,
)
from firce.novelty.decision_rules import NOVELTY_RULES
from firce.novelty.thresholds import DEFAULT_FA_TARGETS

KNOWN = ['benign', 'dos']
HELD_OUT = ['scan']


def _split(n_known, n_unknown, seed, devices=('d1', 'd2')):
    rng = np.random.default_rng(seed)
    n = n_known + n_unknown
    y_true = np.array(['benign', 'dos'] * (n_known // 2) + ['scan'] * n_unknown)
    unknown = y_true == 'scan'
    top = np.where(unknown, rng.uniform(0.4, 0.8, n), rng.uniform(0.6, 1.0, n))
    probas = np.stack([top, 1.0 - top], axis=1)
    flip = y_true == 'dos'
    probas[flip] = probas[flip][:, ::-1]
    p_values = {name: np.where(unknown, rng.uniform(0.0, 0.3, n), rng.uniform(0.0, 1.0, n)) for name in KNOWN}
    return ScoredSplit(
        probas=probas,
        p_values=p_values,
        y_true=y_true,
        device=np.array([devices[i % len(devices)] for i in range(n)]),
        timestamp=np.arange(n, dtype=float) * 60.0,
    )


@pytest.fixture(scope='module')
def fixture_rows():
    val = _split(6000, 0, seed=1)
    stream = _split(6000, 300, seed=2)
    return val, stream, evaluate_seed(val, stream, HELD_OUT, KNOWN)


def test_study_a_one_row_per_rule_and_target(fixture_rows):
    _, _, rows = fixture_rows
    assert [(r['rule'], r['target_fa']) for r in rows] == [(r, t) for r in NOVELTY_RULES for t in DEFAULT_FA_TARGETS]
    for row in rows:
        assert abs(row['stream_known_fa'] - row['target_fa']) <= 0.03
        assert row['val_fa'] <= row['target_fa']


def test_study_a_closed_world_maps_argmax_to_class_names(fixture_rows):
    val, stream, rows = fixture_rows
    assert rows[0]['closed_macro_f1'] == pytest.approx(1.0)
    swapped = evaluate_seed(val, stream, HELD_OUT, list(reversed(KNOWN)))
    assert swapped[0]['closed_macro_f1'] == pytest.approx(0.0)


def test_study_a_rejects_misaligned_class_names(fixture_rows):
    val, stream, _ = fixture_rows
    with pytest.raises(ValueError, match='class_names'):
        evaluate_seed(val, stream, HELD_OUT, ['benign'])


def _timeline(times, labels, device='d1'):
    n = len(times)
    return ScoredSplit(
        probas=np.full((n, 2), 0.5),
        p_values={'a': np.full(n, 0.5)},
        y_true=np.array(labels),
        device=np.array([device] * n),
        timestamp=np.array(times, dtype=float),
    )


def test_study_a_two_bursts_make_two_attack_sessions():
    times = [0, 60, 600, 660, 10000, 10060, 12000]
    labels = ['benign', 'benign', 'scan', 'scan', 'scan', 'scan', 'benign']
    sessions = build_sessions(_timeline(times, labels), HELD_OUT, lookback_s=1800)
    attacks = [s for s in sessions if s['session_type'] == 'attack']
    assert [s['change_point'] for s in attacks] == [600.0, 10000.0]
    assert list(attacks[0]['index']) == [0, 1, 2, 3]
    assert list(attacks[1]['index']) == [4, 5]


def test_study_a_stable_sessions_contain_no_held_out_flow():
    times = list(range(0, 8000, 100))
    labels = ['scan' if 3000 <= t < 3500 else 'benign' for t in times]
    stream = _timeline(times, labels)
    stable = [s for s in build_sessions(stream, HELD_OUT, lookback_s=1800) if s['session_type'] == 'stable']
    assert stable
    for session in stable:
        assert 'scan' not in stream.y_true[session['index']]
        assert session['change_point'] is None


def test_study_a_delay_metrics_on_known_burst(fixture_rows):
    _, _, rows = fixture_rows
    for row in rows:
        assert row['n_attack_sessions'] >= 1
        assert 0 <= row['n_detected'] <= row['n_attack_sessions']
        assert row['stable_fa_rate'] is not None


def test_study_a_write_results_layout_and_roundtrip(tmp_path, fixture_rows):
    val, stream, rows = fixture_rows
    other = evaluate_seed(val, _split(6000, 300, seed=3), HELD_OUT, KNOWN)
    frame = build_scores_frame(stream, rows)
    out = tmp_path / 'run'
    write_results(out, {'seeds': [0, 1]}, {0: rows, 1: other}, {0: frame, 1: frame})

    assert json.loads((out / 'config.json').read_text()) == {'seeds': [0, 1]}
    assert json.loads((out / 'seed_0' / 'metrics.json').read_text())[0]['rule'] == 'confidence'
    summary = pd.read_csv(out / 'summary.csv')
    assert len(summary) == len(NOVELTY_RULES) * len(DEFAULT_FA_TARGETS)
    assert {'auroc_mean', 'auroc_std', 'stream_known_fa_mean', 'median_delay_s_mean', 'n_seeds'} <= set(summary.columns)
    assert (summary['n_seeds'] == 2).all()

    back = pd.read_parquet(out / 'seed_1' / 'scores.parquet')
    pd.testing.assert_frame_equal(back, frame)
    flag_columns = [c for c in back.columns if c.startswith('flag_')]
    assert len(flag_columns) == len(rows)
    assert {'device', 'timestamp', 'y_true', 'max_softmax', 'max_p'} <= set(back.columns)


def test_study_a_make_run_id_format():
    assert re.fullmatch(r'\d{8}T\d{6}Z-([0-9a-f]{4,40}|nogit)', make_run_id())


def test_summarize_non_null_counts():
    base = {'rule': 'r', 'target_fa': 0.05}
    per_seed = {
        0: [{**base, 'median_delay_s': None, 'never_set': None}],
        1: [{**base, 'median_delay_s': 4.0, 'never_set': None}],
        2: [{**base, 'median_delay_s': None, 'never_set': None}],
    }
    row = summarize(per_seed).iloc[0]
    assert row['median_delay_s_n'] == 1
    assert row['median_delay_s_mean'] == 4.0
    assert row['never_set_n'] == 0
    assert np.isnan(row['never_set_mean'])
    assert row['n_seeds'] == 3
