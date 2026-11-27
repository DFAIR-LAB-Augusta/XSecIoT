"""Tests for the Direction 3 experiment driver and its command line (issue #181)."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from firce.novelty.decision_rules import NOVELTY_RULES
from firce.novelty.thresholds import DEFAULT_FA_TARGETS

LABELS = ['Benign', 'TCPAttack', 'UDPAttack', 'HTTPAttack']
FEATURES = ['flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'flow_pkts_s']


def _make_csv(path, devices=3, n=600, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for device in range(devices):
        labels = rng.choice(LABELS, size=n, p=[0.4, 0.25, 0.25, 0.1])
        shift = np.array([LABELS.index(label) for label in labels], dtype=float)
        minutes = np.sort(rng.integers(0, 12 * 60, size=n))
        stamps = pd.to_datetime('2024-08-13') + pd.to_timedelta(minutes, unit='m')
        frame = pd.DataFrame({
            'device_id': device,
            'session_id': np.arange(n) + device * n,
            'src_ip': '10.0.0.1',
            'dst_ip': '10.0.0.2',
            'src_port': rng.integers(1024, 65535, size=n),
            'dst_port': rng.integers(1, 1024, size=n),
            'protocol': rng.integers(0, 2, size=n),
            'timestamp': stamps.strftime('%d-%m-%Y %H:%M'),
        })
        for i, name in enumerate(FEATURES):
            frame[name] = rng.random(n) * 10 + shift * (30 + 10 * i)
        frame['Label'] = labels
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


@pytest.fixture(scope='module')
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('d3_driver')
    csv = tmp / 'flows.csv'
    _make_csv(csv)
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'firce.experiments',
            '--preset',
            'dfair',
            '--dataPath',
            str(csv),
            '--seeds',
            '0',
            '1',
            '--out',
            str(tmp / 'out'),
            '--studyB',
            '--studyBSamples',
            '3',
            '--studyC',
            '--modelVariant',
            'dt',
        ],
        cwd=tmp,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    printed = result.stdout.strip().splitlines()[-1]
    return tmp, tmp / 'out' / 'dfair', printed


def _run(run_dir):
    tmp, root, printed = run_dir
    (path,) = [p for p in root.iterdir() if p.is_dir()]
    assert str(path) == printed
    return path


def test_d3_driver_writes_results_layout(run_dir):
    path = _run(run_dir)
    for seed in (0, 1):
        for name in ('metrics.json', 'scores.parquet', 'study_b.json', 'study_c.jsonl'):
            assert (path / f'seed_{seed}' / name).is_file()
    summary = pd.read_csv(path / 'summary.csv')
    assert len(summary) == len(NOVELTY_RULES) * len(DEFAULT_FA_TARGETS) == 9
    assert set(summary['n_seeds']) == {2}


def test_d3_driver_config_records_arguments(run_dir):
    config = json.loads((_run(run_dir) / 'config.json').read_text())
    assert config['seeds'] == [0, 1]
    assert config['model_variant'] == 'dt'
    assert config['study_b'] is True
    assert config['study_c'] is True
    assert config['study_b_samples'] == 3
    assert config['git_sha']
    assert set(config['val_strategy']) == {'0', '1'}
    assert set(config['val_strategy'].values()) <= {'block', 'stratified'}


def test_d3_driver_study_outputs(run_dir):
    path = _run(run_dir)
    study_b = json.loads((path / 'seed_0' / 'study_b.json').read_text())
    assert 'latency_mean_seconds' in study_b
    assert 1 <= study_b['n_explained'] <= 3
    reports = [json.loads(line) for line in (path / 'seed_0' / 'study_c.jsonl').read_text().splitlines()]
    assert len(reports) == study_b['n_explained']
    assert all(report['template_report']['summary'] for report in reports)
    assert all(report['llm_report'] is None for report in reports)


def test_d3_driver_class_names_match_p_value_keys(run_dir):
    from firce.experiments.driver import class_names_of
    from firce.experiments.presets import PRESETS
    from firce.experiments.scoring import build_runtime, score_frame
    from firce.experiments.split import split_device_time_blocks

    tmp, _, _ = run_dir
    preset = PRESETS['dfair']
    data = pd.read_csv(tmp / 'flows.csv')
    split = split_device_time_blocks(data, preset, 0)
    runtime = build_runtime(split.train, preset, tmp / 'names_work', 0, 'dt')
    scored = score_frame(runtime, split.val, preset)
    assert set(class_names_of(runtime)) == set(scored.p_values)
    assert scored.features.shape[0] == len(split.val)
