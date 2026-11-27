import numpy as np
import pandas as pd
import pytest

from firce.experiments.presets import PRESETS, DatasetPreset
from firce.experiments.scoring import build_runtime, score_frame
from firce.experiments.split import split_device_time_blocks

KNOWN = ['Benign', 'TCP_SYN_Flood', 'UDP_Flood']
HELD_OUT = 'HTTP_Flood'
PRESET = DatasetPreset('toy', 'unused.csv', 'Label', (HELD_OUT,), '%d-%m-%Y %H:%M', 'device_id')


def _make_flows(n_per_device=300, devices=3, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for device in range(devices):
        n = n_per_device
        labels = rng.choice(KNOWN + [HELD_OUT], size=n, p=[0.4, 0.25, 0.25, 0.1])
        shift = np.array([KNOWN.index(lbl) if lbl in KNOWN else 3 for lbl in labels], dtype=float)
        minutes = np.sort(rng.integers(0, 24 * 60, size=n))
        stamps = pd.to_datetime('2024-08-13') + pd.to_timedelta(minutes, unit='m')
        frames.append(
            pd.DataFrame({
                'device_id': device,
                'session_id': np.arange(n) + device * n,
                'src_ip': '10.0.0.1',
                'dst_ip': '10.0.0.2',
                'src_port': rng.integers(1024, 65535, size=n),
                'dst_port': rng.integers(1, 1024, size=n),
                'protocol': rng.integers(0, 2, size=n),
                'timestamp': stamps.strftime('%d-%m-%Y %H:%M'),
                'flow_duration': rng.random(n) * 10 + shift * 50,
                'tot_fwd_pkts': rng.integers(1, 20, size=n) + (shift * 10).astype(int),
                'tot_bwd_pkts': rng.integers(0, 20, size=n),
                'totlen_fwd_pkts': rng.random(n) * 100 + shift * 300,
                'totlen_bwd_pkts': rng.random(n) * 100,
                'Label': labels,
            })
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope='module')
def flows():
    return _make_flows()


def _ids(frame):
    return set(frame['session_id'])


def test_experiment_scoring_preset_path_is_repo_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    resolved = PRESETS['dfair'].resolved_path
    assert resolved.is_absolute()
    assert resolved.name == 'combined_data_with_okpVacc_modified.csv'
    assert resolved.exists()


def test_experiment_scoring_split_disjoint_and_complete(flows):
    result = split_device_time_blocks(flows, PRESET, seed=0, block_minutes=120)
    parts = [_ids(result.train), _ids(result.val), _ids(result.stream)]
    assert sum(len(p) for p in parts) == len(flows)
    assert parts[0].isdisjoint(parts[1])
    assert parts[0].isdisjoint(parts[2])
    assert parts[1].isdisjoint(parts[2])
    assert set(result.train['MC_Label']) <= set(KNOWN)
    assert len(result.train) > 0
    assert len(result.val) > 0
    assert len(result.stream) > 0


def test_experiment_scoring_held_out_only_in_stream(flows):
    result = split_device_time_blocks(flows, PRESET, seed=1, block_minutes=120)
    assert HELD_OUT in set(result.stream['MC_Label'])
    assert HELD_OUT not in set(result.train['MC_Label'])
    assert HELD_OUT not in set(result.val['MC_Label'])


def test_experiment_scoring_split_deterministic_per_seed(flows):
    a = split_device_time_blocks(flows, PRESET, seed=3, block_minutes=120)
    b = split_device_time_blocks(flows, PRESET, seed=3, block_minutes=120)
    c = split_device_time_blocks(flows, PRESET, seed=4, block_minutes=120)
    for name in ('train', 'val', 'stream'):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))
    assert _ids(a.stream) != _ids(c.stream)


def test_experiment_scoring_stratified_fallback_on_starved_val(flows):
    result = split_device_time_blocks(flows, PRESET, seed=0, block_minutes=120, val_frac=0.0)
    assert result.val_strategy == 'stratified'
    assert set(result.val['MC_Label']) == set(KNOWN)
    assert _ids(result.train).isdisjoint(_ids(result.val))
    assert HELD_OUT not in set(result.val['MC_Label'])


def test_experiment_scoring_score_frame_shapes(flows, tmp_path):
    result = split_device_time_blocks(flows, PRESET, seed=0, block_minutes=120)
    runtime = build_runtime(result.train, PRESET, tmp_path, seed=0, model_variant='dt')
    assert (tmp_path / 'CETrain' / 'train.csv').exists()
    assert (tmp_path / 'multi_class_models').is_dir()

    scored = score_frame(runtime, result.stream, PRESET)
    n = len(result.stream)
    assert scored.probas.shape == (n, len(KNOWN))
    assert set(scored.p_values) == set(KNOWN)
    assert all(v.shape == (n,) for v in scored.p_values.values())
    assert scored.y_true.shape == scored.device.shape == scored.timestamp.shape == (n,)
    assert HELD_OUT in set(scored.y_true)
    assert np.array_equal(scored.device, result.stream['device_id'].to_numpy())
