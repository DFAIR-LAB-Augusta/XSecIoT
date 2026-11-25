from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from firce.ce_model_training import train_ce_multiclass
from firce.classes import DEFAULT_HELD_OUT, KNOWN_CLASSES, canonicalize_label
from firce.runtime.bootstrap import load_training_frame
from firce.runtime.inference import _prepare_chunk
from firce.utils.config import ModelVariant

DFAIR_LABELS = ['Benign', 'XMasAttack', 'TCPAttack', 'UDPAttack', 'HTTPAttack']


def _labeled_csv(path, n=100, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'flow_duration': rng.random(n) * 100,
        'tot_bwd_pkts': rng.integers(0, 50, size=n),
        'totlen_fwd_pkts': rng.random(n) * 1000,
        'Label': [DFAIR_LABELS[i % len(DFAIR_LABELS)] for i in range(n)],
    })
    df.to_csv(path, index=False)
    return df


def test_class_registry_sizes_and_aliases():
    assert len(KNOWN_CLASSES) == 13
    assert len(DEFAULT_HELD_OUT) == 8
    assert canonicalize_label('HTTPAttack') == 'HTTP_Flood'
    assert canonicalize_label('XMasAttack') == 'XMAS_Flood'
    assert canonicalize_label('Slowloris') == 'Slowloris'


def test_class_registry_unmapped_label_raises():
    with pytest.raises(ValueError, match='NopeAttack'):
        canonicalize_label('NopeAttack')


def test_class_registry_config_rejects_non_canonical_held_out(tmp_path, sim_config_factory):
    with pytest.raises(ValueError, match='held_out'):
        sim_config_factory(tmp_path, held_out=('HTTPAttack',))


def test_heldout_class_excluded_from_train_and_calibration(tmp_path, monkeypatch, sim_config_factory):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    _labeled_csv(csv_path)
    config = sim_config_factory(
        tmp_path,
        aggregated_path=csv_path,
        flows_path=csv_path,
        label_column='Label',
        held_out=('HTTP_Flood',),
    )

    frame = load_training_frame(config)
    assert 'HTTP_Flood' not in set(frame['MC_Label'])
    assert {'Benign', 'XMAS_Flood', 'TCP_SYN_Flood', 'UDP_Flood'} == set(frame['MC_Label'])

    model_dir = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False)
    encoder = joblib.load(model_dir / 'label_encoder_multi.pkl')
    assert 'HTTP_Flood' not in encoder.classes_

    stream = _labeled_csv(tmp_path / 'stream.csv', n=20, seed=1)
    _, ground_truth = _prepare_chunk(SimpleNamespace(config=config), stream)
    assert 'HTTP_Flood' in set(ground_truth)
    assert len(ground_truth) == len(stream)
