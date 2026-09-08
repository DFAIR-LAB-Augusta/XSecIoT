import numpy as np
import pandas as pd
import pytest
import torch

from firce.ce_model_training import train_ce_multiclass
from firce.utils.config import CEType, ModelType, ModelVariant, SimulationConfig

DEVICE = torch.device('cpu')


def _make_config(tmp_path, **overrides):
    dummy = tmp_path / 'dummy.csv'
    if not dummy.exists():
        dummy.write_text('a\n1\n')
    defaults = dict(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant.DT,
        ce_type=CEType.NONE,
        aggregated_path=dummy,
        flows_path=dummy,
        is_unsw=False,
        seed=0,
        device=DEVICE,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def _make_multiclass_csv(tmp_path, n=60, seed=0, dirname='DS'):
    rng = np.random.default_rng(seed)
    ds_dir = tmp_path / dirname
    ds_dir.mkdir(exist_ok=True)
    csv_path = ds_dir / 'flows.csv'
    labels = np.array(['Benign', 'PortScan', 'XMasAttack'])
    idx = rng.integers(0, 3, size=n)
    df = pd.DataFrame({
        'device_id': range(n),
        'session_id': range(n),
        'src_ip': ['192.168.1.1'] * n,
        'dst_ip': ['192.168.1.2'] * n,
        'src_port': rng.integers(1024, 65535, size=n),
        'dst_port': rng.integers(1, 1024, size=n),
        'protocol': rng.integers(0, 2, size=n),
        'timestamp': ['01-01-2020 00:00'] * n,
        'flow_duration': rng.random(n) * 100,
        'tot_fwd_pkt': rng.integers(1, 50, size=n),
        'tot_bwd_pkts': rng.integers(0, 50, size=n),
        'totlen_fwd_pkts': rng.random(n) * 1000,
        'totlen_bwd_pkts': rng.random(n) * 1000,
        'MC_Label': labels[idx],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_unsw_multiclass_csv(tmp_path, n=60, seed=0, dirname='UNSW_DS'):
    rng = np.random.default_rng(seed)
    ds_dir = tmp_path / dirname
    ds_dir.mkdir(exist_ok=True)
    csv_path = ds_dir / 'flows.csv'
    labels = np.array(['Benign', 'DoS', 'Reconnaissance'])
    idx = rng.integers(0, 3, size=n)
    df = pd.DataFrame({
        'src_ip': ['10.0.0.1'] * n,
        'dst_ip': ['10.0.0.2'] * n,
        'src_port': rng.integers(1024, 65535, size=n),
        'dst_port': rng.integers(1, 1024, size=n),
        'protocol': rng.integers(0, 2, size=n),
        'flow_duration': rng.random(n) * 100,
        'in_pkts': rng.integers(1, 50, size=n),
        'out_pkts': rng.integers(0, 50, size=n),
        'in_bytes': rng.random(n) * 1000,
        'out_bytes': rng.random(n) * 1000,
        'Label': (idx != 0).astype(int),
        'Attack': labels[idx],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.parametrize('variant', [ModelVariant.DT, ModelVariant.KNN, ModelVariant.RF, ModelVariant.SVM])
def test_train_ce_multiclass_classical_variants(tmp_path, monkeypatch, variant):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=variant)

    outdir = train_ce_multiclass(config, str(csv_path), variant=variant, use_pca=False)

    assert outdir.resolve() == (tmp_path / 'multi_class_models' / 'DS').resolve()
    assert (outdir / 'scaler_multi.pkl').exists()
    assert (outdir / 'label_encoder_multi.pkl').exists()
    assert (outdir / f'{variant.value}_model_multi.pkl').exists()
    assert not (outdir / 'pca_multi.pkl').exists()


def test_train_ce_multiclass_with_pca_writes_pca_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)

    outdir = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=True)

    assert (outdir / 'pca_multi.pkl').exists()


def test_train_ce_multiclass_feedforward(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.FEEDFORWARD)

    outdir = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.FEEDFORWARD, use_pca=False)

    ckpt_path = outdir / 'feedforward_model_multi.pt'
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert ckpt['num_classes'] == 3
    assert ckpt['input_dim'] > 0


def test_train_ce_multiclass_xgb(tmp_path, monkeypatch):
    pytest.importorskip('xgboost')
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.XGB)

    outdir = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.XGB, use_pca=False)

    assert (outdir / 'xgb_model_multi.pkl').exists()


def test_train_ce_multiclass_unsw_uses_attack_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_unsw_multiclass_csv(tmp_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT, is_unsw=True)

    outdir = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False)

    assert outdir.resolve() == (tmp_path / 'multi_class_models' / 'UNSW_DS').resolve()
    assert (outdir / 'label_encoder_multi.pkl').exists()
    import joblib

    encoder = joblib.load(outdir / 'label_encoder_multi.pkl')
    assert sorted(encoder.classes_.tolist()) == ['Benign', 'DoS', 'Reconnaissance']


def test_train_ce_multiclass_missing_label_column_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    pd.DataFrame({'flow_duration': [1, 2, 3], 'tot_fwd_pkt': [1, 2, 3]}).to_csv(csv_path, index=False)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)

    with pytest.raises(ValueError, match="must contain an 'MC_Label' column"):
        train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False)


def test_train_ce_multiclass_too_few_classes_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds_dir = tmp_path / 'DS'
    ds_dir.mkdir()
    csv_path = ds_dir / 'flows.csv'
    pd.DataFrame({
        'flow_duration': [1, 2, 3],
        'tot_fwd_pkt': [1, 2, 3],
        'MC_Label': ['Benign', 'Benign', 'Benign'],
    }).to_csv(csv_path, index=False)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)

    with pytest.raises(ValueError, match='at least 2 distinct MC_Label classes'):
        train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False)


def test_train_ce_multiclass_with_df_log_writes_retraining_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    df_log = pd.read_csv(csv_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)

    outdir = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False, df_log=df_log)

    assert outdir.name.startswith('Model_dt_Retraining_')
    assert (outdir / 'dt_model_multi.pkl').exists()
    assert (outdir / 'label_encoder_multi.pkl').exists()


def test_train_ce_multiclass_with_df_log_cleans_up_old_retraining_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_multiclass_csv(tmp_path)
    df_log = pd.read_csv(csv_path)
    config = _make_config(tmp_path, model_variant=ModelVariant.DT)

    first = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False, df_log=df_log)
    second = train_ce_multiclass(config, str(csv_path), variant=ModelVariant.DT, use_pca=False, df_log=df_log)

    assert not first.exists()
    assert second.exists()
