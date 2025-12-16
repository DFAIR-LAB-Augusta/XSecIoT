# tests/test_ce_model_training.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.ce_model_training import train_ce_binary, train_ce_multiclass
from src.core.config import ModelVariant
from src.core.perf_stats import PerformanceStats

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class DummyConfig:
    """Minimal config stub with only attributes used by ce_model_training."""

    model_variant: ModelVariant
    use_pca: bool = False
    is_unsw: bool = False
    device: torch.device = torch.device('cpu')


def _patch_shortuuid(monkeypatch) -> None:
    import src.core.ce_model_training as ce_training

    def _fixed_random(self, length: int = 8) -> str:
        return 'TESTUUID'

    monkeypatch.setattr(ce_training.shortuuid.ShortUUID, 'random', _fixed_random)


def _toy_binary_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    labels = (['Benign'] * (n // 2)) + (['Attack'] * (n - n // 2))

    df = pd.DataFrame({
        'src_ip': ['1.1.1.1'] * n,
        'dst_ip': ['2.2.2.2'] * n,
        'src_port': rng.integers(1024, 65535, size=n),
        'dst_port': rng.integers(1, 1024, size=n),
        'protocol': rng.integers(0, 2, size=n),
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='s'),
        'flow_duration': rng.normal(1000, 10, size=n),
        'flow_byts_s': rng.normal(100, 5, size=n),
        'Label': labels,
    })

    df.loc[0, 'flow_byts_s'] = np.nan
    df.loc[1, 'flow_duration'] = np.nan
    return df


def _toy_binary_df_unsw_subset(n: int = 40) -> pd.DataFrame:
    """
    UNSW path enforces "no extra features beyond FINAL_LOG_COLUMNS" after _unsw_clean.
    We keep a strict subset of FINAL_LOG_COLUMNS plus Label (which _unsw_clean drops),
    and rely on the code to create BinLabel from Label before cleaning.
    """
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=n)

    return pd.DataFrame({
        'src_ip': ['10.0.0.1'] * n,
        'dst_ip': ['10.0.0.2'] * n,
        'src_port': rng.integers(1024, 65535, size=n),
        'dst_port': rng.integers(1, 1024, size=n),
        'protocol': rng.integers(0, 2, size=n),
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='s'),
        'flow_duration': rng.normal(10, 1, size=n),
        'flow_byts_s': rng.normal(20, 2, size=n),
        'Label': y,
    })


def _toy_multiclass_df(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    classes = np.array(['Benign', 'Mirai', 'Bashlite'])
    y = classes[rng.integers(0, len(classes), size=n)]

    return pd.DataFrame({
        'src_ip': ['3.3.3.3'] * n,
        'dst_ip': ['4.4.4.4'] * n,
        'src_port': rng.integers(1024, 65535, size=n),
        'dst_port': rng.integers(1, 1024, size=n),
        'protocol': rng.integers(0, 2, size=n),
        'timestamp': pd.date_range('2024-01-02', periods=n, freq='s'),
        'flow_duration': rng.normal(500, 20, size=n),
        'flow_byts_s': rng.normal(55, 3, size=n),
        'Attack': y,
    })


def test_train_ce_binary_dt_writes_artifacts_and_logs_metrics(tmp_path: Path, monkeypatch) -> None:
    _patch_shortuuid(monkeypatch)
    monkeypatch.chdir(tmp_path)

    df = _toy_binary_df()
    perf = PerformanceStats()
    cfg = DummyConfig(model_variant=ModelVariant.DT, use_pca=False, is_unsw=False)

    outdir = train_ce_binary(cfg, flow_path='unused.csv', perf_stats=perf, df_log=df)  # type: ignore

    assert outdir.exists()
    assert outdir.parent.name == 'binary_models'
    assert (outdir / 'scaler_binary.pkl').exists()
    assert (outdir / 'dt_model_binary.pkl').exists()

    assert len(perf.classifier_stats.accuracies) == 1
    assert len(perf.classifier_stats.f1s) == 1


def test_train_ce_binary_dt_with_pca_writes_pca_artifact(tmp_path: Path, monkeypatch) -> None:
    _patch_shortuuid(monkeypatch)
    monkeypatch.chdir(tmp_path)

    df = _toy_binary_df()
    perf = PerformanceStats()
    cfg = DummyConfig(model_variant=ModelVariant.DT, use_pca=True, is_unsw=False)

    outdir = train_ce_binary(cfg, flow_path='unused.csv', perf_stats=perf, df_log=df)  # type: ignore

    assert (outdir / 'pca_binary.pkl').exists()


def test_train_ce_binary_unsw_subset_columns_smoke(tmp_path: Path, monkeypatch) -> None:
    _patch_shortuuid(monkeypatch)
    monkeypatch.chdir(tmp_path)

    df = _toy_binary_df_unsw_subset()
    perf = PerformanceStats()
    cfg = DummyConfig(model_variant=ModelVariant.DT, use_pca=False, is_unsw=True)

    outdir = train_ce_binary(cfg, flow_path='unused.csv', perf_stats=perf, df_log=df)  # type: ignore

    assert (outdir / 'scaler_binary.pkl').exists()
    assert (outdir / 'dt_model_binary.pkl').exists()


def test_train_ce_multiclass_dt_writes_artifacts_and_logs_metrics(tmp_path: Path, monkeypatch) -> None:
    _patch_shortuuid(monkeypatch)
    monkeypatch.chdir(tmp_path)

    df = _toy_multiclass_df()
    perf = PerformanceStats()
    cfg = DummyConfig(model_variant=ModelVariant.DT, use_pca=False, is_unsw=False)

    outdir = train_ce_multiclass(cfg, perf_stats=perf, flow_path='unused.csv', df_log=df)  # type: ignore

    assert outdir.exists()
    assert outdir.parent.name == 'multi_class_models'
    assert (outdir / 'scaler_multi.pkl').exists()
    assert (outdir / 'label_encoder_mc.pkl').exists()
    assert (outdir / 'dt_model_binary.pkl').exists()

    assert len(perf.classifier_stats.accuracies) == 1
    assert len(perf.classifier_stats.f1s) == 1


def test_train_ce_multiclass_raises_without_label_column(tmp_path: Path, monkeypatch) -> None:
    _patch_shortuuid(monkeypatch)
    monkeypatch.chdir(tmp_path)

    df = _toy_multiclass_df().drop(columns=['Attack'])
    perf = PerformanceStats()
    cfg = DummyConfig(model_variant=ModelVariant.DT, use_pca=False, is_unsw=False)

    with pytest.raises(ValueError, match='MC_Label'):
        train_ce_multiclass(cfg, perf_stats=perf, flow_path='unused.csv', df_log=df)  # type: ignore


@pytest.mark.slow
def test_train_ce_binary_feedforward_smoke(tmp_path: Path, monkeypatch) -> None:
    _patch_shortuuid(monkeypatch)
    monkeypatch.chdir(tmp_path)

    df = _toy_binary_df(n=80)
    perf = PerformanceStats()
    cfg = DummyConfig(
        model_variant=ModelVariant.FEEDFORWARD,
        use_pca=False,
        is_unsw=False,
        device=torch.device('cpu'),
    )

    outdir = train_ce_binary(cfg, flow_path='unused.csv', perf_stats=perf, df_log=df)  # type: ignore

    assert (outdir / 'scaler_binary.pkl').exists()
    assert (outdir / 'feedforward_model_binary.pt').exists()
    assert len(perf.classifier_stats.accuracies) == 1


@pytest.mark.slow
def test_train_ce_multiclass_feedforward_smoke(tmp_path: Path, monkeypatch) -> None:
    _patch_shortuuid(monkeypatch)
    monkeypatch.chdir(tmp_path)

    df = _toy_multiclass_df(n=90)
    perf = PerformanceStats()
    cfg = DummyConfig(
        model_variant=ModelVariant.FEEDFORWARD,
        use_pca=False,
        is_unsw=False,
        device=torch.device('cpu'),
    )

    outdir = train_ce_multiclass(cfg, perf_stats=perf, flow_path='unused.csv', df_log=df)  # type: ignore

    assert (outdir / 'scaler_multi.pkl').exists()
    assert (outdir / 'label_encoder_mc.pkl').exists()
    assert (outdir / 'feedforward_multi.pt').exists()
    assert len(perf.classifier_stats.accuracies) == 1
