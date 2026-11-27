"""In-process FIRCE scoring: per-flow class probabilities and per-class conformal p-values."""

from __future__ import annotations

import os

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from firce.classes import derive_mc_label
from firce.experiments.presets import PRESETS
from firce.novelty.decision_rules import compute_all_class_p_values
from firce.runtime.bootstrap import initialize_simulation_runtime
from firce.runtime.inference import _prepare_monitor_chunk_features
from firce.utils.config import CEType, ModelType, ModelVariant, MonitorType, SimulationConfig
from fire.preprocessing import clean_data

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np

    from firce.experiments.presets import DatasetPreset
    from firce.runtime.sim_types import SimulationRuntime


@dataclass(frozen=True)
class ScoredSplit:
    """Per-flow model outputs for one frame, row-aligned with the scored dataframe."""

    probas: np.ndarray
    p_values: dict[str, np.ndarray]
    y_true: np.ndarray
    device: np.ndarray
    timestamp: np.ndarray


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Temporarily change the process cwd."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_runtime(
    train_df: pd.DataFrame,
    preset: DatasetPreset,
    work_dir: Path,
    seed: int,
    model_variant: str = 'rf',
    ce_type: str = 'ice',
) -> SimulationRuntime:
    """
    Train the multiclass model and calibrate the CE monitor on `train_df`.

    Args:
        train_df: Raw training flows (held-out classes are dropped by the runtime).
        preset: Dataset preset naming the label column and held-out classes.
        work_dir: Directory receiving `CETrain/train.csv` and `multi_class_models/`.
        seed: Model seed.
        model_variant: Classifier variant name.
        ce_type: Conformal evaluator type name.

    Returns:
        Initialized simulation runtime.
    """
    work_dir = work_dir.resolve()
    train_dir = work_dir / 'CETrain'
    train_dir.mkdir(parents=True, exist_ok=True)
    train_path = train_dir / 'train.csv'
    train_df.to_csv(train_path, index=False)

    config = SimulationConfig(
        model_type=ModelType.MULTI,
        model_variant=ModelVariant(model_variant),
        ce_type=CEType(ce_type),
        aggregated_path=train_path,
        flows_path=train_path,
        monitor_type=MonitorType.CE,
        label_column=preset.label_column,
        held_out=preset.held_out,
        novelty_enabled=False,
        seed=seed,
    )
    with _working_directory(work_dir):
        return initialize_simulation_runtime(config)


def score_frame(runtime: SimulationRuntime, df: pd.DataFrame, preset: DatasetPreset = PRESETS['dfair']) -> ScoredSplit:
    """
    Score every row of `df` in one batch.

    Args:
        runtime: Runtime from `build_runtime`.
        df: Raw flows to score; held-out classes are kept.
        preset: Dataset preset naming the label, device and timestamp columns.

    Returns:
        Probabilities, per-class p-values, canonical labels, device and epoch-second timestamps.
    """
    frame = derive_mc_label(df, preset.label_column, drop_held_out=False).reset_index(drop=True)
    device = frame[preset.device_column].to_numpy()
    parsed = pd.to_datetime(frame['timestamp'], format=preset.timestamp_format)
    timestamp = (parsed - pd.Timestamp(0)).dt.total_seconds().to_numpy()
    y_true = frame['MC_Label'].to_numpy()

    clean = clean_data(frame.drop(columns=['MC_Label', preset.label_column]), runtime.config.is_unsw)
    x_monitor = _prepare_monitor_chunk_features(runtime, clean)
    model = runtime.monitor.model
    probas = model.predict_proba(x_monitor)
    p_values = compute_all_class_p_values(model, runtime.monitor.calibration_scores, x_monitor)
    return ScoredSplit(probas=probas, p_values=p_values, y_true=y_true, device=device, timestamp=timestamp)
