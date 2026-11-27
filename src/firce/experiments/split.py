"""Seeded device-time block split into train, val and stream partitions."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from firce.classes import derive_mc_label

if TYPE_CHECKING:
    from firce.experiments.presets import DatasetPreset

logger = logging.getLogger(__name__)

MIN_VAL_ROWS = 100
STRATIFIED_VAL_FRAC = 0.2


@dataclass(frozen=True)
class SplitResult:
    """Train, val and stream partitions plus how val was chosen ('block' or 'stratified')."""

    train: pd.DataFrame
    val: pd.DataFrame
    stream: pd.DataFrame
    val_strategy: str = 'block'


def split_device_time_blocks(
    df: pd.DataFrame,
    preset: DatasetPreset,
    seed: int,
    block_minutes: int = 60,
    stream_frac: float = 0.3,
    val_frac: float = 0.2,
) -> SplitResult:
    """
    Split flows into train, val and stream by contiguous per-device time blocks.

    Each (device, block) goes to the stream with probability `stream_frac`; held-out-class rows always go
    to the stream. Of the remaining known-class blocks, `val_frac` go to val and the rest to train. When a
    known class is starved in val, val is rebuilt as a stratified 20% sample of the non-stream known rows.

    Args:
        df: Raw flows with the preset's label, device and timestamp columns.
        preset: Dataset preset naming those columns and the held-out classes.
        seed: Seed for all random draws.
        block_minutes: Width of a time block in minutes.
        stream_frac: Probability a block goes to the stream.
        val_frac: Fraction of non-stream blocks that go to val.

    Returns:
        Disjoint train, val and stream frames (carrying canonical `MC_Label`) and the val strategy used.
    """
    rng = np.random.default_rng(seed)
    labeled = derive_mc_label(df, preset.label_column, preset.held_out, drop_held_out=False).reset_index(drop=True)

    ts = pd.to_datetime(labeled['timestamp'], format=preset.timestamp_format)
    minutes = (ts - ts.min()).dt.total_seconds().to_numpy() // 60
    block = (minutes // block_minutes).astype(np.int64)
    keys = pd.MultiIndex.from_arrays([labeled[preset.device_column].to_numpy(), block])
    codes, uniques = pd.factorize(keys, sort=True)
    n_blocks = len(uniques)

    block_to_stream = rng.random(n_blocks) < stream_frac
    block_to_val = rng.random(n_blocks) < val_frac

    held_out = labeled['MC_Label'].isin(preset.held_out).to_numpy()
    to_stream = held_out | block_to_stream[codes]
    to_val = ~to_stream & block_to_val[codes]
    val_strategy = 'block'

    known = ~held_out
    known_counts = labeled.loc[known, 'MC_Label'].value_counts()
    val_counts = labeled.loc[to_val, 'MC_Label'].value_counts()
    starved = [
        cls for cls, total in known_counts.items() if val_counts.get(cls, 0) < min(MIN_VAL_ROWS, int(total) // 5)
    ]
    if starved:
        logger.warning('Block val split starved classes %s; falling back to stratified val split', starved)
        val_strategy = 'stratified'
        to_val = np.zeros(len(labeled), dtype=bool)
        candidates = labeled.loc[~to_stream, 'MC_Label']
        for _, idx in candidates.groupby(candidates).groups.items():
            positions = np.asarray(idx)
            n_val = int(round(len(positions) * STRATIFIED_VAL_FRAC))
            to_val[rng.choice(positions, size=n_val, replace=False)] = True

    return SplitResult(
        train=labeled.loc[~to_stream & ~to_val],
        val=labeled.loc[to_val],
        stream=labeled.loc[to_stream],
        val_strategy=val_strategy,
    )
