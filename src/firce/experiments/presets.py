"""Dataset presets for the Direction 3 experiment driver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DatasetPreset:
    """Dataset location and column conventions for an experiment run."""

    name: str
    path: str
    label_column: str
    held_out: tuple[str, ...]
    timestamp_format: str
    device_column: str

    @property
    def resolved_path(self) -> Path:
        """Dataset path resolved against the repository root, independent of the cwd."""
        return REPO_ROOT / self.path


PRESETS: dict[str, DatasetPreset] = {
    'dfair': DatasetPreset(
        name='dfair',
        path='datasets/DFAIR/combined_data_with_okpVacc_modified.csv',
        label_column='Label',
        held_out=('HTTP_Flood',),
        timestamp_format='%d-%m-%Y %H:%M',
        device_column='device_id',
    ),
}
