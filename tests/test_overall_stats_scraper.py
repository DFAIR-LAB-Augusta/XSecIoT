from __future__ import annotations

import importlib
import sys

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _write_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_overall_stats_scraper_parses_and_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    overall_stats_scraper executes on import:
      - walks ./logging (excluding dirs starting with 'old')
      - parses [==OVERALL SIM STATS==] lines, applying CE/Classifier prefixes
      - writes ./logging/overall_stats.log

    This test builds a tiny ./logging tree in tmp_path and imports the module
    after chdir so it only touches temp files.
    """
    monkeypatch.chdir(tmp_path)

    log_dir = tmp_path / 'logging'
    exp1 = log_dir / 'exp1'
    exp2 = log_dir / 'exp2'
    old = log_dir / 'old_runs'

    _write_log(
        exp1 / 'a.log',
        [
            'INFO start',
            '[==OVERALL SIM STATS==] Total Drift Detections: 5',
            '[==OVERALL SIM STATS==] Drift Detection Rate: 10.0%',
            '=== CE Model Calibration Summary ===',
            '[==OVERALL SIM STATS==] Calibrations: 2',
            '[==OVERALL SIM STATS==] Avg Accuracy: 90.0%',
            '[==OVERALL SIM STATS==] Std Accuracy: 0.10',
            '=== Classifier Model Performance Summary ===',
            '[==OVERALL SIM STATS==] Calibrations: 2',
            '[==OVERALL SIM STATS==] Avg Accuracy: 95.0%',
            '[==OVERALL SIM STATS==] Std Accuracy: 0.05',
            '[==OVERALL SIM STATS==] Average Chunk Size: 10',
            '[==OVERALL SIM STATS==] Median Chunk Size: 9',
            '[==OVERALL SIM STATS==] Standard Deviation of Chunk Sizes: 2',
        ],
    )

    _write_log(
        exp2 / 'b.log',
        [
            'INFO start',
            '[==OVERALL SIM STATS==] Total Drift Detections: 2',
            '[==OVERALL SIM STATS==] Drift Detection Rate: 5.0%',
            '=== CE Model Calibration Summary ===',
            '[==OVERALL SIM STATS==] Calibrations: 2',
            '[==OVERALL SIM STATS==] Avg Accuracy: 80.0%',
            '[==OVERALL SIM STATS==] Std Accuracy: 0.20',
            '=== Classifier Model Performance Summary ===',
            '[==OVERALL SIM STATS==] Calibrations: 2',
            '[==OVERALL SIM STATS==] Avg Accuracy: 85.0%',
            '[==OVERALL SIM STATS==] Std Accuracy: 0.10',
            '[==OVERALL SIM STATS==] Average Chunk Size: 20',
            '[==OVERALL SIM STATS==] Median Chunk Size: 19',
            '[==OVERALL SIM STATS==] Standard Deviation of Chunk Sizes: 4',
        ],
    )

    _write_log(
        old / 'ignored.log',
        [
            '[==OVERALL SIM STATS==] Total Drift Detections: 999',
            '=== CE Model Calibration Summary ===',
            '[==OVERALL SIM STATS==] Avg Accuracy: 0.0%',
        ],
    )

    modname = 'src.utils.overall_stats_scraper'
    sys.modules.pop(modname, None)

    mod = importlib.import_module(modname)

    out_file = log_dir / 'overall_stats.log'
    assert out_file.exists()
    text = out_file.read_text(encoding='utf-8')

    assert 'Subfolder: exp1' in text
    assert 'File: a.log' in text
    assert 'Subfolder: exp2' in text
    assert 'File: b.log' in text

    assert '[CE Model] Calibrations: 2' in text
    assert '[CE Model] Avg Accuracy: 90.0%' in text
    assert '[Classifier Model] Avg Accuracy: 95.0%' in text

    assert 'Average Chunk Size: 10' in text
    assert 'Median Chunk Size: 9' in text
    assert 'Standard Deviation of Chunk Sizes: 2' in text

    assert 'old_runs' not in text
    assert 'ignored.log' not in text

    assert 'Total Drift Detections:' in text
    assert '5 in exp1/a.log' in text
    assert '2 in exp2/b.log' in text

    assert '[CE Model] Avg Accuracy:' in text
    assert 'Highest = 90.0% in exp1/a.log' in text

    assert '[CE Model] Std Accuracy:' in text
    assert 'Lowest = 0.10 in exp1/a.log' in text

    assert 'Adaptive Chunker Average Chunk Size:' in text
    assert '10 in exp1/a.log' in text
    assert '20 in exp2/b.log' in text

    assert 'exp1' in mod.stats_by_run
    assert 'a.log' in mod.stats_by_run['exp1']
