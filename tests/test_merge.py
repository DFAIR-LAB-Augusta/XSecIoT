from __future__ import annotations

import sys

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from src.utils import merge as merge_mod

if TYPE_CHECKING:
    from pathlib import Path


def _write_csv(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_validate_directory_missing(tmp_path: Path) -> None:
    missing = tmp_path / 'nope'
    with pytest.raises(FileNotFoundError, match='Directory not found'):
        merge_mod._validate_directory(missing)


def test_validate_directory_not_a_dir(tmp_path: Path) -> None:
    f = tmp_path / 'file.txt'
    f.write_text('x', encoding='utf-8')
    with pytest.raises(ValueError, match='not a directory'):
        merge_mod._validate_directory(f)


def test_merge_and_sort_no_csvs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='No CSV files found'):
        merge_mod._merge_and_sort_csvs(tmp_path)


def test_merge_and_sort_skips_bad_csv_and_merges_good(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_csv(
        tmp_path / 'good.csv',
        [
            {'timestamp': '2020-01-01 00:00:00', 'val': 1},
            {'timestamp': '2020-01-02 00:00:00', 'val': 2},
        ],
    )

    _write_csv(
        tmp_path / 'bad.csv',
        [
            {'not_timestamp': '2020-01-03 00:00:00', 'val': 999},
        ],
    )

    out = merge_mod._merge_and_sort_csvs(tmp_path)
    assert out.exists()
    assert out.name == f'{tmp_path.name}_merged.csv'

    captured = capsys.readouterr()
    assert 'Skipping bad.csv' in captured.err

    merged = pd.read_csv(out, parse_dates=['timestamp'])
    assert merged.shape[0] == 2
    assert merged['timestamp'].iloc[0] > merged['timestamp'].iloc[1]
    assert merged['val'].tolist() == [2, 1]


def test_merge_and_sort_all_invalid_raises(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_csv(tmp_path / 'a.csv', [{'x': 1}])
    _write_csv(tmp_path / 'b.csv', [{'y': 2}])

    with pytest.raises(ValueError, match='No valid CSV files were loaded'):
        merge_mod._merge_and_sort_csvs(tmp_path)

    captured = capsys.readouterr()
    assert 'Skipping a.csv' in captured.err
    assert 'Skipping b.csv' in captured.err


def test_main_success_prints_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_csv(
        tmp_path / 'one.csv',
        [{'timestamp': '2020-01-01 00:00:00', 'val': 1}],
    )

    monkeypatch.setattr(sys, 'argv', ['prog', '--dataset_path', str(tmp_path)])
    merge_mod.main()

    out = tmp_path / f'{tmp_path.name}_merged.csv'
    assert out.exists()

    captured = capsys.readouterr()
    assert 'Merged CSV saved to:' in captured.out
    assert str(out) in captured.out


def test_main_failure_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    missing = tmp_path / 'missing_dir'
    monkeypatch.setattr(sys, 'argv', ['prog', '--dataset_path', str(missing)])

    with pytest.raises(SystemExit) as excinfo:
        merge_mod.main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert 'Error:' in captured.err
