from __future__ import annotations

import runpy

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import src.utils.bin_labeling as bl

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    'ip, expected',
    [
        ('0.0.0.0', True),
        ('1.2.3.4', True),
        ('255.255.255.255', True),
        ('256.0.0.1', False),
        ('192.168.1', False),
        ('192.168.1.999', False),
        ('-1.2.3.4', False),
        ('1.2.3.4.5', False),
        ('abc', False),
        ('', False),
        (' 1.2.3.4', False),
        ('1.2.3.4 ', False),
    ],
)
def test_is_valid_ip(ip: str, expected: bool) -> None:
    assert bl._is_valid_ip(ip) is expected


def test_validate_inputs_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / 'nope.csv'
    with pytest.raises(FileNotFoundError, match='File does not exist'):
        bl._validate_inputs(missing, '1.2.3.4', '5.6.7.8')


def test_validate_inputs_path_is_directory(tmp_path: Path) -> None:
    d = tmp_path / 'adir'
    d.mkdir()
    with pytest.raises(ValueError, match='Provided path is not a file'):
        bl._validate_inputs(d, '1.2.3.4', '5.6.7.8')


def test_validate_inputs_requires_csv_suffix(tmp_path: Path) -> None:
    p = tmp_path / 'data.txt'
    p.write_text('hi', encoding='utf-8')
    with pytest.raises(ValueError, match='Only CSV files are supported'):
        bl._validate_inputs(p, '1.2.3.4', '5.6.7.8')


def test_validate_inputs_invalid_src_ip(tmp_path: Path) -> None:
    p = tmp_path / 'data.csv'
    p.write_text('x,y\n1,2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid source IP address'):
        bl._validate_inputs(p, '999.1.2.3', '5.6.7.8')


def test_validate_inputs_invalid_dest_ip(tmp_path: Path) -> None:
    p = tmp_path / 'data.csv'
    p.write_text('x,y\n1,2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid destination IP address'):
        bl._validate_inputs(p, '1.2.3.4', '5.6.7.999')


def test_add_binary_labels_happy_path(tmp_path: Path) -> None:
    df = pd.DataFrame({
        'src_ip': ['1.1.1.1', '1.1.1.1', '2.2.2.2'],
        'dst_ip': ['9.9.9.9', '8.8.8.8', '9.9.9.9'],
        'f1': [1, 2, 3],
    })
    p = tmp_path / 'flows.csv'
    df.to_csv(p, index=False)

    out = bl._add_binary_labels(p, src_ip='1.1.1.1', dest_ip='9.9.9.9')
    assert out.exists()
    assert out.name == 'flows_labeled.csv'

    df2 = pd.read_csv(out)
    assert 'BinLabel' in df2.columns
    assert df2['BinLabel'].tolist() == [1, 0, 0]


def test_add_binary_labels_missing_required_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({'src_ip': ['1.1.1.1'], 'nope': ['x']})
    p = tmp_path / 'bad.csv'
    df.to_csv(p, index=False)

    with pytest.raises(ValueError, match=r"CSV must contain 'src_ip' and 'dst_ip' columns"):
        bl._add_binary_labels(p, '1.1.1.1', '2.2.2.2')


def test_add_binary_labels_read_csv_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / 'flows.csv'
    p.write_text('not,a,csv', encoding='utf-8')

    def boom(*args, **kwargs):
        raise RuntimeError('kaboom')

    monkeypatch.setattr(pd, 'read_csv', boom)

    with pytest.raises(ValueError, match=r'Failed to read CSV file:'):
        bl._add_binary_labels(p, '1.1.1.1', '2.2.2.2')


def test_main_success_prints_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    df = pd.DataFrame({'src_ip': ['1.1.1.1'], 'dst_ip': ['2.2.2.2']})
    p = tmp_path / 'flows.csv'
    df.to_csv(p, index=False)

    monkeypatch.setattr(bl.sys, 'argv', ['prog', str(p), '1.1.1.1', '2.2.2.2'])

    # Should not exit
    bl.main()

    out = capsys.readouterr()
    assert 'Labeled dataset saved to:' in out.out
    assert 'flows_labeled.csv' in out.out
    assert out.err == ''


def test_main_failure_exits_1_and_writes_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    missing = tmp_path / 'missing.csv'
    monkeypatch.setattr(bl.sys, 'argv', ['prog', str(missing), '1.1.1.1', '2.2.2.2'])

    with pytest.raises(SystemExit) as excinfo:
        bl.main()

    assert excinfo.value.code == 1
    out = capsys.readouterr()
    assert out.out == ''
    assert out.err.startswith('Error:')


def test_module_main_guard_runs_and_exits_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bl.sys, 'argv', ['prog', 'nope.csv', '1.1.1.1', '2.2.2.2'])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module('src.utils.bin_labeling', run_name='__main__')

    assert excinfo.value.code == 1
