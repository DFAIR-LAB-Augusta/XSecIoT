from __future__ import annotations

import argparse
import logging
import re

from pathlib import Path

import pandas as pd
import pytest

import src.utils.mc_labeling as mcl


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def test_configure_logging_sets_level(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_basicConfig(*, level, format):
        calls['level'] = level
        calls['format'] = format

    monkeypatch.setattr(logging, 'basicConfig', fake_basicConfig)
    mcl._configure_logging(verbose=False)
    assert calls['level'] == logging.INFO
    assert 'asctime' in str(calls['format'])

    calls.clear()
    mcl._configure_logging(verbose=True)
    assert calls['level'] == logging.DEBUG


def test_validate_dataset_dir_errors(tmp_path: Path) -> None:
    missing = tmp_path / 'nope'
    with pytest.raises(FileNotFoundError, match='dataset_path does not exist'):
        mcl._validate_dataset_dir(missing)

    f = tmp_path / 'file.txt'
    f.write_text('x', encoding='utf-8')
    with pytest.raises(ValueError, match='dataset_path must be a directory'):
        mcl._validate_dataset_dir(f)


def test_find_helpers(tmp_path: Path) -> None:
    d = tmp_path / 'data'
    d.mkdir()

    _write_csv(d / 'a_labeled.csv', pd.DataFrame({'Bin_Label': [0], 'MC_Label': ['Benign']}))
    _write_csv(d / 'nested' / 'b_labeled.csv', pd.DataFrame({'BinLabel': [1]}))
    (d / 'nested' / 'note.txt').write_text('hi\n', encoding='utf-8')
    (d / 'ignore.csv').write_text('x,y\n1,2\n', encoding='utf-8')

    csvs = mcl._find_labeled_csvs(d)
    txts = mcl._find_txt_files(d)

    assert [p.name for p in csvs] == ['a_labeled.csv', 'b_labeled.csv']
    assert [p.name for p in txts] == ['note.txt']


@pytest.mark.parametrize(
    'cols, expected',
    [
        (['Bin_Label'], 'Bin_Label'),
        (['BinLabel'], 'BinLabel'),
        (['Bin_Label', 'BinLabel'], 'Bin_Label'),
        (['x'], None),
    ],
)
def test_resolve_bin_label_column(cols: list[str], expected: str | None) -> None:
    df = pd.DataFrame({c: [0] for c in cols})
    assert mcl._resolve_bin_label_column(df) == expected


def test_update_csv_mc_label_skips_if_no_bin_column(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = tmp_path / 'x_labeled.csv'
    _write_csv(p, pd.DataFrame({'src_ip': ['1.1.1.1']}))

    caplog.set_level(logging.WARNING)
    changed = mcl._update_csv_mc_label(p, dry_run=False)
    assert changed is False
    assert 'Skipping CSV' in caplog.text


def test_update_csv_mc_label_updates_bin_label_to_mc_label(tmp_path: Path) -> None:
    p = tmp_path / 'x_labeled.csv'
    _write_csv(p, pd.DataFrame({'Bin_Label': [0, 1, 0]}))

    changed = mcl._update_csv_mc_label(p, dry_run=False)
    assert changed is True

    df2 = _read_csv(p)
    assert df2['MC_Label'].tolist() == ['Benign', 'FILL_ME', 'Benign']


def test_update_csv_mc_label_accepts_binlabel_legacy(tmp_path: Path) -> None:
    p = tmp_path / 'x_labeled.csv'
    _write_csv(p, pd.DataFrame({'BinLabel': [1]}))

    assert mcl._update_csv_mc_label(p, dry_run=False) is True
    df2 = _read_csv(p)
    assert df2['MC_Label'].tolist() == ['FILL_ME']


def test_update_csv_mc_label_unknown_values_fill_me(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = tmp_path / 'x_labeled.csv'
    _write_csv(p, pd.DataFrame({'Bin_Label': [0, 2, 1, -1]}))

    caplog.set_level(logging.WARNING)
    assert mcl._update_csv_mc_label(p, dry_run=False) is True
    assert 'unexpected Bin_Label values' in caplog.text

    df2 = _read_csv(p)
    assert df2['MC_Label'].tolist() == ['Benign', 'FILL_ME', 'FILL_ME', 'FILL_ME']


def test_update_csv_mc_label_no_change_when_already_matches(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = tmp_path / 'x_labeled.csv'
    _write_csv(p, pd.DataFrame({'Bin_Label': [0, 1], 'MC_Label': ['Benign', 'FILL_ME']}))

    caplog.set_level(logging.INFO)
    changed = mcl._update_csv_mc_label(p, dry_run=False)
    assert changed is False
    assert 'No change needed' in caplog.text


def test_update_csv_mc_label_dry_run_does_not_modify(tmp_path: Path) -> None:
    p = tmp_path / 'x_labeled.csv'
    _write_csv(p, pd.DataFrame({'Bin_Label': [0, 1]}))

    assert mcl._update_csv_mc_label(p, dry_run=True) is True
    df2 = _read_csv(p)
    assert 'MC_Label' not in df2.columns


def test_update_csv_mc_label_read_failure_bubbles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / 'x_labeled.csv'
    p.write_text('not,a,csv', encoding='utf-8')

    def boom(*args, **kwargs):
        raise RuntimeError('kaboom')

    monkeypatch.setattr(pd, 'read_csv', boom)
    with pytest.raises(RuntimeError, match='kaboom'):
        mcl._update_csv_mc_label(p, dry_run=False)


def test_extract_unix_time_from_third_field() -> None:
    assert mcl._extract_unix_time_from_third_field('Unix Time: 1751471126.992594') == pytest.approx(1751471126.992594)
    assert mcl._extract_unix_time_from_third_field('Unix Time:') is None
    assert mcl._extract_unix_time_from_third_field('Unix Time: nope') is None
    assert mcl._extract_unix_time_from_third_field('something else') is None


def test_format_unix_time_returns_expected_shape() -> None:
    s = mcl._format_unix_time(0.0)
    assert len(s) == 19
    assert s[4] == '-'
    assert s[7] == '-'
    assert ':' in s


def test_update_txt_file_appends_datetime(tmp_path: Path) -> None:
    p = tmp_path / 'labels.txt'
    p.write_text('a, b, Unix Time: 0.0\n', encoding='utf-8')

    assert mcl._update_txt_file(p, dry_run=False) is True
    out = p.read_text(encoding='utf-8')
    assert out.startswith('a, b, Unix Time: 0.0, ')
    assert len(out.strip().split(',')) >= 4


def test_update_txt_file_skips_if_already_has_datetime(tmp_path: Path) -> None:
    p = tmp_path / 'labels.txt'
    p.write_text('a, b, Unix Time: 0.0, 1970-01-01 00:00:00\n', encoding='utf-8')

    assert mcl._update_txt_file(p, dry_run=False) is False
    assert p.read_text(encoding='utf-8') == 'a, b, Unix Time: 0.0, 1970-01-01 00:00:00\n'


def test_update_txt_file_ignores_short_or_unparseable_lines(tmp_path: Path) -> None:
    p = tmp_path / 'labels.txt'
    p.write_text(
        'onlyonefield\na, b, no time here\na, b, Unix Time: nope\n',
        encoding='utf-8',
    )
    assert mcl._update_txt_file(p, dry_run=False) is False


def test_update_txt_file_dry_run_does_not_modify(tmp_path: Path) -> None:
    p = tmp_path / 'labels.txt'
    p.write_text('a, b, Unix Time: 0.0\n', encoding='utf-8')

    assert mcl._update_txt_file(p, dry_run=True) is True
    assert p.read_text(encoding='utf-8') == 'a, b, Unix Time: 0.0\n'


def test_update_txt_file_read_failure_bubbles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / 'labels.txt'
    p.write_text('x\n', encoding='utf-8')

    def boom(*args, **kwargs):
        raise OSError('kaboom')

    monkeypatch.setattr(Path, 'read_text', boom, raising=True)
    with pytest.raises(OSError, match='kaboom'):
        mcl._update_txt_file(p, dry_run=False)


def test_main_success_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    dataset = tmp_path / 'ds'
    dataset.mkdir()

    _write_csv(dataset / 'a_labeled.csv', pd.DataFrame({'Bin_Label': [0, 1]}))
    (dataset / 'labels.txt').write_text('a, b, Unix Time: 0.0\n', encoding='utf-8')

    monkeypatch.setattr(
        mcl,
        '_parse_arguments',
        lambda: argparse.Namespace(dataset_path=str(dataset), dry_run=False, verbose=False),
    )

    caplog.set_level(logging.INFO)
    mcl.main()
    assert 'Found 1 labeled CSVs' in caplog.text
    assert 'Found 1 txt files' in caplog.text
    assert 'Done. CSVs: found=1 updated=1' in caplog.text
    assert 'TXTs: found=1 updated=1' in caplog.text

    df2 = _read_csv(dataset / 'a_labeled.csv')
    assert df2['MC_Label'].tolist() == ['Benign', 'FILL_ME']
    txt = (dataset / 'labels.txt').read_text(encoding='utf-8')
    assert re.search(r', \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\n$', txt) is not None


def test_main_invalid_dataset_path_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bogus = tmp_path / 'does_not_exist'

    monkeypatch.setattr(
        mcl,
        '_parse_arguments',
        lambda: argparse.Namespace(dataset_path=str(bogus), dry_run=False, verbose=False),
    )

    with pytest.raises(SystemExit) as excinfo:
        mcl.main()

    assert excinfo.value.code == 1
