from __future__ import annotations

import importlib
import sys

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _import_module():
    """
    Prefer src.utils.overall_perf_stats, but allow the script to be named something else
    (your pasted file looks like perf_stats_ce_only_plots).
    """
    try:
        return importlib.import_module('src.utils.overall_perf_stats')
    except ModuleNotFoundError:
        return importlib.import_module('src.utils.perf_stats_ce_only_plots')


def _write_log(
    log_path: Path,
    *,
    model_variant: str = 'DecisionTreeClassifier',
    ce_type: str = 'approx-cce',
    ce_acc: str = '0.9, 0.8',
    ce_prec: str = '0.7, 0.6',
    ce_rec: str = '0.5',
    ce_f1: str = '0.55',
    clf_acc: str = '0.95, 0.94',
    clf_prec: str = '0.85, 0.84',
    clf_rec: str = '0.75',
    clf_f1: str = '0.80',
) -> None:
    perf_line = (
        '2025-01-01 00:00:00 INFO __main__: Full performance stats: perf_stats = PerformanceStats('
        f'ce_stats = ModelStats(accuracies=[{ce_acc}], precisions=[{ce_prec}], recalls=[{ce_rec}], f1s=[{ce_f1}]), '
        f'classifier_stats = ModelStats(accuracies=[{clf_acc}], precisions=[{clf_prec}], recalls=[{clf_rec}], f1s=[{clf_f1}])'  # noqa: E501
        ')'
    )
    content = '\n'.join([
        '2025-01-01 00:00:00 INFO __main__: start',
        f'2025-01-01 00:00:01 INFO __main__: modelVariant={model_variant}',
        f'2025-01-01 00:00:02 INFO __main__: ceType={ce_type}',
        perf_line,
        '2025-01-01 00:00:03 INFO __main__: done',
        '',
    ])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding='utf-8')


def test_parse_float_list_handles_percent_nan_and_junk():
    m = _import_module()
    got = m._parse_float_list('0.5, 50%, nan, nope, , 1')
    assert got == [0.5, 50.0, 1.0]


def test_parse_both_series_happy_path():
    m = _import_module()
    line = (
        'INFO __main__: Full performance stats: perf_stats = PerformanceStats('
        'ce_stats = ModelStats(accuracies=[0.9, 0.8], precisions=[0.7], recalls=[0.6], f1s=[0.65]), '
        'classifier_stats = ModelStats(accuracies=[0.95], precisions=[0.85], recalls=[0.75], f1s=[0.80])'
        ')'
    )
    both = m._parse_both_series(line)
    assert both is not None
    assert both['ce']['accuracies'] == [0.9, 0.8]
    assert both['classifier']['precisions'] == [0.85]


def test_scan_logs_group_by_classifier_infers_labels_from_content(tmp_path: Path):
    m = _import_module()

    log_dir = tmp_path / 'logs'
    log_path = log_dir / 'runA' / 'dt_approx_cce.log'
    _write_log(log_path, model_variant='DecisionTreeClassifier', ce_type='approx-cce')

    grouped = m._scan_logs_group_by_classifier(log_dir)

    assert 'dt' in grouped
    assert 'approx-cce' in grouped['dt']

    run_rel = log_path.relative_to(log_dir).as_posix()
    assert run_rel in grouped['dt']['approx-cce']

    series_pair = grouped['dt']['approx-cce'][run_rel]
    assert series_pair['ce']['accuracies'] == [0.9, 0.8]
    assert series_pair['classifier']['f1s'] == [0.8]


def test_write_classifier_csv_writes_expected_rows(tmp_path: Path):
    m = _import_module()

    runs_by_ce = {
        'approx-cce': {
            'run1.log': {
                'ce': {'accuracies': [0.9, 0.8], 'precisions': [], 'recalls': [], 'f1s': []},
                'classifier': {'accuracies': [0.95, 0.94], 'precisions': [], 'recalls': [], 'f1s': []},
            }
        }
    }

    out_csv = tmp_path / 'dt_ce_metrics.csv'
    m._write_classifier_csv('dt', runs_by_ce, out_csv)

    assert out_csv.exists()
    rows = out_csv.read_text(encoding='utf-8').splitlines()
    assert rows[0].split(',') == ['classifier', 'series', 'run', 'ce_type', 'metric', 'step', 'value']

    assert len(rows) == 1 + 4


def test_plot_classifier_grid_writes_png(tmp_path: Path):
    m = _import_module()

    runs_by_ce = {
        'approx-cce': {
            'run1.log': {
                'ce': {'accuracies': [0.9, 0.8], 'precisions': [0.7], 'recalls': [0.6], 'f1s': [0.65]},
                'classifier': {'accuracies': [0.95], 'precisions': [0.85], 'recalls': [0.75], 'f1s': [0.80]},
            }
        }
    }

    out_png = tmp_path / 'dt_ce_grid.png'
    m._plot_classifier_grid('dt', runs_by_ce, out_png)

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_main_smoke_emits_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    m = _import_module()

    log_dir = tmp_path / 'logs'
    out_dir = tmp_path / 'out'

    _write_log(log_dir / 'runA.log', model_variant='DecisionTreeClassifier', ce_type='approx-cce')

    monkeypatch.setattr(
        sys,
        'argv',
        ['prog', '--log-dir', str(log_dir), '--out-dir', str(out_dir)],
    )

    m.main()

    assert (out_dir / 'dt_ce_grid.png').exists()
    assert (out_dir / 'dt_ce_metrics.csv').exists()
