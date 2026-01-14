from __future__ import annotations

import runpy

import pytest

from src.core.perf_stats import ModelStats, PerformanceStats


def test_modelstats_log_metrics_appends_all_fields() -> None:
    ms = ModelStats()
    ms.log_metrics(acc=0.9, prec=0.8, rec=0.7, f1=0.6)

    assert ms.accuracies == [0.9]
    assert ms.precisions == [0.8]
    assert ms.recalls == [0.7]
    assert ms.f1s == [0.6]


def test_modelstats_summarize_metrics_warns_when_empty(caplog: pytest.LogCaptureFixture) -> None:
    ms = ModelStats()

    with caplog.at_level('WARNING'):
        ms.summarize_metrics()

    assert 'No CE training metrics available.' in caplog.text


def test_modelstats_summarize_metrics_logs_stats_when_present(caplog: pytest.LogCaptureFixture) -> None:
    ms = ModelStats()
    ms.log_metrics(0.5, 0.4, 0.3, 0.2)
    ms.log_metrics(0.7, 0.6, 0.5, 0.4)

    with caplog.at_level('INFO'):
        ms.summarize_metrics()

    assert 'Calibrations: 2' in caplog.text
    assert 'Avg Accuracy' in caplog.text
    assert 'Avg Precision' in caplog.text
    assert 'Avg Recall' in caplog.text
    assert 'Avg F1 Score' in caplog.text
    assert 'Std Accuracy' in caplog.text


def test_performance_stats_log_drift_intervals() -> None:
    ps = PerformanceStats()

    ps.log_drift(10)
    assert ps.drift_detected_indices == [10]
    assert ps.drift_intervals == []

    ps.log_drift(17)
    assert ps.drift_detected_indices == [10, 17]
    assert ps.drift_intervals == [7]

    ps.log_drift(30)
    assert ps.drift_detected_indices == [10, 17, 30]
    assert ps.drift_intervals == [7, 13]


def test_performance_stats_logs_to_correct_substats() -> None:
    ps = PerformanceStats()

    ps.log_ce_metrics(0.9, 0.8, 0.7, 0.6)
    ps.log_classifier_metrics(0.1, 0.2, 0.3, 0.4)

    assert ps.ce_stats.accuracies == [0.9]
    assert ps.ce_stats.precisions == [0.8]
    assert ps.ce_stats.recalls == [0.7]
    assert ps.ce_stats.f1s == [0.6]

    assert ps.classifier_stats.accuracies == [0.1]
    assert ps.classifier_stats.precisions == [0.2]
    assert ps.classifier_stats.recalls == [0.3]
    assert ps.classifier_stats.f1s == [0.4]


def test_performance_stats_summarize_ce_and_classifier_metrics_logs_headers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ps = PerformanceStats()
    ps.log_ce_metrics(0.5, 0.5, 0.5, 0.5)
    ps.log_classifier_metrics(0.6, 0.6, 0.6, 0.6)

    with caplog.at_level('INFO'):
        ps.summarize_ce_metrics()
        ps.summarize_classifier_metrics()

    assert '=== CE Model Calibration Summary ===' in caplog.text
    assert '=== Classifier Model Performance Summary ===' in caplog.text
    assert 'Calibrations: 1' in caplog.text


def test_perf_stats_module_main_guard_raises_notimplementederror() -> None:
    with pytest.raises(NotImplementedError, match='not intended to be run directly'):
        runpy.run_module('src.core.perf_stats', run_name='__main__')
