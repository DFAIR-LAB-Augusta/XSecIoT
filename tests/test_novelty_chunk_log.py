import logging

from types import SimpleNamespace

import numpy as np
import pandas as pd

from firce.runtime import inference
from firce.runtime.inference import _log_novelty_counts, process_chunk


def test_novelty_chunk_log_counts_flagged_by_class(caplog):
    flags = np.array([True, False, True, True, False])
    truth = pd.Series(['HTTP_Flood', 'Benign', 'HTTP_Flood', 'UDP_Flood', 'Benign'])

    with caplog.at_level(logging.INFO, logger=inference.logger.name):
        _log_novelty_counts(3, flags, truth)

    assert "[novelty] chunk=3 rows=5 flagged=3 by_class={'HTTP_Flood': 2, 'UDP_Flood': 1}" in caplog.messages


def test_novelty_chunk_log_without_ground_truth(caplog):
    with caplog.at_level(logging.INFO, logger=inference.logger.name):
        _log_novelty_counts(0, np.array([True, False]), None)

    assert '[novelty] chunk=0 rows=2 flagged=1 by_class={}' in caplog.messages


def test_novelty_chunk_log_emitted_by_process_chunk(caplog, monkeypatch):
    chunk = pd.DataFrame({'x': [1, 2, 3]})
    truth = pd.Series(['Benign', 'HTTP_Flood', 'HTTP_Flood'])
    flags = np.array([False, True, True])
    monkeypatch.setattr(inference, '_prepare_chunk', lambda runtime, c: (c, truth))
    monkeypatch.setattr(inference, '_process_chunk_rows', lambda runtime, c, gt: None)
    monkeypatch.setattr(inference, '_score_chunk_novelty', lambda runtime, c: (flags, c))
    monkeypatch.setattr(inference, '_generate_novelty_reports', lambda runtime, c, x, f: None)
    monkeypatch.setattr(inference, '_detect_chunk_drift', lambda runtime, c: False)
    runtime = SimpleNamespace(perf_stats=SimpleNamespace(iteration_times=[]))

    with caplog.at_level(logging.INFO, logger=inference.logger.name):
        process_chunk(runtime, chunk, chunk_num=7)

    assert "[novelty] chunk=7 rows=3 flagged=2 by_class={'HTTP_Flood': 2}" in caplog.messages
