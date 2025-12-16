from __future__ import annotations

import logging

import pytest

from src.core.adaptive_chunking import AdaptiveChunkController
from src.core.config import AdaptiveChunkConfig
from src.core.perf_stats import PerformanceStats


def _cfg(
    *,
    init_chunk_size: int = 100,
    min_chunk_size: int = 10,
    max_chunk_size: int = 1000,
    ema_decay: float = 0.0,
    cooldown_period: int = 0,
    step_size: int = 5,
) -> AdaptiveChunkConfig:
    # ema_decay=0.0 makes EMA track raw drift rate immediately (easy deterministic tests)
    return AdaptiveChunkConfig(
        init_chunk_size=init_chunk_size,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        ema_decay=ema_decay,
        cooldown_period=cooldown_period,
        step_size=step_size,
    )


def test_init_sets_expected_fields() -> None:
    cfg = _cfg(
        init_chunk_size=123, min_chunk_size=7, max_chunk_size=999, ema_decay=0.5, cooldown_period=2, step_size=11
    )
    c = AdaptiveChunkController(cfg)

    assert c.get_chunk_size() == 123
    assert c.min_chunk_size == 7
    assert c.max_chunk_size == 999
    assert c.ema_decay == 0.5
    assert c.cooldown_period == 2
    assert c.step_size == 11

    # internal counters
    assert c._cooldown_counter == 0
    assert c._drift_rate_ema == 0.0
    assert c._total_chunks == 0
    assert c._total_drifts == 0


def test_update_appends_chunk_size_only_when_adjustment_runs() -> None:
    # cooldown_period=2 => after an adjustment, next 2 updates should early-return (no append)
    cfg = _cfg(init_chunk_size=100, cooldown_period=2, ema_decay=0.0, step_size=10)
    c = AdaptiveChunkController(cfg)
    ps = PerformanceStats()

    c.update(drift_detected=False, perf_stats=ps)
    assert ps.chunk_sizes == [110]  # drift_ema=0 => increase by step_size

    c.update(drift_detected=False, perf_stats=ps)
    c.update(drift_detected=False, perf_stats=ps)
    assert ps.chunk_sizes == [110]  # no new appends during cooldown

    c.update(drift_detected=False, perf_stats=ps)
    assert ps.chunk_sizes == [110, 120]  # adjustment runs again


def test_halves_chunk_size_when_drift_rate_high() -> None:
    cfg = _cfg(init_chunk_size=100, min_chunk_size=30, ema_decay=0.0, cooldown_period=0)
    c = AdaptiveChunkController(cfg)
    ps = PerformanceStats()

    c.update(drift_detected=True, perf_stats=ps)  # raw drift rate = 1.0 => halve
    assert c.get_chunk_size() == 50
    assert ps.chunk_sizes == [50]

    # keep halving, but never below min_chunk_size
    c.update(drift_detected=True, perf_stats=ps)
    assert c.get_chunk_size() == 30
    assert ps.chunk_sizes[-1] == 30


def test_increases_chunk_size_when_drift_rate_low() -> None:
    cfg = _cfg(init_chunk_size=10, max_chunk_size=17, step_size=5, ema_decay=0.0, cooldown_period=0)
    c = AdaptiveChunkController(cfg)
    ps = PerformanceStats()

    c.update(drift_detected=False, perf_stats=ps)  # drift_ema=0 => +5
    assert c.get_chunk_size() == 15

    c.update(drift_detected=False, perf_stats=ps)  # would go to 20, but capped to 17
    assert c.get_chunk_size() == 17


def test_adjust_no_change_logs_debug(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _cfg(init_chunk_size=100, ema_decay=0.0)
    c = AdaptiveChunkController(cfg)

    # force a mid-range drift EMA so neither branch triggers:
    # > 0.2 => halve, < 0.05 => increase, else no change
    c._drift_rate_ema = 0.10

    with caplog.at_level(logging.DEBUG):
        c._adjust_chunk_size()

    assert c.get_chunk_size() == 100
    assert any('No change in chunk size' in rec.message for rec in caplog.records)


def test_adjust_change_logs_info(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _cfg(init_chunk_size=100, min_chunk_size=1, ema_decay=0.0)
    c = AdaptiveChunkController(cfg)

    c._drift_rate_ema = 0.9  # force halve branch

    with caplog.at_level(logging.INFO):
        c._adjust_chunk_size()

    assert c.get_chunk_size() == 50
    assert any('Chunk size changed' in rec.message for rec in caplog.records)


def test_ema_updates_as_expected() -> None:
    # Use decay=0.5 and cooldown_period very large so only test EMA math,
    # not chunk resizing effects.
    cfg = _cfg(init_chunk_size=100, ema_decay=0.5, cooldown_period=10_000)
    c = AdaptiveChunkController(cfg)
    ps = PerformanceStats()

    # Step 1: drift True => raw=1/1=1.0, ema=0.5*0 + 0.5*1 = 0.5
    c.update(drift_detected=True, perf_stats=ps)
    assert c._total_chunks == 1
    assert c._total_drifts == 1
    assert c._drift_rate_ema == pytest.approx(0.5, abs=1e-12)

    # Step 2: drift False => raw=1/2=0.5, ema=0.5*0.5 + 0.5*0.5 = 0.5
    c.update(drift_detected=False, perf_stats=ps)
    assert c._total_chunks == 2
    assert c._total_drifts == 1
    assert c._drift_rate_ema == pytest.approx(0.5, abs=1e-12)

    # Step 3: drift False => raw=1/3=0.333..., ema=0.5*0.5 + 0.5*(1/3)=0.416666...
    c.update(drift_detected=False, perf_stats=ps)
    assert c._total_chunks == 3
    assert c._total_drifts == 1
    assert c._drift_rate_ema == pytest.approx(0.4166666666666667, abs=1e-9)


def test_reset_clears_internal_state() -> None:
    cfg = _cfg(init_chunk_size=100, ema_decay=0.0, cooldown_period=0)
    c = AdaptiveChunkController(cfg)
    ps = PerformanceStats()

    c.update(drift_detected=True, perf_stats=ps)
    c.update(drift_detected=False, perf_stats=ps)

    assert c._total_chunks == 2
    assert c._total_drifts == 1
    assert c._drift_rate_ema != 0.0

    c.reset()
    assert c._cooldown_counter == 0
    assert c._drift_rate_ema == 0.0
    assert c._total_chunks == 0
    assert c._total_drifts == 0
