from __future__ import annotations

import importlib
import sys

from collections import deque

import numpy as np
import pytest


def _import_adaptive_sig_module(monkeypatch: pytest.MonkeyPatch):
    """
    adaptive_significance_controller loads TOML at import-time. This helper makes the import robust by
    monkeypatching load_conformal_config() if the TOML file isn't available.
    """
    mod_name = 'src.core.conformalEval.adaptive_significance_controller'

    try:
        return importlib.import_module(mod_name)
    except FileNotFoundError:
        sys.modules.pop(mod_name, None)

        import src.core.conformalEval.utils as utils

        fake_cfg = {
            'adaptive_significance': {
                'decay': 0.9,
                'max_alpha': 0.3,
                'min_alpha': 0.1,
                'window_size': 10,
                'alpha_step': 0.05,
                'increase_threshold': 0.6,
                'decrease_threshold': 0.1,
            }
        }
        monkeypatch.setattr(utils, 'load_conformal_config', lambda *a, **k: fake_cfg, raising=True)

        return importlib.import_module(mod_name)


def test_update_creates_history_and_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _import_adaptive_sig_module(monkeypatch)
    ASC = m.AdaptiveSignificanceController

    ctlr = ASC(
        window_size=5,
        min_alpha=0.10,
        max_alpha=0.30,
        decay=0.9,
        alpha_step=0.05,
        increase_threshold=0.60,
        decrease_threshold=0.10,
    )

    classes = np.array(['A', 'A', 'B'], dtype=object)
    pvals = np.array([0.01, 0.02, 0.50], dtype=float)
    ctlr.update(classes=classes, p_values=pvals)

    assert 'A' in ctlr.pvalue_history
    assert 'B' in ctlr.pvalue_history
    assert isinstance(ctlr.pvalue_history['A'], deque)
    assert isinstance(ctlr.pvalue_history['B'], deque)

    # For A: current_alpha starts 0.10; drift_rate = mean(p < 0.10) = 1.0 -> increases
    assert ctlr.adaptive_thresholds['A'] == pytest.approx(0.15)
    # For B: drift_rate = mean([0.50] < 0.10) = 0.0 -> tries to decrease, but already at min
    assert ctlr.adaptive_thresholds['B'] == pytest.approx(0.10)

    assert list(ctlr.pvalue_history['A']) == [0.01, 0.02]
    assert list(ctlr.pvalue_history['B']) == [0.50]


def test_recompute_noop_for_empty_history(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _import_adaptive_sig_module(monkeypatch)
    ASC = m.AdaptiveSignificanceController

    ctlr = ASC(
        window_size=3,
        min_alpha=0.10,
        max_alpha=0.30,
        alpha_step=0.05,
        increase_threshold=0.60,
        decrease_threshold=0.10,
    )

    ctlr.pvalue_history['X'] = deque(maxlen=3)
    ctlr.adaptive_thresholds['X'] = 0.20

    ctlr._recompute_thresholds()
    assert ctlr.adaptive_thresholds['X'] == pytest.approx(0.20)


def test_thresholds_saturate_at_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _import_adaptive_sig_module(monkeypatch)
    ASC = m.AdaptiveSignificanceController

    ctlr = ASC(
        window_size=5,
        min_alpha=0.10,
        max_alpha=0.20,
        alpha_step=0.05,
        increase_threshold=0.60,
        decrease_threshold=0.10,
    )

    for _ in range(5):
        ctlr.update(
            classes=np.array(['A'] * 5, dtype=object),
            p_values=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        )
    assert ctlr.adaptive_thresholds['A'] <= 0.20 + 1e-9
    assert ctlr.adaptive_thresholds['A'] == pytest.approx(0.20)

    for _ in range(5):
        ctlr.update(
            classes=np.array(['A'] * 5, dtype=object),
            p_values=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
        )
    assert ctlr.adaptive_thresholds['A'] >= 0.10 - 1e-9
    assert ctlr.adaptive_thresholds['A'] == pytest.approx(0.10)


def test_stable_drift_keeps_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _import_adaptive_sig_module(monkeypatch)
    ASC = m.AdaptiveSignificanceController

    ctlr = ASC(
        window_size=10,
        min_alpha=0.10,
        max_alpha=0.30,
        alpha_step=0.05,
        increase_threshold=0.80,
        decrease_threshold=0.10,
    )

    ctlr.pvalue_history['C'] = deque([0.11, 0.20, 0.14, 0.16], maxlen=10)
    ctlr.adaptive_thresholds['C'] = 0.15

    # drift_rate = mean(p < 0.15) = (0.11, 0.14 are < 0.15) => 0.5
    # With thresholds (increase=0.80, decrease=0.10), this is "stable" -> no change.
    ctlr._recompute_thresholds()
    assert ctlr.adaptive_thresholds['C'] == pytest.approx(0.15)


def test_get_thresholds_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _import_adaptive_sig_module(monkeypatch)
    ASC = m.AdaptiveSignificanceController

    ctlr = ASC(min_alpha=0.10, max_alpha=0.30)
    ctlr.update(classes=np.array(['A'], dtype=object), p_values=np.array([0.2], dtype=float))

    thr = ctlr.get_thresholds()
    assert isinstance(thr, dict)
    assert 'A' in thr
    assert thr['A'] >= 0.10
