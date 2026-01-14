from __future__ import annotations

import gc
import os

import pytest
import torch

import src.core.models.torch_device as td


def test_mps_usable_false_on_non_mps_platforms() -> None:
    out = td._mps_usable()
    assert isinstance(out, bool)


def test_smoke_test_cpu_success() -> None:
    ok, why = td._smoke_test(torch.device('cpu'))
    assert ok is True
    assert why == ''


def test_smoke_test_failure_returns_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError('boom')

    monkeypatch.setattr(torch, 'randn', _boom)
    ok, why = td._smoke_test(torch.device('cpu'))
    assert ok is False
    assert 'RuntimeError' in why
    assert 'boom' in why


def test_set_precision_safe_noop_if_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args, **kwargs):
        raise AttributeError('nope')

    monkeypatch.setattr(torch, 'set_float32_matmul_precision', _raise)
    td._set_precision_safe()


def test_pick_device_cpu_when_no_cuda_no_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(td, '_mps_usable', lambda: False)

    called = {'prec': 0}

    def _prec() -> None:
        called['prec'] += 1

    monkeypatch.setattr(td, '_set_precision_safe', _prec)

    dev = td.pick_device()
    assert dev.type == 'cpu'
    assert called['prec'] == 1


def test_pick_device_cuda_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(td, '_mps_usable', lambda: False)

    seen = {'dev': None, 'prec': 0}

    def _smoke(d: torch.device):
        seen['dev'] = d.type
        return True, ''

    def _prec() -> None:
        seen['prec'] += 1

    monkeypatch.setattr(td, '_smoke_test', _smoke)
    monkeypatch.setattr(td, '_set_precision_safe', _prec)

    dev = td.pick_device()
    assert dev.type == 'cuda'
    assert seen['dev'] == 'cuda'
    assert seen['prec'] == 1


def test_pick_device_cuda_fails_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(td, '_mps_usable', lambda: False)

    def _smoke(_d: torch.device):
        return False, 'nope'

    monkeypatch.setattr(td, '_smoke_test', _smoke)

    dev = td.pick_device()
    assert dev.type == 'cpu'


@pytest.mark.slow
def test_pick_device_mps_success_simulated_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(td, '_mps_usable', lambda: True)

    monkeypatch.setattr(td, '_smoke_test', lambda _d: (True, ''))

    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK', raising=False)

    dev = td.pick_device()
    assert dev.type == 'mps'
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'


@pytest.mark.slow
def test_pick_device_mps_fails_calls_empty_cache_and_gc_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(td, '_mps_usable', lambda: True)
    monkeypatch.setattr(td, '_smoke_test', lambda _d: (False, 'mps nope'))

    class _MPS:
        def __init__(self):
            self.calls = 0

        def empty_cache(self):
            self.calls += 1

    mps = _MPS()
    monkeypatch.setattr(torch, 'mps', mps, raising=False)

    gc_calls = {'n': 0}

    def _gc():
        gc_calls['n'] += 1
        return 0

    monkeypatch.setattr(gc, 'collect', _gc)

    dev = td.pick_device()
    assert dev.type == 'cpu'
    assert mps.calls == 1
    assert gc_calls['n'] == 1
