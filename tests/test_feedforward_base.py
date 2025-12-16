from __future__ import annotations

import numpy as np
import pytest
import torch

from src.core.models.feedforward_base import FeedForwardBase


def test_feedforward_base_rejects_invalid_input_dim() -> None:
    with pytest.raises(ValueError, match=r'input_dim must be >= 1'):
        FeedForwardBase(input_dim=0, output_dim=1)


def test_feedforward_base_rejects_invalid_output_dim() -> None:
    with pytest.raises(ValueError, match=r'output_dim must be >= 1'):
        FeedForwardBase(input_dim=4, output_dim=0)


def test_feedforward_base_forward_shape_binary_output() -> None:
    m = FeedForwardBase(input_dim=8, output_dim=1, p_drop=0.0)
    x = torch.randn(5, 8, dtype=torch.float32)
    y = m(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (5, 1)
    assert y.dtype == torch.float32


def test_feedforward_base_forward_shape_multiclass_output() -> None:
    m = FeedForwardBase(input_dim=8, output_dim=5, p_drop=0.0)
    x = torch.from_numpy(np.random.default_rng(0).normal(size=(7, 8)).astype(np.float32))
    y = m(x)

    assert y.shape == (7, 5)
    assert y.dtype == torch.float32


def test_feedforward_base_stores_hparams() -> None:
    m = FeedForwardBase(input_dim=3, output_dim=2, p_drop=0.25)
    assert m.input_dim == 3
    assert m.output_dim == 2
    assert pytest.approx(m.p_drop) == 0.25
