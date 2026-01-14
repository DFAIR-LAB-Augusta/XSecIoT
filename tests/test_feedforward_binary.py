from __future__ import annotations

import pytest
import torch

from src.core.models.feedforward_binary import FeedForwardBinary


def test_feedforward_binary_forward_squeezes_last_dim() -> None:
    m = FeedForwardBinary(input_dim=6, p_drop=0.0)
    x = torch.randn(4, 6, dtype=torch.float32)
    y = m(x)

    # override behavior: (B, 1) -> (B,)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (4,)
    assert y.dtype == torch.float32


def test_feedforward_binary_is_base_with_output_dim_1() -> None:
    m = FeedForwardBinary(input_dim=3, p_drop=0.25)
    assert m.input_dim == 3
    assert m.output_dim == 1
    assert pytest.approx(m.p_drop) == 0.25
