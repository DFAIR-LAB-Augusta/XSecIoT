from __future__ import annotations

import pytest
import torch

from src.core.models.feedforward_multiclass import FeedForwardMulticlass


def test_feedforward_multiclass_rejects_num_classes_lt_2() -> None:
    with pytest.raises(ValueError, match=r'num_classes must be >= 2'):
        FeedForwardMulticlass(input_dim=4, num_classes=1)


def test_feedforward_multiclass_forward_shape_and_attrs() -> None:
    m = FeedForwardMulticlass(input_dim=6, num_classes=5, p_drop=0.0)
    x = torch.randn(3, 6, dtype=torch.float32)
    y = m(x)

    assert m.num_classes == 5
    assert m.input_dim == 6
    assert m.output_dim == 5

    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 5)
    assert y.dtype == torch.float32
