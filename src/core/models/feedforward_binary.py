from __future__ import annotations

from typing import TYPE_CHECKING

from .feedforward_base import FeedForwardBase

if TYPE_CHECKING:
    import torch


class FeedForwardBinary(FeedForwardBase):
    """
    Feedforward MLP for binary classification.

    Outputs raw logits of shape (batch_size,), intended for BCEWithLogitsLoss.
    """

    def __init__(self, input_dim: int, p_drop: float = 0.3) -> None:
        super().__init__(input_dim=input_dim, output_dim=1, p_drop=p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1) -> (B,)
        return super().forward(x).squeeze(1)
