from __future__ import annotations

from typing import TYPE_CHECKING

from .feedforward_base import FeedForwardBase

if TYPE_CHECKING:
    import torch


class FeedForwardMulticlass(FeedForwardBase):
    """
    Feedforward MLP for multiclass classification.

    Outputs raw logits of shape (batch_size, num_classes), intended for CrossEntropyLoss.
    """

    def __init__(self, input_dim: int, num_classes: int, p_drop: float = 0.3) -> None:
        if num_classes < 2:
            raise ValueError(f'num_classes must be >= 2, got {num_classes}')
        self.num_classes = int(num_classes)
        super().__init__(input_dim=input_dim, output_dim=self.num_classes, p_drop=p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C)
        return super().forward(x)
