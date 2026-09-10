# src/firce/models/feedforward_base
"""
feedforward_base
=================

Shared trunk for the binary and multiclass feedforward (FFN) CE models.
`FeedForwardBase` builds the two-hidden-layer MLP trunk (mirroring the
former Keras topology) and lets subclasses attach their own output head
via `_build_head`.
"""

import torch
import torch.nn as nn


class FeedForwardBase(nn.Module):
    """
    Shared feedforward trunk: Dense(64, ReLU) -> Dropout -> Dense(32, ReLU) -> Dropout.

    Subclasses implement `_build_head` to attach an output layer sized for
    their task (single logit for binary, K logits for multiclass).

    Args:
        input_dim (int): Dimensionality of the input features.
        p_drop (float, optional): Dropout probability applied after each
            hidden layer. Defaults to 0.3.
    """

    def __init__(self, input_dim: int, p_drop: float = 0.3) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p_drop),
        )
        self.head = self._build_head()

    def _build_head(self) -> nn.Module:
        """Build the output layer. Must be implemented by subclasses."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Raw output of the head (shape depends on subclass).
        """
        return self.head(self.trunk(x))


if __name__ == '__main__':
    raise NotImplementedError('This module is not intended to be run directly. ')
