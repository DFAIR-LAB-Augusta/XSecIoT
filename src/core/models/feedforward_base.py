from __future__ import annotations

import torch
import torch.nn as nn


class FeedForwardBase(nn.Module):
    """
    Shared feedforward MLP backbone.

    Architecture:
        Input -> Linear(64) -> ReLU -> Dropout ->
        Linear(32) -> ReLU -> Dropout -> Linear(output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int, p_drop: float = 0.3) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f'input_dim must be >= 1, got {input_dim}')
        if output_dim < 1:
            raise ValueError(f'output_dim must be >= 1, got {output_dim}')

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.p_drop = float(p_drop)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.p_drop),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.p_drop),
            nn.Linear(32, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
