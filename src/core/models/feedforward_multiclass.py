"""
feedforward_multiclass
======================

This module defines a simple feedforward neural network (MLP) for multiclass
classification tasks, implemented in PyTorch.

Architecture:
    Input -> Dense(64, ReLU) -> Dropout(p) ->
    Dense(32, ReLU) -> Dropout(p) -> Dense(num_classes)

The model outputs raw logits of shape (batch_size, num_classes), intended for
use with ``torch.nn.CrossEntropyLoss`` during training. Softmax may be applied
at inference time to obtain class probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForwardMulticlass(nn.Module):
    """
    Feedforward multilayer perceptron for multiclass classification.

    Two hidden layers with ReLU activations and dropout regularization.
    The final output layer produces ``num_classes`` logits.

    Args:
        input_dim (int): Dimensionality of the input features.
        num_classes (int): Number of classes.
        p_drop (float, optional): Dropout probability applied after each
            hidden layer. Defaults to 0.3.
    """

    def __init__(self, input_dim: int, num_classes: int, p_drop: float = 0.3) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        return self.net(x)


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly.")
