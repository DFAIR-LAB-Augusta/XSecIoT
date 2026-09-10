# src/firce/models/feedforward_binary
"""
feedforward_binary
==================

This module defines a simple feedforward neural network (MLP) for binary
classification tasks. The network architecture mirrors the former Keras
topology used in earlier experiments and is implemented using PyTorch.

Architecture:
    Input -> Dense(64, ReLU) -> Dropout(0.3) ->
    Dense(32, ReLU) -> Dropout(0.3) -> Dense(1)

The model outputs raw logits, intended for use with
``torch.nn.BCEWithLogitsLoss`` during training. A sigmoid activation
should be applied at inference to obtain probabilities.
"""

import torch
import torch.nn as nn

from firce.models.feedforward_base import FeedForwardBase


class FeedForwardBinary(FeedForwardBase):
    """
    Feedforward multilayer perceptron for binary classification.

    This network consists of two hidden layers with ReLU activations
    and dropout regularization. The final output layer produces a
    single logit, which should be passed through a sigmoid function
    during inference.

    Args:
        input_dim (int): Dimensionality of the input features.
        p_drop (float, optional): Dropout probability applied after each
            hidden layer. Defaults to 0.3.
    """

    def _build_head(self) -> nn.Module:
        return nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size,).
        """
        return super().forward(x).squeeze(1)


if __name__ == '__main__':
    raise NotImplementedError('This module is not intended to be run directly. ')
