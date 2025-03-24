# src/core/models/feedforward_binary
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


class FeedForwardBinary(nn.Module):
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

    def __init__(self, input_dim: int, p_drop: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size,).
        """
        return self.net(x).squeeze(1)


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )
