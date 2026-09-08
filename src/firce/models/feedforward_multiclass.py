# src/firce/models/feedforward_multiclass
"""
feedforward_multiclass
=======================

Feedforward neural network (MLP) for multiclass classification, sharing
the same two-hidden-layer trunk as `FeedForwardBinary` via `FeedForwardBase`.

Architecture:
    Input -> Dense(64, ReLU) -> Dropout(0.3) ->
    Dense(32, ReLU) -> Dropout(0.3) -> Dense(num_classes)

The model outputs raw logits, intended for use with
``torch.nn.CrossEntropyLoss`` during training. A softmax activation
should be applied at inference to obtain probabilities.
"""

import torch.nn as nn

from firce.models.feedforward_base import FeedForwardBase


class FeedForwardMulticlass(FeedForwardBase):
    """
    Feedforward multilayer perceptron for multiclass classification.

    Args:
        input_dim (int): Dimensionality of the input features.
        num_classes (int): Number of output classes.
        p_drop (float, optional): Dropout probability applied after each
            hidden layer. Defaults to 0.3.
    """

    def __init__(self, input_dim: int, num_classes: int, p_drop: float = 0.3) -> None:
        self.num_classes = int(num_classes)
        super().__init__(input_dim, p_drop)

    def _build_head(self) -> nn.Module:
        return nn.Linear(32, self.num_classes)


if __name__ == '__main__':
    raise NotImplementedError('This module is not intended to be run directly. ')
