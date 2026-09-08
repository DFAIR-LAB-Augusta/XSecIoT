# src/firce/models/mlp_ce_multiclass
"""
CE-ready multilayer perceptron (MLP) for multiclass classification.

This module defines `MLPCEMulticlass`, the multiclass sibling of `MLP_CE`:
K output logits, cross-entropy loss, and softmax `predict_proba`. Shared
trunk/training/persistence scaffolding lives in `MLPCEBase`.
"""

import logging

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from firce.models.mlp_ce_base import MLPCEBase

logger = logging.getLogger(__name__)


class MLPCEMulticlass(MLPCEBase):
    """
    MLP for multiclass classification with CE-friendly APIs.

    Exposes scikit-learn–style `fit`, `predict_proba`, `predict` so
    ICE/CCE/Approx-CCE can use it directly, mirroring `MLP_CE`'s binary
    interface but with K output logits, cross-entropy loss, and softmax
    probabilities.

    Args:
        input_dim: Number of input features.
        classes: The K known class labels (sortable, hashable values, e.g.
            attack-type strings). Fixes the output layer size and the
            column order of ``predict_proba``. Must contain at least 2
            distinct values.
        device: Preferred torch device.
        widths: Hidden layer widths (applied in order). Defaults to (256, 128, 64).
        p_drop: Dropout probability after each hidden layer. Defaults to 0.2.
        lr: Learning rate for Adam optimizer during ``fit``. Defaults to 1e-3.
        epochs: Number of training epochs for ``fit``. Defaults to 20.
        batch_size: Mini-batch size used in ``fit`` and inference. If ``None``,
            an appropriate value is chosen based on dataset size. Defaults to ``None``.
        random_state: Seed for deterministic training. Defaults to 42.
    """

    def __init__(
        self,
        input_dim: int,
        classes: Sequence,
        device: torch.device,
        widths: Tuple[int, ...] = (256, 128, 64),
        p_drop: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: Optional[int] = None,
        random_state: int = 42,
    ):
        classes_arr = np.array(sorted(set(classes)))
        if len(classes_arr) < 2:
            raise ValueError(f'classes must contain at least 2 distinct labels, got {classes_arr.tolist()}')
        self.classes_ = classes_arr
        self.num_classes = len(classes_arr)
        self._label_to_index = {label: i for i, label in enumerate(classes_arr.tolist())}
        super().__init__(
            input_dim=input_dim,
            device=device,
            widths=widths,
            p_drop=p_drop,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
        )

    def _build_head(self, in_features: int) -> nn.Module:
        return nn.Linear(in_features, self.num_classes)

    def _loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _prepare_targets(self, y: np.ndarray) -> torch.Tensor:
        y_arr = np.asarray(y)
        try:
            idx = np.array([self._label_to_index[label] for label in y_arr.tolist()], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f'Label {exc} not found in known classes {self.classes_.tolist()}') from exc
        return torch.from_numpy(idx)

    def _finalize_fit(self, y: np.ndarray) -> None:
        pass  # classes_ is fixed at construction time, not derived from fit data.

    def _extra_params(self) -> dict:
        return {'classes': self.classes_.tolist()}

    @torch.no_grad()
    def predict_proba(
        self,
        X: np.ndarray | torch.Tensor,
        batch_size: int = 4096,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Predict class probabilities for input samples.

        Args:
            X: Input features of shape ``(N, D)`` or a single sample ``(D,)``.
            batch_size: Inference batch size. Defaults to ``4096``.
            device: Override device for inference. Defaults to the model device.

        Returns:
            Array of shape ``(N, K)`` with probabilities in ``classes_`` order.
        """
        logits = self._forward_batches(X, batch_size, device)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = exp / exp.sum(axis=1, keepdims=True)
        logger.debug(f'[mlp_ce_multiclass.predict_proba] proba.shape={proba.shape}')
        return proba

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray | torch.Tensor,
        batch_size: int = 4096,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Predict class labels for input samples via argmax over ``predict_proba``.

        Args:
            X: Input features of shape ``(N, D)`` or a single sample ``(D,)``.
            batch_size: Inference batch size. Defaults to ``4096``.
            device: Override device for inference. Defaults to the model device.

        Returns:
            Array of shape ``(N,)`` with values drawn from ``classes_``.
        """
        proba = self.predict_proba(X, batch_size=batch_size, device=device)
        idx = np.argmax(proba, axis=1)
        preds = self.classes_[idx]
        logger.debug(f'[mlp_ce_multiclass.predict] shape={preds.shape}')
        return preds


if __name__ == '__main__':
    raise NotImplementedError('This module is not intended to be run directly. ')
