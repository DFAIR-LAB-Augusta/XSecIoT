# src/firce/models/mlp_ce
"""
CE-ready multilayer perceptron (MLP) for binary classification.

This module defines `MLP_CE`, an MLP that can act as the *CE model*
in your pipeline. It exposes scikit-learn–style methods (`fit`,
`predict_proba`, `predict`) so it can be used directly by ICE/CCE/Approx-CCE
without wrappers. Shared trunk/training/persistence scaffolding lives in
`MLPCEBase`; see `mlp_ce_multiclass.py` for the multiclass sibling.
"""

import logging

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from firce.models.mlp_ce_base import MLPCEBase

logger = logging.getLogger(__name__)


class MLP_CE(MLPCEBase):
    """
    MLP for binary classification with CE-friendly APIs.

    This class is intended to be the CE model. It exposes scikit-learn–style
    methods (`fit`, `predict_proba`, `predict`) so your ICE/CCE/Approx-CCE
    code can use it directly.

    Args:
        input_dim: Number of input features.
        device: Preferred torch device.
        widths: Hidden layer widths (applied in order). Defaults to (256, 128, 64).
        p_drop: Dropout probability after each hidden layer. Defaults to 0.2.
        threshold: Positive-class decision threshold used by ``predict``. Defaults to 0.5.
        lr: Learning rate for Adam optimizer during ``fit``. Defaults to 1e-3.
        epochs: Number of training epochs for ``fit``. Defaults to 20.
        batch_size: Mini-batch size used in ``fit`` and inference. If ``None``,
            an appropriate value is chosen based on dataset size. Defaults to ``None``.
        random_state: Seed for deterministic training. Defaults to 42.
    """

    def __init__(
        self,
        input_dim: int,
        device: torch.device,
        widths: Tuple[int, ...] = (256, 128, 64),
        p_drop: float = 0.2,
        threshold: float = 0.5,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: Optional[int] = None,
        random_state: int = 42,
    ):
        self.threshold = float(threshold)
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
        return nn.Linear(in_features, 1)

    def _loss_fn(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def _prepare_targets(self, y: np.ndarray) -> torch.Tensor:
        y_arr = np.asarray(y).astype(np.float32, copy=False)
        return torch.from_numpy(y_arr)

    def _finalize_fit(self, y: np.ndarray) -> None:
        classes = np.unique(np.asarray(y).astype(int))
        if classes.tolist() != [0, 1]:
            classes = np.array(sorted(classes.tolist()))
        self.classes_ = classes

    def _extra_params(self) -> dict:
        return {'threshold': self.threshold}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute raw logits.

        Args:
            x: Input tensor of shape ``(N, input_dim)``.

        Returns:
            Logits of shape ``(N,)``.
        """
        return super().forward(x).squeeze(1)

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
            Array of shape ``(N, 2)`` with probabilities ``[P(0), P(1)]``.
        """
        logits = self._forward_batches(X, batch_size, device).reshape(-1)
        p1 = 1.0 / (1.0 + np.exp(-logits))
        logger.debug(f'[mlp_ce.predict_proba] p1.shape={p1.shape}')
        return np.stack([1.0 - p1, p1], axis=1)

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray | torch.Tensor,
        batch_size: int = 4096,
        device: Optional[torch.device] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict binary labels for input samples by thresholding ``P(class=1)``.

        Args:
            X: Input features of shape ``(N, D)`` or a single sample ``(D,)``.
            batch_size: Inference batch size. Defaults to ``4096``.
            device: Override device for inference. Defaults to the model device.
            threshold: Override decision threshold. Defaults to the instance value.

        Returns:
            Integer labels of shape ``(N,)`` with values in ``{0, 1}``.
        """
        thr = self.threshold if threshold is None else float(threshold)
        proba = self.predict_proba(X, batch_size=batch_size, device=device)[:, 1]
        y = (proba > thr).astype(np.int32, copy=False)
        logger.debug(f'[mlp_ce.predict] shape={y.shape}, mean_prob={float(proba.mean()):.6f}, thr={thr}')
        return y


if __name__ == '__main__':
    raise NotImplementedError('This module is not intended to be run directly. ')
