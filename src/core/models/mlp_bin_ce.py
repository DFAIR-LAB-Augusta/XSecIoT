# src/core/models/mlp_ce_binary.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.core.models.mlp_ce_base import MLP_CE_Base


class MLP_CE_Binary(MLP_CE_Base):
    """
    CE-ready MLP for binary classification.

    - Output logits: (B, 1)
    - Loss: BCEWithLogitsLoss
    - predict_proba: returns (N, 2) as [P(class=0), P(class=1)]
    - predict: threshold on P(class=1)
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
    ) -> None:
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
        self.threshold = float(threshold)

        # binary output_dim = 1
        self.net = self._build_net(output_dim=1)
        self.to(self._device)
        self.eval()

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        # Make targets shape (N, 1) float32 to match logits shape (B, 1)
        y_arr = np.asarray(y)
        return y_arr.reshape(-1, 1).astype(np.float32, copy=False)

    def _criterion(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def _logits_to_proba(self, logits: torch.Tensor) -> np.ndarray:
        # logits: (B, 1) -> p1: (B,)
        p1 = torch.sigmoid(logits).to('cpu').numpy().reshape(-1)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)  # (B, 2)

    def predict(
        self,
        X: np.ndarray | torch.Tensor,
        batch_size: int = 4096,
        device: Optional[torch.device] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        thr = self.threshold if threshold is None else float(threshold)
        proba = self.predict_proba(X, batch_size=batch_size, device=device)
        return (proba[:, 1] > thr).astype(np.int32)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'widths': self.widths,
            'p_drop': self.p_drop,
            'threshold': self.threshold,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
            'device': self._device,
        }

    def clone(self) -> 'MLP_CE_Binary':
        return MLP_CE_Binary(**self.get_params(deep=True))
