# src/core/models/mlp_ce_multiclass.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.mlp_ce_base import MLP_CE_Base


class MLP_CE_Multiclass(MLP_CE_Base):
    """
    CE-ready MLP for multiclass classification.

    - Output logits: (B, C)
    - Loss: CrossEntropyLoss
    - y targets: int64 indices in [0..C-1]
    - predict_proba: softmax(logits) -> (N, C)
    - predict: argmax -> class label (or indices)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        device: torch.device,
        classes: Optional[np.ndarray] = None,
        widths: Tuple[int, ...] = (256, 128, 64),
        p_drop: float = 0.2,
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
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError(f'num_classes must be >= 2, got {self.num_classes}')

        self.classes_: Optional[np.ndarray] = None if classes is None else np.asarray(classes, dtype=object)
        self._class_to_index: Optional[dict[object, int]] = None

        self.net = self._build_net(output_dim=self.num_classes)
        self.to(self._device)
        self.eval()

    def _ensure_mapping(self, y_raw: np.ndarray) -> None:
        if self.classes_ is None:
            self.classes_ = np.unique(y_raw.astype(object))
        if int(self.classes_.size) != self.num_classes:
            raise ValueError(
                f'class count mismatch: num_classes={self.num_classes}, len(classes_)={int(self.classes_.size)}'
            )
        self._class_to_index = {c: i for i, c in enumerate(self.classes_)}

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        y_raw = np.asarray(y, dtype=object).reshape(-1)
        self._ensure_mapping(y_raw)
        assert self._class_to_index is not None

        try:
            y_idx = np.fromiter(
                (self._class_to_index[v] for v in y_raw),
                dtype=np.int64,
                count=y_raw.shape[0],
            )
        except KeyError as e:
            raise ValueError(f'Unknown class label encountered: {e!s}') from e

        return y_idx

    def _criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _logits_to_proba(self, logits: torch.Tensor) -> np.ndarray:
        # logits: (B, C) -> probs: (B, C)
        probs = F.softmax(logits, dim=1).to('cpu').numpy()
        return probs

    def predict(
        self,
        X: np.ndarray | torch.Tensor,
        batch_size: int = 4096,
        device: Optional[torch.device] = None,
        return_indices: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        proba = self.predict_proba(X, batch_size=batch_size, device=device)
        idx = np.argmax(proba, axis=1).astype(np.int32, copy=False)
        if return_indices:
            return idx
        if self.classes_ is None:
            return idx
        return self.classes_[idx]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'classes': None if self.classes_ is None else self.classes_.copy(),
            'widths': self.widths,
            'p_drop': self.p_drop,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
            'device': self._device,
        }

    def clone(self) -> 'MLP_CE_Multiclass':
        return MLP_CE_Multiclass(**self.get_params(deep=True))
