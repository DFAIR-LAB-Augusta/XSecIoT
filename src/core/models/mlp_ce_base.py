from __future__ import annotations

import threading

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, TensorDataset


class MLP_CE_Base(nn.Module, ABC):
    def __init__(
        self,
        input_dim: int,
        device: torch.device,
        widths: Tuple[int, ...] = (256, 128, 64),
        p_drop: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.widths = tuple(int(w) for w in widths)
        self.p_drop = float(p_drop)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.random_state = int(random_state)

        self._device = device
        self._lock = threading.RLock()
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False

        # subclass sets self.net via _build_net(...)
        self.net: nn.Sequential

    def _build_net(self, output_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = self.input_dim
        for w in self.widths:
            layers += [
                nn.Linear(d, w),
                nn.GELU(),
                nn.LayerNorm(w),
                nn.Dropout(self.p_drop),
            ]
            d = w
        layers += [nn.Linear(d, output_dim)]
        return nn.Sequential(*layers)

    @abstractmethod
    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        """Convert raw labels to the training target format (binary float or MC int indices)."""

    @abstractmethod
    def _criterion(self) -> nn.Module:
        """Return loss function instance."""

    @abstractmethod
    def _logits_to_proba(self, logits: torch.Tensor) -> np.ndarray:
        """Convert model logits to numpy probabilities with the correct shape."""

    @abstractmethod
    def predict(self, X: np.ndarray | torch.Tensor, **kwargs: Any) -> np.ndarray:
        """Binary thresholding vs multiclass argmax."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLP_CE_Base':
        X = np.asarray(X, dtype=np.float32, order='C')
        y = np.asarray(y)

        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f'Expected X shape (N, {self.input_dim}), got {X.shape}')

        y_train = self._prepare_y(y)

        self.train()
        torch.manual_seed(self.random_state)
        if self._device.type == 'cuda':
            torch.cuda.manual_seed_all(self.random_state)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y_train)

        N = X_tensor.shape[0]
        bs = self.batch_size or (2048 if N >= 8192 else 512)

        ds = TensorDataset(X_tensor, y_tensor)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = self._criterion()

        for epoch in range(self.epochs):
            rng = np.random.default_rng(self.random_state + epoch)
            idx = rng.permutation(N).tolist()
            subset = Subset(ds, idx)
            loader = DataLoader(subset, batch_size=bs, shuffle=False, num_workers=0)

            for xb, yb in loader:
                xb = xb.to(self._device, dtype=torch.float32, non_blocking=False)
                yb = yb.to(self._device, non_blocking=False)

                optimizer.zero_grad(set_to_none=True)
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        self.eval()
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def predict_proba(
        self,
        X: np.ndarray | torch.Tensor,
        batch_size: int = 4096,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        dev = device if device is not None else self._device

        if not isinstance(X, torch.Tensor):
            X = np.asarray(X, dtype=np.float32, order='C')
            if X.ndim == 1:
                X = X.reshape(1, -1)

        n = X.shape[0]  # type: ignore[union-attr]

        probs: list[np.ndarray] = []
        with self._lock:
            self.eval()
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                xb = X[s:e]  # type: ignore[index]
                xb = (
                    xb.to(dev, dtype=torch.float32, non_blocking=False).contiguous()
                    if isinstance(xb, torch.Tensor)
                    else torch.from_numpy(xb).to(dev)
                )
                logits = self.forward(xb)
                probs.append(self._logits_to_proba(logits))

        return np.concatenate(probs, axis=0)
