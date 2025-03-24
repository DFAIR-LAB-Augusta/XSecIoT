# src/core/models/mlp_ce
"""
CE-ready multilayer perceptron (MLP) for binary classification.

This module defines `MLP_CE`, an MLP that can act as the *CE model*
in your pipeline. It exposes scikit-learn–style methods (`fit`,
`predict_proba`, `predict`) so it can be used directly by ICE/CCE/Approx-CCE
without wrappers.

Key features:
- Deterministic per-epoch CPU shuffling (avoids `torch.randperm` on MPS).
- Thread-safe inference for use with threaded Approx-CCE calibration.
- `get_params`/`set_params` so cloning utilities can re-instantiate models.
- `save`/`load` checkpoint helpers.
"""
import logging
import threading

from typing import Any, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, TensorDataset

from src.core.models.torch_device import pick_device

logger = logging.getLogger(__name__)


class MLP_CE(nn.Module):
    """
    MLP for binary classification with CE-friendly APIs.

    This class is intended to be the CE model. It exposes scikit-learn–style
    methods (`fit`, `predict_proba`, `predict`) so your ICE/CCE/Approx-CCE
    code can use it directly.

    Args:
        input_dim: Number of input features.
        widths: Hidden layer widths (applied in order). Defaults to (256, 128, 64).
        p_drop: Dropout probability after each hidden layer. Defaults to 0.2.
        threshold: Positive-class decision threshold used by ``predict``. Defaults to 0.5.
        lr: Learning rate for Adam optimizer during ``fit``. Defaults to 1e-3.
        epochs: Number of training epochs for ``fit``. Defaults to 20.
        batch_size: Mini-batch size used in ``fit`` and inference. If ``None``,
            an appropriate value is chosen based on dataset size. Defaults to ``None``.
        random_state: Seed for deterministic training. Defaults to 42.
        device: Preferred torch device.
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
        super().__init__()
        self.input_dim = int(input_dim)
        self.widths = tuple(int(w) for w in widths)
        self.p_drop = float(p_drop)
        self.threshold = float(threshold)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.random_state = int(random_state)

        layers: list[nn.Module] = []
        d = self.input_dim
        for w in self.widths:
            layers += [nn.Linear(d, w), nn.GELU(),
                       nn.LayerNorm(w), nn.Dropout(self.p_drop)]
            d = w
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

        self._device = device
        self.to(self._device)
        self.eval()

        self._lock = threading.RLock()
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP_CE":
        """
        Train or retrain the CE model on labeled data.

        Training uses deterministic, CPU-only shuffling per epoch to avoid
        device-specific randomness.

        Args:
            X: Feature matrix of shape ``(N, D)``.
            y: Binary labels in ``{0, 1}`` with shape ``(N,)``.

        Returns:
            The fitted model (``self``).
        """
        X = np.asarray(X, dtype=np.float32, order="C")
        y = np.asarray(y).astype(np.float32, copy=False)

        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected X shape (N, {self.input_dim}), got {X.shape}")

        self.train()
        torch.manual_seed(self.random_state)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        N = X_tensor.shape[0]
        bs = self.batch_size or (2048 if N >= 8192 else 512)

        ds = TensorDataset(X_tensor, y_tensor)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        logger.debug(
            f"[mlp_ce.fit] device={self._device}, N={N}, bs={bs}, epochs={self.epochs}, lr={self.lr}")

        for epoch in range(self.epochs):
            rng = np.random.default_rng(self.random_state + epoch)
            idx = rng.permutation(N).tolist()
            subset = Subset(ds, idx)
            loader = DataLoader(subset, batch_size=bs,
                                shuffle=False, num_workers=0)

            running = 0.0
            for b, (xb, yb) in enumerate(loader):
                xb = xb.to(self._device, dtype=torch.float32,
                           non_blocking=False)
                yb = yb.to(self._device, dtype=torch.float32,
                           non_blocking=False).view(-1)
                optimizer.zero_grad(set_to_none=True)
                logits = self.net(xb).squeeze(1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * xb.size(0)

            epoch_loss = running / N
            logger.debug(
                f"[mlp_ce.fit] epoch {epoch + 1}/{self.epochs} loss={epoch_loss:.6f}")

        self.eval()
        self.classes_ = np.unique(y.astype(int))
        if self.classes_.tolist() != [0, 1]:
            self.classes_ = np.array(sorted(self.classes_.tolist()))
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        logger.debug(
            f"[mlp_ce.fit] fitted: classes_={self.classes_}, n_features_in_={self.n_features_in_}")
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute raw logits.

        Args:
            x: Input tensor of shape ``(N, input_dim)``.

        Returns:
            Logits of shape ``(N,)``.
        """
        return self.net(x).squeeze(1)

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
        if isinstance(X, torch.Tensor):
            n = X.shape[0] if X.ndim == 2 else 1
            D = X.shape[1] if X.ndim == 2 else X.shape[0]
        else:
            X = np.asarray(X, dtype=np.float32, order="C")
            if X.ndim == 1:
                X = X.reshape(1, -1)
            n, D = X.shape

        if self.n_features_in_ is not None and D != self.n_features_in_:
            logger.debug(
                f"[mlp_ce.predict_proba] expected D={self.n_features_in_}, got D={D}")

        dev = device if device is not None else self._device

        def batch_iter_np() -> Iterable[torch.Tensor]:
            assert not isinstance(X, torch.Tensor)
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                yield torch.from_numpy(X[s:e])

        def batch_iter_t() -> Iterable[torch.Tensor]:
            assert isinstance(X, torch.Tensor)
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                yield X[s:e]

        it = batch_iter_t() if isinstance(X, torch.Tensor) else batch_iter_np()

        probs: list[np.ndarray] = []
        with self._lock:
            self.eval()
            for i, xb in enumerate(it):
                xb = xb.to(dev, dtype=torch.float32,
                           non_blocking=False).contiguous()
                logger.debug(
                    f"[mlp_ce.predict_proba] batch {i} xb.shape={tuple(xb.shape)} device={xb.device}")
                logits = self.forward(xb)
                pb = torch.sigmoid(logits).to("cpu").numpy().reshape(-1)
                logger.debug(
                    f"[mlp_ce.predict_proba] batch {i} probs.shape={pb.shape}")
                probs.append(pb)

        p1 = np.concatenate(probs, axis=0).reshape(-1)
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
        proba = self.predict_proba(
            X, batch_size=batch_size, device=device)[:, 1]
        y = (proba > thr).astype(np.int32, copy=False)
        logger.debug(
            f"[mlp_ce.predict] shape={y.shape}, mean_prob={float(proba.mean()):.6f}, thr={thr}")
        return y

    def get_params(self, deep: bool = True) -> dict:
        """
        Return initialization parameters for cloning utilities.

        Args:
            deep: Ignored; provided for scikit-learn compatibility.

        Returns:
            A dictionary with keys matching the constructor signature.
        """
        return {
            "input_dim": self.input_dim,
            "widths": self.widths,
            "p_drop": self.p_drop,
            "threshold": self.threshold,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "device": self._device,
        }

    def set_params(self, **params: Any) -> "MLP_CE":
        """
        Set parameters; rebuild layers if architecture-affecting values change.

        Returns:
            The updated instance (``self``).
        """
        arch_changed = False
        for k, v in params.items():
            if k == "device":
                setattr(self, "_device", v)
                continue
            if hasattr(self, k):
                if k in ("input_dim", "widths", "p_drop") and getattr(self, k) != v:
                    arch_changed = True
                setattr(self, k, v)

        if arch_changed:
            layers: list[nn.Module] = []
            d = int(self.input_dim)
            for w in tuple(self.widths):
                layers += [nn.Linear(d, int(w)), nn.GELU(),
                           nn.LayerNorm(int(w)), nn.Dropout(float(self.p_drop))]
                d = int(w)
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)
            self.to(self._device)
            self.eval()
        return self

    def clone(self) -> "MLP_CE":
        """
        Create a fresh, untrained copy with the same hyperparameters.

        Returns:
            A new instance with randomly initialized weights.
        """
        params = self.get_params(deep=True)
        model = MLP_CE(**params)
        model.is_fitted_ = False
        model.classes_ = None
        model.n_features_in_ = None
        return model

    def save(self, path: str) -> None:
        """
        Save model weights and configuration to disk.

        The checkpoint includes model weights and constructor parameters.

        Args:
            path: Destination file path (e.g., ``.pt`` file).
        """
        ckpt = {
            "state_dict": self.state_dict(),
            "params": self.get_params(deep=True),
            "classes_": None if self.classes_ is None else self.classes_.tolist(),
            "n_features_in_": self.n_features_in_,
        }
        torch.save(ckpt, path)
        logger.debug(f"[mlp_ce.save] wrote checkpoint to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        map_location: str | torch.device = "cpu",
        device: Optional[torch.device] = None,
    ) -> "MLP_CE":
        """
        Load a model checkpoint created by :meth:`save`.

        Args:
            path: Checkpoint file path.
            map_location: Map location passed to :func:`torch.load`. Defaults to ``"cpu"``.
            device: Final device for the restored model. If ``None``, uses ``pick_device()``.

        Returns:
            An evaluation-ready :class:`MLP_CE` instance.
        """
        ckpt = torch.load(path, map_location=map_location)
        params = ckpt.get("params", {})
        if device is not None:
            params["device"] = device
        else:
            params["device"] = params.get("device", pick_device())
        model = cls(**params)
        missing, unexpected = model.load_state_dict(
            ckpt["state_dict"], strict=False)
        if missing:
            logger.debug(f"[mlp_ce.load] missing keys: {missing}")
        if unexpected:
            logger.debug(f"[mlp_ce.load] unexpected keys: {unexpected}")
        if ckpt.get("classes_") is not None:
            model.classes_ = np.array(ckpt["classes_"])
        model.n_features_in_ = ckpt.get("n_features_in_")
        model.eval()
        logger.debug(
            f"[mlp_ce.load] loaded on device={next(model.parameters()).device}")
        return model


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )
