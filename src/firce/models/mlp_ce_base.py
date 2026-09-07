# src/firce/models/mlp_ce_base
"""
CE-ready multilayer perceptron (MLP) base class.

This module defines `MLPCEBase`, the shared scaffolding for binary and
multiclass CE-model MLPs: hidden-layer trunk construction, the training
loop, device handling, and save/load/get_params/set_params/clone support.
Subclasses implement `_build_head`, `_loss_fn`, `_prepare_targets`, and
`_finalize_fit` to specialize output size, loss function, and label
handling.
"""

import logging
import threading

from typing import Any, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, TensorDataset

from firce.models.torch_device import pick_device

logger = logging.getLogger(__name__)


class MLPCEBase(nn.Module):
    """
    Shared trunk, training loop, and persistence scaffolding for CE-model MLPs.

    Subclasses must implement `_build_head`, `_loss_fn`, `_prepare_targets`,
    and `_finalize_fit` to specialize the output layer, loss function, target
    encoding, and post-fit bookkeeping (e.g. `classes_`).

    Args:
        input_dim: Number of input features.
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
        device: torch.device,
        widths: Tuple[int, ...] = (256, 128, 64),
        p_drop: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: Optional[int] = None,
        random_state: int = 42,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.widths = tuple(int(w) for w in widths)
        self.p_drop = float(p_drop)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.random_state = int(random_state)

        self.trunk = self._build_trunk()
        self.head = self._build_head(self._trunk_out_dim())

        self._device = device
        self.to(self._device)
        self.eval()

        self._lock = threading.RLock()
        if not hasattr(self, 'classes_'):
            self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False

    def _build_trunk(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = self.input_dim
        for w in self.widths:
            layers += [nn.Linear(d, w), nn.GELU(), nn.LayerNorm(w), nn.Dropout(self.p_drop)]
            d = w
        return nn.Sequential(*layers)

    def _trunk_out_dim(self) -> int:
        return self.widths[-1] if self.widths else self.input_dim

    def _build_head(self, in_features: int) -> nn.Module:
        """Build the output layer(s). Must be implemented by subclasses."""
        raise NotImplementedError

    def _loss_fn(self) -> nn.Module:
        """Return the loss criterion. Must be implemented by subclasses."""
        raise NotImplementedError

    def _prepare_targets(self, y: np.ndarray) -> torch.Tensor:
        """Convert raw labels to the tensor shape/dtype the loss expects."""
        raise NotImplementedError

    def _finalize_fit(self, y: np.ndarray) -> None:
        """Set post-fit bookkeeping (e.g. `classes_`) after training completes."""
        raise NotImplementedError

    def _extra_params(self) -> dict:
        """Subclass-specific constructor kwargs (e.g. `threshold`, `classes`)."""
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPCEBase':
        """
        Train or retrain the CE model on labeled data.

        Training uses deterministic, CPU-only shuffling per epoch to avoid
        device-specific randomness.

        Args:
            X: Feature matrix of shape ``(N, D)``.
            y: Labels of shape ``(N,)``; semantics defined by the subclass.

        Returns:
            The fitted model (``self``).
        """
        X = np.asarray(X, dtype=np.float32, order='C')

        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f'Expected X shape (N, {self.input_dim}), got {X.shape}')

        self.train()
        torch.manual_seed(self.random_state)
        if self._device.type == 'cuda':
            torch.cuda.manual_seed_all(self.random_state)

        X_tensor = torch.from_numpy(X)
        y_tensor = self._prepare_targets(y)
        criterion = self._loss_fn()

        N = X_tensor.shape[0]
        bs = self.batch_size or (2048 if N >= 8192 else 512)

        ds = TensorDataset(X_tensor, y_tensor)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        logger.debug(f'[mlp_ce.fit] device={self._device}, N={N}, bs={bs}, epochs={self.epochs}, lr={self.lr}')

        for epoch in range(self.epochs):
            rng = np.random.default_rng(self.random_state + epoch)
            idx = rng.permutation(N).tolist()
            subset = Subset(ds, idx)
            loader = DataLoader(subset, batch_size=bs, shuffle=False, num_workers=0)

            running = 0.0
            for b, (xb, yb) in enumerate(loader):
                xb = xb.to(self._device, dtype=torch.float32, non_blocking=False)
                yb = yb.to(self._device, non_blocking=False)
                optimizer.zero_grad(set_to_none=True)
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * xb.size(0)

            epoch_loss = running / N
            logger.debug(f'[mlp_ce.fit] epoch {epoch + 1}/{self.epochs} loss={epoch_loss:.6f}')

        self.eval()
        self._finalize_fit(y)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        logger.debug(f'[mlp_ce.fit] fitted: classes_={self.classes_}, n_features_in_={self.n_features_in_}')
        return self

    def _forward_batches(
        self,
        X: np.ndarray | torch.Tensor,
        batch_size: int,
        device: Optional[torch.device],
    ) -> np.ndarray:
        """Run forward passes over X in batches, returning concatenated raw outputs."""
        if isinstance(X, torch.Tensor):
            n = X.shape[0] if X.ndim == 2 else 1
            D = X.shape[1] if X.ndim == 2 else X.shape[0]
        else:
            X = np.asarray(X, dtype=np.float32, order='C')
            if X.ndim == 1:
                X = X.reshape(1, -1)
            n, D = X.shape

        if self.n_features_in_ is not None and D != self.n_features_in_:
            logger.debug(f'[mlp_ce._forward_batches] expected D={self.n_features_in_}, got D={D}')

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

        outputs: list[np.ndarray] = []
        with self._lock:
            self.eval()
            with torch.no_grad():
                for xb in it:
                    xb = xb.to(dev, dtype=torch.float32, non_blocking=False).contiguous()
                    out = self.forward(xb).to('cpu').numpy()
                    outputs.append(out)

        return np.concatenate(outputs, axis=0)

    def get_params(self, deep: bool = True) -> dict:
        """
        Return initialization parameters for cloning utilities.

        Args:
            deep: Ignored; provided for scikit-learn compatibility.

        Returns:
            A dictionary with keys matching the constructor signature.
        """
        params = {
            'input_dim': self.input_dim,
            'widths': self.widths,
            'p_drop': self.p_drop,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
            'device': self._device,
        }
        params.update(self._extra_params())
        return params

    def set_params(self, **params: Any) -> 'MLPCEBase':
        """
        Set parameters; rebuild layers if architecture-affecting values change.

        Returns:
            The updated instance (``self``).
        """
        arch_changed = False
        for k, v in params.items():
            if k == 'device':
                setattr(self, '_device', v)
                continue
            if hasattr(self, k):
                if k in ('input_dim', 'widths', 'p_drop') and getattr(self, k) != v:
                    arch_changed = True
                setattr(self, k, v)

        if arch_changed:
            self.trunk = self._build_trunk()
            self.head = self._build_head(self._trunk_out_dim())
            self.to(self._device)
            self.eval()
        return self

    def clone(self) -> 'MLPCEBase':
        """
        Create a fresh, untrained copy with the same hyperparameters.

        Returns:
            A new instance with randomly initialized weights.
        """
        params = self.get_params(deep=True)
        model = type(self)(**params)
        model.is_fitted_ = False
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
            'state_dict': self.state_dict(),
            'params': self.get_params(deep=True),
            'classes_': None if self.classes_ is None else np.asarray(self.classes_).tolist(),
            'n_features_in_': self.n_features_in_,
        }
        torch.save(ckpt, path)
        logger.debug(f'[mlp_ce.save] wrote checkpoint to {path}')

    @classmethod
    def load(
        cls,
        path: str,
        map_location: str | torch.device = 'cpu',
        device: Optional[torch.device] = None,
    ) -> 'MLPCEBase':
        """
        Load a model checkpoint created by :meth:`save`.

        Args:
            path: Checkpoint file path.
            map_location: Map location passed to :func:`torch.load`. Defaults to ``"cpu"``.
            device: Final device for the restored model. If ``None``, uses ``pick_device()``.

        Returns:
            An evaluation-ready instance of ``cls``.
        """
        ckpt = torch.load(path, map_location=map_location)
        params = dict(ckpt.get('params', {}))
        if device is not None:
            params['device'] = device
        else:
            params['device'] = params.get('device', pick_device())
        model = cls(**params)
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        if missing:
            logger.debug(f'[mlp_ce.load] missing keys: {missing}')
        if unexpected:
            logger.debug(f'[mlp_ce.load] unexpected keys: {unexpected}')
        if ckpt.get('classes_') is not None:
            model.classes_ = np.array(ckpt['classes_'])
        model.n_features_in_ = ckpt.get('n_features_in_')
        model.eval()
        logger.debug(f'[mlp_ce.load] loaded on device={next(model.parameters()).device}')
        return model


if __name__ == '__main__':
    raise NotImplementedError('This module is not intended to be run directly. ')
