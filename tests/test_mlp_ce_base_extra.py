from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.core.models.mlp_ce_base import MLP_CE_Base


class _DummyCE(MLP_CE_Base):
    """
    Minimal concrete subclass to exercise base-class branches cheaply.
    """

    def __init__(self, input_dim: int, device: torch.device) -> None:
        super().__init__(input_dim=input_dim, device=device,
                         widths=(4,), p_drop=0.0, lr=1e-3, epochs=0)
        self.net = self._build_net(output_dim=1)
        self.to(self._device)
        self.eval()

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y).reshape(-1, 1).astype(np.float32, copy=False)

    def _criterion(self) -> nn.Module:
        return nn.MSELoss()

    def _logits_to_proba(self, logits: torch.Tensor) -> np.ndarray:
        p1 = torch.sigmoid(logits).to("cpu").numpy().reshape(-1)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    def predict(self, X: np.ndarray | torch.Tensor, **kwargs) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(np.int32)


def test_fit_raises_on_wrong_shape() -> None:
    m = _DummyCE(input_dim=3, device=torch.device("cpu"))
    X_bad = np.zeros((5, 2), dtype=np.float32)
    y = np.zeros((5,), dtype=np.int32)
    with pytest.raises(ValueError, match=r"Expected X shape"):
        m.fit(X_bad, y)


def test_predict_proba_reshapes_1d_input() -> None:
    m = _DummyCE(input_dim=3, device=torch.device("cpu"))

    x1d = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    proba = m.predict_proba(x1d)

    assert proba.shape == (1, 2)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_tensor_branch_contiguous() -> None:
    m = _DummyCE(input_dim=3, device=torch.device("cpu"))

    xt = torch.randn(7, 3, dtype=torch.float32)
    proba = m.predict_proba(xt, batch_size=4)

    assert proba.shape == (7, 2)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
