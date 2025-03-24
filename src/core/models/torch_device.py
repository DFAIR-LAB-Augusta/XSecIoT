# src/core/torch_device
"""
torch_device
============

This module provides utilities for selecting and testing the best
available PyTorch device (CUDA, MPS, or CPU). It performs a smoke test
to ensure the device is usable for forward and backward passes, and
falls back gracefully if issues are detected.

Functions:
    pick_device: Select the best device available with smoke testing.
"""

from __future__ import annotations

import gc
import logging

from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _mps_usable() -> bool:
    """
    Check if the Metal Performance Shaders (MPS) backend is available.

    Returns:
        bool: True if MPS is built and reports availability, False otherwise.
    """
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and torch.backends.mps.is_built() and torch.backends.mps.is_available())


def _smoke_test(device: torch.device) -> Tuple[bool, str]:
    """
    Run a small forward and backward pass to confirm device usability.

    Uses broadly supported float32 operations.

    Args:
        device (torch.device): The device to test.

    Returns:
        Tuple[bool, str]: (True, "") if usable; (False, error_message) otherwise.
    """
    try:
        torch.manual_seed(123)

        model = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        ).to(device)

        x = torch.randn(32, 8, device=device, dtype=torch.float32)
        y = torch.randint(0, 2, (32,), device=device,
                          dtype=torch.float32)  # 0/1

        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()

        opt.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)  # (B,)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def pick_device() -> torch.device:
    """
    Select the best available device (CUDA → MPS → CPU) with smoke testing.

    Returns:
        torch.device: A working device verified with a smoke test.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        ok, why = _smoke_test(dev)
        if ok:
            _set_precision_safe()
            return dev
        logger.warning("CUDA failed smoke test; falling back. Reason: %s", why)

    if _mps_usable():
        try:
            import os
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        except Exception:
            pass

        dev = torch.device("mps")
        ok, why = _smoke_test(dev)
        if ok:
            _set_precision_safe()
            return dev
        logger.warning("MPS failed smoke test; falling back. Reason: %s", why)
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()

    dev = torch.device("cpu")
    _set_precision_safe()
    return dev


def _set_precision_safe() -> None:
    """
    Attempt to set float32 matmul precision for better numerical stability.

    This is a no-op if the option is not supported.
    """
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )
