# src/core/config
"""Pydantic-based configuration for CE simulation and training.

This module defines enums for model type, model variant, and conformal
evaluation (CE) strategy, and provides a Pydantic v2 `SimulationConfig`
model that validates inputs, supports enum coercion, and offers
JSON-friendly serialization.

Key benefits over plain dataclasses:
- Type and value validation with clear errors.
- Enum coercion from strings (e.g., "binary" → `ModelType.BINARY`).
- Immutability via Pydantic's `frozen` config.
- Easy serialization with `model_dump()` and `model_dump_json()`.
"""
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import torch

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from src.core.models.torch_device import pick_device


class ModelType(str, Enum):
    """Classification task type."""

    BINARY = "binary"
    MULTI = "multi"


class ModelVariant(str, Enum):
    """Model architecture choices."""

    DT = "dt"
    KNN = "knn"
    RF = "rf"
    SVM = "svm"
    FEEDFORWARD = "feedforward"
    XGB = "xgb"


class CEType(str, Enum):
    """Conformal evaluation strategy."""

    CCE = "cce"
    APPROX_TCE = "approx_tce"
    ICE = "ice"
    NONE = "none"
    APPROX_CCE = "approx_cce"


class AdaptiveChunkConfig(BaseModel):
    """Configuration for adaptive chunking behavior."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    init_chunk_size: int = 10
    min_chunk_size: int = 1
    max_chunk_size: int = 1000
    ema_decay: float = 0.9
    cooldown_period: int = 0
    step_size: int = 5

    @field_validator(
        "init_chunk_size",
        "min_chunk_size",
        "max_chunk_size",
        "cooldown_period",
        "step_size",
    )
    @classmethod
    def _non_negative(cls, value: int) -> int:
        """Validate that integer parameters are non-negative."""
        if value < 0:
            raise ValueError("must be non-negative")
        return value

    @field_validator("ema_decay")
    @classmethod
    def _ema_in_unit_interval(cls, value: float) -> float:
        """Validate that `ema_decay` lies within the unit interval."""
        if not (0.0 <= value <= 1.0):
            raise ValueError("ema_decay must be in [0, 1]")
        return value

    @field_validator("max_chunk_size")
    @classmethod
    def _max_ge_min(cls, value: int, info) -> int:
        """Validate that `max_chunk_size` is not smaller than `min_chunk_size`."""
        min_value = info.data.get("min_chunk_size", 1)
        if value < min_value:
            raise ValueError("max_chunk_size must be >= min_chunk_size")
        return value

    @field_validator("init_chunk_size")
    @classmethod
    def _init_between_bounds(cls, value: int, info) -> int:
        """Validate that `init_chunk_size` lies within [min_chunk_size, max_chunk_size]."""
        min_value = info.data.get("min_chunk_size", 1)
        max_value = info.data.get("max_chunk_size", 1000)
        if not (min_value <= value <= max_value):
            raise ValueError(
                "init_chunk_size must be within [min_chunk_size, max_chunk_size]")
        return value


class SimulationConfig(BaseModel):
    """Configuration model for CE simulation and training runs."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    model_type: ModelType
    model_variant: ModelVariant
    ce_type: CEType
    aggregated_path: Path
    flows_path: Path

    adaptive_chunk_config: AdaptiveChunkConfig = Field(
        default_factory=AdaptiveChunkConfig)

    ce_kwargs: Dict[str, Any] = Field(default_factory=dict)

    threshold: float = 0.5
    is_unsw: bool = False
    chunk_size: int = 1000
    use_pca: bool = False
    use_ASC: bool = False
    log_path: Path = Path("ce_log.csv.gz")
    max_rows: int = 10000
    use_circular_logger: bool = False
    debug: bool = False
    log_to_file: bool = False
    use_svm: bool = False
    use_mlp: bool = False
    use_adaptive_chunking: bool = False
    use_cuml: bool = False

    device: torch.device = Field(default_factory=pick_device)

    @field_validator("threshold")
    @classmethod
    def _threshold_in_unit_interval(cls, value: float) -> float:
        """Validate that `threshold` lies within the unit interval."""
        if not (0.0 <= value <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        return value

    @field_validator("chunk_size", "max_rows")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        """Validate that integer sizing parameters are strictly positive."""
        if value <= 0:
            raise ValueError("must be > 0")
        return value

    @field_validator("aggregated_path", "flows_path")
    @classmethod
    def _file_must_exist(cls, path: Path) -> Path:
        """Validate that provided paths exist and refer to files."""
        if not path.exists():
            raise ValueError(f"path does not exist: {path}")
        if path.is_dir():
            raise ValueError(f"expected a file, got directory: {path}")
        return path

    @field_validator("ce_kwargs")
    @classmethod
    def _validate_ce_kwargs(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate common CE kwargs if present."""
        folds = params.get("folds", None)
        if folds is not None and int(folds) < 2:
            raise ValueError("ce_kwargs.folds must be >= 2")

        significance = params.get("significance", None)
        if significance is not None and not (0.0 < float(significance) < 1.0):
            raise ValueError("ce_kwargs.significance must be in (0, 1)")

        return params

    @field_serializer("device")
    def _serialize_device(self, value: torch.device) -> str:
        """Serialize `torch.device` as a simple string (e.g., 'cuda:0', 'cpu')."""
        return str(value)


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly.")
