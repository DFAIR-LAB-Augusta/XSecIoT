from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CadeMonitorConfig(BaseModel):
    """Typed configuration for the CADE drift monitor."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    dims: list[int] = Field(..., min_length=2)
    margin: float = 10.0
    mad_threshold: float = 3.5
    min_drift_ratio: float = 0.05
    min_drift_count: int = 1

    batch_size: int = 64
    epochs: int = 250
    lr: float = 1e-3
    cae_lambda_1: float = 1e-1
    similar_ratio: float = 0.25
    display_interval: int = 10
    force_retrain: bool = False
    weights_path: str | None = None
    device: str = "/CPU:0"

    @field_validator("dims")
    @classmethod
    def _validate_dims(cls, value: list[int]) -> list[int]:
        if len(value) < 2:
            raise ValueError(
                "dims must contain at least input and latent dimensions"
            )
        if any(v <= 0 for v in value):
            raise ValueError("all dims entries must be positive")
        return value

    @field_validator("mad_threshold", "margin", "min_drift_ratio", "lr", "cae_lambda_1", "similar_ratio")
    @classmethod
    def _non_negative_float(cls, value: float, info) -> float:
        if value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value

    @field_validator("min_drift_ratio", "similar_ratio")
    @classmethod
    def _ratio_in_unit_interval(cls, value: float, info) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{info.field_name} must be in [0, 1]")
        return value

    @field_validator("min_drift_count", "batch_size", "epochs", "display_interval")
    @classmethod
    def _positive_int(cls, value: int, info) -> int:
        if value < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return value

    @field_validator("batch_size")
    @classmethod
    def _batch_size_multiple_of_4(cls, value: int) -> int:
        if value < 4 or value % 4 != 0:
            raise ValueError("batch_size must be a multiple of 4 and >= 4")
        return value
