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

    @field_validator("dims")
    @classmethod
    def _validate_dims(cls, value: list[int]) -> list[int]:
        if len(value) < 2:
            raise ValueError(
                "dims must contain at least input and latent dimensions")
        if any(v <= 0 for v in value):
            raise ValueError("all dims entries must be positive")
        return value

    @field_validator("mad_threshold", "margin", "min_drift_ratio")
    @classmethod
    def _non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be non-negative")
        return value

    @field_validator("min_drift_count")
    @classmethod
    def _positive_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("min_drift_count must be >= 1")
        return value
