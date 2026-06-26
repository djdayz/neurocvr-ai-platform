from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class SyntheticBenchmarkConfig(BaseModel):
    """Configuration for a synthetic GLM CVR benchmark run."""

    spatial_shape: tuple[int, int, int] = (4, 5, 3)
    n_timepoints: int = 220
    tr_seconds: float = 1.55
    tcnr: float | None = 5.0
    seed: int = 42
    delay_min_seconds: float = 0.0
    delay_max_seconds: float = 8.0
    delay_step_seconds: float = 1.0
    n_baseline_volumes: int = 30
    output_dir: Path = Field(default=Path("outputs/benchmark"))

    @field_validator("n_timepoints", "n_baseline_volumes")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("tr_seconds", "delay_step_seconds")
    @classmethod
    def positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("spatial_shape")
    @classmethod
    def validate_spatial_shape(
        cls,
        value: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        if len(value) != 3:
            raise ValueError("spatial_shape must contain exactly three dimensions")

        if any(dim <= 0 for dim in value):
            raise ValueError("all spatial_shape dimensions must be positive")

        return value


def load_benchmark_config(path: str | Path) -> SyntheticBenchmarkConfig:
    """Load a synthetic benchmark config from a YAML file."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        raw_config: dict[str, Any] = yaml.safe_load(f) or {}

    return SyntheticBenchmarkConfig(**raw_config)


def save_default_benchmark_config(path: str | Path) -> Path:
    """Save a default benchmark config to YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = SyntheticBenchmarkConfig()
    data = config.model_dump(mode="json")

    with output_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return output_path
