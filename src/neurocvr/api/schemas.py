from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SyntheticBenchmarkRequest(BaseModel):
    """Request body for a synthetic GLM benchmark run."""

    spatial_shape: tuple[int, int, int] = (4, 5, 3)
    n_timepoints: int = Field(default=220, gt=0)
    tr_seconds: float = Field(default=1.55, gt=0)
    tcnr: float | None = 5.0
    seed: int = 42
    delay_min_seconds: float = 0.0
    delay_max_seconds: float = 8.0
    delay_step_seconds: float = Field(default=1.0, gt=0)
    n_baseline_volumes: int = Field(default=30, gt=0)

    @field_validator("spatial_shape")
    @classmethod
    def validate_spatial_shape(
        cls,
        value: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        if len(value) != 3:
            raise ValueError("spatial_shape must have exactly three dimensions")

        if any(dim <= 0 for dim in value):
            raise ValueError("spatial_shape dimensions must be positive")

        return value

    @field_validator("tcnr")
    @classmethod
    def validate_tcnr(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("tcnr must be positive or null")

        return value


class MetricResponse(BaseModel):
    """Regression metric response."""

    rmse: float
    mae: float
    bias: float
    pcc: float
    n_voxels: int


class SyntheticBenchmarkResponse(BaseModel):
    """Response body for a synthetic GLM benchmark run."""

    run_name: str
    spatial_shape: tuple[int, int, int]
    n_timepoints: int
    tr_seconds: float
    tcnr: float | None
    seed: int
    cvr_magnitude: MetricResponse
    delay: MetricResponse


class QcReportResponse(BaseModel):
    """Technical QC report response."""

    status: str
    summary: str
    cvr_assessment: str
    delay_assessment: str
    recommendations: list[str]
    safety_notice: str


class SyntheticBenchmarkQcResponse(BaseModel):
    """Response body for benchmark metrics plus QC report."""

    benchmark: SyntheticBenchmarkResponse
    qc_report: QcReportResponse
