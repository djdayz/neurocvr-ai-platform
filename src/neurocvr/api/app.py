from __future__ import annotations

from fastapi import FastAPI

from neurocvr.api.schemas import (
    MetricResponse,
    QcReportResponse,
    SyntheticBenchmarkQcResponse,
    SyntheticBenchmarkRequest,
    SyntheticBenchmarkResponse,
)
from neurocvr.evaluation.benchmark import run_synthetic_glm_benchmark
from neurocvr.reporting.qc_report import generate_qc_report

app = FastAPI(
    title="NeuroCVR-AI API",
    version="0.1.0",
    description=(
        "Research API for synthetic BOLD-fMRI cerebrovascular reactivity "
        "benchmarking. Not for clinical use."
    ),
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health status."""
    return {
        "status": "ok",
        "service": "neurocvr-ai",
    }


@app.post(
    "/benchmarks/synthetic-glm",
    response_model=SyntheticBenchmarkResponse,
)
def run_synthetic_glm_endpoint(
    request: SyntheticBenchmarkRequest,
) -> SyntheticBenchmarkResponse:
    """Run a synthetic GLM CVR benchmark and return evaluation metrics."""
    result = run_synthetic_glm_benchmark(
        spatial_shape=request.spatial_shape,
        n_timepoints=request.n_timepoints,
        tr_seconds=request.tr_seconds,
        tcnr=request.tcnr,
        seed=request.seed,
        delay_min_seconds=request.delay_min_seconds,
        delay_max_seconds=request.delay_max_seconds,
        delay_step_seconds=request.delay_step_seconds,
        n_baseline_volumes=request.n_baseline_volumes,
    )

    return SyntheticBenchmarkResponse(
        run_name="synthetic_glm_benchmark_api",
        spatial_shape=request.spatial_shape,
        n_timepoints=request.n_timepoints,
        tr_seconds=request.tr_seconds,
        tcnr=request.tcnr,
        seed=request.seed,
        cvr_magnitude=MetricResponse(
            rmse=result.cvr_metrics.rmse,
            mae=result.cvr_metrics.mae,
            bias=result.cvr_metrics.bias,
            pcc=result.cvr_metrics.pcc,
            n_voxels=result.cvr_metrics.n_voxels,
        ),
        delay=MetricResponse(
            rmse=result.delay_metrics.rmse,
            mae=result.delay_metrics.mae,
            bias=result.delay_metrics.bias,
            pcc=result.delay_metrics.pcc,
            n_voxels=result.delay_metrics.n_voxels,
        ),
    )


@app.post(
    "/reports/synthetic-glm-qc",
    response_model=SyntheticBenchmarkQcResponse,
)
def run_synthetic_glm_qc_report_endpoint(
    request: SyntheticBenchmarkRequest,
) -> SyntheticBenchmarkQcResponse:
    """Run a synthetic GLM benchmark and return metrics plus technical QC report."""
    result = run_synthetic_glm_benchmark(
        spatial_shape=request.spatial_shape,
        n_timepoints=request.n_timepoints,
        tr_seconds=request.tr_seconds,
        tcnr=request.tcnr,
        seed=request.seed,
        delay_min_seconds=request.delay_min_seconds,
        delay_max_seconds=request.delay_max_seconds,
        delay_step_seconds=request.delay_step_seconds,
        n_baseline_volumes=request.n_baseline_volumes,
    )

    benchmark_response = SyntheticBenchmarkResponse(
        run_name="synthetic_glm_benchmark_qc_api",
        spatial_shape=request.spatial_shape,
        n_timepoints=request.n_timepoints,
        tr_seconds=request.tr_seconds,
        tcnr=request.tcnr,
        seed=request.seed,
        cvr_magnitude=MetricResponse(
            rmse=result.cvr_metrics.rmse,
            mae=result.cvr_metrics.mae,
            bias=result.cvr_metrics.bias,
            pcc=result.cvr_metrics.pcc,
            n_voxels=result.cvr_metrics.n_voxels,
        ),
        delay=MetricResponse(
            rmse=result.delay_metrics.rmse,
            mae=result.delay_metrics.mae,
            bias=result.delay_metrics.bias,
            pcc=result.delay_metrics.pcc,
            n_voxels=result.delay_metrics.n_voxels,
        ),
    )

    report = generate_qc_report(
        cvr_rmse=result.cvr_metrics.rmse,
        cvr_mae=result.cvr_metrics.mae,
        cvr_bias=result.cvr_metrics.bias,
        cvr_pcc=result.cvr_metrics.pcc,
        delay_rmse=result.delay_metrics.rmse,
        delay_mae=result.delay_metrics.mae,
        delay_bias=result.delay_metrics.bias,
        delay_pcc=result.delay_metrics.pcc,
        n_voxels=result.cvr_metrics.n_voxels,
    )

    return SyntheticBenchmarkQcResponse(
        benchmark=benchmark_response,
        qc_report=QcReportResponse(
            status=report.status,
            summary=report.summary,
            cvr_assessment=report.cvr_assessment,
            delay_assessment=report.delay_assessment,
            recommendations=report.recommendations,
            safety_notice=report.safety_notice,
        ),
    )