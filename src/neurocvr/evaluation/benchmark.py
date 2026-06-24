from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurocvr.cvr.glm import GlmResult, fit_glm_delay_search
from neurocvr.evaluation.metrics import RegressionMetrics, compute_regression_metrics
from neurocvr.preprocessing.bold import (
    apply_brain_mask,
    compute_global_baseline,
    compute_voxel_baseline,
    reshape_voxel_values_to_volume,
)
from neurocvr.simulation.synthetic import SyntheticCvrDataset, simulate_glm_cvr_dataset


@dataclass(frozen=True)
class SyntheticGlmBenchmarkResult:
    """End-to-end synthetic GLM benchmark result."""

    dataset: SyntheticCvrDataset
    glm_result: GlmResult
    estimated_cvr_map: np.ndarray
    estimated_delay_map: np.ndarray
    cvr_metrics: RegressionMetrics
    delay_metrics: RegressionMetrics


def run_synthetic_glm_benchmark(
    spatial_shape: tuple[int, int, int] = (4, 5, 3),
    n_timepoints: int = 220,
    tr_seconds: float = 1.55,
    tcnr: float | None = 5.0,
    seed: int = 42,
    delay_min_seconds: float = 0.0,
    delay_max_seconds: float = 8.0,
    delay_step_seconds: float = 1.0,
    n_baseline_volumes: int = 30,
) -> SyntheticGlmBenchmarkResult:
    """
    Run an end-to-end synthetic GLM CVR benchmark.

    Pipeline:
    synthetic BOLD generation -> GLM delay search -> map reconstruction -> metrics.
    """
    if delay_step_seconds <= 0:
        raise ValueError("delay_step_seconds must be positive.")

    dataset = simulate_glm_cvr_dataset(
        spatial_shape=spatial_shape,
        n_timepoints=n_timepoints,
        tr_seconds=tr_seconds,
        tcnr=tcnr,
        seed=seed,
    )

    bold_matrix = apply_brain_mask(
        bold_data=dataset.bold_4d,
        mask=dataset.mask,
    )

    n_baseline = min(n_baseline_volumes, bold_matrix.n_timepoints)
    voxel_baseline = compute_voxel_baseline(
        bold_matrix=bold_matrix.data,
        n_baseline_volumes=n_baseline,
    )
    global_baseline = compute_global_baseline(voxel_baseline)

    delay_candidates = np.arange(
        delay_min_seconds,
        delay_max_seconds + 0.5 * delay_step_seconds,
        delay_step_seconds,
    )

    glm_result = fit_glm_delay_search(
        bold_matrix=bold_matrix.data,
        etco2_regressor=dataset.etco2_regressor,
        time_seconds=dataset.time_seconds,
        bold_baseline=global_baseline,
        delay_candidates_seconds=delay_candidates,
    )

    estimated_cvr_map = reshape_voxel_values_to_volume(
        voxel_values=glm_result.cvr_magnitude,
        mask=dataset.mask,
    )
    estimated_delay_map = reshape_voxel_values_to_volume(
        voxel_values=glm_result.delay_seconds,
        mask=dataset.mask,
    )

    true_cvr = dataset.cvr_magnitude_map[dataset.mask]
    true_delay = dataset.delay_map[dataset.mask]

    cvr_metrics = compute_regression_metrics(
        y_true=true_cvr,
        y_pred=glm_result.cvr_magnitude,
    )
    delay_metrics = compute_regression_metrics(
        y_true=true_delay,
        y_pred=glm_result.delay_seconds,
    )

    return SyntheticGlmBenchmarkResult(
        dataset=dataset,
        glm_result=glm_result,
        estimated_cvr_map=estimated_cvr_map,
        estimated_delay_map=estimated_delay_map,
        cvr_metrics=cvr_metrics,
        delay_metrics=delay_metrics,
    )
