import math

import pytest

from neurocvr.evaluation.benchmark import run_synthetic_glm_benchmark


def test_run_synthetic_glm_benchmark_shapes() -> None:
    result = run_synthetic_glm_benchmark(
        spatial_shape=(2, 2, 1),
        n_timepoints=220,
        tr_seconds=1.55,
        tcnr=None,
        seed=42,
        delay_step_seconds=2.0,
    )

    assert result.dataset.bold_4d.shape == (2, 2, 1, 220)
    assert result.estimated_cvr_map.shape == (2, 2, 1)
    assert result.estimated_delay_map.shape == (2, 2, 1)
    assert result.cvr_metrics.n_voxels == 4
    assert result.delay_metrics.n_voxels == 4
    assert math.isfinite(result.cvr_metrics.rmse)
    assert math.isfinite(result.delay_metrics.rmse)


def test_run_synthetic_glm_benchmark_rejects_bad_delay_step() -> None:
    with pytest.raises(ValueError):
        run_synthetic_glm_benchmark(delay_step_seconds=0.0)
