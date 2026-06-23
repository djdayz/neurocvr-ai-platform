import numpy as np
import pytest

from neurocvr.cvr.glm import (
    compute_cvr_magnitude,
    fit_glm_delay_search,
    fit_glm_fixed_delay,
    make_design_matrix,
    shift_regressor_by_delay,
)


def test_make_design_matrix() -> None:
    regressor = np.array([0.0, 1.0, 2.0])
    time = np.array([0.0, 1.0, 2.0])

    design = make_design_matrix(regressor, time)

    assert design.shape == (3, 3)
    np.testing.assert_allclose(design[:, 0], np.ones(3))
    np.testing.assert_allclose(design[:, 1], regressor)


def test_compute_cvr_magnitude() -> None:
    beta = np.array([0.5, 1.0])

    cvr = compute_cvr_magnitude(beta, bold_baseline=100.0)

    np.testing.assert_allclose(cvr, np.array([0.5, 1.0]))


def test_compute_cvr_magnitude_rejects_zero_baseline() -> None:
    with pytest.raises(ValueError):
        compute_cvr_magnitude(np.array([0.5]), bold_baseline=0.0)


def test_shift_regressor_by_delay_zero_delay() -> None:
    time = np.array([0.0, 1.0, 2.0])
    regressor = np.array([0.0, 10.0, 20.0])

    shifted = shift_regressor_by_delay(time, regressor, delay_seconds=0.0)

    np.testing.assert_allclose(shifted, regressor)


def test_fit_glm_fixed_delay_recovers_cvr() -> None:
    time = np.arange(6, dtype=float)
    etco2 = np.array([0.0, 0.0, 5.0, 10.0, 10.0, 10.0])
    bold_baseline = 100.0

    true_cvr = np.array([0.5, 1.0])
    true_beta = true_cvr / 100.0 * bold_baseline

    bold = bold_baseline + np.outer(etco2, true_beta)

    result = fit_glm_fixed_delay(
        bold_matrix=bold,
        etco2_regressor=etco2,
        time_seconds=time,
        bold_baseline=bold_baseline,
        delay_seconds=0.0,
    )

    np.testing.assert_allclose(result.cvr_magnitude, true_cvr, atol=1e-10)
    np.testing.assert_allclose(result.delay_seconds, np.array([0.0, 0.0]))
    assert result.n_voxels == 2


def test_fit_glm_delay_search_recovers_known_delay() -> None:
    time = np.arange(10, dtype=float)
    etco2 = np.array([0.0, 0.0, 0.0, 5.0, 10.0, 10.0, 5.0, 0.0, 0.0, 0.0])
    bold_baseline = 100.0
    true_delay = 2.0
    true_cvr = np.array([0.8])

    delayed_etco2 = shift_regressor_by_delay(
        time_seconds=time,
        regressor=etco2,
        delay_seconds=true_delay,
    )

    true_beta = true_cvr / 100.0 * bold_baseline
    bold = bold_baseline + np.outer(delayed_etco2, true_beta)

    result = fit_glm_delay_search(
        bold_matrix=bold,
        etco2_regressor=etco2,
        time_seconds=time,
        bold_baseline=bold_baseline,
        delay_candidates_seconds=np.array([0.0, 1.0, 2.0, 3.0]),
    )

    np.testing.assert_allclose(result.cvr_magnitude, true_cvr, atol=1e-10)
    np.testing.assert_allclose(result.delay_seconds, np.array([2.0]))
