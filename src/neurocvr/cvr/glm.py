from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GlmResult:
    """Voxelwise GLM CVR estimation result."""

    intercept: np.ndarray
    etco2_beta: np.ndarray
    drift_beta: np.ndarray
    cvr_magnitude: np.ndarray
    delay_seconds: np.ndarray
    ssr: np.ndarray

    @property
    def n_voxels(self) -> int:
        return int(self.cvr_magnitude.shape[0])


def validate_time_by_voxel_matrix(bold_matrix: np.ndarray) -> None:
    """Validate BOLD matrix shape: time by voxels."""
    if bold_matrix.ndim != 2:
        raise ValueError("bold_matrix must have shape (time, voxels).")


def validate_regressor(regressor: np.ndarray, n_timepoints: int) -> None:
    """Validate regressor shape."""
    if regressor.ndim != 1:
        raise ValueError("regressor must be a 1D array.")

    if regressor.shape[0] != n_timepoints:
        raise ValueError(
            "regressor length must match the number of BOLD time points."
        )


def make_design_matrix(
    etco2_regressor: np.ndarray,
    time_seconds: np.ndarray,
) -> np.ndarray:
    """
    Build GLM design matrix.

    Columns:
    1. intercept
    2. baseline-corrected ETCO2 regressor
    3. linear drift term
    """
    regressor = np.asarray(etco2_regressor, dtype=float)
    time = np.asarray(time_seconds, dtype=float)

    if regressor.ndim != 1:
        raise ValueError("etco2_regressor must be 1D.")

    if time.ndim != 1:
        raise ValueError("time_seconds must be 1D.")

    if regressor.shape != time.shape:
        raise ValueError("etco2_regressor and time_seconds must have the same shape.")

    centred_time = time - np.nanmean(time)

    return np.column_stack(
        [
            np.ones_like(regressor),
            regressor,
            centred_time,
        ]
    )


def fit_ols(
    bold_matrix: np.ndarray,
    design_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit ordinary least squares for all voxels at once.

    Parameters
    ----------
    bold_matrix:
        Shape (time, voxels).
    design_matrix:
        Shape (time, regressors).

    Returns
    -------
    betas:
        Shape (regressors, voxels).
    ssr:
        Sum of squared residuals per voxel, shape (voxels,).
    """
    y = np.asarray(bold_matrix, dtype=float)
    x = np.asarray(design_matrix, dtype=float)

    validate_time_by_voxel_matrix(y)

    if x.ndim != 2:
        raise ValueError("design_matrix must be 2D.")

    if x.shape[0] != y.shape[0]:
        raise ValueError("design_matrix and bold_matrix must have same time length.")

    betas = np.linalg.pinv(x) @ y
    predictions = x @ betas
    residuals = y - predictions
    ssr = np.sum(residuals**2, axis=0)

    return betas, ssr


def compute_cvr_magnitude(
    etco2_beta: np.ndarray,
    bold_baseline: float,
) -> np.ndarray:
    """
    Compute CVR magnitude in %BOLD/mmHg.

    CVR = beta_ETCO2 / BOLD_baseline * 100
    """
    if bold_baseline == 0:
        raise ValueError("bold_baseline must be non-zero.")

    beta = np.asarray(etco2_beta, dtype=float)
    return (beta / bold_baseline) * 100.0


def shift_regressor_by_delay(
    time_seconds: np.ndarray,
    regressor: np.ndarray,
    delay_seconds: float,
) -> np.ndarray:
    """
    Shift ETCO2 regressor for a candidate CVR delay.

    Positive delay means:
        BOLD(t) is fitted using ETCO2(t - delay)
    """
    time = np.asarray(time_seconds, dtype=float)
    values = np.asarray(regressor, dtype=float)

    validate_regressor(values, n_timepoints=time.shape[0])

    shifted_query_time = time - delay_seconds

    return np.interp(
        shifted_query_time,
        time,
        values,
        left=values[0],
        right=values[-1],
    )


def fit_glm_fixed_delay(
    bold_matrix: np.ndarray,
    etco2_regressor: np.ndarray,
    time_seconds: np.ndarray,
    bold_baseline: float,
    delay_seconds: float = 0.0,
) -> GlmResult:
    """
    Fit voxelwise GLM at one fixed delay.
    """
    y = np.asarray(bold_matrix, dtype=float)
    regressor = np.asarray(etco2_regressor, dtype=float)
    time = np.asarray(time_seconds, dtype=float)

    validate_time_by_voxel_matrix(y)
    validate_regressor(regressor, n_timepoints=y.shape[0])

    shifted_regressor = shift_regressor_by_delay(
        time_seconds=time,
        regressor=regressor,
        delay_seconds=delay_seconds,
    )

    design = make_design_matrix(
        etco2_regressor=shifted_regressor,
        time_seconds=time,
    )

    betas, ssr = fit_ols(
        bold_matrix=y,
        design_matrix=design,
    )

    cvr_magnitude = compute_cvr_magnitude(
        etco2_beta=betas[1, :],
        bold_baseline=bold_baseline,
    )

    delay = np.full(y.shape[1], delay_seconds, dtype=float)

    return GlmResult(
        intercept=betas[0, :],
        etco2_beta=betas[1, :],
        drift_beta=betas[2, :],
        cvr_magnitude=cvr_magnitude,
        delay_seconds=delay,
        ssr=ssr,
    )


def fit_glm_delay_search(
    bold_matrix: np.ndarray,
    etco2_regressor: np.ndarray,
    time_seconds: np.ndarray,
    bold_baseline: float,
    delay_candidates_seconds: np.ndarray,
) -> GlmResult:
    """
    Fit voxelwise GLM across candidate delays and keep the best delay per voxel.

    Best delay is selected by minimum SSR.
    """
    y = np.asarray(bold_matrix, dtype=float)
    candidates = np.asarray(delay_candidates_seconds, dtype=float)

    validate_time_by_voxel_matrix(y)

    if candidates.ndim != 1:
        raise ValueError("delay_candidates_seconds must be a 1D array.")

    if candidates.shape[0] == 0:
        raise ValueError("At least one delay candidate is required.")

    best_ssr = np.full(y.shape[1], np.inf, dtype=float)
    best_intercept = np.zeros(y.shape[1], dtype=float)
    best_etco2_beta = np.zeros(y.shape[1], dtype=float)
    best_drift_beta = np.zeros(y.shape[1], dtype=float)
    best_delay = np.zeros(y.shape[1], dtype=float)

    for delay in candidates:
        result = fit_glm_fixed_delay(
            bold_matrix=y,
            etco2_regressor=etco2_regressor,
            time_seconds=time_seconds,
            bold_baseline=bold_baseline,
            delay_seconds=float(delay),
        )

        improved = result.ssr < best_ssr

        best_ssr[improved] = result.ssr[improved]
        best_intercept[improved] = result.intercept[improved]
        best_etco2_beta[improved] = result.etco2_beta[improved]
        best_drift_beta[improved] = result.drift_beta[improved]
        best_delay[improved] = delay

    cvr_magnitude = compute_cvr_magnitude(
        etco2_beta=best_etco2_beta,
        bold_baseline=bold_baseline,
    )

    return GlmResult(
        intercept=best_intercept,
        etco2_beta=best_etco2_beta,
        drift_beta=best_drift_beta,
        cvr_magnitude=cvr_magnitude,
        delay_seconds=best_delay,
        ssr=best_ssr,
    )