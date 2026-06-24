from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurocvr.cvr.glm import shift_regressor_by_delay
from neurocvr.preprocessing.bold import reshape_voxel_values_to_volume


@dataclass(frozen=True)
class SyntheticCvrDataset:
    """Synthetic BOLD-CVR dataset with ground-truth parameter maps."""

    bold_4d: np.ndarray
    cvr_magnitude_map: np.ndarray
    delay_map: np.ndarray
    etco2_regressor: np.ndarray
    time_seconds: np.ndarray
    mask: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.bold_4d.shape


def make_block_etco2_regressor(
    n_timepoints: int,
    tr_seconds: float,
    baseline_mmhg: float = 40.0,
    hypercapnia_mmhg: float = 50.0,
    normocapnia_duration_seconds: float = 120.0,
    hypercapnia_duration_seconds: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a baseline-corrected ETCO2 block-paradigm regressor.

    Returns
    -------
    time_seconds:
        BOLD time axis.
    etco2_regressor:
        Baseline-corrected ETCO2 in mmHg.
    """
    if n_timepoints <= 0:
        raise ValueError("n_timepoints must be positive.")

    if tr_seconds <= 0:
        raise ValueError("tr_seconds must be positive.")

    if hypercapnia_mmhg <= baseline_mmhg:
        raise ValueError("hypercapnia_mmhg must be greater than baseline_mmhg.")

    if normocapnia_duration_seconds <= 0 or hypercapnia_duration_seconds <= 0:
        raise ValueError("Block durations must be positive.")

    time_seconds = np.arange(n_timepoints, dtype=float) * tr_seconds
    cycle_duration = normocapnia_duration_seconds + hypercapnia_duration_seconds
    cycle_position = np.mod(time_seconds, cycle_duration)

    etco2_mmhg = np.where(
        cycle_position < normocapnia_duration_seconds,
        baseline_mmhg,
        hypercapnia_mmhg,
    )

    etco2_regressor = etco2_mmhg - baseline_mmhg

    return time_seconds, etco2_regressor


def sample_ground_truth_maps(
    spatial_shape: tuple[int, int, int],
    cvr_range: tuple[float, float] = (0.2, 1.5),
    delay_range_seconds: tuple[float, float] = (0.0, 8.0),
    seed: int = 42,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample synthetic ground-truth CVR magnitude and delay maps.

    CVR magnitude unit:
        %BOLD/mmHg

    Delay unit:
        seconds
    """
    rng = np.random.default_rng(seed)

    if mask is None:
        mask = np.ones(spatial_shape, dtype=bool)
    else:
        mask = mask.astype(bool)

    if mask.shape != spatial_shape:
        raise ValueError("mask shape must match spatial_shape.")

    if not np.any(mask):
        raise ValueError("mask contains no voxels.")

    cvr_low, cvr_high = cvr_range
    delay_low, delay_high = delay_range_seconds

    if cvr_high <= cvr_low:
        raise ValueError("cvr_range must be increasing.")

    if delay_high < delay_low:
        raise ValueError("delay_range_seconds must be increasing.")

    n_voxels = int(mask.sum())

    cvr_values = rng.uniform(cvr_low, cvr_high, size=n_voxels)
    delay_values = rng.uniform(delay_low, delay_high, size=n_voxels)

    cvr_map = np.full(spatial_shape, np.nan, dtype=float)
    delay_map = np.full(spatial_shape, np.nan, dtype=float)

    cvr_map[mask] = cvr_values
    delay_map[mask] = delay_values

    return cvr_map, delay_map, mask


def simulate_glm_bold_matrix(
    cvr_magnitude: np.ndarray,
    delay_seconds: np.ndarray,
    etco2_regressor: np.ndarray,
    time_seconds: np.ndarray,
    bold_baseline: float = 100.0,
) -> np.ndarray:
    """
    Simulate BOLD time series using the GLM-style forward model.

    BOLD(t) = baseline + beta_ETCO2 * ETCO2(t - delay)

    where:
        beta_ETCO2 = CVR / 100 * baseline
    """
    cvr = np.asarray(cvr_magnitude, dtype=float).ravel()
    delays = np.asarray(delay_seconds, dtype=float).ravel()
    regressor = np.asarray(etco2_regressor, dtype=float)
    time = np.asarray(time_seconds, dtype=float)

    if cvr.shape != delays.shape:
        raise ValueError("cvr_magnitude and delay_seconds must have the same shape.")

    if regressor.shape != time.shape:
        raise ValueError("etco2_regressor and time_seconds must have the same shape.")

    if bold_baseline <= 0:
        raise ValueError("bold_baseline must be positive.")

    bold_matrix = np.empty((time.shape[0], cvr.shape[0]), dtype=float)

    for voxel_index, delay in enumerate(delays):
        shifted_etco2 = shift_regressor_by_delay(
            time_seconds=time,
            regressor=regressor,
            delay_seconds=float(delay),
        )
        beta = cvr[voxel_index] / 100.0 * bold_baseline
        bold_matrix[:, voxel_index] = bold_baseline + beta * shifted_etco2

    return bold_matrix


def add_tcnr_noise(
    bold_matrix: np.ndarray,
    tcnr: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Add voxelwise Gaussian noise controlled by approximate tCNR.

    sigma = signal_range / tCNR
    """
    if tcnr <= 0:
        raise ValueError("tcnr must be positive.")

    rng = np.random.default_rng(seed)
    matrix = np.asarray(bold_matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("bold_matrix must have shape (time, voxels).")

    signal_range = np.nanmax(matrix, axis=0) - np.nanmin(matrix, axis=0)
    sigma = signal_range / tcnr
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)

    noise = rng.normal(loc=0.0, scale=sigma[None, :], size=matrix.shape)

    return matrix + noise


def simulate_glm_cvr_dataset(
    spatial_shape: tuple[int, int, int] = (4, 5, 3),
    n_timepoints: int = 220,
    tr_seconds: float = 1.55,
    tcnr: float | None = 5.0,
    seed: int = 42,
    bold_baseline: float = 100.0,
) -> SyntheticCvrDataset:
    """
    Generate a small synthetic BOLD-CVR dataset.

    This is intentionally lightweight so it can run quickly in tests and demos.
    """
    cvr_map, delay_map, mask = sample_ground_truth_maps(
        spatial_shape=spatial_shape,
        seed=seed,
    )

    time_seconds, etco2_regressor = make_block_etco2_regressor(
        n_timepoints=n_timepoints,
        tr_seconds=tr_seconds,
    )

    bold_matrix = simulate_glm_bold_matrix(
        cvr_magnitude=cvr_map[mask],
        delay_seconds=delay_map[mask],
        etco2_regressor=etco2_regressor,
        time_seconds=time_seconds,
        bold_baseline=bold_baseline,
    )

    if tcnr is not None:
        bold_matrix = add_tcnr_noise(
            bold_matrix=bold_matrix,
            tcnr=tcnr,
            seed=seed,
        )

    bold_4d = np.zeros((*spatial_shape, n_timepoints), dtype=float)

    for time_index in range(n_timepoints):
        bold_4d[..., time_index] = reshape_voxel_values_to_volume(
            voxel_values=bold_matrix[time_index, :],
            mask=mask,
        )

    return SyntheticCvrDataset(
        bold_4d=bold_4d,
        cvr_magnitude_map=cvr_map,
        delay_map=delay_map,
        etco2_regressor=etco2_regressor,
        time_seconds=time_seconds,
        mask=mask,
    )
