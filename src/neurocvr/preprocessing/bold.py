from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoldMatrix:
    """Masked BOLD data represented as a time-by-voxel matrix."""

    data: np.ndarray
    mask: np.ndarray
    spatial_shape: tuple[int, int, int]

    @property
    def n_timepoints(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_voxels(self) -> int:
        return int(self.data.shape[1])


def validate_bold_4d(bold_data: np.ndarray) -> None:
    """Validate that BOLD data are 4D: x, y, z, time."""
    if bold_data.ndim != 4:
        raise ValueError(
            f"Expected 4D BOLD data with shape (x, y, z, time), "
            f"got shape {bold_data.shape}."
        )


def validate_mask_3d(mask: np.ndarray, spatial_shape: tuple[int, int, int]) -> None:
    """Validate that a brain mask is 3D and matches BOLD spatial dimensions."""
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}.")

    if mask.shape != spatial_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match BOLD spatial shape "
            f"{spatial_shape}."
        )


def create_full_brain_mask(spatial_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Create a simple all-True mask.

    This is useful for tests and demos. Real analysis should use a proper brain mask.
    """
    return np.ones(spatial_shape, dtype=bool)


def apply_brain_mask(bold_data: np.ndarray, mask: np.ndarray) -> BoldMatrix:
    """
    Convert 4D BOLD data into a masked time-by-voxel matrix.

    Input shape:
        bold_data: (x, y, z, time)
        mask:      (x, y, z)

    Output shape:
        matrix:    (time, voxels)
    """
    validate_bold_4d(bold_data)

    spatial_shape = bold_data.shape[:3]
    validate_mask_3d(mask, spatial_shape)

    mask_bool = mask.astype(bool)

    if not np.any(mask_bool):
        raise ValueError("Brain mask contains no voxels.")

    voxel_by_time = bold_data[mask_bool, :]
    time_by_voxel = voxel_by_time.T

    return BoldMatrix(
        data=np.asarray(time_by_voxel, dtype=float),
        mask=mask_bool,
        spatial_shape=spatial_shape,
    )


def compute_voxel_baseline(
    bold_matrix: np.ndarray,
    n_baseline_volumes: int = 30,
) -> np.ndarray:
    """
    Compute voxelwise baseline BOLD signal.

    The baseline is the mean over the first n_baseline_volumes.
    """
    if bold_matrix.ndim != 2:
        raise ValueError("bold_matrix must have shape (time, voxels).")

    if n_baseline_volumes <= 0:
        raise ValueError("n_baseline_volumes must be positive.")

    if n_baseline_volumes > bold_matrix.shape[0]:
        raise ValueError(
            "n_baseline_volumes cannot exceed the number of time points."
        )

    return np.nanmean(bold_matrix[:n_baseline_volumes, :], axis=0)


def compute_global_baseline(voxel_baseline: np.ndarray) -> float:
    """Compute global BOLD baseline from voxelwise baselines."""
    values = np.asarray(voxel_baseline, dtype=float)

    if values.ndim != 1:
        raise ValueError("voxel_baseline must be a 1D array.")

    return float(np.nanmean(values))


def compute_percent_signal_change(
    bold_matrix: np.ndarray,
    voxel_baseline: np.ndarray,
) -> np.ndarray:
    """
    Convert BOLD signal to percentage signal change.

    PSC = (S(t) - S_baseline) / S_baseline * 100
    """
    if bold_matrix.ndim != 2:
        raise ValueError("bold_matrix must have shape (time, voxels).")

    baseline = np.asarray(voxel_baseline, dtype=float)

    if baseline.ndim != 1:
        raise ValueError("voxel_baseline must be a 1D array.")

    if baseline.shape[0] != bold_matrix.shape[1]:
        raise ValueError(
            "voxel_baseline length must match the number of voxels."
        )

    safe_baseline = np.where(baseline == 0, np.nan, baseline)

    return ((bold_matrix - safe_baseline) / safe_baseline) * 100.0


def reshape_voxel_values_to_volume(
    voxel_values: np.ndarray,
    mask: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Put voxelwise values back into a 3D volume using a mask.
    """
    values = np.asarray(voxel_values, dtype=float)
    mask_bool = mask.astype(bool)

    if values.ndim != 1:
        raise ValueError("voxel_values must be a 1D array.")

    if values.shape[0] != int(mask_bool.sum()):
        raise ValueError(
            "Number of voxel values must match number of voxels in mask."
        )

    volume = np.full(mask_bool.shape, fill_value, dtype=float)
    volume[mask_bool] = values

    return volume