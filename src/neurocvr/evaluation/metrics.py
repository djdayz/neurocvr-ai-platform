from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    """Common voxelwise map comparison metrics."""

    rmse: float
    mae: float
    bias: float
    pcc: float
    n_voxels: int


def finite_pair_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return mask where both arrays have finite values."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    if true.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {true.shape}, "
            f"but y_pred has shape {pred.shape}."
        )

    return np.isfinite(true) & np.isfinite(pred)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    mask = finite_pair_mask(y_true, y_pred)

    if not np.any(mask):
        raise ValueError(
            "No finite paired values available for RMSE."
            )

    error = np.asarray(
        y_pred, dtype=float
        )[mask] - np.asarray(
            y_true, dtype=float
            )[mask]

    return float(np.sqrt(np.mean(error**2)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    mask = finite_pair_mask(y_true, y_pred)

    if not np.any(mask):
        raise ValueError(
            "No finite paired values available for MAE."
            )

    error = np.asarray(
        y_pred, dtype=float
        )[mask] - np.asarray(
            y_true, dtype=float
            )[mask]

    return float(np.mean(np.abs(error)))


def mean_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean prediction bias.

    bias = mean(y_pred - y_true)
    """
    mask = finite_pair_mask(y_true, y_pred)

    if not np.any(mask):
        raise ValueError("No finite paired values available for bias.")

    error = np.asarray(
        y_pred, dtype=float
        )[mask] - np.asarray(
            y_true, dtype=float
            )[mask]

    return float(np.mean(error))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    mask = finite_pair_mask(y_true, y_pred)

    if int(mask.sum()) < 2:
        raise ValueError("At least two finite paired values are required for PCC.")

    true = np.asarray(y_true, dtype=float)[mask]
    pred = np.asarray(y_pred, dtype=float)[mask]

    if np.nanstd(true) == 0 or np.nanstd(pred) == 0:
        return float("nan")

    return float(np.corrcoef(true, pred)[0, 1])


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """Compute RMSE, MAE, bias, PCC, and number of valid voxels."""
    mask = finite_pair_mask(y_true, y_pred)

    if not np.any(mask):
        raise ValueError(
            "No finite paired values available for metric calculation."
            )

    return RegressionMetrics(
        rmse=root_mean_squared_error(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        bias=mean_bias(y_true, y_pred),
        pcc=pearson_correlation(y_true, y_pred),
        n_voxels=int(mask.sum()),
    )
