import math

import numpy as np
import pytest

from neurocvr.evaluation.metrics import (
    compute_regression_metrics,
    finite_pair_mask,
    mean_absolute_error,
    mean_bias,
    pearson_correlation,
    root_mean_squared_error,
)


def test_finite_pair_mask() -> None:
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, np.inf, 5.0])

    mask = finite_pair_mask(y_true, y_pred)

    np.testing.assert_array_equal(mask, np.array([True, False, False, True]))


def test_finite_pair_mask_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        finite_pair_mask(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))


def test_root_mean_squared_error() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 4.0, 3.0])

    rmse = root_mean_squared_error(y_true, y_pred)

    assert rmse == pytest.approx(math.sqrt(4.0 / 3.0))


def test_mean_absolute_error() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 6.0])

    mae = mean_absolute_error(y_true, y_pred)

    assert mae == pytest.approx(4.0 / 3.0)


def test_mean_bias() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])

    bias = mean_bias(y_true, y_pred)

    assert bias == pytest.approx(1.0)


def test_pearson_correlation() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 4.0, 6.0])

    pcc = pearson_correlation(y_true, y_pred)

    assert pcc == pytest.approx(1.0)


def test_pearson_correlation_returns_nan_for_constant_array() -> None:
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([2.0, 3.0, 4.0])

    pcc = pearson_correlation(y_true, y_pred)

    assert math.isnan(pcc)


def test_compute_regression_metrics() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 4.0, 3.0])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics.n_voxels == 3
    assert metrics.rmse == pytest.approx(math.sqrt(4.0 / 3.0))
    assert metrics.mae == pytest.approx(2.0 / 3.0)
    assert metrics.bias == pytest.approx(2.0 / 3.0)
