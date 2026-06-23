import numpy as np
import pytest

from neurocvr.preprocessing.bold import (
    apply_brain_mask,
    compute_global_baseline,
    compute_percent_signal_change,
    compute_voxel_baseline,
    create_full_brain_mask,
    reshape_voxel_values_to_volume,
    validate_bold_4d,
)


def test_validate_bold_4d_rejects_3d() -> None:
    data = np.zeros((2, 3, 4))

    with pytest.raises(ValueError):
        validate_bold_4d(data)


def test_create_full_brain_mask() -> None:
    mask = create_full_brain_mask((2, 3, 4))

    assert mask.shape == (2, 3, 4)
    assert mask.dtype == bool
    assert mask.sum() == 24


def test_apply_brain_mask_returns_time_by_voxel_matrix() -> None:
    bold = np.zeros((2, 2, 1, 3), dtype=float)
    bold[..., 0] = 10.0
    bold[..., 1] = 20.0
    bold[..., 2] = 30.0

    mask = np.array([[[True], [False]], [[True], [False]]])

    matrix = apply_brain_mask(bold, mask)

    assert matrix.data.shape == (3, 2)
    assert matrix.n_timepoints == 3
    assert matrix.n_voxels == 2
    np.testing.assert_allclose(matrix.data[:, 0], np.array([10.0, 20.0, 30.0]))


def test_apply_brain_mask_rejects_empty_mask() -> None:
    bold = np.zeros((2, 2, 1, 3), dtype=float)
    mask = np.zeros((2, 2, 1), dtype=bool)

    with pytest.raises(ValueError):
        apply_brain_mask(bold, mask)


def test_compute_voxel_baseline() -> None:
    bold_matrix = np.array(
        [
            [10.0, 20.0],
            [12.0, 22.0],
            [100.0, 200.0],
        ]
    )

    baseline = compute_voxel_baseline(bold_matrix, n_baseline_volumes=2)

    np.testing.assert_allclose(baseline, np.array([11.0, 21.0]))


def test_compute_global_baseline() -> None:
    voxel_baseline = np.array([10.0, 20.0, 30.0])

    global_baseline = compute_global_baseline(voxel_baseline)

    assert global_baseline == 20.0


def test_compute_percent_signal_change() -> None:
    bold_matrix = np.array(
        [
            [100.0, 200.0],
            [110.0, 220.0],
        ]
    )
    baseline = np.array([100.0, 200.0])

    psc = compute_percent_signal_change(bold_matrix, baseline)

    np.testing.assert_allclose(psc, np.array([[0.0, 0.0], [10.0, 10.0]]))


def test_reshape_voxel_values_to_volume() -> None:
    mask = np.array(
        [
            [[True], [False]],
            [[True], [False]],
        ]
    )
    values = np.array([1.5, 2.5])

    volume = reshape_voxel_values_to_volume(values, mask)

    assert volume.shape == mask.shape
    assert volume[0, 0, 0] == 1.5
    assert volume[1, 0, 0] == 2.5
    assert np.isnan(volume[0, 1, 0])
