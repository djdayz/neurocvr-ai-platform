import numpy as np
import pytest

from neurocvr.simulation.synthetic import (
    add_tcnr_noise,
    make_block_etco2_regressor,
    sample_ground_truth_maps,
    simulate_glm_bold_matrix,
    simulate_glm_cvr_dataset,
)


def test_make_block_etco2_regressor() -> None:
    time, regressor = make_block_etco2_regressor(
        n_timepoints=6,
        tr_seconds=1.0,
        baseline_mmhg=40.0,
        hypercapnia_mmhg=50.0,
        normocapnia_duration_seconds=2.0,
        hypercapnia_duration_seconds=2.0,
    )

    np.testing.assert_allclose(time, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_allclose(regressor, np.array([0.0, 0.0, 10.0, 10.0, 0.0, 0.0]))


def test_sample_ground_truth_maps_shapes() -> None:
    cvr_map, delay_map, mask = sample_ground_truth_maps(
        spatial_shape=(2, 3, 1),
        seed=123,
    )

    assert cvr_map.shape == (2, 3, 1)
    assert delay_map.shape == (2, 3, 1)
    assert mask.shape == (2, 3, 1)
    assert mask.sum() == 6
    assert np.nanmin(cvr_map) >= 0.2
    assert np.nanmax(cvr_map) <= 1.5


def test_simulate_glm_bold_matrix_without_delay() -> None:
    time = np.array([0.0, 1.0, 2.0])
    etco2 = np.array([0.0, 5.0, 10.0])
    cvr = np.array([1.0])
    delay = np.array([0.0])

    bold = simulate_glm_bold_matrix(
        cvr_magnitude=cvr,
        delay_seconds=delay,
        etco2_regressor=etco2,
        time_seconds=time,
        bold_baseline=100.0,
    )

    np.testing.assert_allclose(bold[:, 0], np.array([100.0, 105.0, 110.0]))


def test_add_tcnr_noise_is_reproducible() -> None:
    bold = np.array(
        [
            [100.0, 100.0],
            [105.0, 110.0],
            [110.0, 120.0],
        ]
    )

    noisy_1 = add_tcnr_noise(bold, tcnr=5.0, seed=42)
    noisy_2 = add_tcnr_noise(bold, tcnr=5.0, seed=42)

    np.testing.assert_allclose(noisy_1, noisy_2)


def test_add_tcnr_noise_rejects_bad_tcnr() -> None:
    with pytest.raises(ValueError):
        add_tcnr_noise(np.zeros((3, 2)), tcnr=0.0)


def test_simulate_glm_cvr_dataset_shapes() -> None:
    dataset = simulate_glm_cvr_dataset(
        spatial_shape=(2, 3, 1),
        n_timepoints=8,
        tr_seconds=1.0,
        tcnr=None,
        seed=42,
    )

    assert dataset.bold_4d.shape == (2, 3, 1, 8)
    assert dataset.cvr_magnitude_map.shape == (2, 3, 1)
    assert dataset.delay_map.shape == (2, 3, 1)
    assert dataset.etco2_regressor.shape == (8,)
    assert dataset.time_seconds.shape == (8,)
    assert dataset.mask.sum() == 6
