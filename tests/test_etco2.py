import numpy as np
import pandas as pd
import pytest

from neurocvr.preprocessing.etco2 import (
    convert_percent_co2_to_mmhg,
    estimate_baseline,
    interpolate_trace,
    make_bold_time_axis,
    prepare_etco2_regressor,
    shift_trace_time,
)


def test_convert_percent_co2_to_mmhg() -> None:
    percent = np.array([0.0, 5.0, 10.0])

    mmhg = convert_percent_co2_to_mmhg(percent)

    np.testing.assert_allclose(mmhg, np.array([0.0, 38.0, 76.0]))


def test_estimate_baseline_median_with_window() -> None:
    time = np.array([0.0, 1.0, 2.0, 3.0])
    etco2 = np.array([40.0, 41.0, 50.0, 55.0])

    baseline = estimate_baseline(
        etco2_mmhg=etco2,
        time_seconds=time,
        baseline_end_seconds=1.0,
        method="median",
    )

    assert baseline == 40.5


def test_estimate_baseline_rejects_bad_method() -> None:
    with pytest.raises(ValueError):
        estimate_baseline(np.array([40.0, 41.0]), method="mode")


def test_interpolate_trace() -> None:
    source_time = np.array([0.0, 2.0, 4.0])
    source_values = np.array([40.0, 44.0, 48.0])
    target_time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    interpolated = interpolate_trace(source_time, source_values, target_time)

    np.testing.assert_allclose(interpolated, np.array([40.0, 42.0, 44.0, 46.0, 48.0]))


def test_shift_trace_time() -> None:
    time = np.array([0.0, 1.0, 2.0])

    shifted = shift_trace_time(time, shift_seconds=5.0)

    np.testing.assert_allclose(shifted, np.array([5.0, 6.0, 7.0]))


def test_make_bold_time_axis() -> None:
    time = make_bold_time_axis(n_volumes=4, tr_seconds=1.55)

    np.testing.assert_allclose(time, np.array([0.0, 1.55, 3.10, 4.65]))


def test_prepare_etco2_regressor_mmhg() -> None:
    df = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0],
            "etco2": [40.0, 40.0, 50.0, 50.0],
        }
    )

    trace = prepare_etco2_regressor(
        etco2_df=df,
        n_bold_volumes=4,
        tr_seconds=1.0,
        baseline_end_seconds=1.0,
        input_units="mmhg",
    )

    assert trace.n_samples == 4
    np.testing.assert_allclose(trace.etco2_mmhg, np.array([0.0, 0.0, 10.0, 10.0]))


def test_prepare_etco2_regressor_percent() -> None:
    df = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "co2_percent": [5.0, 5.0, 6.0],
        }
    )

    trace = prepare_etco2_regressor(
        etco2_df=df,
        n_bold_volumes=3,
        tr_seconds=1.0,
        etco2_column="co2_percent",
        baseline_end_seconds=1.0,
        input_units="percent",
    )

    np.testing.assert_allclose(trace.etco2_mmhg, np.array([0.0, 0.0, 7.6]))
