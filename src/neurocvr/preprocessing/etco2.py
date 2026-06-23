from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MMHG_PER_PERCENT_CO2 = 760.0 / 100.0


@dataclass(frozen=True)
class Etco2Trace:
    """Processed ETCO2 trace on a known time grid."""

    time_seconds: np.ndarray
    etco2_mmhg: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.time_seconds.shape[0])


def convert_percent_co2_to_mmhg(percent_co2: np.ndarray) -> np.ndarray:
    """
    Convert CO2 percentage to mmHg.

    The conversion follows:
        CO2_mmHg = CO2_percent * 760 / 100
    """
    values = np.asarray(percent_co2, dtype=float)
    return values * MMHG_PER_PERCENT_CO2


def estimate_baseline(
    etco2_mmhg: np.ndarray,
    time_seconds: np.ndarray | None = None,
    baseline_end_seconds: float | None = None,
    method: str = "median",
) -> float:
    """
    Estimate the ETCO2 baseline.

    Parameters
    ----------
    etco2_mmhg:
        ETCO2 values in mmHg.
    time_seconds:
        Time axis in seconds. Required if baseline_end_seconds is used.
    baseline_end_seconds:
        Use only samples with time <= baseline_end_seconds.
    method:
        Either 'median' or 'mean'.
    """
    values = np.asarray(etco2_mmhg, dtype=float)

    if values.ndim != 1:
        raise ValueError("ETCO2 values must be a 1D array.")

    if baseline_end_seconds is not None:
        if time_seconds is None:
            raise ValueError(
                "time_seconds is required when baseline_end_seconds is provided."
            )

        time = np.asarray(time_seconds, dtype=float)

        if time.shape != values.shape:
            raise ValueError("time_seconds and etco2_mmhg must have the same shape.")

        mask = time <= baseline_end_seconds

        if not np.any(mask):
            raise ValueError("No ETCO2 samples found in the requested baseline window.")

        values = values[mask]

    if method == "median":
        return float(np.nanmedian(values))

    if method == "mean":
        return float(np.nanmean(values))

    raise ValueError("method must be either 'median' or 'mean'.")


def interpolate_trace(
    source_time_seconds: np.ndarray,
    source_values: np.ndarray,
    target_time_seconds: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate a time series onto a target time axis.
    """
    source_time = np.asarray(source_time_seconds, dtype=float)
    values = np.asarray(source_values, dtype=float)
    target_time = np.asarray(target_time_seconds, dtype=float)

    if source_time.ndim != 1 or values.ndim != 1 or target_time.ndim != 1:
        raise ValueError("All inputs must be 1D arrays.")

    if source_time.shape != values.shape:
        raise ValueError(
            "source_time_seconds and source_values must have the same shape."
            )

    if np.any(np.diff(source_time) <= 0):
        raise ValueError("source_time_seconds must be strictly increasing.")

    return np.interp(target_time, source_time, values)


def shift_trace_time(time_seconds: np.ndarray, shift_seconds: float) -> np.ndarray:
    """
    Shift a time axis by a given number of seconds.

    Positive shift means the source trace is moved later in time.
    """
    time = np.asarray(time_seconds, dtype=float)
    return time + shift_seconds


def make_bold_time_axis(n_volumes: int, tr_seconds: float) -> np.ndarray:
    """
    Create a BOLD acquisition time axis from number of volumes and TR.
    """
    if n_volumes <= 0:
        raise ValueError("n_volumes must be positive.")

    if tr_seconds <= 0:
        raise ValueError("tr_seconds must be positive.")

    return np.arange(n_volumes, dtype=float) * tr_seconds


def prepare_etco2_regressor(
    etco2_df: pd.DataFrame,
    n_bold_volumes: int,
    tr_seconds: float,
    global_shift_seconds: float = 0.0,
    time_column: str = "time",
    etco2_column: str = "etco2",
    baseline_end_seconds: float | None = None,
    baseline_method: str = "median",
    input_units: str = "mmhg",
) -> Etco2Trace:
    """
    Prepare an ETCO2 regressor on the BOLD time axis.

    This function:
    1. reads ETCO2 values from a dataframe,
    2. optionally converts percent CO2 to mmHg,
    3. applies a global temporal shift,
    4. interpolates onto the BOLD acquisition time axis,
    5. subtracts the ETCO2 baseline.

    Returns
    -------
    Etco2Trace
        Baseline-corrected ETCO2 regressor on the BOLD time axis.
    """
    if time_column not in etco2_df.columns:
        raise ValueError(f"Missing time column: {time_column}")

    if etco2_column not in etco2_df.columns:
        raise ValueError(f"Missing ETCO2 column: {etco2_column}")

    source_time = etco2_df[time_column].to_numpy(dtype=float)
    source_etco2 = etco2_df[etco2_column].to_numpy(dtype=float)

    if input_units == "percent":
        source_etco2 = convert_percent_co2_to_mmhg(source_etco2)
    elif input_units == "mmhg":
        pass
    else:
        raise ValueError("input_units must be either 'mmhg' or 'percent'.")

    baseline = estimate_baseline(
        etco2_mmhg=source_etco2,
        time_seconds=source_time,
        baseline_end_seconds=baseline_end_seconds,
        method=baseline_method,
    )

    shifted_time = shift_trace_time(source_time, global_shift_seconds)
    bold_time = make_bold_time_axis(n_bold_volumes, tr_seconds)

    interpolated = interpolate_trace(
        source_time_seconds=shifted_time,
        source_values=source_etco2,
        target_time_seconds=bold_time,
    )

    baseline_corrected = interpolated - baseline

    return Etco2Trace(
        time_seconds=bold_time,
        etco2_mmhg=baseline_corrected,
    )
