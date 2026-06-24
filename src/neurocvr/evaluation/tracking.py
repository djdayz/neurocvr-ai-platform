from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from neurocvr.evaluation.metrics import RegressionMetrics


def to_python_scalar(value: Any) -> Any:
    """Convert NumPy scalars and Paths into JSON/CSV-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value


def clean_metric_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a metric record into JSON/CSV-safe values."""
    return {key: to_python_scalar(value) for key, value in record.items()}


def regression_metrics_to_dict(
    metrics: RegressionMetrics,
    prefix: str,
) -> dict[str, float | int]:
    """Flatten RegressionMetrics into a dictionary with a prefix."""
    clean_prefix = f"{prefix}_" if prefix else ""

    return {
        f"{clean_prefix}rmse": metrics.rmse,
        f"{clean_prefix}mae": metrics.mae,
        f"{clean_prefix}bias": metrics.bias,
        f"{clean_prefix}pcc": metrics.pcc,
        f"{clean_prefix}n_voxels": metrics.n_voxels,
    }


def save_metrics_json(
    record: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Save a metric record as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cleaned = clean_metric_record(record)

    with path.open("w") as f:
        json.dump(cleaned, f, indent=2, sort_keys=True)

    return path


def save_metrics_csv(
    record: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Save a single metric record as a one-row CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cleaned = clean_metric_record(record)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cleaned.keys()))
        writer.writeheader()
        writer.writerow(cleaned)

    return path
