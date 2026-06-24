import csv
import json
from pathlib import Path

import numpy as np

from neurocvr.evaluation.metrics import RegressionMetrics
from neurocvr.evaluation.tracking import (
    clean_metric_record,
    regression_metrics_to_dict,
    save_metrics_csv,
    save_metrics_json,
)


def test_regression_metrics_to_dict() -> None:
    metrics = RegressionMetrics(
        rmse=0.1,
        mae=0.2,
        bias=-0.3,
        pcc=0.9,
        n_voxels=100,
    )

    record = regression_metrics_to_dict(metrics, prefix="cvr")

    assert record["cvr_rmse"] == 0.1
    assert record["cvr_mae"] == 0.2
    assert record["cvr_bias"] == -0.3
    assert record["cvr_pcc"] == 0.9
    assert record["cvr_n_voxels"] == 100


def test_clean_metric_record_converts_numpy_scalars_and_paths(tmp_path: Path) -> None:
    record = {
        "rmse": np.float64(0.1),
        "n_voxels": np.int64(10),
        "path": tmp_path / "metrics.json",
    }

    cleaned = clean_metric_record(record)

    assert isinstance(cleaned["rmse"], float)
    assert isinstance(cleaned["n_voxels"], int)
    assert isinstance(cleaned["path"], str)


def test_save_metrics_json(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    record = {"run_name": "demo", "rmse": 0.1}

    saved_path = save_metrics_json(record, path)

    with saved_path.open() as f:
        loaded = json.load(f)

    assert loaded["run_name"] == "demo"
    assert loaded["rmse"] == 0.1


def test_save_metrics_csv(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    record = {"run_name": "demo", "rmse": 0.1}

    saved_path = save_metrics_csv(record, path)

    with saved_path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["run_name"] == "demo"
    assert rows[0]["rmse"] == "0.1"
