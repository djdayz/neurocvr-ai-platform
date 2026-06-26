from pathlib import Path

from neurocvr.evaluation.benchmark import run_synthetic_glm_benchmark
from neurocvr.evaluation.mlflow_tracking import (
    benchmark_metrics_to_mlflow_dict,
    clean_mlflow_param,
    log_benchmark_result_to_mlflow,
    make_sqlite_tracking_uri,
)


def test_clean_mlflow_param() -> None:
    assert clean_mlflow_param("demo") == "demo"
    assert clean_mlflow_param(42) == 42
    assert clean_mlflow_param(5.0) == 5.0
    assert clean_mlflow_param(None) is None
    assert clean_mlflow_param((2, 2, 1)) == "(2, 2, 1)"


def test_make_sqlite_tracking_uri(tmp_path: Path) -> None:
    tracking_dir = tmp_path / "mlruns"
    uri = make_sqlite_tracking_uri(tracking_dir)

    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")
    assert tracking_dir.exists()
    assert (tracking_dir / "mlflow.db").exists()


def test_benchmark_metrics_to_mlflow_dict() -> None:
    result = run_synthetic_glm_benchmark(
        spatial_shape=(2, 2, 1),
        n_timepoints=220,
        tr_seconds=1.55,
        tcnr=5.0,
        seed=42,
        delay_step_seconds=2.0,
    )

    metrics = benchmark_metrics_to_mlflow_dict(result)

    assert metrics["cvr_rmse"] >= 0
    assert metrics["delay_rmse"] >= 0
    assert "cvr_pcc" in metrics
    assert "delay_pcc" in metrics


def test_log_benchmark_result_to_mlflow(tmp_path: Path) -> None:
    result = run_synthetic_glm_benchmark(
        spatial_shape=(2, 2, 1),
        n_timepoints=220,
        tr_seconds=1.55,
        tcnr=5.0,
        seed=42,
        delay_step_seconds=2.0,
    )

    tracking_dir = tmp_path / "mlruns"

    run_id = log_benchmark_result_to_mlflow(
        result=result,
        params={
            "spatial_shape": (2, 2, 1),
            "n_timepoints": 220,
            "tr_seconds": 1.55,
            "tcnr": 5.0,
            "seed": 42,
            "delay_step_seconds": 2.0,
        },
        tracking_dir=tracking_dir,
        experiment_name="NeuroCVR-AI-Test",
        run_name="test-run",
    )

    assert isinstance(run_id, str)
    assert len(run_id) > 0
    assert tracking_dir.exists()
    assert (tracking_dir / "mlflow.db").exists()
