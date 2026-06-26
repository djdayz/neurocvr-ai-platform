from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from neurocvr.evaluation.benchmark import SyntheticGlmBenchmarkResult


def clean_mlflow_param(value: Any) -> str | int | float | bool | None:
    """Convert Python objects into MLflow-safe parameter values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value

    return str(value)


def make_sqlite_tracking_uri(tracking_dir: str | Path) -> str:
    """Create a SQLite MLflow tracking URI inside the tracking directory."""
    tracking_path = Path(tracking_dir)
    tracking_path.mkdir(parents=True, exist_ok=True)

    db_path = tracking_path / "mlflow.db"
    db_path.touch(exist_ok=True)
    return f"sqlite:///{db_path.resolve()}"


def make_mlflow_ui_command(tracking_dir: str | Path, port: int = 5000) -> str:
    """Build the MLflow UI command for the local SQLite tracking store."""
    tracking_path = Path(tracking_dir)
    artifact_path = tracking_path / "artifacts"
    artifact_path.mkdir(parents=True, exist_ok=True)

    return (
        "mlflow ui "
        f"--backend-store-uri {make_sqlite_tracking_uri(tracking_path)} "
        f"--default-artifact-root {artifact_path.resolve().as_uri()} "
        f"--port {port}"
    )


def make_mlflow_runs_url(
    experiment_name: str,
    tracking_dir: str | Path,
    host: str = "http://127.0.0.1",
    port: int = 5000,
) -> str | None:
    """Build a direct URL to the MLflow runs table for an experiment."""
    mlflow.set_tracking_uri(make_sqlite_tracking_uri(tracking_dir))
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        return None

    return f"{host}:{port}/#/experiments/{experiment.experiment_id}/runs"


def benchmark_metrics_to_mlflow_dict(
    result: SyntheticGlmBenchmarkResult,
) -> dict[str, float]:
    """Flatten benchmark metrics into an MLflow metric dictionary."""
    return {
        "cvr_rmse": result.cvr_metrics.rmse,
        "cvr_mae": result.cvr_metrics.mae,
        "cvr_bias": result.cvr_metrics.bias,
        "cvr_pcc": result.cvr_metrics.pcc,
        "delay_rmse": result.delay_metrics.rmse,
        "delay_mae": result.delay_metrics.mae,
        "delay_bias": result.delay_metrics.bias,
        "delay_pcc": result.delay_metrics.pcc,
    }


def set_mlflow_experiment(
    tracking_dir: str | Path,
    experiment_name: str,
) -> None:
    """Configure MLflow to use a local SQLite backend and artifact directory."""
    tracking_path = Path(tracking_dir)
    artifact_path = tracking_path / "artifacts"
    artifact_path.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(make_sqlite_tracking_uri(tracking_path))

    existing_experiment = mlflow.get_experiment_by_name(experiment_name)

    if existing_experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_path.resolve().as_uri(),
        )

    mlflow.set_experiment(experiment_name)


def log_benchmark_result_to_mlflow(
    result: SyntheticGlmBenchmarkResult,
    params: dict[str, Any],
    tracking_dir: str | Path = "mlruns",
    experiment_name: str = "NeuroCVR-AI",
    run_name: str = "synthetic_glm_benchmark",
) -> str:
    """Log benchmark parameters and metrics to a local MLflow SQLite store."""
    set_mlflow_experiment(
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
    )

    cleaned_params = {
        key: clean_mlflow_param(value) for key, value in params.items()
    }
    metrics = benchmark_metrics_to_mlflow_dict(result)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(cleaned_params)
        mlflow.log_metrics(metrics)

        mlflow.set_tag("project", "neurocvr-ai")
        mlflow.set_tag("pipeline", "synthetic-glm-benchmark")
        mlflow.set_tag("clinical_use", "not-for-clinical-use")

        return run.info.run_id
