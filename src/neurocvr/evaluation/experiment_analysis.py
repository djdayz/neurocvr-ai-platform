from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def load_mlflow_runs(
    tracking_db_path: str | Path = "mlruns/mlflow.db",
    experiment_name: str = "NeuroCVR-AI",
) -> pd.DataFrame:
    """Load MLflow runs for an experiment from a local SQLite tracking database."""
    db_path = Path(tracking_db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"MLflow tracking database not found: {db_path}")

    mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve()}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = mlflow.search_runs([experiment.experiment_id])

    if runs.empty:
        raise ValueError(f"No MLflow runs found for experiment: {experiment_name}")

    return runs


def extract_sweep_columns(runs: pd.DataFrame) -> pd.DataFrame:
    """Extract key parameters and metrics from raw MLflow runs."""
    required_columns = [
        "tags.mlflow.runName",
        "params.tcnr",
        "params.seed",
        "metrics.cvr_rmse",
        "metrics.cvr_mae",
        "metrics.cvr_bias",
        "metrics.cvr_pcc",
        "metrics.delay_rmse",
        "metrics.delay_mae",
        "metrics.delay_bias",
        "metrics.delay_pcc",
    ]

    missing = [column for column in required_columns if column not in runs.columns]
    if missing:
        raise ValueError(f"Missing expected MLflow columns: {missing}")

    sweep = runs[required_columns].copy()

    sweep = sweep.rename(
        columns={
            "tags.mlflow.runName": "run_name",
            "params.tcnr": "tcnr",
            "params.seed": "seed",
            "metrics.cvr_rmse": "cvr_rmse",
            "metrics.cvr_mae": "cvr_mae",
            "metrics.cvr_bias": "cvr_bias",
            "metrics.cvr_pcc": "cvr_pcc",
            "metrics.delay_rmse": "delay_rmse",
            "metrics.delay_mae": "delay_mae",
            "metrics.delay_bias": "delay_bias",
            "metrics.delay_pcc": "delay_pcc",
        }
    )

    sweep["tcnr"] = sweep["tcnr"].astype(float)
    sweep["seed"] = sweep["seed"].astype(int)

    return sweep.sort_values(["tcnr", "seed"]).reset_index(drop=True)


def summarise_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    """Summarise sweep metrics by tCNR."""
    metric_columns = [
        "cvr_rmse",
        "cvr_mae",
        "cvr_bias",
        "cvr_pcc",
        "delay_rmse",
        "delay_mae",
        "delay_bias",
        "delay_pcc",
    ]

    summary = (
        sweep.groupby("tcnr")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]

    return summary


def plot_metric_vs_tcnr(
    summary: pd.DataFrame,
    mean_column: str,
    std_column: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """Plot a sweep metric against tCNR with standard deviation error bars."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.errorbar(
        summary["tcnr"],
        summary[mean_column],
        yerr=summary[std_column],
        marker="o",
        capsize=4,
    )
    ax.set_xlabel("tCNR")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs tCNR")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def export_mlflow_sweep_report(
    tracking_db_path: str | Path = "mlruns/mlflow.db",
    experiment_name: str = "NeuroCVR-AI",
    output_dir: str | Path = "reports/mlflow_sweep",
) -> tuple[Path, Path, Path, Path]:
    """Export MLflow sweep runs, summary tables, and plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_mlflow_runs(
        tracking_db_path=tracking_db_path,
        experiment_name=experiment_name,
    )
    sweep = extract_sweep_columns(runs)
    summary = summarise_sweep(sweep)

    runs_path = output_dir / "sweep_runs.csv"
    summary_path = output_dir / "sweep_summary.csv"
    cvr_plot_path = output_dir / "cvr_rmse_vs_tcnr.png"
    delay_plot_path = output_dir / "delay_rmse_vs_tcnr.png"

    sweep.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_metric_vs_tcnr(
        summary=summary,
        mean_column="cvr_rmse_mean",
        std_column="cvr_rmse_std",
        ylabel="CVR magnitude RMSE",
        output_path=cvr_plot_path,
    )

    plot_metric_vs_tcnr(
        summary=summary,
        mean_column="delay_rmse_mean",
        std_column="delay_rmse_std",
        ylabel="Delay RMSE",
        output_path=delay_plot_path,
    )

    return runs_path, summary_path, cvr_plot_path, delay_plot_path
