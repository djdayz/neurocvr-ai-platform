from pathlib import Path

import pandas as pd

from neurocvr.evaluation.experiment_analysis import (
    extract_sweep_columns,
    plot_metric_vs_tcnr,
    summarise_sweep,
)


def make_raw_runs_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tags.mlflow.runName": [
                "synthetic_glm_tcnr_0.5_seed_42",
                "synthetic_glm_tcnr_0.5_seed_43",
                "synthetic_glm_tcnr_5.0_seed_42",
                "synthetic_glm_tcnr_5.0_seed_43",
            ],
            "params.tcnr": ["0.5", "0.5", "5.0", "5.0"],
            "params.seed": ["42", "43", "42", "43"],
            "metrics.cvr_rmse": [0.2, 0.22, 0.03, 0.04],
            "metrics.cvr_mae": [0.15, 0.16, 0.02, 0.03],
            "metrics.cvr_bias": [0.01, 0.02, 0.001, 0.002],
            "metrics.cvr_pcc": [0.8, 0.82, 0.99, 0.98],
            "metrics.delay_rmse": [2.0, 2.2, 0.4, 0.5],
            "metrics.delay_mae": [1.5, 1.6, 0.3, 0.35],
            "metrics.delay_bias": [0.2, 0.3, 0.01, 0.02],
            "metrics.delay_pcc": [0.7, 0.72, 0.98, 0.97],
        }
    )


def test_extract_sweep_columns() -> None:
    raw_runs = make_raw_runs_dataframe()

    sweep = extract_sweep_columns(raw_runs)

    assert list(sweep.columns) == [
        "run_name",
        "tcnr",
        "seed",
        "cvr_rmse",
        "cvr_mae",
        "cvr_bias",
        "cvr_pcc",
        "delay_rmse",
        "delay_mae",
        "delay_bias",
        "delay_pcc",
    ]
    assert sweep["tcnr"].dtype == float
    assert sweep["seed"].dtype == int
    assert sweep.shape[0] == 4


def test_summarise_sweep() -> None:
    sweep = extract_sweep_columns(make_raw_runs_dataframe())

    summary = summarise_sweep(sweep)

    assert summary.shape[0] == 2
    assert "cvr_rmse_mean" in summary.columns
    assert "cvr_rmse_std" in summary.columns
    assert "delay_rmse_mean" in summary.columns
    assert "delay_rmse_std" in summary.columns


def test_plot_metric_vs_tcnr(tmp_path: Path) -> None:
    sweep = extract_sweep_columns(make_raw_runs_dataframe())
    summary = summarise_sweep(sweep)

    output_path = tmp_path / "plot.png"

    plot_metric_vs_tcnr(
        summary=summary,
        mean_column="cvr_rmse_mean",
        std_column="cvr_rmse_std",
        ylabel="CVR magnitude RMSE",
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
