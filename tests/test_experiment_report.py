from pathlib import Path

import pandas as pd

from neurocvr.reporting.experiment_report import (
    format_float,
    identify_best_rows,
    load_sweep_summary,
    make_experiment_report_markdown,
    write_experiment_report,
)


def make_summary_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tcnr": [0.5, 5.0],
            "cvr_rmse_mean": [0.20, 0.03],
            "cvr_rmse_std": [0.01, 0.005],
            "cvr_mae_mean": [0.15, 0.02],
            "cvr_mae_std": [0.01, 0.002],
            "cvr_bias_mean": [0.02, 0.001],
            "cvr_bias_std": [0.01, 0.001],
            "cvr_pcc_mean": [0.80, 0.99],
            "cvr_pcc_std": [0.02, 0.001],
            "delay_rmse_mean": [2.0, 0.4],
            "delay_rmse_std": [0.2, 0.05],
            "delay_mae_mean": [1.5, 0.3],
            "delay_mae_std": [0.1, 0.04],
            "delay_bias_mean": [0.3, 0.01],
            "delay_bias_std": [0.1, 0.01],
            "delay_pcc_mean": [0.70, 0.98],
            "delay_pcc_std": [0.03, 0.01],
        }
    )


def test_format_float() -> None:
    assert format_float(0.123456) == "0.1235"
    assert format_float(0.123456, digits=2) == "0.12"


def test_load_sweep_summary(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    make_summary_dataframe().to_csv(path, index=False)

    loaded = load_sweep_summary(path)

    assert loaded.shape[0] == 2
    assert "cvr_rmse_mean" in loaded.columns


def test_identify_best_rows() -> None:
    summary = make_summary_dataframe()

    best_cvr, best_delay = identify_best_rows(summary)

    assert best_cvr["tcnr"] == 5.0
    assert best_delay["tcnr"] == 5.0


def test_make_experiment_report_markdown() -> None:
    summary = make_summary_dataframe()

    markdown = make_experiment_report_markdown(summary)

    assert "# NeuroCVR-AI Experiment Report" in markdown
    assert "Best CVR magnitude recovery" in markdown
    assert "not a diagnostic tool" in markdown


def test_write_experiment_report(tmp_path: Path) -> None:
    summary_path = tmp_path / "sweep_summary.csv"
    output_path = tmp_path / "experiment_report.md"

    make_summary_dataframe().to_csv(summary_path, index=False)

    written_path = write_experiment_report(
        summary_path=summary_path,
        output_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()
    assert "NeuroCVR-AI Experiment Report" in output_path.read_text()
