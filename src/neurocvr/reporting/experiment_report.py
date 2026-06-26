from __future__ import annotations

from pathlib import Path

import pandas as pd


def format_float(value: float, digits: int = 4) -> str:
    """Format a float for report text."""
    return f"{value:.{digits}f}"


def load_sweep_summary(summary_path: str | Path) -> pd.DataFrame:
    """Load an exported MLflow sweep summary CSV."""
    summary_path = Path(summary_path)

    if not summary_path.exists():
        raise FileNotFoundError(f"Sweep summary not found: {summary_path}")

    return pd.read_csv(summary_path)


def identify_best_rows(summary: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Identify best tCNR rows for CVR RMSE and delay RMSE."""
    required = ["cvr_rmse_mean", "delay_rmse_mean"]

    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise ValueError(f"Missing expected summary columns: {missing}")

    best_cvr = summary.loc[summary["cvr_rmse_mean"].idxmin()]
    best_delay = summary.loc[summary["delay_rmse_mean"].idxmin()]

    return best_cvr, best_delay


def dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
    """Render a DataFrame as a basic GitHub-flavored Markdown table."""
    columns = [str(column) for column in dataframe.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in dataframe.itertuples(index=False, name=None)
    ]

    return "\n".join([header, separator, *rows])


def make_experiment_report_markdown(
    summary: pd.DataFrame,
    cvr_plot_path: str | Path = "cvr_rmse_vs_tcnr.png",
    delay_plot_path: str | Path = "delay_rmse_vs_tcnr.png",
) -> str:
    """Generate a Markdown experiment report from a sweep summary."""
    best_cvr, best_delay = identify_best_rows(summary)

    table = dataframe_to_markdown_table(summary)

    lines = [
        "# NeuroCVR-AI Experiment Report",
        "",
        "This report summarises the synthetic GLM benchmark sweep for "
        "NeuroCVR-AI.",
        "",
        "The sweep evaluates how cerebrovascular reactivity magnitude and "
        "delay recovery change across temporal contrast-to-noise ratio "
        "conditions.",
        "",
        "## Experiment setup",
        "",
        "The benchmark sweep varies:",
        "",
        "- tCNR noise level",
        "- random seed",
        "",
        "For each run, the pipeline:",
        "",
        "1. Generates synthetic ground-truth CVR magnitude and delay maps.",
        "2. Simulates BOLD-fMRI data from an ETCO2 stimulus.",
        "3. Fits a voxelwise GLM with delay search.",
        "4. Estimates CVR magnitude and delay maps.",
        "5. Compares estimates against synthetic ground truth.",
        "6. Logs parameters and metrics to MLflow.",
        "",
        "## Key metrics",
        "",
        "The tracked metrics are:",
        "",
        "- CVR magnitude RMSE",
        "- CVR magnitude MAE",
        "- CVR magnitude bias",
        "- CVR magnitude PCC",
        "- Delay RMSE",
        "- Delay MAE",
        "- Delay bias",
        "- Delay PCC",
        "",
        "## Best CVR magnitude recovery",
        "",
        "Best mean CVR RMSE occurred at:",
        "",
        f"- tCNR: {format_float(float(best_cvr['tcnr']))}",
        f"- mean CVR RMSE: {format_float(float(best_cvr['cvr_rmse_mean']))}",
        f"- mean CVR PCC: {format_float(float(best_cvr['cvr_pcc_mean']))}",
        "",
        "## Best delay recovery",
        "",
        "Best mean delay RMSE occurred at:",
        "",
        f"- tCNR: {format_float(float(best_delay['tcnr']))}",
        "- mean delay RMSE: "
        f"{format_float(float(best_delay['delay_rmse_mean']))} s",
        f"- mean delay PCC: {format_float(float(best_delay['delay_pcc_mean']))}",
        "",
        "## Summary table",
        "",
        table,
        "",
        "## Figures",
        "",
        "### CVR magnitude RMSE vs tCNR",
        "",
        f"![CVR magnitude RMSE vs tCNR]({cvr_plot_path})",
        "",
        "### Delay RMSE vs tCNR",
        "",
        f"![Delay RMSE vs tCNR]({delay_plot_path})",
        "",
        "## Interpretation",
        "",
        "Higher tCNR generally improves CVR magnitude and delay recovery "
        "because the simulated BOLD response is less dominated by noise.",
        "",
        "Low-tCNR runs are expected to show larger errors and lower "
        "correlations, making them useful stress tests for the estimation "
        "pipeline.",
        "",
        "## Engineering relevance",
        "",
        "This report demonstrates:",
        "",
        "- experiment tracking with MLflow",
        "- repeatable benchmark sweeps",
        "- CSV export of run-level and summary metrics",
        "- automated report generation",
        "- portfolio-ready visualisation of model behaviour",
        "",
        "## Safety and limitations",
        "",
        "This project is a research and portfolio prototype.",
        "",
        "The reported metrics are based on synthetic data with known ground "
        "truth. They should not be interpreted as clinical validation.",
        "",
        "This software is not a diagnostic tool and must not be used for "
        "patient care.",
        "",
    ]

    return "\n".join(lines)


def write_experiment_report(
    summary_path: str | Path = "reports/mlflow_sweep/sweep_summary.csv",
    output_path: str | Path = "reports/mlflow_sweep/experiment_report.md",
) -> Path:
    """Write an experiment report Markdown file from a sweep summary CSV."""
    summary_path = Path(summary_path)
    output_path = Path(output_path)

    summary = load_sweep_summary(summary_path)

    markdown = make_experiment_report_markdown(
        summary=summary,
        cvr_plot_path="cvr_rmse_vs_tcnr.png",
        delay_plot_path="delay_rmse_vs_tcnr.png",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)

    return output_path
