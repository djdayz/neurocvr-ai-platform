from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QcStatus = Literal["pass", "warning", "fail"]


@dataclass(frozen=True)
class QcThresholds:
    """Thresholds for technical QC reporting."""

    cvr_rmse_warning: float = 0.10
    cvr_rmse_fail: float = 0.20
    delay_rmse_warning_seconds: float = 1.0
    delay_rmse_fail_seconds: float = 2.0
    pcc_warning: float = 0.90
    pcc_fail: float = 0.75


@dataclass(frozen=True)
class QcReport:
    """Technical QC report for a synthetic CVR benchmark run."""

    status: QcStatus
    summary: str
    cvr_assessment: str
    delay_assessment: str
    recommendations: list[str]
    safety_notice: str


def classify_metric(
    value: float,
    warning_threshold: float,
    fail_threshold: float,
    lower_is_better: bool = True,
) -> QcStatus:
    """Classify a metric as pass, warning, or fail."""
    if lower_is_better:
        if value >= fail_threshold:
            return "fail"
        if value >= warning_threshold:
            return "warning"
        return "pass"

    if value <= fail_threshold:
        return "fail"
    if value <= warning_threshold:
        return "warning"
    return "pass"


def combine_statuses(statuses: list[QcStatus]) -> QcStatus:
    """Combine multiple QC statuses into one overall status."""
    if "fail" in statuses:
        return "fail"

    if "warning" in statuses:
        return "warning"

    return "pass"


def generate_qc_report(
    cvr_rmse: float,
    cvr_mae: float,
    cvr_bias: float,
    cvr_pcc: float,
    delay_rmse: float,
    delay_mae: float,
    delay_bias: float,
    delay_pcc: float,
    n_voxels: int,
    thresholds: QcThresholds | None = None,
) -> QcReport:
    """Generate a technical QC report from benchmark metrics."""
    if thresholds is None:
        thresholds = QcThresholds()

    cvr_rmse_status = classify_metric(
        value=cvr_rmse,
        warning_threshold=thresholds.cvr_rmse_warning,
        fail_threshold=thresholds.cvr_rmse_fail,
        lower_is_better=True,
    )
    cvr_pcc_status = classify_metric(
        value=cvr_pcc,
        warning_threshold=thresholds.pcc_warning,
        fail_threshold=thresholds.pcc_fail,
        lower_is_better=False,
    )
    delay_rmse_status = classify_metric(
        value=delay_rmse,
        warning_threshold=thresholds.delay_rmse_warning_seconds,
        fail_threshold=thresholds.delay_rmse_fail_seconds,
        lower_is_better=True,
    )
    delay_pcc_status = classify_metric(
        value=delay_pcc,
        warning_threshold=thresholds.pcc_warning,
        fail_threshold=thresholds.pcc_fail,
        lower_is_better=False,
    )

    overall_status = combine_statuses(
        [
            cvr_rmse_status,
            cvr_pcc_status,
            delay_rmse_status,
            delay_pcc_status,
        ]
    )

    if overall_status == "pass":
        summary = (
            "The synthetic benchmark passed the technical QC thresholds. "
            "CVR magnitude and delay recovery were both consistent with the "
            "known simulated ground truth."
        )
    elif overall_status == "warning":
        summary = (
            "The synthetic benchmark completed with QC warnings. The pipeline "
            "ran successfully, but at least one metric suggests reduced "
            "estimation quality."
        )
    else:
        summary = (
            "The synthetic benchmark failed one or more technical QC thresholds. "
            "The run should be inspected before being used as evidence of model "
            "or pipeline performance."
        )

    cvr_assessment = (
        f"CVR magnitude recovery: RMSE={cvr_rmse:.4f}, MAE={cvr_mae:.4f}, "
        f"bias={cvr_bias:.4f}, PCC={cvr_pcc:.4f}. "
        f"RMSE status={cvr_rmse_status}; PCC status={cvr_pcc_status}."
    )

    delay_assessment = (
        f"Delay recovery: RMSE={delay_rmse:.4f} s, MAE={delay_mae:.4f} s, "
        f"bias={delay_bias:.4f} s, PCC={delay_pcc:.4f}. "
        f"RMSE status={delay_rmse_status}; PCC status={delay_pcc_status}."
    )

    recommendations = [
        "Inspect generated CVR and delay maps visually before interpreting results.",
        "Compare performance across multiple random seeds and tCNR levels.",
        "Use real BOLD/ETCO2 data only after validating the synthetic benchmark.",
    ]

    if overall_status != "pass":
        recommendations.append(
            "Review the ETCO2 regressor, delay search range, noise level, and "
            "baseline estimation settings."
        )

    safety_notice = (
        "This report is a technical QC summary for research and engineering "
        "validation only. It is not a diagnosis and must not be used for "
        "clinical decision-making."
    )

    return QcReport(
        status=overall_status,
        summary=summary,
        cvr_assessment=cvr_assessment,
        delay_assessment=delay_assessment,
        recommendations=recommendations,
        safety_notice=safety_notice,
    )


def qc_report_to_markdown(report: QcReport) -> str:
    """Render a QC report as Markdown."""
    recommendations = "\n".join(
        f"- {recommendation}" for recommendation in report.recommendations
    )

    return (
        "# NeuroCVR-AI Technical QC Report\n\n"
        f"## Overall status\n\n{report.status.upper()}\n\n"
        f"## Summary\n\n{report.summary}\n\n"
        f"## CVR magnitude assessment\n\n{report.cvr_assessment}\n\n"
        f"## Delay assessment\n\n{report.delay_assessment}\n\n"
        f"## Recommendations\n\n{recommendations}\n\n"
        f"## Safety notice\n\n{report.safety_notice}\n"
    )
