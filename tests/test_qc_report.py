from neurocvr.reporting.qc_report import (
    classify_metric,
    combine_statuses,
    generate_qc_report,
    qc_report_to_markdown,
)


def test_classify_metric_lower_is_better() -> None:
    assert classify_metric(0.05, 0.10, 0.20, lower_is_better=True) == "pass"
    assert classify_metric(0.15, 0.10, 0.20, lower_is_better=True) == "warning"
    assert classify_metric(0.25, 0.10, 0.20, lower_is_better=True) == "fail"


def test_classify_metric_higher_is_better() -> None:
    assert classify_metric(0.95, 0.90, 0.75, lower_is_better=False) == "pass"
    assert classify_metric(0.85, 0.90, 0.75, lower_is_better=False) == "warning"
    assert classify_metric(0.70, 0.90, 0.75, lower_is_better=False) == "fail"


def test_combine_statuses() -> None:
    assert combine_statuses(["pass", "pass"]) == "pass"
    assert combine_statuses(["pass", "warning"]) == "warning"
    assert combine_statuses(["pass", "fail"]) == "fail"


def test_generate_qc_report_pass() -> None:
    report = generate_qc_report(
        cvr_rmse=0.03,
        cvr_mae=0.02,
        cvr_bias=0.00,
        cvr_pcc=0.99,
        delay_rmse=0.40,
        delay_mae=0.30,
        delay_bias=0.01,
        delay_pcc=0.98,
        n_voxels=60,
    )

    assert report.status == "pass"
    assert "passed" in report.summary
    assert "not a diagnosis" in report.safety_notice


def test_generate_qc_report_warning() -> None:
    report = generate_qc_report(
        cvr_rmse=0.15,
        cvr_mae=0.10,
        cvr_bias=0.00,
        cvr_pcc=0.99,
        delay_rmse=0.40,
        delay_mae=0.30,
        delay_bias=0.01,
        delay_pcc=0.98,
        n_voxels=60,
    )

    assert report.status == "warning"


def test_generate_qc_report_fail() -> None:
    report = generate_qc_report(
        cvr_rmse=0.30,
        cvr_mae=0.20,
        cvr_bias=0.00,
        cvr_pcc=0.99,
        delay_rmse=0.40,
        delay_mae=0.30,
        delay_bias=0.01,
        delay_pcc=0.98,
        n_voxels=60,
    )

    assert report.status == "fail"


def test_qc_report_to_markdown() -> None:
    report = generate_qc_report(
        cvr_rmse=0.03,
        cvr_mae=0.02,
        cvr_bias=0.00,
        cvr_pcc=0.99,
        delay_rmse=0.40,
        delay_mae=0.30,
        delay_bias=0.01,
        delay_pcc=0.98,
        n_voxels=60,
    )

    markdown = qc_report_to_markdown(report)

    assert "# NeuroCVR-AI Technical QC Report" in markdown
    assert "Overall status" in markdown
    assert "Safety notice" in markdown
