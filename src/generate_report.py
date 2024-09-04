"""Target/model drift report generation using evidently."""

import pandas as pd
from evidently.metric_preset import RegressionPreset, TargetDriftPreset
from evidently.report import Report


def generate_report(
    historical_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_name,
    report_location,
) -> None:
    """Evidently target/model drift generation."""
    # Initialize Evidently's report with
    # target drift and regression performance metrics
    report = Report(metrics=[RegressionPreset(), TargetDriftPreset()])

    # Generate the report comparing historical data to current data
    report.run(reference_data=historical_data, current_data=current_data)

    # Save the report as an HTML file
    report.save_html(report_name)
