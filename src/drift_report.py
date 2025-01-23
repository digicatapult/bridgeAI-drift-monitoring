"""Target/model drift report generation using evidently."""

from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.utils import load_yaml_config


def generate_report(
    historical_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_save_path,
) -> None:
    """Evidently target/model drift generation."""
    # Initialize Evidently's report with data drift report metric
    report = Report(metrics=[DataDriftPreset()])

    # Generate the report comparing historical data to current data
    report.run(reference_data=historical_data, current_data=current_data)

    # Save the report as an HTML file
    report.save_html(str(report_save_path))


if __name__ == "__main__":
    config = load_yaml_config()

    historical_data_save_path = Path(
        config["historical_data_save_path"]
    ).resolve()
    new_data_save_path = Path(config["new_data_save_path"]).resolve()
    report_save_path = Path(config["report_save_path"]).resolve()

    historical_data = pd.read_csv(historical_data_save_path)
    new_data = pd.read_csv(new_data_save_path)

    generate_report(historical_data, new_data, report_save_path)
