"""Unit tests for evidently report generation."""

import os

import pandas as pd

from src.drift_report import generate_report


def test_generate_report():
    """Test evidently report generation"""

    # Dummy arguments for generate_report function
    historical_data = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "target": [10, 20, 30],
            "prediction": [11, 22, 33],
        }
    )
    current_data = pd.DataFrame(
        {
            "feature1": [4, 10, 12],
            "target": [40, 50, 60],
            "prediction": [44, 55, 66],
        }
    )
    report_name = "test_report.html"

    # Assertions
    try:
        # Generate the report
        generate_report(historical_data, current_data, report_name)

        # Assert if the file is created
        assert os.path.exists(
            report_name
        ), f"{report_name} file was not created."

    finally:
        # Clean up: Remove the file after the test
        if os.path.exists(report_name):
            os.remove(report_name)
