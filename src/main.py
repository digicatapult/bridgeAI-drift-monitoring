"""Main entrypoint for drift monitoring for bridgeai regression model data."""

import os
import warnings

from src.generate_report import generate_report
from src.get_data import load_data
from src.inference import predict
from src.utils import load_yaml_config

warnings.filterwarnings("ignore")


def main():
    """Main entry point."""
    # load the config file
    config = load_yaml_config()

    current_data_local_path = config["current_data_local_path"]
    new_data_local_path = config["new_data_local_path"]
    feature_columns = config["feature_columns"]
    model_endpoint = os.getenv("MODEL_ENDPOINT", config["model_endpoint"])
    report_location = os.getenv("REPORT_LOCATION", config["report_location"])
    report_name = os.getenv("REPORT_NAME", config["report_name"])
    label_column = config["label_column"]

    # load the datasets
    historical_data, current_data = load_data(
        current_data_local_path, new_data_local_path, label_column
    )

    # Model predictions for both datasets
    historical_data["prediction"] = predict(
        model_endpoint, historical_data[feature_columns]
    )
    current_data["prediction"] = predict(
        model_endpoint, current_data[feature_columns]
    )

    generate_report(
        historical_data, current_data, report_name, report_location
    )


if __name__ == "__main__":
    main()
