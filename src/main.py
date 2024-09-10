"""Main entrypoint for drift monitoring for bridgeai regression model data."""

import os
import warnings
from pathlib import Path

from src.drift_report import generate_report
from src.get_data import fetch_data
from src.inference import load_data, predict
from src.upload_report import upload
from src.utils import load_yaml_config

warnings.filterwarnings("ignore")


def main():
    """Main entry point."""
    # load the config file
    config = load_yaml_config()

    historical_data_version = os.getenv(
        "HISTORICAL_DATA_VERSION", config["historical_data_version"]
    )
    new_data_version = os.getenv(
        "NEW_DATA_VERSION", config["new_data_version"]
    )
    feature_columns = config["feature_columns"]
    model_endpoint = os.getenv("MODEL_ENDPOINT", config["model_endpoint"])
    report_save_path = Path(config["report_save_path"]).resolve()

    historical_data_save_path = Path(
        config["historical_data_save_path"]
    ).resolve()
    new_data_save_path = Path(config["new_data_save_path"]).resolve()

    print(f"historical_data_save_path: {historical_data_save_path}")
    print(f"new_data_save_path: {new_data_save_path}")

    # Fetch datasets from dvc
    fetch_data(config, historical_data_version, historical_data_save_path)
    fetch_data(config, new_data_version, new_data_save_path)

    # load the datasets
    historical_data, current_data = load_data(
        historical_data_save_path, new_data_save_path, config
    )

    # Model predictions for both datasets
    historical_data["prediction"] = predict(
        model_endpoint, historical_data[feature_columns]
    )
    current_data["prediction"] = predict(
        model_endpoint, current_data[feature_columns]
    )

    generate_report(historical_data, current_data, report_save_path)

    bucket_name = os.getenv(
        "DRIFT_REPORT_BUCKET", config["report_save_bucket"]
    )
    s3_client = get_s3_client()
    upload(s3_client, report_save_path, bucket_name)


if __name__ == "__main__":
    main()
