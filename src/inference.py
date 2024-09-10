"""Inference on reference data using model prediction endpoints."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pandera import Column, DataFrameSchema

from src.utils import load_yaml_config

headers = {"Content-Type": "application/json"}


# Pandera schema for validating the data
schema = DataFrameSchema(
    {
        "area": Column(float),
        "bedrooms": Column(int),
        "bathrooms": Column(int),
        "stories": Column(int),
        "mainroad": Column(str),
        "guestroom": Column(str),
        "basement": Column(str),
        "hotwaterheating": Column(str),
        "airconditioning": Column(str),
        "parking": Column(int),
        "prefarea": Column(str),
        "furnishingstatus": Column(str),
    },
    coerce=True,
)


def load_data(historical_data_path, new_data_path, config):
    """Load current(new) data and historical(used for training) data."""

    label_column = config["label_column"]

    # Historical data
    historical_data = schema.validate(pd.read_csv(historical_data_path))

    # Current data (new incoming data)
    current_data = schema.validate(pd.read_csv(new_data_path))

    current_data.rename(columns={label_column: "target"}, inplace=True)
    historical_data.rename(columns={label_column: "target"}, inplace=True)

    return historical_data, current_data


def predict(model_endpoint: str, data: pd.DataFrame) -> np.ndarray:
    """Get the model output for the given data using the model endpoing."""
    payload = {"dataframe_records": data.to_dict(orient="records")}
    response = requests.post(model_endpoint, headers=headers, json=payload)
    predictions = np.array(response.json()["predictions"]).flatten()
    return predictions


if __name__ == "__main__":
    # load the config file
    config = load_yaml_config()

    feature_columns = config["feature_columns"]
    model_endpoint = os.getenv("MODEL_ENDPOINT", config["model_endpoint"])

    historical_data_save_path = Path(
        config["historical_data_save_path"]
    ).resolve()
    new_data_save_path = Path(config["new_data_save_path"]).resolve()

    # load the datasets
    historical_data, new_data = load_data(
        historical_data_save_path, new_data_save_path, config
    )

    # Model predictions for both datasets
    historical_data["prediction"] = predict(
        model_endpoint, historical_data[feature_columns]
    )
    new_data["prediction"] = predict(model_endpoint, new_data[feature_columns])

    historical_data.to_csv(historical_data_save_path, index=False)
    new_data.to_csv(new_data_save_path, index=False)
