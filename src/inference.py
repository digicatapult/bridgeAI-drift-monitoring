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


def transform_data_for_batch_processing(data: pd.DataFrame) -> list:
    """Transform batch of data for kserve batch inference."""
    # Map pandas dtypes to Kserve Inference Server data types
    dtype_map = {
        "float64": "FP64",
        "float32": "FP32",
        "int64": "INT64",
        "int32": "INT32",
        "int16": "INT16",
        "int8": "INT8",
        "object": "BYTES",
        "bool": "BOOL",
    }
    batch_size = len(data)
    inputs = []

    # Iteratively prepare the list of inputs by column
    for column_name in data.columns:
        column_data = data[column_name]
        # Get the data type mapping for Kserve
        pandas_dtype = str(column_data.dtype)
        kserve_dtype = dtype_map.get(pandas_dtype)
        if kserve_dtype is None:
            raise ValueError(
                f"Unsupported data type '{pandas_dtype}' "
                f"for column '{column_name}'"
            )

        # Prepare individual input tensor
        input_tensor = {
            "name": column_name,
            "shape": [batch_size],
            "datatype": kserve_dtype,
            "data": column_data.tolist(),
        }
        inputs.append(input_tensor)
    return inputs


def predict(model_endpoint: str, data: pd.DataFrame) -> np.ndarray:
    """Get the model output for batch data using the KServe model endpoint."""
    payload = {"inputs": transform_data_for_batch_processing(data)}

    headers = {
        "Content-Type": "application/json",
    }
    print(
        f"Sending batch inference to the model endpoint: {model_endpoint} "
        f"for {len(data)} data points"
    )

    # POST request
    response = requests.post(model_endpoint, headers=headers, json=payload)
    response.raise_for_status()

    # Extract predictions from the response's "outputs" key
    outputs = response.json().get("outputs")
    if outputs is None:
        raise ValueError("No outputs found in the response.")

    # Assuming the output is a single tensor
    predictions = outputs[0].get("data")
    if predictions is None:
        raise ValueError("No predictions found in the outputs.")

    return np.array(predictions)


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
