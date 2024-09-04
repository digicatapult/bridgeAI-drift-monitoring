"""Inference on reference data using model prediction endpoints."""

import numpy as np
import pandas as pd
import requests

headers = {"Content-Type": "application/json"}


def predict(model_endpoint, data: pd.DataFrame):
    """Get the model output for the given data using the model endpoing."""
    payload = {"dataframe_records": data.to_dict(orient="records")}
    response = requests.post(model_endpoint, headers=headers, json=payload)
    predictions = np.array(response.json()["predictions"]).flatten()
    return predictions
