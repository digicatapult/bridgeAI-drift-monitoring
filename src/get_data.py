"""Get the data to be compared for drift detection."""

import os

import pandas as pd
from pandera import Column, DataFrameSchema

data_repo = os.getenv("DATA_REPO")
repo_branch = os.getenv("REPO_BRANCH")
current_data_version = os.getenv("CURRENT_DATA_VERSION")
new_data_version = os.getenv("NEW_DATA_VERSION")


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


def load_data(current_data_local_path, new_data_local_path, label_column):
    """Load current(new) data and historical(used for training) data."""

    historical_data = schema.validate(pd.read_csv(current_data_local_path))

    # Current data (new incoming data)
    current_data = schema.validate(pd.read_csv(new_data_local_path))

    current_data.rename(columns={label_column: "target"}, inplace=True)
    historical_data.rename(columns={label_column: "target"}, inplace=True)

    return historical_data, current_data
