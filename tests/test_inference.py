"""Unit tests for batch inference."""

from unittest.mock import ANY, patch

import numpy as np
import pandas as pd
import pytest
import requests

from src.inference import predict, transform_data_for_batch_processing

# define dummy endpoint, sample test data, and expected output
model_endpoint = "http://localhost/v2/models/house_price_prediction_prod/infer"
sample_data = pd.DataFrame(
    {
        "area": [1000, 500],
        "bedrooms": [3, 2],
        "bathrooms": [2, 1],
        "stories": [2, 3],
        "mainroad": ["YES", "NO"],
        "guestroom": ["YES", "NO"],
        "basement": ["YES", "NO"],
        "hotwaterheating": ["YES", "NO"],
        "airconditioning": ["YES", "NO"],
        "parking": [2, 1],
        "prefarea": ["YES", "NO"],
        "furnishingstatus": ["furnished", "unfurnished"],
    }
)
mock_predictions = np.array([[250000.0], [100000.0]])


@patch("src.inference.requests.post")
def test_predict_normal_case(mock_post):
    """Passing test case with mocked response."""
    # mock the requests.post response to mimic KServe V2 response structure
    mock_post.return_value.json.return_value = {
        "outputs": [{"data": mock_predictions.flatten().tolist()}]
    }

    result = predict(model_endpoint, sample_data)

    # assert that the result is numpy array
    assert isinstance(result, np.ndarray)

    # check if the response matches
    np.testing.assert_array_equal(result, np.array(mock_predictions).flatten())

    # transform the data and prepare the payload
    expected_payload = {
        "inputs": transform_data_for_batch_processing(sample_data)
    }

    # Check if the endpoint was called with the expected payload
    mock_post.assert_called_once_with(
        model_endpoint,
        headers=ANY,
        json=expected_payload,
    )


@patch("src.inference.requests.post")
def test_predict_incorrect_response_structure(mock_post):
    """Test with incorrect payload."""
    # Mocking an incorrect response structure
    mock_post.return_value.json.return_value = {"wrong_key": []}

    # Check if the function raises an exception or handles it gracefully
    with pytest.raises(ValueError):
        predict(model_endpoint, sample_data)


@patch("src.inference.requests.post")
def test_predict_missing_data_in_outputs(mock_post):
    """Test response with missing data key in outputs."""
    # Mocking a response with "outputs" but without "data" key
    mock_post.return_value.json.return_value = {"outputs": [{}]}

    # Expect the function to raise a ValueError due to missing data field
    with pytest.raises(ValueError):
        predict(model_endpoint, sample_data)


def test_predict_http_error():
    """Failed request (HTTP error)"""
    # Check if the function raises an exception when the request fails
    with pytest.raises(requests.exceptions.RequestException):
        # a request that will definitely fail (the endpoint doesn't exist)
        predict(model_endpoint, sample_data)
