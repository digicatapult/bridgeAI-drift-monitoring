"""Unit tests for inference."""

from unittest.mock import ANY, patch

import numpy as np
import pandas as pd
import pytest
import requests

from src.inference import predict

# Sample test data and expected output
model_endpoint = "http://test-endpoint.com/predict"
sample_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
mock_predictions = np.array([[0.9], [0.1], [0.8]])


@patch("src.inference.requests.post")
def test_predict_normal_case(mock_post):
    """Normal case with mocked response."""
    # Mocking the requests.post response
    mock_post.return_value.json.return_value = {
        "predictions": mock_predictions
    }

    # Call the predict function
    result = predict(model_endpoint, sample_data)

    # Assert that the result is an ndarray
    assert isinstance(result, np.ndarray)

    # Assert that the result is flattened ndarray
    np.testing.assert_array_equal(result, np.array(mock_predictions).flatten())

    # Check if the mocked requests.post was called with the expected arguments
    mock_post.assert_called_once_with(
        model_endpoint,
        headers=ANY,
        json={"dataframe_records": sample_data.to_dict(orient="records")},
    )


@patch("src.inference.requests.post")
def test_predict_incorrect_response_structure(mock_post):
    """Test with incorrect payload."""
    # Mocking an incorrect response structure
    mock_post.return_value.json.return_value = {"wrong_key": []}

    # Check if the function raises an exception or handles it gracefully
    with pytest.raises(KeyError):
        predict(model_endpoint, sample_data)


def test_predict_http_error():
    """Failed request (HTTP error)"""
    # Check if the function raises an exception when the request fails
    with pytest.raises(requests.exceptions.RequestException):
        # a request that will definitely fail (the endpoint doesn't exist)
        predict(model_endpoint, sample_data)
