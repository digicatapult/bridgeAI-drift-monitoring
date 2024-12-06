"""Unit tests for batch inference."""

from unittest.mock import call, patch

import numpy as np
import pandas as pd
import pytest
import requests

from src.inference import predict, prepare_single_record_payload

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
mock_predictions = np.array([250000.0, 100000.0])


class MockResponse:
    def __init__(self, prediction, status_code=200):
        self._prediction = prediction
        self.status_code = status_code

    def json(self):
        return {
            "status": 200,
            "message": "House price prediction successful",
            "response": {"prediction": self._prediction, "unit": "GBP(£)"},
        }

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Error", response=self
            )


@patch("src.inference.requests.post")
def test_predict_normal_case(mock_post):
    """Passing test case with mocked response."""

    # Set the side_effect to return a different MockResponse for each call
    mock_post.side_effect = [
        MockResponse(mock_predictions[0]),
        MockResponse(mock_predictions[1]),
    ]

    result = predict(model_endpoint, sample_data)

    # assert that the result is numpy array
    assert isinstance(result, np.ndarray)

    # check if the response matches
    np.testing.assert_array_equal(result, np.array(mock_predictions))

    # Prepare the expected payload for the single record
    expected_payload_first = prepare_single_record_payload(sample_data.iloc[0])
    expected_payload_second = prepare_single_record_payload(
        sample_data.iloc[1]
    )

    # Check calls to the mock post
    mock_post.assert_has_calls(
        [
            call(
                model_endpoint,
                json=expected_payload_first,
                headers={"Content-Type": "application/json"},
            ),
            call(
                model_endpoint,
                json=expected_payload_second,
                headers={"Content-Type": "application/json"},
            ),
        ],
        any_order=False,
    )


@patch("src.inference.requests.post")
def test_predict_incorrect_response_structure(mock_post):
    """Test with incorrect payload."""
    # Mocking a response with wrong key
    mock_post.return_value.json.return_value = {
        "status": 200,
        "response": {"wrong_key": []},
    }

    # Check if the function raises a ValueError due to missing "prediction"
    with pytest.raises(KeyError):
        predict(model_endpoint, sample_data)


@patch("src.inference.requests.post")
def test_predict_invalid_prediction(mock_post):
    """Test response missing the 'prediction' key."""
    # Mocking a response with invalid "prediction" value
    mock_post.return_value.json.return_value = {
        "status": 200,
        "response": {"prediction": "NA"},
    }

    # Expect the function to raise a ValueError due invalid prediction
    with pytest.raises(ValueError):
        predict(model_endpoint, sample_data)


def test_predict_http_error():
    """Failed request (HTTP error)"""
    # Check if the function raises an exception when the request fails
    with pytest.raises(requests.exceptions.RequestException):
        # a request that will definitely fail (the endpoint doesn't exist)
        predict(model_endpoint, sample_data)
