"""
Root Test Suite Runner forwarding to tests/test_api.py.
"""

from tests.test_api import (
    test_batch_prediction,
    test_empty_batch_validation,
    test_explain_endpoint_valid,
    test_explain_endpoint_validation_error,
    test_health_check,
    test_health_endpoint,
    test_predict_invalid_enum,
    test_predict_valid_payload,
    test_predict_validation_error,
    test_single_prediction_valid,
    test_single_prediction_validation_failure,
)

__all__ = [
    "test_health_endpoint",
    "test_health_check",
    "test_predict_valid_payload",
    "test_single_prediction_valid",
    "test_predict_validation_error",
    "test_single_prediction_validation_failure",
    "test_predict_invalid_enum",
    "test_batch_prediction",
    "test_empty_batch_validation",
    "test_explain_endpoint_valid",
    "test_explain_endpoint_validation_error",
]
