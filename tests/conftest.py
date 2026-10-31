"""
Shared Pytest Fixtures for Customer Churn Prediction Test Suites.

Provides reusable fixtures for mock data payloads, trained pipelines,
and FastAPI test clients with lifespans enabled.
"""

from typing import Any, Dict, List

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import app
from src.pipeline import build_pipeline


@pytest.fixture(scope="session")
def sample_customer_record() -> Dict[str, Any]:
    """
    Provide a canonical single customer record with valid schema types and boundaries.
    """
    return {
        "Tenure_Months": 12,
        "MonthlyCharges": 70.5,
        "TotalTickets": 2,
        "ContractType": "One year",
        "TechSupport": "Yes",
        "PaymentMethod": "Credit card",
        "Age": 42,
        "PaperlessBilling": "Yes",
    }


@pytest.fixture(scope="session")
def sample_batch_records() -> List[Dict[str, Any]]:
    """
    Provide a representative batch of customer records spanning Low, Medium, and High risk profiles.
    """
    return [
        {
            "Tenure_Months": 2,
            "MonthlyCharges": 98.5,
            "TotalTickets": 5,
            "ContractType": "Month-to-month",
            "TechSupport": "No",
            "PaymentMethod": "Electronic check",
            "Age": 28,
            "PaperlessBilling": "Yes",
        },
        {
            "Tenure_Months": 36,
            "MonthlyCharges": 65.0,
            "TotalTickets": 1,
            "ContractType": "One year",
            "TechSupport": "Yes",
            "PaymentMethod": "Bank transfer",
            "Age": 45,
            "PaperlessBilling": "No",
        },
        {
            "Tenure_Months": 68,
            "MonthlyCharges": 25.0,
            "TotalTickets": 0,
            "ContractType": "Two year",
            "TechSupport": "Yes",
            "PaymentMethod": "Credit card",
            "Age": 62,
            "PaperlessBilling": "No",
        },
    ]


@pytest.fixture(scope="session")
def mock_trained_pipeline():
    """
    Provide a fitted pipeline instance trained on deterministic synthetic samples
    for unit testing isolated pipeline operations without depending on disk artifacts.
    """
    from src.config import ALL_FEATURES

    # Create minimal deterministic training data
    data = {
        "Tenure_Months": [1, 24, 60, 4, 36, 72, 8, 12, 48, 6],
        "MonthlyCharges": [85.0, 45.0, 20.0, 105.0, 60.0, 25.0, 95.0, 50.0, 30.0, 110.0],
        "TotalTickets": [4, 1, 0, 5, 2, 0, 3, 2, 0, 4],
        "Age": [25, 45, 65, 30, 50, 70, 35, 40, 55, 28],
        "ContractType": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
        ],
        "TechSupport": [
            "No",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "Yes",
            "No",
        ],
        "PaymentMethod": [
            "Electronic check",
            "Credit card",
            "Bank transfer",
            "Electronic check",
            "Mailed check",
            "Credit card",
            "Electronic check",
            "Credit card",
            "Bank transfer",
            "Electronic check",
        ],
        "PaperlessBilling": [
            "Yes",
            "No",
            "No",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "No",
            "No",
            "Yes",
        ],
    }
    df = pd.DataFrame(data)[ALL_FEATURES]
    y = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]

    pipe = build_pipeline(n_estimators=10, max_depth=3, random_state=42)
    pipe.fit(df, y)
    return pipe


@pytest.fixture(scope="session")
def test_client():
    """
    Provide a FastAPI TestClient with application lifespan context manager triggered.
    """
    with TestClient(app, headers={"X-API-Key": "test-api-key"}) as client:
        yield client
