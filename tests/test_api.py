"""
Automated Pytest Suite for FastAPI Customer Churn Prediction Service.

Tests health check, single customer inference, Pydantic field validation boundaries,
batch prediction throughput, SHAP explainability endpoint, and error handling.
"""

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, headers={"X-API-Key": "test-api-key"})


def test_health_endpoint():
    """
    Verify health probe responds with HTTP 200 and indicates model and explainer are loaded.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["explainer_loaded"] is True
    assert "model_artifact" in data


def test_health_check():
    """
    Alias test verifying /health backward compatibility.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_payload():
    """
    Verify single customer prediction returns valid schema, risk tier, and recommendation.
    """
    payload = {
        "Tenure_Months": 3,
        "MonthlyCharges": 99.5,
        "TotalTickets": 4,
        "ContractType": "Month-to-month",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert isinstance(data["recommended_action"], str)
    assert len(data["recommended_action"]) > 0


def test_single_prediction_valid():
    """
    Alias test verifying /predict with valid payload.
    """
    test_predict_valid_payload()


def test_predict_validation_error():
    """
    Verify that invalid inputs (e.g., negative tenure, out-of-range values) trigger HTTP 422.
    """
    payload = {
        "Tenure_Months": -5, # Constraint: ge=0
        "MonthlyCharges": 50.0,
        "TotalTickets": 0,
        "ContractType": "One year",
        "TechSupport": "Yes",
        "PaymentMethod": "Bank transfer",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_single_prediction_validation_failure():
    """
    Alias test verifying validation error behavior.
    """
    test_predict_validation_error()


def test_predict_invalid_enum():
    """
    Verify that an illegal categorical enum value triggers HTTP 422.
    """
    payload = {
        "Tenure_Months": 12,
        "MonthlyCharges": 65.0,
        "TotalTickets": 1,
        "ContractType": "Five year", # Invalid enum
        "TechSupport": "Yes",
        "PaymentMethod": "Credit card",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_prediction():
    """
    Verify batch prediction processes multiple customer profiles and returns structured list.
    """
    payload = {
        "customers": [
            {
                "Tenure_Months": 2,
                "MonthlyCharges": 105.0,
                "TotalTickets": 5,
                "ContractType": "Month-to-month",
                "TechSupport": "No",
                "PaymentMethod": "Electronic check",
            },
            {
                "Tenure_Months": 60,
                "MonthlyCharges": 25.0,
                "TotalTickets": 0,
                "ContractType": "Two year",
                "TechSupport": "Yes",
                "PaymentMethod": "Credit card",
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2

    cust_high = data["predictions"][0]
    assert cust_high["churn_prediction"] == 1
    assert cust_high["risk_level"] in ["Medium", "High"]

    cust_low = data["predictions"][1]
    assert cust_low["churn_prediction"] == 0
    assert cust_low["risk_level"] == "Low"


def test_empty_batch_validation():
    """
    Verify that an empty customer batch triggers HTTP 400.
    """
    payload = {"customers": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400


def test_explain_endpoint_valid():
    """
    Verify /explain endpoint computes and returns structured SHAP contributions.
    """
    payload = {
        "Tenure_Months": 2,
        "MonthlyCharges": 105.0,
        "TotalTickets": 4,
        "ContractType": "Month-to-month",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "churn_probability" in data
    assert "base_value" in data
    assert "top_risk_drivers" in data
    assert "top_retention_drivers" in data
    assert "summary_narrative" in data

    assert 0.0 <= data["churn_probability"] <= 1.0
    assert len(data["top_risk_drivers"]) > 0
    first_driver = data["top_risk_drivers"][0]
    assert "feature_name" in first_driver
    assert "shap_value" in first_driver
    assert first_driver["shap_value"] > 0
    assert isinstance(data["summary_narrative"], str)
    assert len(data["summary_narrative"]) > 0


def test_explain_endpoint_validation_error():
    """
    Verify that invalid customer input to /explain triggers HTTP 422.
    """
    payload = {
        "Tenure_Months": -1,
        "MonthlyCharges": 50.0,
        "TotalTickets": 2,
        "ContractType": "Month-to-month",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 422


def test_lifespan_context_and_fixtures(test_client, sample_customer_record):
    """
    Verify that application lifespan initializes model and explainer instances
    and handles customer prediction queries via fixture-injected client.
    """
    health = test_client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    resp = test_client.post("/predict", json=sample_customer_record)
    assert resp.status_code == 200
    assert "churn_probability" in resp.json()
