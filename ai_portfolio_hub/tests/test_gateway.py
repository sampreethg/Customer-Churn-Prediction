"""
Integration Tests for Unified Multi-Domain AI Portfolio Gateway.

Tests endpoints across all 4 domains:
- GET /health
- POST /api/v1/tabular/predict
- POST /api/v1/vision/classify
- POST /api/v1/nlp/analyze
- POST /api/v1/rl/playback
"""

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai_portfolio_hub.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers={"X-API-Key": "test-api-key"}) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health returns 200 and telemetry data."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "device" in data
    assert "models_loaded" in data
    assert "uptime_seconds" in data


def test_tabular_churn_prediction(client):
    """Verify /api/v1/tabular/predict returns churn probabilities and SHAP values."""
    payload = {
        "Tenure_Months": 12,
        "MonthlyCharges": 85.0,
        "TotalTickets": 3,
        "ContractType": "Month-to-month",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "Age": 45,
        "PaperlessBilling": "Yes",
    }
    response = client.post("/api/v1/tabular/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "recommended_action" in data
    assert "shap_values" in data
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_vision_image_classification(client):
    """Verify /api/v1/vision/classify accepts image upload and returns predictions."""
    # Generate test 32x32 image
    img = np.full((32, 32, 3), [100, 150, 200], dtype=np.uint8)
    cv2.circle(img, (16, 16), 8, (255, 255, 255), -1)
    pil_img = Image.fromarray(img)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test_circle.png", buf.getvalue(), "image/png")}
    response = client.post("/api/v1/vision/classify", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "top_class" in data
    assert "top_confidence" in data
    assert "predictions" in data
    assert len(data["predictions"]) > 0


def test_nlp_sentiment_analysis(client):
    """Verify /api/v1/nlp/analyze returns classification label and confidence scores."""
    payload = {"text": "The platform exceeded our expectations! Highly recommended."}
    response = client.post("/api/v1/nlp/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["POSITIVE", "NEGATIVE"]
    assert "confidence" in data
    assert "probabilities" in data
    assert "latency_ms" in data


def test_rl_cartpole_playback(client):
    """Verify /api/v1/rl/playback runs CartPole simulation and records trajectories."""
    payload = {"episodes": 1, "seed": 42}
    response = client.post("/api/v1/rl/playback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "total_reward" in data
    assert "steps" in data
    assert "state_history" in data
    assert "q_values_history" in data
    assert data["steps"] > 0
    assert len(data["state_history"]) == data["steps"]
    assert len(data["state_history"][0]) == 4 # 4D state vector


def test_rag_ingest_and_query(client):
    """Verify RAG document ingestion and grounded question answering."""
    # Test Ingest
    ingest_payload = {
        "doc_id": "test_policy",
        "text": "Model safety guardrails mandate strict adherence to privacy rules. All logs are purged after 90 days.",
    }
    resp_ingest = client.post("/api/v1/rag/ingest", json=ingest_payload)
    assert resp_ingest.status_code == 200
    ingest_data = resp_ingest.json()
    assert ingest_data["chunks_created"] >= 1
    assert ingest_data["total_indexed_chunks"] >= 1

    # Test Query
    query_payload = {
        "question": "What is the policy for log purging and privacy?",
        "dense_top_k": 4,
        "rerank_top_n": 2,
    }
    resp_query = client.post("/api/v1/rag/query", json=query_payload)
    assert resp_query.status_code == 200
    query_data = resp_query.json()
    assert "answer" in query_data
    assert "citations" in query_data
    assert "confidence_score" in query_data
    assert "latency_ms" in query_data

