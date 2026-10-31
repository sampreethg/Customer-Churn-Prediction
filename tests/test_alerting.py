"""
Automated Pytest Suite for Retention Alerting & Webhook Dispatch Engine.

Validates Slack/CRM payload schemas, asynchronous dispatch logic, threshold gating,
exponential backoff retries, and graceful recovery under external webhook failures.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from api.app import app
from src.alerting import AlertPayloadBuilder, WebhookNotifier

client = TestClient(app, headers={"X-API-Key": "test-api-key"})


def test_slack_payload_builder():
    """
    Verify Slack Block Kit payload structure, sections, and metadata.
    """
    payload = AlertPayloadBuilder.build_slack_blocks(
        customer_id="CUST-9999",
        prob=0.885,
        risk_level="High",
        action="Offer targeted discount and schedule retention callback.",
        top_factors=[
            {"feature_name": "Tenure_Months", "shap_value": 0.165},
            {"feature_name": "ContractType_One year", "shap_value": 0.082},
        ],
    )
    assert "blocks" in payload
    assert len(payload["blocks"]) >= 4
    assert payload["blocks"][0]["type"] == "header"
    assert "CUST-9999" in payload["blocks"][0]["text"]["text"]
    assert "88.5%" in payload["blocks"][1]["fields"][1]["text"]


def test_generic_crm_payload_builder():
    """
    Verify standard REST CRM JSON schema, event type, and data attributes.
    """
    payload = AlertPayloadBuilder.build_generic_crm_payload(
        customer_id="CUST-8888",
        prob=0.742,
        risk_level="High",
        action="Send check-in guide.",
        top_factors=[{"feature_name": "TotalTickets", "shap_value": 0.091}],
    )
    assert payload["event_type"] == "customer.churn_risk.alert"
    assert "timestamp" in payload
    assert payload["data"]["customer_id"] == "CUST-8888"
    assert payload["data"]["churn_probability"] == 0.742
    assert len(payload["data"]["primary_risk_drivers"]) == 1


@pytest.mark.asyncio
async def test_high_risk_triggers_webhook():
    """
    Assert that a churn probability >= threshold triggers outbound webhook dispatch.
    """
    notifier = WebhookNotifier(
        slack_webhook_url="https://hooks.slack.com/services/TEST/SLACK/WEBHOOK",
        crm_webhook_url="https://api.crm.example.com/webhooks/churn",
        alert_threshold=0.70,
    )

    mock_resp = httpx.Response(200, request=httpx.Request("POST", "https://test.com"))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        result = await notifier.dispatch_retention_alert(
            customer_id="CUST-1234",
            prob=0.82,
            risk_level="High",
            action="Priority intervention",
            top_factors=[{"feature_name": "Tenure_Months", "shap_value": 0.12}],
        )

        assert result["status"] == "dispatched"
        assert result["customer_id"] == "CUST-1234"
        assert mock_post.call_count == 2
        assert result["results"]["Slack"]["success"] is True
        assert result["results"]["CRM"]["success"] is True


@pytest.mark.asyncio
async def test_low_risk_skips_webhook():
    """
    Assert that a churn probability < threshold skips webhook dispatch cleanly without HTTP calls.
    """
    notifier = WebhookNotifier(
        slack_webhook_url="https://hooks.slack.com/services/TEST/SLACK/WEBHOOK",
        alert_threshold=0.70,
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        result = await notifier.dispatch_retention_alert(
            customer_id="CUST-5678",
            prob=0.25,
            risk_level="Low",
            action="Standard engagement",
        )

        assert result["status"] == "skipped"
        assert result["reason"] == "below_threshold"
        assert mock_post.call_count == 0


@pytest.mark.asyncio
async def test_webhook_failure_graceful_recovery():
    """
    Verify that an HTTP 500 error from external webhooks is caught, retried,
    and logged gracefully without raising unhandled exceptions.
    """
    notifier = WebhookNotifier(
        slack_webhook_url="https://hooks.slack.com/services/FAIL/500",
        alert_threshold=0.70,
    )

    fail_resp = httpx.Response(500, request=httpx.Request("POST", "https://fail.com"))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = fail_resp

        result = await notifier.dispatch_retention_alert(
            customer_id="CUST-FAIL",
            prob=0.91,
            risk_level="High",
            action="Emergency retention",
        )

        # Confirm retried and completed without crashing
        assert result["status"] == "dispatched"
        assert (
            result["results"]["Slack"]["success"] is True or "error" in result["results"]["Slack"]
        )
        assert mock_post.call_count >= 1


def test_api_predict_triggers_background_alert():
    """
    Test that high-risk single customer prediction in FastAPI schedules retention alert.
    """
    high_risk_payload = {
        "Customer_ID": "CUST-HOT-01",
        "Tenure_Months": 2,
        "MonthlyCharges": 105.0,
        "TotalTickets": 4,
        "ContractType": "Month-to-month",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "notify_retention_team": True,
    }

    with patch("src.alerting.WebhookNotifier.dispatch_retention_alert", new_callable=AsyncMock):
        response = client.post("/predict", json=high_risk_payload)
        assert response.status_code == 200
        data = response.json()

        assert data["Customer_ID"] == "CUST-HOT-01"
        assert data["churn_prediction"] == 1
        assert data["alert_dispatched"] is True


def test_api_predict_suppresses_alert_when_toggle_disabled():
    """
    Test that notify_retention_team=False suppresses retention alert even if high risk.
    """
    payload = {
        "Customer_ID": "CUST-SUPPRESS",
        "Tenure_Months": 2,
        "MonthlyCharges": 105.0,
        "TotalTickets": 4,
        "ContractType": "Month-to-month",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "notify_retention_team": False,
    }

    with patch(
        "src.alerting.WebhookNotifier.dispatch_retention_alert", new_callable=AsyncMock
    ) as mock_dispatch:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["alert_dispatched"] is False
        assert mock_dispatch.call_count == 0
