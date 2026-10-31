"""
Automated Security Remediation Verification Suite.

Comprehensive security tests validating:
1. Tiered In-Memory Sliding-Window Rate Limiting (HTTP 429 and Retry-After).
2. API Key and Bearer Token Authentication (HTTP 401 on missing/invalid keys, 200 on authorized keys).
3. SSRF Protection on Webhooks (HTTPS enforcement, RFC 1918/link-local/loopback rejection, domain allowlisting).
4. Safe PyTorch Deserialization (weights_only=True enforcement).
"""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from api.app import app
from src.alerting import validate_webhook_url
from src.rate_limiter import rate_limiter

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def reset_rate_limiter_buckets():
    """Clear in-memory rate limiter buckets before each test run."""
    rate_limiter._buckets.clear()
    rate_limiter.enabled = True
    yield
    rate_limiter._buckets.clear()


SAMPLE_PAYLOAD = {
    "Tenure_Months": 6,
    "MonthlyCharges": 85.0,
    "TotalTickets": 3,
    "ContractType": "Month-to-month",
    "TechSupport": "No",
    "PaymentMethod": "Electronic check",
    "Age": 35,
    "PaperlessBilling": "Yes",
}


def test_health_probe_unauthenticated():
    """Verify /health remains accessible without API credentials for orchestrators."""
    unauth_client = TestClient(app)
    resp = unauth_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_auth_protection_rejects_unauthenticated_requests():
    """Verify endpoints reject requests lacking authentication credentials with HTTP 401."""
    unauth_client = TestClient(app)

    predict_resp = unauth_client.post("/predict", json=SAMPLE_PAYLOAD)
    assert predict_resp.status_code == 401
    assert "Invalid or missing API key" in predict_resp.json()["detail"]

    batch_resp = unauth_client.post("/predict/batch", json={"customers": [SAMPLE_PAYLOAD]})
    assert batch_resp.status_code == 401

    explain_resp = unauth_client.post("/explain", json=SAMPLE_PAYLOAD)
    assert explain_resp.status_code == 401


def test_auth_protection_rejects_invalid_api_key():
    """Verify endpoints reject forged or unauthorized credentials with HTTP 401."""
    invalid_client = TestClient(app, headers={"X-API-Key": "unauthorized-malicious-key"})
    resp = invalid_client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 401
    assert "Invalid API key supplied" in resp.json()["detail"]


def test_auth_protection_accepts_valid_x_api_key():
    """Verify authorized request via X-API-Key header succeeds with HTTP 200."""
    valid_client = TestClient(app, headers={"X-API-Key": "test-api-key"})
    resp = valid_client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    assert "churn_probability" in resp.json()


def test_auth_protection_accepts_valid_bearer_token():
    """Verify authorized request via Authorization: Bearer token succeeds with HTTP 200."""
    valid_client = TestClient(app, headers={"Authorization": "Bearer test-api-key"})
    resp = valid_client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    assert "churn_probability" in resp.json()


def test_rate_limiting_burst_explain_endpoint():
    """
    Verify burst requests against compute-heavy /explain (10 req/min limit)
    trigger HTTP 429 Too Many Requests with Retry-After header on the 11th call.
    """
    valid_client = TestClient(app, headers={"X-API-Key": "test-api-key"})

    # Send 10 successful requests (quota limit)
    for i in range(10):
        resp = valid_client.post("/explain", json=SAMPLE_PAYLOAD)
        assert resp.status_code == 200, f"Request {i + 1} failed unexpectedly with status {resp.status_code}"

    # 11th request must exceed quota and return HTTP 429
    excess_resp = valid_client.post("/explain", json=SAMPLE_PAYLOAD)
    assert excess_resp.status_code == 429
    assert "Retry-After" in excess_resp.headers
    retry_after = int(excess_resp.headers["Retry-After"])
    assert retry_after >= 1

    detail = excess_resp.json()["detail"]
    assert "Rate limit exceeded" in detail["error"]
    assert detail["retry_after"] == retry_after


def test_ssrf_rejects_non_https_schemes():
    """Verify webhook validator rejects non-HTTPS schemes (http, file, gopher, ftp)."""
    with pytest.raises(ValueError, match="Only 'https' is permitted"):
        validate_webhook_url("http://hooks.slack.com/services/T00/B00/X00")

    with pytest.raises(ValueError, match="Only 'https' is permitted"):
        validate_webhook_url("file:///etc/passwd")

    with pytest.raises(ValueError, match="Only 'https' is permitted"):
        validate_webhook_url("ftp://hooks.slack.com/webhook")


def test_ssrf_rejects_private_and_metadata_ip_targets():
    """Verify webhook validator strictly blocks RFC 1918, link-local metadata, and loopback IPs."""
    # Cloud metadata service (AWS/GCP/Azure)
    with pytest.raises(ValueError, match="forbidden non-global or private IP"):
        validate_webhook_url("https://169.254.169.254/latest/meta-data")

    # Localhost loopback
    with pytest.raises(ValueError, match="forbidden non-global or private IP"):
        validate_webhook_url("https://127.0.0.1:8000/webhook")

    # RFC 1918 subnets
    with pytest.raises(ValueError, match="forbidden non-global or private IP"):
        validate_webhook_url("https://10.0.0.1/webhook")

    with pytest.raises(ValueError, match="forbidden non-global or private IP"):
        validate_webhook_url("https://192.168.1.1/webhook")

    with pytest.raises(ValueError, match="forbidden non-global or private IP"):
        validate_webhook_url("https://172.16.0.1/webhook")


def test_ssrf_rejects_unapproved_external_domains():
    """Verify webhook validator blocks unapproved external domains."""
    with pytest.raises(ValueError, match="not in the allowed domains"):
        validate_webhook_url("https://malicious-exfiltration-target.org/hook")


def test_ssrf_permits_allowlisted_domains():
    """Verify webhook validator allows default allowlisted domains."""
    # Slack and Discord allowed endpoints
    validate_webhook_url("https://hooks.slack.com/services/T00/B00/X00")
    validate_webhook_url("https://discord.com/api/webhooks/123/abc")
    # CRM test domain allowed
    validate_webhook_url("https://api.crm.example.com/webhooks/churn")


def test_safe_pytorch_deserialization_weights_only(tmp_path):
    """
    Verify safe deserialization using weights_only=True:
    1. Loads standard tensor weights without error.
    2. Blocks arbitrary pickled code execution payloads.
    """
    # 1. Test existing model artifact weights load with weights_only=True
    model_path = WORKSPACE_ROOT / "ai_portfolio_hub" / "models" / "best_per_d3qn.pt"
    if model_path.is_file():
        weights = torch.load(model_path, map_location="cpu", weights_only=True)
        assert isinstance(weights, dict)
        assert len(weights) > 0

    # 2. Test that weights_only=True rejects arbitrary executable classes
    class MaliciousExploit:
        def __reduce__(self):
            import os
            return (os.system, ("echo exploit",))

    exploit_payload = {"model_state": MaliciousExploit()}
    exploit_path = tmp_path / "exploit.pt"
    torch.save(exploit_payload, exploit_path)
    import pickle

    with pytest.raises((pickle.UnpicklingError, RuntimeError, ValueError)):
        torch.load(exploit_path, weights_only=True)
