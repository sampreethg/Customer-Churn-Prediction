"""
Phase 4 Local Service Smoke Tests
Starts FastAPI server in background, tests endpoints, and terminates cleanly.
"""
import subprocess
import sys
import time
import requests

def main():
    print("==================================================")
    print("        PHASE 4: LOCAL SERVICE SMOKE TESTS        ")
    print("==================================================")

    # 1. Start uvicorn process
    print("\n[1/4] Starting FastAPI server in background...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait up to 15s for server to become responsive
        url_health = "http://127.0.0.1:8000/health"
        server_ready = False
        for attempt in range(15):
            time.sleep(1)
            try:
                r = requests.get(url_health, timeout=2)
                if r.status_code == 200:
                    server_ready = True
                    break
            except Exception:
                pass

        if not server_ready:
            raise RuntimeError("FastAPI server failed to start within 15 seconds.")

        # 2. Test /health
        print("\n[2/4] Verifying /health probe...")
        r_health = requests.get(url_health, timeout=5)
        print(f"Status Code: {r_health.status_code}")
        print(f"Payload: {r_health.json()}")
        assert r_health.status_code == 200
        health_data = r_health.json()
        assert health_data.get("status") == "healthy"
        assert "model_loaded" in health_data

        # 3. Test /predict
        print("\n[3/4] Verifying /predict inference endpoint...")
        sample_customer = {
            "Customer_ID": "CUST-SMOKE-01",
            "Tenure_Months": 12,
            "MonthlyCharges": 85.0,
            "TotalTickets": 4,
            "ContractType": "Month-to-month",
            "TechSupport": "No",
            "PaymentMethod": "Electronic check",
            "Age": 45,
            "PaperlessBilling": "Yes"
        }
        url_predict = "http://127.0.0.1:8000/predict"
        r_predict = requests.post(url_predict, json=sample_customer, timeout=5)
        print(f"Status Code: {r_predict.status_code}")
        pred_data = r_predict.json()
        print(f"Payload: {pred_data}")
        assert r_predict.status_code == 200
        assert "churn_prediction" in pred_data
        assert "churn_probability" in pred_data
        assert "risk_level" in pred_data
        assert "recommended_action" in pred_data
        print(f"-> Churn Prediction: {pred_data['churn_prediction']}")
        print(f"-> Churn Probability: {pred_data['churn_probability']:.4f}")
        print(f"-> Risk Tier: {pred_data['risk_level']}")
        print(f"-> Prescriptive Action: {pred_data['recommended_action']}")

        # 4. Test /explain
        print("\n[4/4] Verifying /explain SHAP attribution endpoint...")
        url_explain = "http://127.0.0.1:8000/explain"
        r_explain = requests.post(url_explain, json=sample_customer, timeout=5)
        print(f"Status Code: {r_explain.status_code}")
        explain_data = r_explain.json()
        print(f"Payload: {explain_data}")
        assert r_explain.status_code == 200
        assert "base_value" in explain_data
        assert "top_risk_drivers" in explain_data
        assert "top_retention_drivers" in explain_data
        assert "summary_narrative" in explain_data
        assert len(explain_data["top_risk_drivers"]) > 0
        print(f"-> Base Value: {explain_data['base_value']}")
        print(f"-> Top Risk Drivers: {explain_data['top_risk_drivers']}")
        print(f"-> Top Retention Drivers: {explain_data['top_retention_drivers']}")
        print(f"-> Summary Narrative: {explain_data['summary_narrative']}")

        print("\n==================================================")
        print("    [SUCCESS] ALL LOCAL SMOKE TESTS PASSED!       ")
        print("==================================================")

    finally:
        print("\nTerminating background uvicorn server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("FastAPI background process cleanly terminated.")

if __name__ == "__main__":
    main()

