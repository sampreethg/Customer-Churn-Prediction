"""Live Integration Smoke-Testing & API Verification Suite for AI Portfolio Hub.

Sends concurrent structured payloads across all four live gateway endpoints:
1. /api/v1/tabular/predict - Verifies risk tiers, probabilities, and SHAP attribution dictionaries.
2. /api/v1/vision/classify - Uploads synthetic image bytes, asserts classification and Base64 Grad-CAM integrity.
3. /api/v1/nlp/analyze - Evaluates review texts, checks Softmax probabilities and bounds.
4. /api/v1/rl/playback - Simulates CartPole episode, validates trajectory dimensions and Q-value shapes.
"""

import argparse
import base64
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
import numpy as np
from PIL import Image

# Terminal ANSI Colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(title: str):
    print(f"\n{BOLD}{BLUE}{'=' * 65}{RESET}")
    print(f"{BOLD}{BLUE} {title}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 65}{RESET}")


def test_health(base_url: str) -> bool:
    print(f"{YELLOW}[PROBE 0/4]{RESET} Checking Gateway Health & Telemetry...")
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{base_url}/health")
            latency = (time.perf_counter() - start) * 1000
            if resp.status_code != 200:
                print(f"  {RED}FAIL{RESET}: /health returned HTTP {resp.status_code}")
                return False
            data = resp.json()
            print(f"  {GREEN}PASS{RESET} (Latency: {latency:.1f}ms)")
            print(f"        Status: {data.get('status')} | Device: {data.get('device')}")
            print(f"        Models: {data.get('models_loaded')}")
            return True
    except Exception as exc:
        print(f"  {RED}ERROR{RESET}: Health check failed with error: {exc}")
        return False


def test_tabular(base_url: str) -> dict:
    start = time.perf_counter()
    payload = {
        "Tenure_Months": 18,
        "MonthlyCharges": 92.5,
        "TotalTickets": 4,
        "ContractType": "Month-to-month",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "Age": 52,
        "PaperlessBilling": "Yes",
    }
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(f"{base_url}/api/v1/tabular/predict", json=payload)
        latency = (time.perf_counter() - start) * 1000

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "churn_probability" in data, "Missing 'churn_probability' key"
        assert 0.0 <= data["churn_probability"] <= 1.0, f"Probability out of range: {data['churn_probability']}"
        assert "risk_tier" in data, "Missing 'risk_tier' key"
        assert "actionable_recommendation" in data, "Missing 'actionable_recommendation' key"
        assert "shap_attributions" in data, "Missing 'shap_attributions' key"
        assert isinstance(data["shap_attributions"], dict), "SHAP attributions must be a dictionary"

        return {
            "endpoint": "/api/v1/tabular/predict",
            "status": "PASS",
            "latency_ms": latency,
            "details": f"Churn Prob: {data['churn_probability']:.3f} | Tier: {data['risk_tier']} | Features Explained: {len(data['shap_attributions'])}",
        }


def test_vision(base_url: str) -> dict:
    start = time.perf_counter()
    # Generate 32x32 synthetic RGB image
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("synthetic_sample.png", buf, "image/png")}
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(f"{base_url}/api/v1/vision/classify", files=files)
        latency = (time.perf_counter() - start) * 1000

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "predicted_class" in data, "Missing 'predicted_class'"
        assert "confidence" in data, "Missing 'confidence'"
        assert 0.0 <= data["confidence"] <= 1.0, "Confidence out of bounds"
        assert "top_k_probabilities" in data, "Missing 'top_k_probabilities'"
        assert len(data["top_k_probabilities"]) == 10, "Expected 10 CIFAR class probabilities"
        assert "gradcam_base64" in data, "Missing 'gradcam_base64'"

        # Verify Base64 string decode
        b64_str = data["gradcam_base64"]
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        decoded = base64.b64decode(b64_str)
        assert len(decoded) > 100, "Base64 Grad-CAM decoded byte buffer is corrupt or empty"

        return {
            "endpoint": "/api/v1/vision/classify",
            "status": "PASS",
            "latency_ms": latency,
            "details": f"Class: {data['predicted_class']} | Conf: {data['confidence']:.3f} | Heatmap Bytes: {len(decoded)}B",
        }


def test_nlp(base_url: str) -> dict:
    start = time.perf_counter()
    payload = {
        "text": "The platform analytics and real-time dashboard are extraordinarily intuitive, highly responsive, and robust!",
        "confidence_threshold": 0.5,
    }
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(f"{base_url}/api/v1/nlp/analyze", json=payload)
        latency = (time.perf_counter() - start) * 1000

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "sentiment" in data, "Missing 'sentiment'"
        assert data["sentiment"] in ["POSITIVE", "NEGATIVE"], f"Invalid sentiment: {data['sentiment']}"
        assert "confidence" in data, "Missing 'confidence'"
        assert 0.0 <= data["confidence"] <= 1.0, "Confidence out of bounds"
        assert "probabilities" in data, "Missing 'probabilities'"
        prob_sum = sum(data["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-3, f"Softmax probabilities do not sum to 1.0: {prob_sum}"

        return {
            "endpoint": "/api/v1/nlp/analyze",
            "status": "PASS",
            "latency_ms": latency,
            "details": f"Sentiment: {data['sentiment']} | Conf: {data['confidence']:.3f} | Latency: {data.get('inference_latency_ms', 0):.1f}ms",
        }


def test_rl(base_url: str) -> dict:
    start = time.perf_counter()
    payload = {
        "max_steps": 500,
        "seed": 42,
    }
    with httpx.Client(timeout=25.0) as client:
        resp = client.post(f"{base_url}/api/v1/rl/playback", json=payload)
        latency = (time.perf_counter() - start) * 1000

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "total_reward" in data, "Missing 'total_reward'"
        assert "steps_completed" in data, "Missing 'steps_completed'"
        assert "state_trajectories" in data, "Missing 'state_trajectories'"
        assert "q_values" in data, "Missing 'q_values'"

        traj = data["state_trajectories"]
        q_vals = data["q_values"]
        assert len(traj) == data["steps_completed"], "State trajectory length mismatch"
        assert len(q_vals) == data["steps_completed"], "Q-values length mismatch"
        if len(traj) > 0:
            assert len(traj[0]) == 4, f"State dimension must be 4, got {len(traj[0])}"
            assert len(q_vals[0]) == 2, f"Action Q-value dimension must be 2, got {len(q_vals[0])}"

        return {
            "endpoint": "/api/v1/rl/playback",
            "status": "PASS",
            "latency_ms": latency,
            "details": f"Steps: {data['steps_completed']} | Total Reward: {data['total_reward']:.1f} | Terminated: {data.get('terminated')}",
        }


def run_concurrent_smoke_test(base_url: str) -> bool:
    print_header(f"CONCURRENT MULTI-DOMAIN SMOKE TEST ({base_url})")

    tasks = {
        "Tabular": (test_tabular, base_url),
        "Vision": (test_vision, base_url),
        "NLP": (test_nlp, base_url),
        "Reinforcement Learning": (test_rl, base_url),
    }

    results = {}
    errors = {}

    print(f"\n{BOLD}Dispatching 4 concurrent domain payloads via ThreadPoolExecutor...{RESET}")
    overall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn, arg): name for name, (fn, arg) in tasks.items()}
        for future in as_completed(futures):
            domain = futures[future]
            try:
                res = future.result()
                results[domain] = res
            except Exception as exc:
                errors[domain] = str(exc)

    total_duration = (time.perf_counter() - overall_start) * 1000

    print_header("EXECUTION SUMMARY & VERIFICATION RESULTS")
    all_passed = True
    for domain, (fn, _) in tasks.items():
        if domain in results:
            r = results[domain]
            print(f"[{GREEN}PASS{RESET}] {BOLD}{domain:<24}{RESET} -> {r['endpoint']}")
            print(f"       Latency: {r['latency_ms']:.1f}ms | {r['details']}")
        else:
            all_passed = False
            print(f"[{RED}FAIL{RESET}] {BOLD}{domain:<24}{RESET}")
            print(f"       Error: {errors.get(domain, 'Unknown error')}")

    print(f"\n{BOLD}Total Concurrent Roundtrip Time:{RESET} {total_duration:.1f}ms")

    if all_passed:
        print(f"\n{GREEN}{BOLD}[SUCCESS] All 4 multi-domain live endpoints verified successfully!{RESET}\n")
    else:
        print(f"\n{RED}{BOLD}[FAILURE] One or more endpoints failed integration verification.{RESET}\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Live Integration Smoke-Tester for AI Portfolio Hub")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of FastAPI Gateway")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # Pre-check health
    if not test_health(base_url):
        print(f"{RED}Health check failed. Ensure the FastAPI server is running at {base_url}{RESET}")
        sys.exit(1)

    # Run concurrent integration verification
    success = run_concurrent_smoke_test(base_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
