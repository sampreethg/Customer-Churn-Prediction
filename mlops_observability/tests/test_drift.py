import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mlops_observability.src.drift_detector import StatisticalDriftDetector, calculate_psi
from mlops_observability.src.alerting import DriftAlertManager


def test_identical_distributions_no_drift():
    """Verify that identical random samples do not trigger drift alarms."""
    np.random.seed(42)
    reference = np.random.normal(loc=50.0, scale=10.0, size=1000)
    current = np.random.normal(loc=50.0, scale=10.0, size=1000)

    psi = calculate_psi(reference, current)
    assert psi < 0.10, f"Identical distributions produced high PSI: {psi}"

    detector = StatisticalDriftDetector()
    report = detector.evaluate_feature_drift(reference, current, "Tenure_Months")
    assert not report["drift_detected"]
    assert report["ks_pvalue"] > 0.01
    assert report["severity"] == "LOW"


def test_shifted_distribution_detected():
    """Verify that distribution shift triggers both K-S rejection and high PSI."""
    np.random.seed(42)
    reference = np.random.normal(loc=50.0, scale=10.0, size=1000)
    # Substantial covariate shift
    current = np.random.normal(loc=80.0, scale=15.0, size=1000)

    psi = calculate_psi(reference, current)
    assert psi >= 0.20, f"Shifted distribution failed to trigger high PSI: {psi}"

    detector = StatisticalDriftDetector()
    report = detector.evaluate_feature_drift(reference, current, "MonthlyCharges")
    assert report["drift_detected"]
    assert report["ks_pvalue"] < 1e-10
    assert report["severity"] == "HIGH"


def test_dataset_drift_retrain_trigger():
    """Verify that multi-feature shift correctly marks dataset_drift_detected and retrain_recommended."""
    np.random.seed(42)
    n = 800
    ref_df = pd.DataFrame({
        "Tenure": np.random.normal(24, 6, n),
        "Charges": np.random.uniform(30, 90, n),
        "Tickets": np.random.poisson(1.5, n),
    })

    # Drastically shift 2 of the 3 features (> 33% ratio)
    cur_df = pd.DataFrame({
        "Tenure": np.random.normal(60, 6, n),  # shifted
        "Charges": np.random.uniform(100, 150, n),  # shifted
        "Tickets": np.random.poisson(1.5, n),  # stable
    })

    detector = StatisticalDriftDetector(dataset_drift_ratio=0.33)
    dataset_report = detector.detect_dataset_drift(ref_df, cur_df)

    assert dataset_report["dataset_drift_detected"] is True
    assert dataset_report["retrain_recommended"] is True
    assert dataset_report["drifted_features_count"] == 2
    assert dataset_report["total_features_evaluated"] == 3

    # Verify alerting manager dispatches critical status
    alert_mgr = DriftAlertManager()
    alert_resp = alert_mgr.dispatch_alert(dataset_report)
    assert alert_resp["status"] == "CRITICAL"
    assert alert_resp["retrain_recommended"] is True
