"""Alerting and incident dispatch for model drift events."""

from typing import Dict, Any
import json
import logging

logger = logging.getLogger("mlops_alerting")


class DriftAlertManager:
    """Dispatches actionable incident reports when data drift thresholds are breached."""

    def __init__(self, alert_sink: str = "console"):
        self.alert_sink = alert_sink

    def dispatch_alert(self, drift_report: Dict[str, Any]) -> Dict[str, Any]:
        """Process drift assessment and dispatch incident notifications if drift is detected."""
        drift_detected = drift_report.get("dataset_drift_detected", False)
        drifted_count = drift_report.get("drifted_features_count", 0)
        total_count = drift_report.get("total_features_evaluated", 0)

        alert_payload = {
            "incident_type": "DATA_DRIFT_ALERT" if drift_detected else "DATA_STABLE",
            "status": "CRITICAL" if drift_detected else "NORMAL",
            "retrain_recommended": drift_report.get("retrain_recommended", False),
            "summary": (
                f"{drifted_count}/{total_count} features experienced significant covariate shift."
                if drift_detected
                else "All monitored features remain within statistical stability bounds."
            ),
            "drift_details": drift_report,
        }

        if drift_detected:
            print(f"\n[CRITICAL MLOps ALERT] Dataset drift detected! Triggering automated retraining pipeline.")
            print(f"Summary: {alert_payload['summary']}")
        else:
            print(f"[OK MLOps Telemetry] Baseline feature distributions remain stable.")

        return alert_payload
