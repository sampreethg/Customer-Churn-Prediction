"""Production MLOps Data Drift & Covariate Shift Detection Engine.

Implements rigorous statistical hypothesis testing:
1. Two-sample Kolmogorov-Smirnov (K-S) test for continuous feature distributions.
2. Population Stability Index (PSI) with epsilon smoothing.
3. 1D Wasserstein Distance (Earth Mover's Distance).
4. Automated dataset-level drift synthesis with retrain triggers.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

EPSILON = 1e-6


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
) -> float:
    """
    Compute the Population Stability Index (PSI) between baseline (expected)
    and production inference (actual) arrays.

    Formula:
        PSI = sum( (actual% - expected%) * ln(actual% / expected%) )

    Thresholds:
        PSI < 0.10: No significant shift (Stable)
        0.10 <= PSI < 0.20: Moderate shift (Monitor closely)
        PSI >= 0.20: Severe shift (Trigger Model Retraining)
    """
    # Remove NaN values
    exp_clean = expected[~np.isnan(expected)]
    act_clean = actual[~np.isnan(actual)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    # Determine quantile bins based on reference data
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(exp_clean, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Bucket frequency counts
    exp_counts = np.histogram(exp_clean, bins=bins)[0]
    act_counts = np.histogram(act_clean, bins=bins)[0]

    # Convert to probability mass distributions with smoothing
    exp_pct = np.maximum(exp_counts / len(exp_clean), EPSILON)
    act_pct = np.maximum(act_counts / len(act_clean), EPSILON)

    # Calculate PSI components
    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_value)


class StatisticalDriftDetector:
    """Production statistical detector monitoring covariate shift in production ML pipelines."""

    def __init__(
        self,
        ks_alpha: float = 0.05,
        psi_threshold: float = 0.20,
        wasserstein_threshold: float = 0.15,
        dataset_drift_ratio: float = 0.33,
    ):
        self.ks_alpha = ks_alpha
        self.psi_threshold = psi_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.dataset_drift_ratio = dataset_drift_ratio

    def evaluate_feature_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str,
        is_categorical: bool = False,
    ) -> Dict[str, Any]:
        """Compute K-S statistic, p-value, PSI, and Wasserstein distance for one feature."""
        ref_arr = np.asarray(reference, dtype=np.float64)
        cur_arr = np.asarray(current, dtype=np.float64)

        # 1. Kolmogorov-Smirnov Test
        ks_stat, ks_pvalue = stats.ks_2samp(ref_arr, cur_arr)

        # 2. Population Stability Index
        psi_score = calculate_psi(ref_arr, cur_arr)

        # 3. Wasserstein Distance
        # Normalize distributions to common scale [0, 1] for scale-invariant distance
        combined_min = min(np.min(ref_arr), np.min(cur_arr))
        combined_max = max(np.max(ref_arr), np.max(cur_arr))
        spread = combined_max - combined_min if combined_max > combined_min else 1.0

        ref_norm = (ref_arr - combined_min) / spread
        cur_norm = (cur_arr - combined_min) / spread
        wasserstein_dist = stats.wasserstein_distance(ref_norm, cur_norm)

        # Decision rule: feature is drifted if KS rejects null hypothesis OR PSI >= threshold
        drift_detected = bool(ks_pvalue < self.ks_alpha or psi_score >= self.psi_threshold)

        return {
            "feature": feature_name,
            "drift_detected": drift_detected,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": float(ks_pvalue),
            "psi_score": round(float(psi_score), 4),
            "wasserstein_distance": round(float(wasserstein_dist), 4),
            "severity": "HIGH" if psi_score >= 0.20 else ("MODERATE" if psi_score >= 0.10 else "LOW"),
        }

    def detect_dataset_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        numeric_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Assess holistic covariate shift across an entire feature set.
        """
        if numeric_features is None:
            numeric_features = [
                c for c in reference_df.columns
                if pd.api.types.is_numeric_dtype(reference_df[c])
            ]

        feature_reports = []
        drifted_count = 0

        for col in numeric_features:
            if col not in current_df.columns:
                continue
            report = self.evaluate_feature_drift(
                reference=reference_df[col].dropna().values,
                current=current_df[col].dropna().values,
                feature_name=col,
            )
            feature_reports.append(report)
            if report["drift_detected"]:
                drifted_count += 1

        total_features = len(feature_reports)
        drift_ratio = (drifted_count / total_features) if total_features > 0 else 0.0
        dataset_drift_detected = bool(drift_ratio >= self.dataset_drift_ratio)

        return {
            "dataset_drift_detected": dataset_drift_detected,
            "drifted_features_count": drifted_count,
            "total_features_evaluated": total_features,
            "drifted_ratio": round(drift_ratio, 4),
            "retrain_recommended": dataset_drift_detected,
            "feature_metrics": feature_reports,
        }
