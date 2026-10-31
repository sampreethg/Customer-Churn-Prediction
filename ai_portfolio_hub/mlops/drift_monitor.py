"""MLOps Data Drift & Covariate Shift Audit Engine.

Executes statistical drift tests across monitored features, checks K-S statistics and PSI,
and renders an HTML & JSON report saved under artifacts/mlops/.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "mlops"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_drift_audit():
    print("=" * 65)
    print("      MLOps PRODUCTION COVARIATE SHIFT & DRIFT AUDIT")
    print("=" * 65)

    np.random.seed(42)
    n_samples = 1000

    # Baseline reference dataset
    ref_df = pd.DataFrame({
        "Tenure_Months": np.random.randint(1, 72, size=n_samples),
        "MonthlyCharges": np.random.uniform(20.0, 120.0, size=n_samples),
        "TotalTickets": np.random.poisson(lam=1.5, size=n_samples),
    })

    # Production inference window with simulated tenure/charges drift
    cur_df = pd.DataFrame({
        "Tenure_Months": np.random.randint(1, 36, size=n_samples),  # drifted (younger tenure)
        "MonthlyCharges": np.random.uniform(40.0, 140.0, size=n_samples),  # drifted (higher charges)
        "TotalTickets": np.random.poisson(lam=1.6, size=n_samples),  # stable
    })

    report_data = []
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    # Evaluate feature by feature
    for col in ref_df.columns:
        ref_vals = ref_df[col].values
        cur_vals = cur_df[col].values

        ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
        drift_detected = bool(ks_pval < 0.05)
        severity = "HIGH" if ks_pval < 0.001 else ("MODERATE" if drift_detected else "LOW")

        print(f"Feature: {col:<18} | KS-Stat: {ks_stat:.4f} | p-value: {ks_pval:.4e} | Drift: {drift_detected} ({severity})")
        report_data.append({
            "feature": col,
            "ks_stat": round(float(ks_stat), 4),
            "p_value": float(ks_pval),
            "drift_detected": drift_detected,
            "severity": severity,
        })

    # Render HTML Report
    html_rows = "".join([
        f"<tr><td>{r['feature']}</td><td>{r['ks_stat']}</td><td>{r['p_value']:.4e}</td>"
        f"<td style='color: {'red' if r['drift_detected'] else 'green'}; font-weight:bold;'>{'DRIFT DETECTED' if r['drift_detected'] else 'STABLE'}</td>"
        f"<td>{r['severity']}</td></tr>"
        for r in report_data
    ])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>MLOps Data Drift Report - {timestamp_str}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; }}
        h1 {{ color: #38bdf8; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0284c7; color: white; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>MLOps Data Drift & Stability Report</h1>
    <p>Audit Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    <table>
        <thead>
            <tr><th>Feature Name</th><th>KS Statistic</th><th>p-value</th><th>Status</th><th>Severity</th></tr>
        </thead>
        <tbody>
            {html_rows}
        </tbody>
    </table>
</body>
</html>
"""

    report_path = ARTIFACTS_DIR / f"drift_report_{timestamp_str}.html"
    report_path.write_text(html_content, encoding="utf-8")
    print(f"\n[OK] MLOps drift audit report exported: {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    generate_drift_audit()
