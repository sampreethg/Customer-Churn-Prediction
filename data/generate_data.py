"""
Deterministic Synthetic Customer Churn Data Generator.

Generates realistic telecom/SaaS customer profiles with statistical risk modeling
and calibrated class imbalance (~20-25% churn rate).
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_churn_data(
    n_samples: int = 2500,
    seed: int = 42,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic customer churn dataset with realistic risk relationships.

    Parameters
    ----------
    n_samples : int, default=2500
        Number of synthetic customer observations (minimum 2000).
    seed : int, default=42
        Random seed for deterministic reproducibility.
    output_path : Path | str | None, default=None
        Optional file path to persist the CSV dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing generated features and binary churn target.
    """
    rng = np.random.default_rng(seed)

    # 1. Numeric Feature Generation
    # Tenure in months: 1 to 72 (weighted towards newer and older customers)
    tenure_months = rng.integers(1, 73, size=n_samples)

    # Monthly charges ($): typical SaaS/telecom plans between $18.50 and $118.75
    monthly_charges = np.round(rng.uniform(18.5, 118.75, size=n_samples), 2)

    # Total customer support tickets logged (0 to 12)
    # Most customers have 0-2 tickets; high tickets correlate with dissatisfaction
    total_tickets = rng.poisson(lam=1.8, size=n_samples)
    total_tickets = np.clip(total_tickets, 0, 15)

    # Age of customer (18 to 82)
    age = rng.integers(18, 83, size=n_samples)

    # 2. Categorical Feature Generation
    contract_choices = ["Month-to-month", "One year", "Two year"]
    contract_probs = [0.55, 0.25, 0.20]
    contract_type = rng.choice(contract_choices, size=n_samples, p=contract_probs)

    tech_support_choices = ["No", "Yes"]
    tech_support_probs = [0.65, 0.35]
    tech_support = rng.choice(tech_support_choices, size=n_samples, p=tech_support_probs)

    payment_choices = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    payment_probs = [0.35, 0.20, 0.23, 0.22]
    payment_method = rng.choice(payment_choices, size=n_samples, p=payment_probs)

    paperless_choices = ["No", "Yes"]
    paperless_probs = [0.41, 0.59]
    paperless_billing = rng.choice(paperless_choices, size=n_samples, p=paperless_probs)

    # 3. Probabilistic Logistic Risk Scoring
    # Base intercept calibrated to yield ~20-25% churn prevalence
    # Negative weights decrease churn risk; positive weights increase churn risk
    logit = -1.85

    # Short tenure has higher churn; long tenure builds loyalty
    # Standardize tenure effect across [1, 72]
    logit += -0.045 * (tenure_months - 36)

    # Higher monthly charges elevate risk of pricing fatigue
    logit += 0.022 * (monthly_charges - 65.0)

    # Frequent support tickets indicate chronic service issues
    logit += 0.42 * (total_tickets - 1.8)

    # Older customers have slightly lower churn tendency
    logit += -0.012 * (age - 45)

    # Contract terms: Month-to-month is high friction; 1-yr / 2-yr are sticky
    contract_risk = np.where(
        contract_type == "Month-to-month",
        0.85,
        np.where(contract_type == "One year", -0.45, -1.10),
    )
    logit += contract_risk

    # Lack of tech support elevates churn
    tech_risk = np.where(tech_support == "No", 0.45, -0.40)
    logit += tech_risk

    # Payment methods: electronic check exhibits historically higher churn
    payment_risk = np.where(
        payment_method == "Electronic check",
        0.40,
        np.where(payment_method == "Mailed check", 0.05, -0.25),
    )
    logit += payment_risk

    # Paperless billing
    paperless_risk = np.where(paperless_billing == "Yes", 0.15, -0.15)
    logit += paperless_risk

    # Compute churn probability via sigmoid function
    churn_prob = 1.0 / (1.0 + np.exp(-logit))

    # Sample binary outcome deterministically with seeded Bernoulli trials
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "Tenure_Months": tenure_months,
            "MonthlyCharges": monthly_charges,
            "TotalTickets": total_tickets,
            "Age": age,
            "ContractType": contract_type,
            "TechSupport": tech_support,
            "PaymentMethod": payment_method,
            "PaperlessBilling": paperless_billing,
            "Churn": churn,
        }
    )

    if output_path is not None:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, index=False)
        print(f"[Data Generator] Successfully saved {len(df)} records to: {target_path}")
        print(
            f"[Data Generator] Churn distribution:\n{df['Churn'].value_counts(normalize=True).mul(100).round(2)}"
        )

    return df


if __name__ == "__main__":
    default_dest = Path(__file__).resolve().parent / "churn_dataset.csv"
    generate_churn_data(n_samples=2500, seed=42, output_path=default_dest)
