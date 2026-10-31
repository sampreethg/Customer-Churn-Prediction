"""
End-to-End Customer Churn Prediction System Orchestrator.

Sequentially executes:
1. Environment and directory bootstrapping (data/, models/, artifacts/)
2. Deterministic synthetic data generation
3. Leakage-free pipeline construction, stratified cross-validation, and GridSearchCV
4. Unseen holdout evaluation and diagnostic artifact generation
5. Post-transformation feature importance extraction and business insights reporting
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_data import generate_churn_data
from src.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.evaluate import evaluate_model
from src.explain import explain_model
from src.train import train_model


def run_pipeline() -> None:
    """
    Execute full customer churn machine learning pipeline lifecycle.
    """
    print("\n" + "#" * 65)
    print("      PRODUCTION-GRADE CUSTOMER CHURN PREDICTION PIPELINE")
    print("#" * 65)
    print(f"Project Root: {PROJECT_ROOT.resolve()}")

    # 1. Directory Bootstrapping
    for directory in [DATA_DIR, MODELS_DIR, ARTIFACTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print("[Setup] Verified system directories: data/, models/, artifacts/.")

    # 2. Data Generation
    print("\n" + "-" * 65)
    print("STEP 1: Synthesizing Realistic Telecom/SaaS Churn Dataset")
    print("-" * 65)
    generate_churn_data(
        n_samples=2500,
        seed=RANDOM_STATE,
        output_path=DATA_PATH,
    )

    # 3. Model Training & Cross-Validation Grid Search
    print("\n" + "-" * 65)
    print("STEP 2: Training Pipeline with Stratified K-Fold & GridSearchCV")
    print("-" * 65)
    best_model, X_train, X_test, y_train, y_test = train_model(
        data_path=DATA_PATH,
        model_output_path=MODEL_PATH,
        random_state=RANDOM_STATE,
    )

    # 4. Holdout Evaluation & Diagnostics
    print("\n" + "-" * 65)
    print("STEP 3: Holdout Evaluation & Diagnostic Plots")
    print("-" * 65)
    evaluate_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        artifacts_dir=ARTIFACTS_DIR,
    )

    # 5. Model Explainability & Driver Reporting
    print("\n" + "-" * 65)
    print("STEP 4: Feature Importance & Business Risk Analysis")
    print("-" * 65)
    explain_model(
        model=best_model,
        artifacts_dir=ARTIFACTS_DIR,
        top_n=10,
    )

    print("\n" + "#" * 65)
    print("       CHURN PREDICTION WORKFLOW SUCCESSFULLY COMPLETED")
    print("#" * 65)
    print("Artifacts generated:")
    print(f"  * Trained Model:      {MODEL_PATH.resolve()}")
    print(f"  * Dataset:            {DATA_PATH.resolve()}")
    print(f"  * Confusion Matrix:   {(ARTIFACTS_DIR / 'confusion_matrix.png').resolve()}")
    print(f"  * ROC Curve:          {(ARTIFACTS_DIR / 'roc_curve.png').resolve()}")
    print(f"  * Feature Importance: {(ARTIFACTS_DIR / 'feature_importance.png').resolve()}")


if __name__ == "__main__":
    run_pipeline()
