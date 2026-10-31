"""
Configuration settings, directory paths, feature schemas, and tuning parameters.
"""

from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# File Paths
DATA_PATH = DATA_DIR / "churn_dataset.csv"
MODEL_PATH = MODELS_DIR / "best_churn_model.joblib"

# Feature Schemas
NUMERIC_FEATURES = [
    "Tenure_Months",
    "MonthlyCharges",
    "TotalTickets",
    "Age",
]

CATEGORICAL_FEATURES = [
    "ContractType",
    "TechSupport",
    "PaymentMethod",
    "PaperlessBilling",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "Churn"

# Reproducibility & Model Validation Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING_METRIC = "f1"

# Hyperparameter Optimization Grid for Random Forest
PARAM_GRID = {
    "classifier__n_estimators": [50, 100, 150],
    "classifier__max_depth": [None, 6, 10, 14],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}
