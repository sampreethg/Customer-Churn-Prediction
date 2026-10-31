"""
Customer Churn Prediction Package.
"""

from .config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DATA_DIR,
    DATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PARAM_GRID,
    PROJECT_ROOT,
    RANDOM_STATE,
    SCORING_METRIC,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .evaluate import evaluate_model
from .explain import explain_model
from .pipeline import build_churn_pipeline
from .train import train_model

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "ARTIFACTS_DIR",
    "DATA_PATH",
    "MODEL_PATH",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "TARGET_COLUMN",
    "RANDOM_STATE",
    "TEST_SIZE",
    "CV_FOLDS",
    "SCORING_METRIC",
    "PARAM_GRID",
    "build_churn_pipeline",
    "train_model",
    "evaluate_model",
    "explain_model",
]
