"""
Model Training and Hyperparameter Optimization Module.

Performs stratified train/test partitioning, Stratified K-Fold cross-validation,
exhaustive Grid Search, and persistence of the optimized pipeline.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .config import (
    CV_FOLDS,
    DATA_PATH,
    MODEL_PATH,
    PARAM_GRID,
    RANDOM_STATE,
    SCORING_METRIC,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .pipeline import build_churn_pipeline


def load_dataset(data_path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load churn dataset from disk and separate predictors and binary target.

    Parameters
    ----------
    data_path : Path | str
        Path to CSV dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at expected location: {path}")

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    return X, y


def train_model(
    data_path: Path | str = DATA_PATH,
    model_output_path: Path | str = MODEL_PATH,
    param_grid: dict | None = None,
    n_splits: int = CV_FOLDS,
    scoring: str = SCORING_METRIC,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Execute stratified train/test split, Grid Search CV tuning, and model persistence.

    Parameters
    ----------
    data_path : Path | str
        Path to the source CSV file.
    model_output_path : Path | str
        Destination path for serializing the best model.
    param_grid : dict | None
        Hyperparameter search space.
    n_splits : int
        Number of Stratified K-Fold CV splits.
    scoring : str
        Scoring metric for evaluating parameter candidates.
    random_state : int
        Seed for reproducibility.
    test_size : float
        Holdout test set fraction.

    Returns
    -------
    tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Fitted best_estimator_, X_train, X_test, y_train, y_test.
    """
    print("=" * 60)
    print(" STAGE 1: MODEL TRAINING & OPTIMIZATION")
    print("=" * 60)

    X, y = load_dataset(data_path=data_path)
    print(f"Loaded dataset: {len(X)} total samples with {X.shape[1]} features.")
    print(f"Target distribution (Churn=1): {y.mean():.2%}")

    # Stratified Train/Test Partitioning
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(
        f"Partitioned into Train set ({len(X_train)} samples) and "
        f"Test set ({len(X_test)} samples) with stratified class ratio."
    )

    # Initialize un-fitted pipeline
    base_pipeline = build_churn_pipeline(random_state=random_state, class_weight="balanced")

    # Configure Stratified K-Fold Cross-Validation
    cv_strategy = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    grid = param_grid if param_grid is not None else PARAM_GRID
    total_candidates = 1
    for values in grid.values():
        total_candidates *= len(values)
    print(
        f"Initiating GridSearchCV across {total_candidates} candidates "
        f"with {n_splits}-fold CV ({total_candidates * n_splits} total fits)..."
    )

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=grid,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    best_pipeline: Pipeline = grid_search.best_estimator_
    print(f"\n[Training Complete] Best CV {scoring.upper()} Score: {grid_search.best_score_:.4f}")
    print("Optimized Hyperparameters:")
    for param_name, param_val in grid_search.best_params_.items():
        clean_param = param_name.replace("classifier__", "")
        print(f" - {clean_param}: {param_val}")

    # Persist Best Estimator
    out_path = Path(model_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, out_path)
    print(f"\nSaved production-ready pipeline artifact to: {out_path.resolve()}")

    return best_pipeline, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_model()
