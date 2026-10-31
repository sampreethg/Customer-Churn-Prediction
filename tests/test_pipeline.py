"""
Automated Pytest Suite for Machine Learning Pipeline Verification.

Validates data leakage absence across train/test splits, pipeline reproducibility
under deterministic random seeds, and expected preprocessor feature dimensions.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.config import ALL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE
from src.pipeline import build_pipeline


@pytest.fixture
def synthetic_dataset():
    """
    Generate deterministic synthetic customer dataset for pipeline unit tests.
    """
    np.random.seed(42)
    n = 200
    data = {
        "Tenure_Months": np.random.randint(1, 72, size=n),
        "MonthlyCharges": np.random.uniform(20.0, 120.0, size=n),
        "TotalTickets": np.random.randint(0, 8, size=n),
        "Age": np.random.randint(18, 80, size=n),
        "ContractType": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
        "TechSupport": np.random.choice(["Yes", "No"], size=n),
        "PaymentMethod": np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            size=n,
        ),
        "PaperlessBilling": np.random.choice(["Yes", "No"], size=n),
    }
    y = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    df = pd.DataFrame(data)[ALL_FEATURES]
    return df, y


def test_data_leakage_absence(synthetic_dataset):
    """
    Verify that transformers (StandardScaler, OneHotEncoder) calculate statistics
    strictly on the training partition and do not leak or shift statistics when
    transforming unseen test data.
    """
    df, y = synthetic_dataset
    x_train, x_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline(n_estimators=10, max_depth=3, random_state=RANDOM_STATE)
    pipeline.fit(x_train, y_train)

    preprocessor = pipeline.named_steps["preprocessor"]
    num_scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    # Verify that the scaler's fitted mean matches the training partition strictly
    train_num_means = x_train[NUMERIC_FEATURES].mean().values
    np.testing.assert_allclose(
        num_scaler.mean_,
        train_num_means,
        rtol=1e-5,
        err_msg="Scaler mean was not calculated strictly on training fold.",
    )

    # Record fitted parameters before transforming test set
    recorded_means = np.copy(num_scaler.mean_)
    recorded_scales = np.copy(num_scaler.scale_)

    # Transform test set and verify preprocessor state remains strictly unaltered
    _ = pipeline.predict_proba(x_test)
    np.testing.assert_array_equal(
        num_scaler.mean_,
        recorded_means,
        err_msg="Preprocessor state mutated during test set transformation (data leakage).",
    )
    np.testing.assert_array_equal(
        num_scaler.scale_,
        recorded_scales,
        err_msg="Preprocessor scale mutated during test set transformation.",
    )


def test_pipeline_reproducibility(synthetic_dataset):
    """
    Verify that fitting the pipeline with the identical random seed on the same data
    yields identical predictions and prediction probabilities.
    """
    df, y = synthetic_dataset
    x_train, x_test, y_train, _ = train_test_split(
        df, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # Run 1
    pipe_1 = build_pipeline(n_estimators=20, max_depth=4, random_state=RANDOM_STATE)
    pipe_1.fit(x_train, y_train)
    probs_1 = pipe_1.predict_proba(x_test)

    # Run 2
    pipe_2 = build_pipeline(n_estimators=20, max_depth=4, random_state=RANDOM_STATE)
    pipe_2.fit(x_train, y_train)
    probs_2 = pipe_2.predict_proba(x_test)

    # Assert exact probability alignment across repeated fits
    np.testing.assert_allclose(
        probs_1,
        probs_2,
        atol=1e-7,
        err_msg="Pipeline predictions failed deterministic reproducibility check.",
    )


def test_expected_feature_dimensions(synthetic_dataset):
    """
    Verify that ColumnTransformer produces the expected dimensional matrix post-encoding
    and matches model input expectations without NaNs or missing columns.
    """
    df, y = synthetic_dataset
    pipeline = build_pipeline(n_estimators=10, max_depth=3, random_state=RANDOM_STATE)
    pipeline.fit(df, y)

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_matrix = preprocessor.transform(df)

    assert not np.isnan(transformed_matrix).any(), (
        "Transformed feature matrix contains unexpected NaNs."
    )

    # Verify column count: 4 numeric features + one-hot encoded categories (drop='first')
    cat_transformer = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_count = sum(len(cats) - 1 for cats in cat_transformer.categories_)
    expected_dim = len(NUMERIC_FEATURES) + cat_feature_count

    assert transformed_matrix.shape[1] == expected_dim, (
        f"Transformed dimension mismatch: got {transformed_matrix.shape[1]}, expected {expected_dim}"
    )
    assert transformed_matrix.shape[0] == len(df), "Row count changed post-transformation."
