"""
Scikit-Learn Pipeline and Preprocessing Architecture.

Encapsulates all feature transformations and estimator configuration within an
integrated Pipeline to prevent data leakage during CV and holdout validation.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Construct ColumnTransformer for numerical standardization and categorical encoding.

    Returns
    -------
    ColumnTransformer
        Transformer applying StandardScaler to numeric columns and OneHotEncoder
        to categorical columns.
    """
    num_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_churn_pipeline(
    random_state: int = RANDOM_STATE,
    class_weight: str = "balanced",
    **classifier_kwargs,
) -> Pipeline:
    """
    Construct full end-to-end Pipeline combining preprocessing and RandomForestClassifier.

    Parameters
    ----------
    random_state : int, default=42
        Seed for reproducible random state.
    class_weight : str, default='balanced'
        Strategy to handle class imbalance at the algorithm level.
    **classifier_kwargs : Any
        Additional keyword arguments passed to the RandomForestClassifier estimator.

    Returns
    -------
    Pipeline
        Executable pipeline ready for fitting or hyperparameter tuning.
    """
    preprocessor = build_preprocessor()

    classifier = RandomForestClassifier(
        random_state=random_state,
        class_weight=class_weight,
        **classifier_kwargs,
    )

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return full_pipeline


# Alias for backward-compatible pipeline initialization
build_pipeline = build_churn_pipeline
