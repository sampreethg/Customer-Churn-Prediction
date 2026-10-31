"""
Unit Tests for Model Training, Evaluation, Diagnostics, and Explanations.

Exercises train_model, evaluate_model, explain_model, and ChurnExplainer visualization
methods to ensure high code quality and test coverage.
"""

import pandas as pd
import pytest

from src.config import DATA_PATH
from src.evaluate import evaluate_model
from src.explain import explain_model, get_feature_names
from src.explainer import ChurnExplainer
from src.train import load_dataset, train_model


def test_load_dataset_success():
    """
    Verify load_dataset loads feature matrix X and binary target y correctly.
    """
    x, y = load_dataset(DATA_PATH)
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(x) == len(y)
    assert set(y.unique()).issubset({0, 1})


def test_load_dataset_file_not_found(tmp_path):
    """
    Verify load_dataset raises FileNotFoundError for missing path.
    """
    non_existent = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(non_existent)


def test_load_dataset_missing_target(tmp_path):
    """
    Verify load_dataset raises ValueError when target column is missing.
    """
    bad_csv = tmp_path / "no_target.csv"
    pd.DataFrame({"colA": [1, 2], "colB": [3, 4]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Target column"):
        load_dataset(bad_csv)


def test_evaluate_and_explain_model(tmp_path, mock_trained_pipeline, sample_batch_records):
    """
    Verify evaluate_model and explain_model correctly compute metrics,
    extract feature names, and export diagnostic visualizations.
    """
    df = pd.DataFrame(sample_batch_records)
    y_test = pd.Series([1, 0, 0])

    metrics = evaluate_model(
        model=mock_trained_pipeline,
        X_test=df,
        y_test=y_test,
        artifacts_dir=tmp_path,
    )

    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1-Score" in metrics
    assert "ROC-AUC" in metrics
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "roc_curve.png").exists()

    feature_names = get_feature_names(mock_trained_pipeline)
    assert len(feature_names) > 0
    assert "Tenure_Months" in feature_names

    importance_df = explain_model(
        model=mock_trained_pipeline,
        artifacts_dir=tmp_path,
        top_n=5,
    )
    assert isinstance(importance_df, pd.DataFrame)
    assert "Feature" in importance_df.columns
    assert "Importance" in importance_df.columns
    assert (tmp_path / "feature_importance.png").exists()


def test_churn_explainer_waterfall_plot(tmp_path, mock_trained_pipeline, sample_customer_record):
    """
    Verify ChurnExplainer.plot_waterfall generates a valid figure and saves an image artifact.
    """
    explainer = ChurnExplainer(model_or_path=mock_trained_pipeline)
    cust_df = pd.DataFrame([sample_customer_record])

    out_file = tmp_path / "waterfall_test.png"
    explainer.plot_waterfall(cust_df, save_path=out_file, top_n=6)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_train_model_minimal_cv(tmp_path):
    """
    Verify train_model runs cross-validation, grid search, and persists pipeline artifact.
    """
    small_data = tmp_path / "small_churn.csv"
    data = {
        "Tenure_Months": [2, 30, 60, 4, 36, 70, 8, 12, 48, 6, 24, 55],
        "MonthlyCharges": [
            85.0,
            45.0,
            20.0,
            105.0,
            60.0,
            25.0,
            95.0,
            50.0,
            30.0,
            110.0,
            40.0,
            35.0,
        ],
        "TotalTickets": [4, 1, 0, 5, 2, 0, 3, 2, 0, 4, 1, 0],
        "Age": [25, 45, 65, 30, 50, 70, 35, 40, 55, 28, 48, 62],
        "ContractType": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "One year",
            "Two year",
        ],
        "TechSupport": [
            "No",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "Yes",
        ],
        "PaymentMethod": [
            "Electronic check",
            "Credit card (automatic)",
            "Bank transfer (automatic)",
            "Electronic check",
            "Mailed check",
            "Credit card (automatic)",
            "Electronic check",
            "Credit card (automatic)",
            "Bank transfer (automatic)",
            "Electronic check",
            "Credit card (automatic)",
            "Bank transfer (automatic)",
        ],
        "PaperlessBilling": [
            "Yes",
            "No",
            "No",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "No",
            "No",
            "Yes",
            "No",
            "No",
        ],
        "Churn": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    }
    pd.DataFrame(data).to_csv(small_data, index=False)

    saved_model = tmp_path / "test_model.joblib"
    fast_grid = {
        "classifier__n_estimators": [5],
        "classifier__max_depth": [3],
    }

    pipeline, X_train, X_test, y_train, y_test = train_model(
        data_path=small_data,
        model_output_path=saved_model,
        param_grid=fast_grid,
        n_splits=2,
        scoring="accuracy",
        test_size=0.25,
    )

    assert saved_model.exists()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert hasattr(pipeline, "predict")


def test_churn_explainer_file_not_found(tmp_path):
    """
    Verify ChurnExplainer raises FileNotFoundError when passed an invalid model file path.
    """
    missing_model = tmp_path / "non_existent_model.joblib"
    with pytest.raises(FileNotFoundError):
        ChurnExplainer(model_or_path=missing_model)


def test_churn_explainer_with_missing_optional_columns(mock_trained_pipeline):
    """
    Verify ChurnExplainer correctly populates default values for missing optional columns
    and generates an in-memory matplotlib Figure when save_path is None.
    """
    explainer = ChurnExplainer(model_or_path=mock_trained_pipeline)

    # Partial observation without Age or PaperlessBilling
    partial_record = {
        "Tenure_Months": 10,
        "MonthlyCharges": 85.0,
        "TotalTickets": 3,
        "ContractType": "Month-to-month",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
    }
    df_partial = pd.DataFrame([partial_record])

    explanation = explainer.explain_instance(df_partial)
    assert "churn_probability" in explanation
    assert "top_risk_drivers" in explanation
    assert "top_retention_drivers" in explanation

    # Render without saving to file (returns in-memory figure)
    fig = explainer.plot_waterfall(df_partial, save_path=None, top_n=5)
    assert fig is not None


def test_api_format_input_dataframe_helper():
    """
    Verify api.app format_input_dataframe helper infers expected columns and defaults.
    """
    from api.app import format_input_dataframe

    raw_records = [
        {
            "Tenure_Months": 6,
            "MonthlyCharges": 75.0,
            "TotalTickets": 2,
            "ContractType": "Month-to-month",
            "TechSupport": "No",
            "PaymentMethod": "Electronic check",
        }
    ]
    df = format_input_dataframe(raw_records)
    assert "Age" in df.columns
    assert "PaperlessBilling" in df.columns
    assert df["Age"].iloc[0] == 45
    assert df["PaperlessBilling"].iloc[0] == "Yes"
