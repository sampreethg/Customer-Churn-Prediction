"""
FastAPI Inference Microservice Engine with Asynchronous Webhook Alerting.

Provides low-latency single, batch, and SHAP explainability endpoints,
startup model decoupling via lifespan context manager, and automated
non-blocking retention alert dispatches to Slack, Discord, and CRMs.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchCustomerInput,
    BatchPredictionResponse,
    ChurnPredictionResponse,
    CustomerFeatures,
    ExplanationResponse,
)
from src.alerting import WebhookNotifier
from src.auth import require_api_key
from src.explainer import ChurnExplainer
from src.rate_limiter import rate_limit

# Resolve default model artifact path
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best_churn_model.joblib"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))

model_pipeline = None
explainer_engine = None
notifier = WebhookNotifier()


def get_model():
    """
    Retrieve loaded model pipeline, loading on-demand if lifespan was bypassed (e.g. TestClient).
    """
    global model_pipeline
    if model_pipeline is None:
        path = Path(MODEL_PATH)
        if path.is_file():
            model_pipeline = joblib.load(path)
    return model_pipeline


def get_explainer():
    """
    Retrieve initialized ChurnExplainer instance, loading on-demand if lifespan was bypassed.
    """
    global explainer_engine
    if explainer_engine is None:
        pipeline = get_model()
        if pipeline is not None:
            explainer_engine = ChurnExplainer(pipeline)
    return explainer_engine


def assign_risk_tier(prob: float) -> tuple[str, str]:
    """
    Map continuous model churn probability into actionable business risk tiers.

    Parameters
    ----------
    prob : float
        Predicted churn probability (0.0 to 1.0).

    Returns
    -------
    tuple[str, str]
        (risk_level, recommended_action)
    """
    if prob >= 0.70:
        return "High", "Offer targeted loyalty discount and schedule priority retention callback."
    elif prob >= 0.35:
        return "Medium", "Send tailored engagement check-in and product optimization guide."
    return "Low", "No retention intervention required. Maintain standard engagement."


def format_input_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Format input feature records into a DataFrame aligned with the model's feature schema.
    """
    df = pd.DataFrame(records)
    pipeline = get_model()
    if pipeline is not None and hasattr(pipeline, "feature_names_in_"):
        for col in pipeline.feature_names_in_:
            if col not in df.columns:
                if col == "Age":
                    df[col] = 45
                elif col == "PaperlessBilling":
                    df[col] = "Yes"
                else:
                    df[col] = 0
    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for startup loading and cleanup.
    """
    global model_pipeline, explainer_engine, notifier
    path = Path(MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at '{path.resolve()}'. Train the pipeline first."
        )
    model_pipeline = joblib.load(path)
    explainer_engine = ChurnExplainer(model_pipeline)
    notifier = WebhookNotifier()
    yield
    model_pipeline = None
    explainer_engine = None


import logging

logger = logging.getLogger("churn.api")

app = FastAPI(
    title="Customer Churn Inference API",
    description="Enterprise-ready low-latency inference microservice for churn prediction, SHAP explainability, and automated retention alerts.",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins_raw = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000",
)
allowed_origins = [orig.strip() for orig in allowed_origins_raw.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Enforce strict baseline security headers across all responses."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Prevent unhandled internal stack traces from leaking to clients."""
    logger.exception("Unhandled server exception processing %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."},
    )



@app.get("/health", status_code=status.HTTP_200_OK, tags=["Monitoring"])
async def health_check():
    """
    Health check probe verifying server status, model readiness, and explainer availability.
    """
    pipeline = get_model()
    explainer = get_explainer()
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "explainer_loaded": explainer is not None,
        "alerting_threshold": notifier.alert_threshold,
        "slack_configured": bool(notifier.slack_webhook_url),
        "crm_configured": bool(notifier.crm_webhook_url),
        "model_artifact": str(MODEL_PATH),
    }


@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    tags=["Inference"],
    dependencies=[Depends(require_api_key), Depends(rate_limit(60))],
)
async def predict_single(customer: CustomerFeatures, background_tasks: BackgroundTasks):
    """
    Perform real-time churn prediction for an individual customer and dispatch non-blocking retention alerts.
    """
    pipeline = get_model()
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized or service is temporarily unavailable.",
        )

    input_df = format_input_dataframe([customer.to_feature_dict()])

    try:
        churn_class = int(pipeline.predict(input_df)[0])
        churn_proba = float(pipeline.predict_proba(input_df)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution error: {str(exc)}",
        ) from exc

    risk_level, recommended_action = assign_risk_tier(churn_proba)

    # Automated asynchronous retention alerting
    alert_dispatched = False
    if churn_proba >= notifier.alert_threshold and customer.notify_retention_team:
        top_factors = []
        try:
            explainer = get_explainer()
            if explainer is not None:
                exp_res = explainer.explain_instance(input_df)
                top_factors = exp_res.get("top_risk_drivers", [])[:3]
        except Exception:
            pass

        background_tasks.add_task(
            notifier.dispatch_retention_alert,
            customer_id=customer.Customer_ID,
            prob=churn_proba,
            risk_level=risk_level,
            action=recommended_action,
            top_factors=top_factors,
        )
        alert_dispatched = True

    return ChurnPredictionResponse(
        Customer_ID=customer.Customer_ID,
        churn_prediction=churn_class,
        churn_probability=round(churn_proba, 4),
        risk_level=risk_level,
        recommended_action=recommended_action,
        alert_dispatched=alert_dispatched,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    dependencies=[Depends(require_api_key), Depends(rate_limit(60))],
)
async def predict_batch(batch: BatchCustomerInput, background_tasks: BackgroundTasks):
    """
    Perform high-throughput vectorized batch churn prediction and dispatch alerts for high-risk accounts.
    """
    pipeline = get_model()
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized or service is temporarily unavailable.",
        )

    if not batch.customers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty customer array provided. At least one customer is required.",
        )

    records = [c.to_feature_dict() for c in batch.customers]
    input_df = format_input_dataframe(records)

    try:
        churn_classes = pipeline.predict(input_df).tolist()
        churn_probas = pipeline.predict_proba(input_df)[:, 1].tolist()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference execution error: {str(exc)}",
        ) from exc

    predictions = []
    for customer, cls, prob in zip(batch.customers, churn_classes, churn_probas, strict=False):
        risk_level, action = assign_risk_tier(prob)
        alert_dispatched = False

        if prob >= notifier.alert_threshold and customer.notify_retention_team:
            background_tasks.add_task(
                notifier.dispatch_retention_alert,
                customer_id=customer.Customer_ID,
                prob=prob,
                risk_level=risk_level,
                action=action,
                top_factors=None,
            )
            alert_dispatched = True

        predictions.append(
            ChurnPredictionResponse(
                Customer_ID=customer.Customer_ID,
                churn_prediction=int(cls),
                churn_probability=round(float(prob), 4),
                risk_level=risk_level,
                recommended_action=action,
                alert_dispatched=alert_dispatched,
            )
        )

    return BatchPredictionResponse(
        total_processed=len(predictions),
        predictions=predictions,
    )


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    tags=["Explainability"],
    dependencies=[Depends(require_api_key), Depends(rate_limit(10))],
)
async def explain_customer(customer: CustomerFeatures):
    """
    Compute local SHAP attributions, feature-level drivers, and plain-English narrative.
    """
    explainer = get_explainer()
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explainability engine is not initialized or service is temporarily unavailable.",
        )

    input_df = format_input_dataframe([customer.to_feature_dict()])

    try:
        explanation = explainer.explain_instance(input_df)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SHAP explanation execution error: {str(exc)}",
        ) from exc

    return ExplanationResponse(
        churn_probability=explanation["churn_probability"],
        base_value=explanation["base_value"],
        top_risk_drivers=explanation["top_risk_drivers"],
        top_retention_drivers=explanation["top_retention_drivers"],
        summary_narrative=explanation["summary_narrative"],
    )
