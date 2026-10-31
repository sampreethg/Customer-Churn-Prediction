"""
Main Unified FastAPI Application for Multi-Domain AI Portfolio Hub.

Orchestrates model lifecycles across Tabular, Vision, NLP, and Reinforcement Learning domains,
exposing modular routers under /api/v1 with comprehensive health and system telemetry.
"""

from contextlib import asynccontextmanager
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import torch

from .config import (
    CHURN_MODEL_PATH,
    DEVICE,
    RL_MODEL_PATH,
    TRANSFORMER_PATH,
    VISION_MODEL_PATH,
)
from .routers import nlp, rag, rl, tabular, vision
from .schemas import SystemHealthResponse

# App Startup Timestamp
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all four specialized AI models into app.state at startup.
    """
    print("\n" + "=" * 65)
    print("      INITIALIZING MULTI-DOMAIN AI PORTFOLIO GATEWAY")
    print("=" * 65)
    print(f"Target Compute Device: {DEVICE}")

    # 1. Load Tabular Churn Model & SHAP Explainer
    try:
        if CHURN_MODEL_PATH.is_file():
            app.state.churn_model = joblib.load(CHURN_MODEL_PATH)
            print(f"[OK] Tabular Model loaded from: {CHURN_MODEL_PATH.name}")
            try:
                import sys
                from pathlib import Path
                churn_src = Path(__file__).resolve().parent.parent.parent / "src"
                if str(churn_src.parent) not in sys.path:
                    sys.path.insert(0, str(churn_src.parent))
                from src.explainer import ChurnExplainer
                app.state.churn_explainer = ChurnExplainer(app.state.churn_model)
                print("[OK] Tabular SHAP Explainer initialized.")
            except Exception as e:
                print(f"[Notice] SHAP Explainer initialization note: {e}")
                app.state.churn_explainer = None
        else:
            print(f"[Warning] Churn model artifact missing at {CHURN_MODEL_PATH}")
            app.state.churn_model = None
            app.state.churn_explainer = None
    except Exception as exc:
        print(f"[Error] Failed to load Tabular model: {exc}")
        app.state.churn_model = None
        app.state.churn_explainer = None

    # 2. Load Vision CNN Model
    try:
        import tensorflow as tf
        if VISION_MODEL_PATH.is_file():
            app.state.vision_model = tf.keras.models.load_model(str(VISION_MODEL_PATH))
            print(f"[OK] Vision CNN Model loaded from: {VISION_MODEL_PATH.name}")
        else:
            # Fallback fresh architecture initialization
            from image_classifier_cnn.src.model import build_cnn_model
            app.state.vision_model = build_cnn_model()
            print("[OK] Vision CNN architecture initialized in memory.")
    except Exception as exc:
        print(f"[Notice] Vision model initialization note: {exc}")
        try:
            from tensorflow.keras import layers, models
            inputs = layers.Input(shape=(32, 32, 3))
            x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="last_conv_layer")(inputs)
            x = layers.Flatten()(x)
            outputs = layers.Dense(10, activation="softmax")(x)
            app.state.vision_model = models.Model(inputs=inputs, outputs=outputs)
            print("[OK] Minimal Vision CNN fallback initialized.")
        except Exception:
            app.state.vision_model = None

    # 3. Load NLP Transformer Model & Tokenizer
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_name_or_path = str(TRANSFORMER_PATH) if TRANSFORMER_PATH.is_dir() else "distilbert-base-uncased-finetuned-sst-2-english"
        app.state.nlp_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        app.state.nlp_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(DEVICE)
        app.state.nlp_model.eval()
        print(f"[OK] NLP Transformer initialized ({model_name_or_path}).")
    except Exception as exc:
        print(f"[Notice] NLP Transformer note: {exc}")
        app.state.nlp_tokenizer = None
        app.state.nlp_model = None

    # 4. Load RL D3QN Policy Network
    try:
        from d3qn_cartpole.src.model import DuelingDQN
        rl_net = DuelingDQN(state_dim=4, action_dim=2).to(DEVICE)
        if RL_MODEL_PATH.is_file():
            checkpoint = torch.load(RL_MODEL_PATH, map_location=DEVICE)
            try:
                rl_net.load_state_dict(checkpoint)
            except Exception:
                renamed = {}
                for k, v in checkpoint.items():
                    if k.startswith("feature_trunk."):
                        renamed[k.replace("feature_trunk.", "feature_network.")] = v
                    elif k.startswith("feature_network."):
                        renamed[k.replace("feature_network.", "feature_trunk.")] = v
                    else:
                        renamed[k] = v
                rl_net.load_state_dict(renamed, strict=False)
            print(f"[OK] RL D3QN Model loaded from: {RL_MODEL_PATH.name}")
        else:
            print("[OK] RL D3QN Model initialized in memory.")
        rl_net.eval()
        app.state.rl_model = rl_net
    except Exception as exc:
        print(f"[Notice] RL D3QN initialization note: {exc}")
        app.state.rl_model = None

    # 5. Initialize Production RAG Pipeline
    try:
        from rag.pipeline import RAGPipeline
        app.state.rag_pipeline = RAGPipeline()
        # Seed core knowledge base
        app.state.rag_pipeline.ingest_document(
            doc_id="ai_portfolio_overview",
            text=(
                "The Unified Multi-Domain AI Portfolio Hub consolidates four enterprise machine learning pillars: "
                "1. Tabular Churn Prediction utilizing CatBoost and Scikit-Learn with cost-optimal Bayes thresholding and SHAP explainability. "
                "2. Computer Vision with Keras CNN and dynamic Grad-CAM heatmaps for interpretable CIFAR-10 classification. "
                "3. NLP Contextual Sentiment Analysis using Hugging Face DistilBERT fine-tuned for high-throughput inference. "
                "4. Continuous and Discrete Deep Reinforcement Learning with PER-D3QN and Soft Actor-Critic (SAC). "
                "The infrastructure is orchestrated using Docker Compose across an isolated bridge network with automated health checks."
            ),
        )
        print("[OK] Production RAG Pipeline initialized and seeded.")
    except Exception as exc:
        print(f"[Notice] RAG Pipeline note: {exc}")
        app.state.rag_pipeline = None

    print("=" * 65 + "\n")
    yield


app = FastAPI(
    title="Unified Multi-Domain AI Portfolio Gateway",
    description="High-performance asynchronous gateway consolidating Tabular, Vision, NLP, RL, and Generative RAG systems.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Domain Routers with /api/v1 Prefix
app.include_router(tabular.router, prefix="/api/v1")
app.include_router(vision.router, prefix="/api/v1")
app.include_router(nlp.router, prefix="/api/v1")
app.include_router(rl.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")


@app.get("/health", response_model=SystemHealthResponse, tags=["Diagnostics"])
async def health_check():
    """
    Global readiness and telemetry endpoint verifying residency for all five AI models/pipelines.
    """
    models_status = {
        "tabular_churn": getattr(app.state, "churn_model", None) is not None,
        "vision_cnn": getattr(app.state, "vision_model", None) is not None,
        "nlp_transformer": getattr(app.state, "nlp_model", None) is not None,
        "rl_d3qn": getattr(app.state, "rl_model", None) is not None,
        "generative_rag": getattr(app.state, "rag_pipeline", None) is not None,
    }
    is_healthy = any(models_status.values())
    return SystemHealthResponse(
        status="healthy" if is_healthy else "degraded",
        device=str(DEVICE),
        models_loaded=models_status,
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "title": "Unified Multi-Domain AI Portfolio Hub API",
        "documentation": "/docs",
        "version": "1.1.0",
        "domains": ["/api/v1/tabular", "/api/v1/vision", "/api/v1/nlp", "/api/v1/rl", "/api/v1/rag"],
    }

