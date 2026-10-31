# Unified Multi-Domain AI Portfolio Showcase Hub

A production-grade, full-stack microservice platform consolidating four foundational AI/ML disciplines into an asynchronous **FastAPI** gateway and an interactive **Streamlit** showcase studio, orchestrated with **Docker Compose**.

---

## 1. Architectural System Layout

```
                                 Client Web Browser
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
      Streamlit Showcase Studio (Port 8501)       FastAPI REST Gateway (Port 8000)
    ┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
    │ - Interactive Tabbed Portfolio       │    │ - Asynchronous Lifespan Model Loader │
    │ - Plotly Risk Gauges & Telemetry     │    │ - OpenAPI / Swagger Documentation    │
    │ - Grad-CAM Visual Heatmaps           │    │ - High-Throughput Microservice Routes│
    │ - Transformer Sentiment Analyzer     │    │ - Cross-Domain Pydantic v2 Contracts │
    │ - CartPole Physics Simulation Player │    └──────────────────┬───────────────────┘
    └──────────────────┬───────────────────┘                       │
                       │ Queries via Container DNS                 │
                       └───────────────────►───────────────────────┘
                                                   │
                ┌──────────────────────────────────┴──────────────────────────────────┐
                │                                                                     │
                ▼                                                                     ▼
    ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
    │ 1. Tabular Churn (SHAP)   │ │ 2. Vision CNN (Grad-CAM)  │ │ 3. Transformer NLP        │ │ 4. RL PER-D3QN            │
    │ CatBoost / LightGBM       │ │ Multi-Block CIFAR-10 CNN  │ │ DistilBERT SST-2 Engine   │ │ Dueling Double Deep Q-Net │
    │ Cost-Benefit Optimization │ │ Saliency Gradient Maps    │ │ Subword Attention Scoring │ │ CartPole-v1 Balancer      │
    └───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
```

---

## 2. Domain Implementations

### Domain 1: Tabular Customer Churn Intelligence (`/api/v1/tabular`)
- **Core Technology**: CatBoost / Scikit-Learn Pipeline with ColumnTransformer preprocessing.
- **Explainability**: Local SHAP feature attribution vectors computing risk-accelerating vs risk-mitigating account factors.
- **Prescriptive Analytics**: Classifies risk tiers (`Low`, `Moderate`, `High`) based on cost-benefit threshold optimization.

### Domain 2: Explainable Computer Vision (`/api/v1/vision`)
- **Core Technology**: 3-Block Convolutional Neural Network with Batch Normalization and Spatial Dropout.
- **Explainability (Grad-CAM)**: Gradient-Weighted Class Activation Mapping using `tf.GradientTape` across final convolutional feature maps.
- **Visual Output**: Returns ranked top-k Softmax probabilities and Superimposed thermal activation heatmaps blended at 40% opacity as Base64 image URIs.

### Domain 3: Contextual Transformer NLP (`/api/v1/nlp`)
- **Core Technology**: Hugging Face DistilBERT Sequence Classification model.
- **Capabilities**: Subword BPE tokenization, calibrated Softmax confidence probability distribution, and millisecond latency tracking.

### Domain 4: Deep Reinforcement Learning (`/api/v1/rl`)
- **Core Technology**: Dueling Double Deep Q-Network (D3QN) on continuous-state Gymnasium `CartPole-v1`.
- **Architectural Mechanics**: Decoupled state-value $V(s)$ and action-advantage $A(s, a)$ streams with mean-centering identifiability and Polyak soft updates.
- **Live Playback**: Real-time simulation returning step-by-step 4D physics trajectories and dynamic dual-action Q-values.

---

## 3. Project Structure

```
ai_portfolio_hub/
├── docker/
│   ├── Dockerfile.api              # Python 3.11-slim FastAPI microservice container
│   └── Dockerfile.dashboard        # Streamlit analytics frontend container
├── docker-compose.yml              # Multi-container isolated bridge orchestration
├── models/                         # Serialized model artifacts
│   ├── best_churn_model.joblib     # Tabular gradient boosting model
│   ├── best_cnn_model.keras        # Keras CNN vision model
│   └── best_per_d3qn.pt            # PyTorch D3QN reinforcement learning weights
├── api/
│   ├── __init__.py
│   ├── app.py                      # Main FastAPI application with lifespan loader
│   ├── config.py                   # Centralized environment and path settings
│   ├── schemas.py                  # Pydantic v2 data contracts across all 4 domains
│   └── routers/
│       ├── __init__.py
│       ├── tabular.py              # /api/v1/tabular router
│       ├── vision.py               # /api/v1/vision router
│       ├── nlp.py                  # /api/v1/nlp router
│       └── rl.py                   # /api/v1/rl router
├── dashboard/
│   ├── __init__.py
│   ├── app.py                      # Streamlit hub entry point with telemetry
│   ├── components.py               # Plotly gauges, SHAP bar charts, trajectory plots
│   └── tabs/
│       ├── __init__.py
│       ├── tab_tabular.py          # Churn prediction & SHAP visualizer
│       ├── tab_vision.py           # Image upload, top-k probabilities, Grad-CAM
│       ├── tab_nlp.py              # Text sentiment analyzer & confidence breakdown
│       └── tab_rl.py               # CartPole PER-D3QN playback & Q-values
├── tests/
│   ├── __init__.py
│   ├── test_gateway.py             # Integration tests for unified API endpoints
│   └── test_schemas.py             # Schema validation and bounds testing
├── requirements.txt                # Unified production dependencies
└── README.md                       # Complete operational documentation
```

---

## 4. Operational Instructions

### A. Run Automated Test Suite
```bash
cd ai_portfolio_hub
pytest tests/ -v
```

### B. Launch via Docker Compose
```bash
docker compose up --build -d
```

Verify service status:
```bash
docker compose ps
```
Both `hub_api` and `hub_dashboard` report `(healthy)`.

### C. Local Development Mode (Without Docker)

#### 1. Start FastAPI Gateway
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
*Swagger UI available at `http://localhost:8000/docs`.*

#### 2. Start Streamlit Dashboard
```bash
streamlit run dashboard/app.py --server.port=8501
```
*Open `http://localhost:8501` to explore all four interactive domains.*

