# Customer Churn Prediction System, API, Dashboard & Alerting Engine

An end-to-end, production-grade Customer Churn Prediction system, enterprise FastAPI inference microservice, interactive Streamlit analytical simulator, SHAP Explainable AI attribution engine, and asynchronous Retention Alerting & Webhook Notification Engine in Python.

---

## 📌 Architecture & Design Highlights

1. **Leakage-Free Preprocessing**:
   - Continuous feature scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) are encapsulated inside a Scikit-Learn `Pipeline` combined with `ColumnTransformer`.
   - Feature transformers are fitted **strictly** within training folds during cross-validation, guaranteeing zero test/holdout distribution leakage.
2. **Algorithm-Level Class Imbalance Handling**:
   - Handled structurally using **Stratified 5-Fold Cross Validation** (`stratify=y`) and algorithmic cost-sensitivity (`class_weight='balanced'`) in the Random Forest ensemble.
3. **Automated Hyperparameter Optimization**:
   - `GridSearchCV` searches optimal tree depth (`max_depth`), ensemble size (`n_estimators`), and leaf constraints (`min_samples_leaf`, `min_samples_split`) optimized for F1 score.
4. **Explainable AI Engine (SHAP)**:
   - Uses `shap.TreeExplainer` on the fitted Random Forest estimator post-`ColumnTransformer`.
   - Computes expected baseline value ($E[\text{Churn}] = 50.08\%$) and per-feature Shapley values for individual queries.
   - Decomposes predictions into positive (risk-increasing) and negative (loyalty-protecting) forces.
   - Generates automated human-readable narratives explaining the score.
5. **Asynchronous Retention Alerting & Webhook Engine**:
   - **Non-Blocking Dispatch**: Uses FastAPI's native `BackgroundTasks` to send webhook dispatches out-of-band, preserving sub-millisecond API response times.
   - **Rich Payload Builders**: Constructs Slack Block Kit and generic REST CRM (HubSpot/Salesforce) payloads detailing Account ID, churn probability, primary SHAP drivers, and concrete retention playbooks.
   - **Resilience & Exponential Backoff**: Wrapped with `tenacity` retries (3 attempts with exponential backoff on HTTP 5xx and connection timeouts) and graceful error recovery so webhook failures never crash primary inference.
6. **Enterprise-Ready FastAPI Inference Service**:
   - **Lifespan Context Management**: Model, explainer, and webhook notifier loaded once at startup into memory.
   - **Pydantic v2 Schema Contracts**: Strict field bounds (`ge`, `le`) and enumerated categorical types.
   - **Endpoints**: `GET /health`, `POST /predict`, `POST /predict/batch`, `POST /explain`.
7. **Interactive Streamlit Analytical Dashboard**:
   - **Dual Inference Mode**: Automatically queries the FastAPI backend over HTTP; falls back to local artifact inference (`models/best_churn_model.joblib`) via `@st.cache_resource` if the API is offline.
   - **Tab 1: Single Customer Risk Simulator & XAI**: Sliders, Plotly gauge visualization, and real-time SHAP factor breakdown chart.
   - **Tab 2: Batch Cohort Scoring & Prioritization**: Upload CSV rosters, auto-score attrition risk, inspect distribution donut charts, filter flight risks, and export prioritized retention task lists.
   - **Tab 3: Model Insights & Performance**: Visualizes training metrics, confusion matrix, ROC curve, and feature importances.

---

## 📁 Project Structure

```text
Customer Churn Prediction/
├── .github/
│   └── workflows/
│       └── ci.yml               # Production-grade GitHub Actions CI/CD automation workflow
├── api/
│   ├── __init__.py               # API package exports
│   ├── schemas.py                # Pydantic v2 request/response contracts, Enums, XAI, and alert fields
│   └── app.py                    # FastAPI application, lifespan loader, background alerting, and routes
├── dashboard/
│   ├── __init__.py               # Dashboard package exports
│   ├── components.py             # UI components, Plotly gauges, SHAP attribution bars, risk cards
│   └── app.py                    # Streamlit entry point, tabs, and dual-mode inference
├── data/
│   ├── generate_data.py          # Deterministic synthetic data generator simulating realistic churn behavior
│   └── churn_dataset.csv         # Generated dataset (2,500 samples, ~26% churn rate)
├── docker/
│   ├── Dockerfile.api            # Slim container definition for FastAPI microservice
│   └── Dockerfile.dashboard      # Container definition for Streamlit analytics dashboard
├── src/
│   ├── __init__.py               # ML package initialization and exports
│   ├── alerting.py               # Asynchronous webhook dispatcher, Slack/CRM formatters, retry logic
│   ├── config.py                 # Paths, hyperparameter grids, random seeds, and feature lists
│   ├── explainer.py              # SHAP TreeExplainer engine, column mapping, and waterfall plotting
│   ├── pipeline.py               # Preprocessing pipelines and model estimator wrapping
│   ├── train.py                  # Stratified K-Fold CV, Grid Search, model persistence
│   ├── evaluate.py               # Evaluation metrics, confusion matrix, ROC-AUC, classification report
│   └── explain.py                # Global feature importance extraction and top driver insights
├── models/
│   └── best_churn_model.joblib   # Persisted optimized pipeline artifact
├── artifacts/
│   ├── confusion_matrix.png      # Holdout confusion matrix plot
│   ├── roc_curve.png             # Receiver Operating Characteristic (ROC) curve plot
│   └── feature_importance.png    # Top driver feature importance horizontal bar chart
├── tests/
│   ├── __init__.py               # Test package initialization
│   ├── conftest.py               # Shared pytest fixtures (mock data, client fixtures, temp dirs)
│   ├── test_alerting.py          # Pytest suite for webhook builders, dispatch, and retry behavior
│   ├── test_api.py               # Pytest suite for API endpoints, validation, and XAI
│   ├── test_pipeline.py          # Leakage absence, reproducibility, and dimensional tests
│   └── test_training_and_evaluation.py # Model training, evaluation metrics, and SHAP tests (31 total)
├── Dockerfile.api                # Production root-level FastAPI container
├── Dockerfile.dashboard          # Production root-level Streamlit container
├── docker-compose.yml            # Multi-service container orchestration
├── pyproject.toml                # Ruff configuration, pytest options, and coverage threshold (>= 80%)
├── requirements.txt              # Pinned production runtime dependencies
├── requirements-dev.txt          # Development, linting, testing, and CI/CD tools
└── README.md                     # Comprehensive setup, architecture, and API documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Recommended: Virtual environment (`venv` or `conda`)

### Step 1: Navigate to Directory
```bash
cd "C:\Users\sampreeth\OneDrive\Documents\Intenship\Customer Churn Prediction"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Variables (Webhook Configuration)

Configure webhook URLs in your environment or a `.env` file:

```bash
# Outbound Webhook Destinations (Optional, defaults to dry-run logging if omitted)
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
export CRM_WEBHOOK_URL="https://api.yourcrm.com/webhooks/churn-alerts"

# Threshold for triggering proactive alerts (Default: 0.70)
export ALERT_THRESHOLD="0.70"

# Application Environment Tag (e.g., production, staging)
export APP_ENV="production"
```

---

## 🚀 Execution Guide

### 1. Run Machine Learning Pipeline
```bash
python main.py
```

### 2. Launch FastAPI Inference Microservice
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation: **`http://localhost:8000/docs`**  
OpenAPI schema: **`http://localhost:8000/openapi.json`**

### 3. Launch Streamlit Analytical Dashboard
```bash
streamlit run dashboard/app.py
```
Opens interactive web application at **`http://localhost:8501`**.

### 4. Run Automated Pytest Suite & Coverage Verification (All 31 Tests)
```bash
# Run full test suite
pytest tests/ -v

# Run with strict coverage threshold gate (>= 80%)
pytest tests/ --cov=src --cov=api --cov-report=term-missing --cov-fail-under=80

# Run Ruff code quality & formatting checks
ruff check .
ruff format --check .
```

---

## 🔄 CI/CD Pipeline & Code Quality Automation

The project includes an enterprise-grade GitHub Actions CI/CD automation workflow (`.github/workflows/ci.yml`) triggered on every push and pull request to `main`/`master`:

1. **Job 1: `lint` (Ruff Static Analysis)**:
   - Enforces Python 3.11 code standards, clean import order, and code quality rules (`E`, `W`, `F`, `I`, `B`).
   - Executes `ruff check .` and `ruff format --check .`.
2. **Job 2: `test` (Pytest & Coverage Gate)**:
   - Bootstraps synthetic data (`python -m data.generate_data`) and trains pipeline artifacts (`python -m src.train`).
   - Executes 31 automated tests covering pipeline data leakage absence, model reproducibility, preprocessor dimensions, FastAPI contracts, and XAI explainers.
   - Enforces a minimum code coverage gate ($\ge 80\%$) across `src/` and `api/` via `pytest-cov`.
   - Generates and uploads `coverage.xml` as a CI workflow artifact.
3. **Job 3: `docker-build` (Build Matrix)**:
   - Utilizes Docker Buildx to concurrently build production container images:
     - `Dockerfile.api` -> `customer-churn-api:latest`
     - `Dockerfile.dashboard` -> `customer-churn-dashboard:latest`


---

## 📡 API Endpoints & Usage

### 1. Single Customer Prediction with Background Alert: `POST /predict`
**Request:**
```json
{
  "Customer_ID": "CUST-HOT-01",
  "Tenure_Months": 2,
  "MonthlyCharges": 105.00,
  "TotalTickets": 4,
  "ContractType": "Month-to-month",
  "TechSupport": "No",
  "PaymentMethod": "Electronic check",
  "notify_retention_team": true
}
```
**Response:**
```json
{
  "Customer_ID": "CUST-HOT-01",
  "churn_prediction": 1,
  "churn_probability": 0.9501,
  "risk_level": "High",
  "recommended_action": "Offer targeted loyalty discount and schedule priority retention callback.",
  "alert_dispatched": true
}
```
*(When `churn_probability >= 0.70` and `notify_retention_team=true`, an alert payload is dispatched to Slack and CRM webhooks out-of-band via FastAPI `BackgroundTasks`.)*

### 2. Explainable AI Attribution: `POST /explain`
**Request:**
```json
{
  "Tenure_Months": 2,
  "MonthlyCharges": 105.00,
  "TotalTickets": 4,
  "ContractType": "Month-to-month",
  "TechSupport": "No",
  "PaymentMethod": "Electronic check"
}
```
**Response:**
```json
{
  "churn_probability": 0.9299,
  "base_value": 0.5008,
  "top_risk_drivers": [
    {
      "feature_name": "Tenure_Months",
      "feature_value": -1.6533,
      "shap_value": 0.1506
    },
    {
      "feature_name": "MonthlyCharges",
      "feature_value": 1.4589,
      "shap_value": 0.1142
    }
  ],
  "top_retention_drivers": [
    {
      "feature_name": "PaperlessBilling_Yes",
      "feature_value": 1.0,
      "shap_value": -0.0124
    }
  ],
  "summary_narrative": "Churn risk is primarily accelerated by Tenure_Months, MonthlyCharges."
}
```

### 3. Batch Prediction: `POST /predict/batch`
Vectorized multi-account inference with automatic per-account alert scheduling.

---

## 🎯 Operational Risk Tiers & Business Actions

| Probability Tier | Operational Risk Level | Prescribed Business Policy | Webhook Alert |
| :--- | :--- | :--- | :--- |
| **P ≥ 0.70** | **High** | Offer targeted loyalty discount and schedule priority retention callback. | 🚨 Dispatched |
| **0.35 ≤ P < 0.70** | **Medium** | Send tailored engagement check-in and product optimization guide. | Standard Log |
| **P < 0.35** | **Low** | No retention intervention required. Maintain standard engagement. | Suppressed |
