# Multi-Container Docker Architecture: Customer Churn Intelligence Platform

This repository provides an enterprise-ready, containerized deployment of the Customer Churn Prediction platform orchestrated via **Docker Compose**. The architecture separates the high-performance **FastAPI inference backend** and the **Streamlit analytical dashboard** into isolated, non-root containers communicating across an internal bridge network with automated healthcheck synchronization and shared volume mounts.

---

## Architecture Overview

```
                      +---------------------------------------+
                      | Host Machine |
                      +---------------------------------------+
                                  | |
                    Port 8501 | | Port 8000
                    (Browser) | | (REST / OpenAPI)
                                  v v
                 +-------------------+ +-------------------+
                 | churn_dashboard | | churn_api |
                 | (Streamlit: 8501) | | (FastAPI: 8000) |
                 | User: dashuser | | User: apiuser |
                 +-------------------+ +-------------------+
                           | ^
                           | HTTP Requests via DNS |
                           +------------------------------+
                               (http://api:8000)
                                        |
                 ===================================================
                       Docker Custom Bridge: churn_network
                 ===================================================
                                        |
                 +-------------------------------------------------+
                 | Host Read-Only Volumes (:ro) |
                 | - ./models -> /app/models:ro |
                 | - ./artifacts -> /app/artifacts:ro |
                 +-------------------------------------------------+
```

### Key Architectural Highlights
1. **Microservice Separation**:
   - `api` (`churn_api`): ASGI service running Uvicorn with 2 workers serving single prediction, batch prediction, SHAP feature attributions, and health check endpoints.
   - `dashboard` (`churn_dashboard`): Interactive analytical dashboard running Streamlit in headless mode, querying the API container via internal DNS name `http://api:8000`.
2. **Container Security & Best Practices**:
   - **Slim Base Image**: Built atop `python:3.11-slim` to minimize image surface area and eliminate unnecessary system tools.
   - **Unprivileged Non-Root Execution**: Both services create dedicated system users (`apiuser` with UID 1000 in API; `dashuser` with UID 1000 in Dashboard) to adhere to the principle of least privilege.
   - **Layer Caching**: `requirements.txt` is copied and installed prior to copying application source code, leveraging Docker cache layers.
   - **Optimized Python Environment**: `PYTHONDONTWRITEBYTECODE=1` prevents `.pyc` clutter; `PYTHONUNBUFFERED=1` guarantees real-time log streaming without buffer delays.
3. **Automated Healthchecks & Dependency Ordering**:
   - `churn_api`: Tests `http://localhost:8000/health` every 15s (5s timeout, 3 retries, 10s start period).
   - `churn_dashboard`: Tests `http://localhost:8501/_stcore/health` with identical intervals.
   - `depends_on: { api: { condition: service_healthy } }` guarantees Streamlit only initializes once the FastAPI model pipeline and SHAP engine are fully loaded and operational.
4. **Volume Decoupling**:
   - Host directories `./models` and `./artifacts` are mounted as read-only (`:ro`) into `/app/models` and `/app/artifacts`, ensuring model updates can be applied without image rebuilds while safeguarding artifacts against container-level modification.

---

## Directory Structure

```
churn_prediction/
 docker/
 Dockerfile.api # Slim container definition for FastAPI inference backend
 Dockerfile.dashboard # Slim container definition for Streamlit UI dashboard
 docker-compose.yml # Multi-service composition and orchestration manifest
 .dockerignore # Build context exclusion rules
 README_DOCKER.md # Docker lifecycle, management, and troubleshooting guide
```

---

## Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/) (v20.10+ or v24+)
- [Docker Compose](https://docs.docker.com/compose/) (v2.20+ or Docker Compose v5+)

Verify local installations:
```bash
docker --version
docker compose version
```

---

## Quickstart & Service Lifecycle

### 1. Build and Launch Containers
Build both container images and run the services in the background (detached mode):
```bash
docker compose up --build -d
```

### 2. Inspect Running Services and Health Probes
Verify that both containers achieve `healthy` status:
```bash
docker compose ps
```
*Expected Output:*
```
NAME IMAGE COMMAND SERVICE CREATED STATUS PORTS
churn_api customer-churn-prediction-api "uvicorn api.app:app…" api 30 seconds ago Up 30 seconds (healthy) 0.0.0.0:8000->8000/tcp
churn_dashboard customer-churn-prediction-dashboard "streamlit run dashb…" dashboard 20 seconds ago Up 20 seconds (healthy) 0.0.0.0:8501->8501/tcp
```

### 3. Stream Live Aggregate Logs
Follow logs from both services simultaneously, or filter by service:
```bash
# Stream logs for all services
docker compose logs -f

# Stream logs specifically for FastAPI backend
docker compose logs -f api

# Stream logs specifically for Streamlit dashboard
docker compose logs -f dashboard
```

### 4. Access Interactive User Interfaces & Endpoints
- **Streamlit Analytical Dashboard**: [http://localhost:8501](http://localhost:8501)
  - Features real-time churn risk simulation, local SHAP waterfall feature attribution, CSV batch cohort scoring, and diagnostic charts.
- **FastAPI OpenAPI Interactive Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **FastAPI ReDoc Schema Explorer**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Service Health Probe**: [http://localhost:8000/health](http://localhost:8000/health)

### 5. Graceful Teardown
Stop and remove containers, networks, and allocated volumes:
```bash
docker compose down -v
```

---

## Verification & API Testing

### Test Health Endpoint
```bash
curl -X GET http://localhost:8000/health
```
*Response:*
```json
{
  "status": "healthy",
  "model_loaded": true,
  "explainer_loaded": true,
  "model_artifact": "/app/models/best_churn_model.joblib"
}
```

### Test Single Customer Inference (`/predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 12,
    "monthly_charges": 85.0,
    "total_charges": 1020.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No"
  }'
```
*Sample Response:*
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.742,
  "risk_level": "High",
  "recommended_action": "Offer targeted loyalty discount and schedule priority retention callback."
}
```

### Test SHAP Local Explainability (`/explain`)
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 6,
    "monthly_charges": 95.0,
    "total_charges": 570.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No"
  }'
```

---

## Production Operations & Troubleshooting

### Validating Compose Configuration Syntax
Before deployment, validate YAML structure and variable resolution:
```bash
docker compose config
```

### Inspecting Container Internals & Shell Access
If debugging is required within an unprivileged container shell:
```bash
# Open interactive bash session in the API container (runs as apiuser)
docker compose exec api /bin/bash

# Open interactive bash session in the Dashboard container (runs as dashuser)
docker compose exec dashboard /bin/bash
```

### Investigating Container Health Probe Failures
Inspect health status details directly from Docker inspect:
```bash
# Check raw healthcheck state for API container
docker inspect --format='{{json .State.Health}}' churn_api

# Check raw healthcheck state for Dashboard container
docker inspect --format='{{json .State.Health}}' churn_dashboard
```

### Hot-Reloading Model Artifacts
Because `./models` and `./artifacts` are mounted as host volumes:
1. Re-train the model pipeline on the host machine (`python -m src.train` or `python main.py`).
2. Restart the API container to reload the lifespan pipeline without rebuilding container images:
   ```bash
   docker compose restart api
   ```

