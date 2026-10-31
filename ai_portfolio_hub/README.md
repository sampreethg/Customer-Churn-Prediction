# Enterprise Multi-Domain AI/ML Systems & Portfolio Hub

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose_v2-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Accelerated-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, containerized AI engineering portfolio hub demonstrating production machine learning architectures across five foundational domains: **Leak-Free Tabular ML & MLOps**, **Explainable Computer Vision (Grad-CAM)**, **Modern NLP & Inference Optimization (DistilBERT + ONNX INT8)**, **Deep Reinforcement Learning (D3QN with PER & Continuous SAC)**, and **Production Retrieval-Augmented Generation (RAG)**. 

The platform provides an asynchronous **FastAPI** gateway microservice, an interactive 5-domain **Streamlit** dashboard, automated data drift monitoring using **Evidently AI**, and zero-downtime containerized orchestration via **Docker Compose**.

---

## System Architecture

```text
                           [ INGRESS CLIENTS ]
                        (Browser UI / REST APIs)
                                   |
                                   v
             +-----------------------------------------------+
             | Streamlit Unified Multi-Page Hub |
             | (Port 8501: UI Layer) |
             +-----------------------------------------------+
                                   |
                                   | HTTP / JSON / Multipart
                                   v
+---------------------------------------------------------------------------------------+
| FastAPI Asynchronous Gateway Service |
| (Port 8000: Gateway Core) |
| +---------------------------------------------------------------------------------+ |
| | Lifespan Preloader: Tabular Model | CNN Model | DistilBERT Pipeline | D3QN Net | |
| +---------------------------------------------------------------------------------+ |
+---------------------------------------------------------------------------------------+
        | | | |
        v v v v
 [ /api/v1/tabular ] [ /api/v1/vision ] [ /api/v1/nlp ] [ /api/v1/rl ]
        | | | |
        v v v v
 +---------------+ +---------------+ +---------------+ +---------------+
 | Scikit-Learn | | TensorFlow | | DistilBERT / | | PyTorch D3QN |
 | Pipeline | | Keras CNN | | ONNX Runtime | | (CartPole-v1) |
 |---------------| |---------------| |---------------| |---------------|
 | • Scaler/OHE | | • Conv2D/Pool | | • HuggingFace | | • SumTree PER |
 | • Cost Tuned | | • Batch Norm | | • FP32 / INT8 | | • Double DQN |
 | • Tree SHAP | | • Grad-CAM | | • Cross-Attn | | • Dueling V/A |
 +---------------+ +---------------+ +---------------+ +---------------+
        \ / \ /
         \ / \ /
          v v v v
 +------------------------------------+ +-------------------------------+
 | Domain 5: Production RAG Hub | | Continuous Deep RL (SAC) |
 |------------------------------------| |-------------------------------|
 | • Bi-Encoder: all-MiniLM-L6-v2 | | • Maximum Entropy Objective |
 | • Vector Store: FAISS IndexFlatIP | | • Clipped Twin Critics |
 | • Cross-Encoder: ms-marco-MiniLM | | • Squashed Gaussian Policy |
 | • Context Grounding & Citations | | • Auto Temperature Alpha |
 +------------------------------------+ +-------------------------------+
                   \ /
                    \ /
                     v v
+---------------------------------------------------------------------------------------+
| Continuous Observability & Hardware Optimization |
| • Evidently AI Drift Engine (Kolmogorov-Smirnov & Population Stability Index) |
| • ONNX Runtime INT8 Post-Training Quantization (PTQ) Engine |
| • Asynchronous Pytest & HTTPX Non-Blocking Smoke-Test Suite |
+---------------------------------------------------------------------------------------+
```

---

## Key Architectural Pillars & Technical Implementations

### 1. Tabular Machine Learning & Cost-Sensitive MLOps
* **Leak-Free Pipelines:** Strict segregation of fit/transform states using Scikit-Learn `ColumnTransformer` (`StandardScaler` + `OneHotEncoder` with `handle_unknown='ignore'`).
* **Benchmarked Estimators:** Comparative validation of `RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`, and `CatBoostClassifier`.
* **Cost-Sensitive Threshold Tuning ($\tau^*$):** Bypasses arbitrary 0.5 decision thresholds via business loss-matrix minimization ($C_{\text{FN}} = \$500$, $C_{\text{FP}} = \$100$).
* **Local Explainability:** Dynamic TreeSHAP attribution extraction returning exact feature importance vectors.

### 2. Computer Vision & Explainable AI (Grad-CAM)
* **Convolutional Topology:** Multi-block 2D CNN with Dropout regularization and Batch Normalization.
* **Data Augmentation:** Real-time spatial affine transforms (random horizontal flips, rotations, and zooms).
* **Explainability Layer:** Explicit mathematical implementation of Gradient-weighted Class Activation Mapping (Grad-CAM):
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
  Heatmaps are dynamically mapped to `cv2.COLORMAP_JET`, blended over the original input, and returned as base64-encoded PNG strings.

### 3. Contextual Transformer NLP & Inference Acceleration
* **Subword Sequence Classification:** Fine-tuned `distilbert-base-uncased` with contextual WordPiece tokenization and custom classification heads.
* **ONNX Acceleration:** Model graph exported with dynamic batch/sequence axes, converted into an optimized compute graph.
* **INT8 Dynamic Quantization:** Post-Training Quantization (PTQ) scaling FP32 parameters to 8-bit integers, reducing binary footprints and slashing CPU inference latency.

### 4. Advanced Reinforcement Learning (PER-D3QN & SAC)
* **Dueling Architecture:** Resolves parameter identifiability by factorizing action-values into scalar state values and mean-centered advantage vectors:
  $$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$
* **Double DQN Target:** Eliminates positive maximization overestimation bias by using the online network for action selection and the target network for valuation.
* **Prioritized Experience Replay (PER):** Complete array-backed binary `SumTree` enabling $O(\log N)$ priority updates and stratified segment sampling with annealed Importance Sampling (IS) weights.
* **Continuous Control (Soft Actor-Critic):** Off-policy maximum entropy formulation over `Pendulum-v1` using clipped dual Q-critics and squashed Gaussian reparameterized policies ($a = \tanh(u) \cdot \text{scale}$).

### 5. Production RAG Document Intelligence
* **Ingestion & Chunking:** Recursive character splitting across PDF and Markdown documentation with boundary-aware sliding-window overlaps.
* **Two-Tier Retrieval:** First-stage dense semantic retrieval via `all-MiniLM-L6-v2` against a normalized `faiss.IndexFlatIP` vector index; second-stage candidate re-ranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **Grounded Attribution:** Enforces factual citation constraints, extracting document names, chunk indices, and exact reference quotes.

---

## CLI Verification Guide

```bash
# 1. Automated Baseline Artifact Generation
python3 scripts/generate_baseline_artifacts.py

# 2. Asynchronous Multi-Service Smoke Tests
python3 scripts/smoke_test_gateway.py

# 3. Benchmark ONNX Runtime INT8 Inference
python3 scripts/optimize_onnx.py

# 4. MLOps Covariate Shift & Drift Audits
python3 mlops/drift_monitor.py

# 5. Run All Evaluations
./run_all_evaluations.sh
```

---

## Security Architecture

The portfolio hub enforces defense-in-depth security hardening across all model inference, execution, serialization, and notification pipelines:

* **Tiered Asynchronous Rate Limiting:** An `asyncio.Lock`-synchronized sliding-window in-memory rate limiter protects compute-intensive endpoints against resource exhaustion and denial-of-service (DoS) bursts:
  - `/api/v1/tabular/predict` & `/api/v1/tabular/explain`: **10-60 req/min** per client IP.
  - `/api/v1/vision/classify` (CNN + Grad-CAM saliency mapping): **15 req/min** per client IP.
  - `/api/v1/rl/playback` (Gym CartPole simulation rollout): **10 req/min** per client IP.
  - `/api/v1/rag/ingest` & `/api/v1/rag/index-docs` (vector indexing): **5 req/min** per client IP.
  - Returns `HTTP 429 Too Many Requests` with dynamic `Retry-After` header. Features proxy-aware IP extraction (`X-Forwarded-For` with `TRUST_PROXIES` gating) and background stale-bucket memory pruning.
* **Unified Authentication Middleware:** Constant-time API Key (`X-API-Key`) and Bearer token (`Authorization: Bearer <token>`) security dependency guards all domain inference and indexing endpoints via `secrets.compare_digest` to prevent timing attacks. Container orchestration and observability probes (`/health`, `/ready`, `/docs`, `/openapi.json`) remain unauthenticated for zero-friction readiness checks.
* **Zero-Trust Webhook Egress (SSRF Defense):** Outbound notification pipelines enforce strict HTTPS-only transport, configurable destination domain allowlists (`WEBHOOK_ALLOWED_DOMAINS`, defaulting to `hooks.slack.com`, `discord.com`, `api.crm.example.com`), and pre-dispatch DNS resolution blocking private subnets (RFC 1918 `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopback (`127.0.0.0/8`), link-local metadata services (`169.254.169.254`), and IPv6 non-global addresses.
* **Safe Weight Deserialization:** Complete elimination of arbitrary Python pickle deserialization and remote code execution (RCE) vectors by enforcing `weights_only=True` across all `torch.load()` call sites in model runtime, optimization, and reinforcement learning agents.
* **Automated Security Verification:** Dedicated regression test suite (`tests/test_security_remediation.py`) verifying all 11 security attack vectors with a 100% test pass rate across 68 automated test cases.

---

## Engineering Credits & Contributors

* **Sampreeth G**
  - **Role:** ML Systems & Application Security Architect
  - **Contributions:** Architected and implemented the tiered in-memory rate limiting engine, constant-time API key and bearer token security dependencies, outbound SSRF validation pipeline, PyTorch deserialization hardening (`weights_only=True`), and the automated security regression test suite.

