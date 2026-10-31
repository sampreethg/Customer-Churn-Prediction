"""
Unified Multi-Domain AI Portfolio Showcase Hub.

Interactive Streamlit front-end integrating:
1. Tabular Churn Intelligence & SHAP
2. Computer Vision CNN Classification & Grad-CAM
3. Contextual Transformer NLP Sentiment Analysis
4. Deep Reinforcement Learning (PER-D3QN CartPole-v1)
"""

import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st

# Configure path so dashboard components and tabs resolve smoothly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.tabs.tab_nlp import render_tab_nlp
from dashboard.tabs.tab_rag import render_rag_tab
from dashboard.tabs.tab_rl import render_tab_rl
from dashboard.tabs.tab_tabular import render_tab_tabular
from dashboard.tabs.tab_vision import render_tab_vision

# Streamlit Page Setup
st.set_page_config(
    page_title="Unified AI Portfolio Showcase Hub",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .hub-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .hub-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-healthy {
        background-color: #dcfce7;
        color: #15803d;
    }
    .badge-degraded {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.svg",
        width=130,
    )
    st.markdown("### Portfolio Architecture Hub")
    st.markdown(
        """
        Consolidating 4 Production AI Domains into a Unified Asynchronous Gateway:
        - **Tabular:** CatBoost / XGBoost / SHAP
        - **Vision:** Multi-Block CNN / Grad-CAM
        - **NLP:** DistilBERT Contextual Transformers
        - **RL:** Dueling Double DQN (CartPole-v1)
        """
    )
    st.markdown("---")
    st.markdown("### Gateway Settings")
    api_base_url = st.text_input(
        "FastAPI Gateway URL",
        value=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="Internal container DNS or external host URL for REST endpoints.",
    )

    # Health check probe
    gateway_status = "Disconnected"
    device_name = "CPU"
    uptime_sec = 0.0
    models_dict = {}

    try:
        t0 = time.perf_counter()
        h_resp = requests.get(f"{api_base_url}/health", timeout=2)
        lat = (time.perf_counter() - t0) * 1000.0
        if h_resp.status_code == 200:
            h_data = h_resp.json()
            gateway_status = h_data.get("status", "healthy")
            device_name = h_data.get("device", "cpu")
            uptime_sec = h_data.get("uptime_seconds", 0.0)
            models_dict = h_data.get("models_loaded", {})
    except Exception:
        lat = 0.0

    st.markdown("---")
    st.markdown("### System Telemetry")
    status_class = "badge-healthy" if gateway_status == "healthy" else "badge-degraded"
    st.markdown(
        f"""
        - **Gateway Status:** <span class="status-badge {status_class}">{gateway_status.upper()}</span>
        - **Compute Device:** `{device_name.upper()}`
        - **Round-Trip Ping:** `{lat:.1f} ms`
        - **Uptime:** `{uptime_sec / 60:.1f} mins`
        """,
        unsafe_allow_html=True,
    )

# Main Application Header
st.markdown('<div class="hub-title"> Unified Multi-Domain AI Portfolio Hub</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hub-subtitle">Production-grade microservice architecture serving Tabular Intelligence, Computer Vision with Grad-CAM, '
    'Transformer NLP, and Deep Reinforcement Learning.</div>',
    unsafe_allow_html=True,
)

# Tabs Navigation
tab_labels = [
    " 1. Tabular Churn (SHAP)",
    " 2. Computer Vision (Grad-CAM)",
    " 3. Transformer NLP (DistilBERT)",
    " 4. Reinforcement Learning (D3QN)",
    " 5. Generative AI (RAG)",
]
t1, t2, t3, t4, t5 = st.tabs(tab_labels)

with t1:
    render_tab_tabular(api_base_url)

with t2:
    render_tab_vision(api_base_url)

with t3:
    render_tab_nlp(api_base_url)

with t4:
    render_tab_rl(api_base_url)

with t5:
    render_rag_tab(api_base_url)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding: 10px;'>"
    "Unified Multi-Domain AI Platform • FastAPI Backend • Streamlit Frontend • Docker Compose Orchestration"
    "</div>",
    unsafe_allow_html=True,
)

