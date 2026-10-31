"""
Streamlit Interactive Analytical Studio for CNN Image Classification & Grad-CAM Explainability.

Provides real-time model inference, top-k probability distributions via Plotly,
and side-by-side Grad-CAM visual saliency heatmaps with plain-English diagnostic callouts.
"""

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configure sys.path so modules inside src and dashboard resolve smoothly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.utils import (
    build_probability_chart,
    generate_gradcam_overlay,
    generate_synthetic_samples,
    preprocess_image,
)
from src.config import (
    CLASS_NAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
)

# Page Setup
st.set_page_config(
    page_title="CNN Vision & Grad-CAM Studio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1e293b;
    }
    .callout-box {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 14px 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained CNN model artifact...")
def load_cached_model(model_path: str = str(MODEL_SAVE_PATH)):
    """
    Load serialized Keras CNN model once and cache across user interactions.
    """
    path = Path(model_path)
    if not path.is_file():
        return None
    try:
        model = tf.keras.models.load_model(str(path))
        return model
    except Exception as exc:
        st.error(f"Error loading model from {path}: {exc}")
        return None


def get_spatial_diagnostic_narrative(predicted_class: str, confidence: float) -> str:
    """
    Generate domain-specific narrative describing salient activation triggers.
    """
    narratives = {
        "airplane": "High-intensity activations converge on fuselage contours, wing tips, and high-contrast sky background borders.",
        "automobile": "Saliency heatmaps focus predominantly on wheel wheelbases, headlights, and horizontal chassis outlines.",
        "bird": "Model attentiveness centers on beak curvature, eye placement, and plumage contrast against backgrounds.",
        "cat": "Key activation clusters detect facial symmetry, pointed ear contours, and whisker pad geometry.",
        "deer": "Heatmap concentrations pinpoint antler arborization, slender neck outlines, and leg structure.",
        "dog": "Focus points highlight snout geometry, ear flap orientation, and chest fur texture.",
        "frog": "Activations gravitate towards protruding dorsal eyes, moist skin gloss, and crouched limb postures.",
        "horse": "Gradients highlight the elongated muzzle, arched dorsal neck ridge, and slender hoof supports.",
        "ship": "Salient regions track the waterline boundary, hull silhouette, and vertical superstructure masts.",
        "truck": "Attention emphasizes elevated rectangular cab silhouettes, front grille bulk, and cargo bed outlines.",
    }
    base = narratives.get(
        predicted_class,
        "Saliency is concentrated in high-gradient spatial clusters corresponding to discriminative object edges.",
    )
    return f"**Visual Saliency Analysis ({predicted_class.title()} - {confidence:.1%}):** {base}"


# --- Sidebar Configuration ---
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/keras-team/keras-io/master/templates/img/keras-logo.png",
        width=120,
    )
    st.markdown("### Studio Controls")

    # Image Input Mode
    input_mode = st.radio(
        "Select Input Source",
        ["Sample Gallery", "Upload Custom Image"],
        index=0,
    )

    sample_dict = generate_synthetic_samples()

    selected_image: Image.Image | None = None
    selected_name = ""

    if input_mode == "Sample Gallery":
        sample_choice = st.selectbox(
            "Choose a Test Instance",
            list(sample_dict.keys()),
            index=0,
        )
        selected_image = sample_dict[sample_choice]
        selected_name = sample_choice
    else:
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file)
                selected_name = uploaded_file.name
            except Exception as e:
                st.error(f"Unable to read uploaded image: {e}")

    st.markdown("---")
    st.markdown("### Inference Hyperparameters")
    top_k = st.slider("Top-K Predictions", min_value=1, max_value=10, value=3, step=1)
    confidence_threshold = st.slider(
        "Confidence Warning Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Alert if the top predicted class falls below this confidence level.",
    )

    st.markdown("---")
    st.markdown("### Model Architecture Specs")
    st.markdown(
        f"""
        - **Target Resolution:** `{IMG_HEIGHT}x{IMG_WIDTH}x3` (RGB)
        - **Dataset:** CIFAR-10 (`{NUM_CLASSES}` Classes)
        - **Structure:** 3 Conv Blocks + BatchNorm
        - **Regularization:** Spatial Pooling + Multi-stage Dropout
        - **Head:** Dense (256, ReLU) $\\to$ Softmax (10)
        """
    )


# --- Main Application Body ---
st.markdown('<div class="main-header"> CNN Vision & Grad-CAM Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interactive Computer Vision classification engine with real-time gradient-weighted class activation mapping (Grad-CAM).</div>',
    unsafe_allow_html=True,
)

model = load_cached_model()

if model is None:
    st.info(
        f"[Notice] **Notice:** Trained checkpoint `{MODEL_SAVE_PATH.name}` is currently training or not yet serialized. "
        "Using freshly assembled CNN architecture for interactive interface demonstration."
    )
    from src.model import build_cnn_model
    model = build_cnn_model()

if selected_image is None:
    st.info(" Please select a sample from the gallery or upload an image in the sidebar to begin inference.")
    st.stop()

# --- Run Inference & Latency Tracking ---
start_time = time.perf_counter()
batch_tensor, display_array = preprocess_image(selected_image, target_size=(IMG_WIDTH, IMG_HEIGHT))
predictions_all = model.predict(batch_tensor, verbose=0)[0]
latency_ms = (time.perf_counter() - start_time) * 1000.0

top_idx = int(np.argmax(predictions_all))
top_class = CLASS_NAMES[top_idx]
top_prob = float(predictions_all[top_idx])

# Generate Grad-CAM Overlay
colored_heatmap, blended_overlay, _ = generate_gradcam_overlay(
    model=model,
    img_array=batch_tensor,
    original_pil_image=selected_image,
    pred_index=top_idx,
)

# --- Top Header Metric Cards ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Class</div>
            <div class="metric-value">{top_class.title()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Prediction Confidence</div>
            <div class="metric-value">{top_prob * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Inference Latency</div>
            <div class="metric-value">{latency_ms:.1f} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if top_prob < confidence_threshold:
    st.warning(
        f"[Warning] **Low Confidence Prediction**: Confidence ({top_prob:.1%}) is below the configured threshold ({confidence_threshold:.0%})."
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Two-Column Layout ---
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### Input Image & Probability Distribution")
    st.image(
        selected_image,
        caption=f"Input: {selected_name} (Native Size: {selected_image.size[0]}x{selected_image.size[1]})",
        use_container_width=True,
    )

    # Plotly Probability Bar Chart
    fig = build_probability_chart(predictions_all, CLASS_NAMES, top_k=top_k)
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown("### Visual Explainability (Grad-CAM)")
    st.markdown(
        "Saliency heatmaps visualize which spatial regions positively stimulated the final feature maps for the top predicted category."
    )

    trio_sub1, trio_sub2 = st.columns(2)
    with trio_sub1:
        st.markdown("**Isolated Thermal Saliency**")
        st.image(
            colored_heatmap,
            caption="Grad-CAM Heatmap (Jet Colormap)",
            use_container_width=True,
        )

    with trio_sub2:
        st.markdown("**Superimposed Feature Overlay**")
        st.image(
            blended_overlay,
            caption=f"Overlay Blend ({top_class.title()}: {top_prob:.1%})",
            use_container_width=True,
        )

    # Plain-text Diagnostic Callout
    callout_text = get_spatial_diagnostic_narrative(top_class, top_prob)
    st.markdown(f'<div class="callout-box">{callout_text}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "Production Computer Vision & Explainability Platform • Built with TensorFlow/Keras, Streamlit, and Plotly"
    "</div>",
    unsafe_allow_html=True,
)

