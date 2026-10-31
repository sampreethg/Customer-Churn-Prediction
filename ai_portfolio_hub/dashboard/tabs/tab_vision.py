"""
Tab 2: Computer Vision & Explainable AI (Grad-CAM).
"""

import io
import os

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

from ..components import build_probability_bar_chart, render_metric_card


def create_sample_gallery() -> dict[str, Image.Image]:
    """Provide synthetic visual samples simulating CIFAR-10 objects."""
    samples = {}
    # Automobile sample
    road = np.full((64, 64, 3), [40, 40, 45], dtype=np.uint8)
    cv2.rectangle(road, (10, 20), (54, 44), (220, 38, 38), -1)
    cv2.circle(road, (20, 46), 6, (20, 20, 20), -1)
    cv2.circle(road, (44, 46), 6, (20, 20, 20), -1)
    samples["Automobile (Car)"] = Image.fromarray(road)

    # Airplane sample
    sky = np.full((64, 64, 3), [135, 206, 235], dtype=np.uint8)
    cv2.line(sky, (8, 32), (56, 32), (200, 200, 215), 5)
    cv2.line(sky, (28, 16), (36, 48), (180, 180, 195), 4)
    samples["Airplane (Jet)"] = Image.fromarray(sky)

    # Frog sample
    pond = np.full((64, 64, 3), [34, 139, 34], dtype=np.uint8)
    cv2.ellipse(pond, (32, 36), (20, 16), 0, 0, 360, (20, 90, 20), -1)
    cv2.circle(pond, (24, 24), 4, (0, 0, 0), -1)
    cv2.circle(pond, (40, 24), 4, (0, 0, 0), -1)
    samples["Frog (Amphibian)"] = Image.fromarray(pond)

    # Ship sample
    sea = np.full((64, 64, 3), [10, 40, 110], dtype=np.uint8)
    pts = np.array([[10, 40], [54, 40], [46, 52], [18, 52]], np.int32)
    cv2.fillPoly(sea, [pts], (240, 240, 245))
    cv2.rectangle(sea, (28, 28), (36, 40), (180, 50, 50), -1)
    samples["Ship (Vessel)"] = Image.fromarray(sea)

    return samples


def render_tab_vision(api_base_url: str):
    st.markdown("### Computer Vision & Explainable AI (Grad-CAM)")
    st.markdown(
        "Classify images into CIFAR-10 object categories using a multi-block CNN and "
        "render gradient-weighted class activation mapping (Grad-CAM) heatmaps revealing the exact spatial triggers."
    )

    col_ctrl, col_view = st.columns([1, 1.2], gap="large")
    gallery = create_sample_gallery()

    with col_ctrl:
        st.markdown("#### Select or Upload Image")
        input_choice = st.radio("Input Source", ["Preloaded Sample Gallery", "Upload Custom Image"], horizontal=True)

        selected_pil = None
        if input_choice == "Preloaded Sample Gallery":
            sample_key = st.selectbox("Select Test Image", list(gallery.keys()))
            selected_pil = gallery[sample_key]
        else:
            upload_file = st.file_uploader("Upload an Image (.png, .jpg, .webp)", type=["png", "jpg", "jpeg", "webp"])
            if upload_file is not None:
                selected_pil = Image.open(upload_file).convert("RGB")

        if selected_pil:
            st.image(selected_pil, caption="Selected Image Input", use_container_width=True)

    with col_view:
        st.markdown("#### Neural Predictions & Spatial Saliency")
        if selected_pil is not None:
            # Convert PIL image to byte stream
            buf = io.BytesIO()
            selected_pil.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            try:
                files = {"file": ("input.png", image_bytes, "image/png")}
                headers = {"X-API-Key": os.getenv("API_KEY", "dev-secret-key-2026")}
                resp = requests.post(f"{api_base_url}/api/v1/vision/classify", files=files, headers=headers, timeout=6)
                if resp.status_code == 200:
                    data = resp.json()
                    top_cls = data["top_class"]
                    top_conf = data["top_confidence"]
                    predictions = data["predictions"]
                    gradcam_b64 = data["gradcam_base64"]

                    # Metric summary
                    m1, m2 = st.columns(2)
                    with m1:
                        render_metric_card("Top Classification", top_cls.title(), "Target Category", border_color="#3b82f6")
                    with m2:
                        render_metric_card("Confidence", f"{top_conf * 100:.1f}%", "Softmax Probability", border_color="#10b981")

                    # Probability breakdown chart
                    st.plotly_chart(build_probability_bar_chart(predictions, title="Top-5 Class Distribution"), use_container_width=True)

                    # Grad-CAM Heatmap overlay
                    if gradcam_b64:
                        st.markdown("**Grad-CAM Saliency Overlay (Jet Colormap Superimposed at 40% Opacity)**")
                        st.image(gradcam_b64, caption=f"Activations identifying salient {top_cls} contours", use_container_width=True)
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
            except Exception as exc:
                st.warning(f"[Warning] Gateway connection note: {exc}. Displaying sample predictions.")
                render_metric_card("Top Classification", "Automobile", "92.4% Confidence", border_color="#3b82f6")
        else:
            st.info(" Please select or upload an image to run CNN inference.")

