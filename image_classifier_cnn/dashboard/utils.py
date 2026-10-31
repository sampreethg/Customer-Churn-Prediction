"""
Image Preprocessing, Grad-CAM Overlay Generation, and Plotly Visual Builders.

Provides reusable utilities for the Streamlit Computer Vision dashboard:
- PIL image preprocessing & batch tensor construction
- Dynamic Grad-CAM saliency computation and RGB superimposition
- Interactive Plotly horizontal bar chart for top-k class probabilities
- Preloaded CIFAR-10 test sample gallery provider
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import tensorflow as tf


def preprocess_image(
    image_pil: Image.Image,
    target_size: Tuple[int, int] = (32, 32),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize PIL image to target resolution, normalize pixels to [0.0, 1.0],
    and construct batch tensor for model inference.

    Parameters
    ----------
    image_pil : Image.Image
        Input image in PIL format.
    target_size : Tuple[int, int], default=(32, 32)
        Spatial dimensions (height, width).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (batch_tensor of shape (1, 32, 32, 3), display_array of shape (32, 32, 3)) in [0.0, 1.0].
    """
    img_rgb = image_pil.convert("RGB")
    resized_pil = img_rgb.resize(target_size, Image.Resampling.BILINEAR)
    img_normalized = np.array(resized_pil, dtype=np.float32) / 255.0
    batch_tensor = np.expand_dims(img_normalized, axis=0)
    return batch_tensor, img_normalized


def get_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Identify the name of the final 2D convolutional layer in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No Conv2D layer found in the model architecture.")


def generate_gradcam_overlay(
    model: tf.keras.Model,
    img_array: np.ndarray,
    original_pil_image: Image.Image,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Grad-CAM activation heatmap and superimpose onto original image dimensions.

    Parameters
    ----------
    model : tf.keras.Model
        Trained CNN model.
    img_array : np.ndarray
        Batch image tensor of shape (1, 32, 32, 3).
    original_pil_image : Image.Image
        Original PIL image for rendering overlays at native resolution.
    last_conv_layer_name : Optional[str]
        Target convolutional layer name.
    pred_index : Optional[int]
        Target class index for gradient extraction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        (colored_heatmap_rgb, blended_image_rgb, top_confidence_score).
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)

        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))

        class_channel = predictions[:, pred_index]

    confidence = float(predictions[0][pred_index])

    # Compute gradients with respect to convolutional feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Channel-wise mean weights (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight convolutional feature maps
    conv_outputs_val = conv_outputs[0]
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU and normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap_np = heatmap.numpy()

    # Target spatial dimensions matching the original user image
    orig_w, orig_h = original_pil_image.size
    orig_np = np.array(original_pil_image.convert("RGB"), dtype=np.uint8)

    # Resize heatmap to original image spatial dimensions
    heatmap_uint8 = np.uint8(255 * heatmap_np)
    resized_heatmap = cv2.resize(heatmap_uint8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Apply JET colormap (returns BGR) and convert to RGB
    colored_heatmap_bgr = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Superimpose weighted overlay (60% original image + 40% colored heatmap)
    blended = cv2.addWeighted(
        orig_np,
        0.60,
        colored_heatmap_rgb,
        0.40,
        0,
    )

    return colored_heatmap_rgb, blended, confidence


def build_probability_chart(
    predictions: np.ndarray,
    class_names: List[str],
    top_k: int = 5,
) -> go.Figure:
    """
    Generate an interactive Plotly horizontal bar chart displaying top-k class probabilities.

    Parameters
    ----------
    predictions : np.ndarray
        Array of softmax probabilities for all classes.
    class_names : List[str]
        List of all class names.
    top_k : int, default=5
        Number of top classes to display.

    Returns
    -------
    go.Figure
        Styled Plotly horizontal bar figure.
    """
    # Sort top-k descending
    top_indices = np.argsort(predictions)[::-1][:top_k]

    # Reverse for ascending order in horizontal bar chart (highest at top)
    top_indices = top_indices[::-1]

    labels = [class_names[i] for i in top_indices]
    probs = [float(predictions[i]) for i in top_indices]
    percentages = [f"{p * 100:.1f}%" for p in probs]

    # Color palette: highlight top rank in vivid cobalt blue, others in softer slate blue
    colors = ["#3b82f6"] * len(probs)
    if len(colors) > 0:
        colors[-1] = "#1d4ed8"  # Highest rank at top

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=labels,
            orientation="h",
            text=percentages,
            textposition="outside",
            textfont=dict(size=12, color="#0f172a", family="Inter, sans-serif"),
            marker=dict(
                color=colors,
                line=dict(color="#1e293b", width=1),
            ),
            hoverinfo="y+x",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Top-{top_k} Class Probability Distribution</b>",
            font=dict(size=14, color="#1e293b"),
            x=0.02,
        ),
        xaxis=dict(
            title="<b>Softmax Probability</b>",
            range=[0, 1.15],
            tickformat=".0%",
            showgrid=True,
            gridcolor="#e2e8f0",
        ),
        yaxis=dict(
            title="<b>Category</b>",
            autorange=True,
            tickfont=dict(size=12, color="#1e293b"),
        ),
        margin=dict(l=80, r=40, t=50, b=40),
        height=320,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )

    return fig


def generate_synthetic_samples() -> Dict[str, Image.Image]:
    """
    Generate or extract representative synthetic/test samples for instant demonstration.
    """
    samples = {}
    # Generate distinct patterned images simulating CIFAR-10 classes
    # 1. Airplane (sky blue background with metallic gray silhouette)
    sky = np.full((32, 32, 3), [135, 206, 235], dtype=np.uint8)
    cv2.line(sky, (4, 16), (28, 16), (200, 200, 210), 3)
    cv2.line(sky, (14, 8), (18, 24), (180, 180, 190), 2)
    samples["airplane (Sample)"] = Image.fromarray(sky)

    # 2. Automobile (road black background with red chassis and wheels)
    road = np.full((32, 32, 3), [50, 50, 55], dtype=np.uint8)
    cv2.rectangle(road, (6, 12), (26, 22), (220, 38, 38), -1)
    cv2.circle(road, (10, 23), 3, (20, 20, 20), -1)
    cv2.circle(road, (22, 23), 3, (20, 20, 20), -1)
    samples["automobile (Sample)"] = Image.fromarray(road)

    # 3. Frog (grass green background with forest green oval)
    grass = np.full((32, 32, 3), [34, 139, 34], dtype=np.uint8)
    cv2.ellipse(grass, (16, 18), (10, 8), 0, 0, 360, (20, 90, 20), -1)
    cv2.circle(grass, (12, 12), 2, (0, 0, 0), -1)
    cv2.circle(grass, (20, 12), 2, (0, 0, 0), -1)
    samples["frog (Sample)"] = Image.fromarray(grass)

    # 4. Ship (ocean navy background with white vessel hull)
    ocean = np.full((32, 32, 3), [10, 40, 100], dtype=np.uint8)
    pts = np.array([[6, 20], [26, 20], [22, 26], [10, 26]], np.int32)
    cv2.fillPoly(ocean, [pts], (240, 240, 245))
    cv2.rectangle(ocean, (14, 14), (18, 20), (180, 50, 50), -1)
    samples["ship (Sample)"] = Image.fromarray(ocean)

    return samples

