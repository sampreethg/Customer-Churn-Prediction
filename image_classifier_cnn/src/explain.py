"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) Visual Explainability Module.

Generates localized visual saliency heatmaps highlighting the exact spatial regions
driving CNN classification decisions on input imagery.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .config import CLASS_NAMES, GRADCAM_SAMPLE_PATH


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """
    Search model layers in reverse order to identify the final 2D convolutional layer.

    Parameters
    ----------
    model : tf.keras.Model
        Keras CNN model.

    Returns
    -------
    str
        Name of the last Conv2D layer.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No Conv2D layer found in the supplied model architecture.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Compute normalized 2D Grad-CAM activation heatmap for a target input image.

    Parameters
    ----------
    img_array : np.ndarray
        Preprocessed image tensor with shape (1, H, W, C) scaled to [0.0, 1.0].
    model : tf.keras.Model
        Trained classification model.
    last_conv_layer_name : Optional[str], default=None
        Name of target convolutional layer. If None, automatically detected.
    pred_index : Optional[int], default=None
        Target class index. If None, the top predicted category is targeted.

    Returns
    -------
    np.ndarray
        2D heatmap array normalized to [0.0, 1.0].
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    # Gradient extraction model mapping inputs to [conv_activations, predictions]
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        # Cast image tensor to float32
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    # Compute gradients of top predicted class with respect to conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Global Average Pooling of gradients across spatial dimensions (height, width)
    # Shape: (num_channels,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map channel by its corresponding pooled gradient
    conv_outputs_val = conv_outputs[0]
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU (only positive contributions increase evidence for class) and normalize
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_gradcam(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Superimpose Grad-CAM heatmap onto the original RGB image.

    Parameters
    ----------
    original_img : np.ndarray
        Original image array in [0.0, 1.0] range, shape (H, W, 3).
    heatmap : np.ndarray
        2D heatmap in [0.0, 1.0] range.
    alpha : float, default=0.45
        Opacity blending factor between heatmap and base image.
    colormap : int, default=cv2.COLORMAP_JET
        OpenCV colormap identifier.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (RGB colored heatmap, Blended superimposed image) both uint8 [0, 255].
    """
    h, w = original_img.shape[0], original_img.shape[1]

    # Convert heatmap to uint8 [0, 255] and upscale to original spatial dimensions
    heatmap_uint8 = np.uint8(255 * heatmap)
    resized_heatmap = cv2.resize(heatmap_uint8, (w, h), interpolation=cv2.INTER_CUBIC)

    # Apply OpenCV color map (returns BGR)
    colored_heatmap_bgr = cv2.applyColorMap(resized_heatmap, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Scale original image to uint8 [0, 255]
    base_img_uint8 = np.uint8(255 * original_img)

    # Superimpose weighted blend
    superimposed = cv2.addWeighted(
        base_img_uint8,
        1.0 - alpha,
        colored_heatmap_rgb,
        alpha,
        0,
    )

    return colored_heatmap_rgb, superimposed


def generate_and_save_gradcam_visual(
    model: tf.keras.Model,
    img: np.ndarray,
    true_label: str,
    class_names: List[str] = CLASS_NAMES,
    save_path: Path = GRADCAM_SAMPLE_PATH,
    last_conv_layer_name: Optional[str] = None,
) -> Tuple[str, float]:
    """
    Generate and save a 3-panel visual explainability figure:
    [1] Original Image | [2] Grad-CAM Heatmap | [3] Superimposed Overlay.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    img : np.ndarray
        Single preprocessed image of shape (32, 32, 3) in [0.0, 1.0].
    true_label : str
        Ground truth class name.
    class_names : List[str]
        Ordered list of class names.
    save_path : Path
        Destination filepath for saved figure.
    last_conv_layer_name : Optional[str]
        Target conv layer.

    Returns
    -------
    Tuple[str, float]
        Predicted class label name and predicted probability confidence.
    """
    img_batch = np.expand_dims(img, axis=0)
    preds = model.predict(img_batch, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = class_names[pred_idx]
    confidence = float(preds[pred_idx])

    # Compute heatmap
    heatmap = make_gradcam_heatmap(
        img_batch,
        model,
        last_conv_layer_name=last_conv_layer_name,
        pred_index=pred_idx,
    )

    # Blend overlay
    colored_heatmap, overlay = overlay_gradcam(img, heatmap, alpha=0.45)

    # Render 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Original Image
    axes[0].imshow(img)
    axes[0].set_title(
        f"Input Image\nTrue: '{true_label}'",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[0].axis("off")

    # Panel 2: Grad-CAM Saliency Heatmap
    im2 = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(
        f"Grad-CAM Heatmap\nLayer: {last_conv_layer_name or get_last_conv_layer_name(model)}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Superimposed Overlay
    axes[2].imshow(overlay)
    pred_status = "Correct" if true_label == pred_label else "Incorrect"
    status_color = "#15803d" if true_label == pred_label else "#b91c1c"
    axes[2].set_title(
        f"Superimposed Saliency Overlay\nPred: '{pred_label}' ({confidence:.1%}) [{pred_status}]",
        fontsize=12,
        fontweight="bold",
        color=status_color,
        pad=10,
    )
    axes[2].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Grad-CAM visual explainability chart exported to: {save_path.resolve()}")
    print(f" Target: Ground Truth='{true_label}' | Predicted='{pred_label}' ({confidence:.2%})")

    return pred_label, confidence

