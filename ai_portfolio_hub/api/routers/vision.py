"""
Computer Vision CNN Classification & Grad-CAM Explainability Router.

Endpoints:
- POST /api/v1/vision/classify: Accepts image uploads, outputs top-k predictions,
  and returns a superimposed Grad-CAM heatmap overlay as a Base64 string.
"""

import base64
import io

import cv2
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from PIL import Image

from ..config import CIFAR10_CLASSES
from ..rate_limiter import rate_limit
from ..schemas import VisionOutput, VisionPredictionItem

router = APIRouter(prefix="/vision", tags=["Computer Vision"])

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}



def get_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the name of the final Conv2D layer in the CNN architecture."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No Conv2D layer found in vision model.")


def compute_gradcam(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
    target_class_idx: int,
) -> np.ndarray:
    """
    Compute 2D normalized Grad-CAM activation heatmap for target class.
    """
    conv_layer_name = get_last_conv_layer(model)
    last_conv_layer = model.get_layer(conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(input_tensor, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        class_channel = predictions[:, target_class_idx]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs_val = conv_outputs[0]
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


@router.post("/classify", response_model=VisionOutput, dependencies=[Depends(rate_limit(15))])
async def classify_image(request: Request, file: UploadFile = File(...)):
    """
    Classify uploaded image (multipart/form-data), compute top-k predictions,
    and generate base64-encoded Grad-CAM saliency overlay.
    """
    model = getattr(request.app.state, "vision_model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision CNN model is not loaded.",
        )

    if file.content_type and file.content_type not in ALLOWED_IMAGE_MIMES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Allowed types: {', '.join(sorted(ALLOWED_IMAGE_MIMES))}",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image file size exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format or corrupted image payload.",
        ) from None

    # Resize to 32x32 for CIFAR-10 model
    resized_pil = pil_img.resize((32, 32), Image.Resampling.BILINEAR)
    norm_array = np.array(resized_pil, dtype=np.float32) / 255.0
    batch_tensor = np.expand_dims(norm_array, axis=0)

    # Model inference
    preds = model.predict(batch_tensor, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1]
    top_idx = int(top_indices[0])
    top_label = CIFAR10_CLASSES[top_idx]
    top_conf = float(preds[top_idx])

    # Build top-k prediction items
    top_items = []
    for rank, idx in enumerate(top_indices[:5], start=1):
        prob = float(preds[idx])
        top_items.append(
            VisionPredictionItem(
                rank=rank,
                label=CIFAR10_CLASSES[idx],
                probability=round(prob, 4),
                percentage=f"{prob * 100:.2f}%",
            )
        )

    # Compute Grad-CAM
    try:
        heatmap_2d = compute_gradcam(model, batch_tensor, top_idx)
        orig_w, orig_h = pil_img.size
        heatmap_uint8 = np.uint8(255 * heatmap_2d)
        resized_heatmap = cv2.resize(heatmap_uint8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        colored_bgr = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        base_np = np.array(pil_img, dtype=np.uint8)

        # Overlay blend (60% base image, 40% heatmap)
        blended = cv2.addWeighted(base_np, 0.60, colored_rgb, 0.40, 0)
        success, buffer = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        gradcam_b64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
    except Exception as exc:
        print(f"[Vision Warning] Grad-CAM generation error: {exc}")
        gradcam_b64 = ""

    return VisionOutput(
        top_class=top_label,
        top_confidence=round(top_conf, 4),
        predictions=top_items,
        gradcam_base64=gradcam_b64,
        image_shape=[32, 32, 3],
    )

