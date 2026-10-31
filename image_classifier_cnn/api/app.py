"""
FastAPI Production REST Inference Microservice.

Serves the trained CNN classifier with endpoints for health inspection,
multipart image classification returning top-k predictions, and on-demand Grad-CAM explanations.
"""

import base64
from contextlib import asynccontextmanager
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
import tensorflow as tf

from src.config import CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH, MODEL_SAVE_PATH
from src.explain import make_gradcam_heatmap, overlay_gradcam

# Global Model Container
model_container: Dict[str, Any] = {"model": None}


def load_inference_model(model_path: Path = MODEL_SAVE_PATH) -> Optional[tf.keras.Model]:
    """
    Load serialized Keras CNN model artifact with architecture fallback.
    """
    if model_path.is_file():
        try:
            model = tf.keras.models.load_model(str(model_path))
            print(f"[API] CNN Model loaded successfully from: {model_path.resolve()}")
            return model
        except Exception as exc:
            print(f"[API ERROR] Failed to load model from {model_path}: {exc}")

    print(f"[API WARNING] Checkpoint not found at: {model_path.resolve()}. Initializing fresh CNN architecture.")
    try:
        from src.model import build_cnn_model
        return build_cnn_model()
    except Exception as exc:
        print(f"[API ERROR] Failed to build fallback model: {exc}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load CNN model at startup.
    """
    model_container["model"] = load_inference_model()
    yield
    model_container["model"] = None


app = FastAPI(
    title="CNN Image Classification API",
    description="High-performance Computer Vision inference microservice with Grad-CAM explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Response Schemas ---
class PredictionItem(BaseModel):
    rank: int = Field(..., description="Rank among predictions (1 to K)")
    class_name: str = Field(..., description="Predicted category label")
    probability: float = Field(..., description="Predicted probability confidence [0.0 - 1.0]")
    percentage: str = Field(..., description="Formatted probability percentage string")


class ClassificationResponse(BaseModel):
    filename: str
    top_prediction: str
    top_confidence: float
    top_predictions: List[PredictionItem]
    image_dimensions: List[int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    target_dimensions: List[int]
    num_classes: int
    class_names: List[str]


def preprocess_image_bytes(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw image bytes to preprocessed model input tensor and normalized display array.

    Parameters
    ----------
    image_bytes : bytes
        Uploaded file payload.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (batch_tensor of shape (1, 32, 32, 3), display_image of shape (32, 32, 3)) in [0.0, 1.0].
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format or unreadable image stream: {str(exc)}",
        ) from exc

    # Resize to target CIFAR-10 spatial resolution (32x32)
    pil_img_resized = pil_img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
    img_array = np.array(pil_img_resized, dtype=np.float32) / 255.0

    batch_tensor = np.expand_dims(img_array, axis=0)
    return batch_tensor, img_array


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """
    Liveness and readiness probe inspecting model availability and configuration.
    """
    model = model_container.get("model")
    is_ready = model is not None
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_path=str(MODEL_SAVE_PATH.resolve()),
        target_dimensions=[IMG_HEIGHT, IMG_WIDTH, 3],
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
    )


@app.post("/classify", response_model=ClassificationResponse, tags=["Inference"])
async def classify_image(file: UploadFile = File(...)):
    """
    Accept an uploaded image (multipart/form-data), resize, normalize,
    and return top-3 predicted class probabilities.
    """
    model = model_container.get("model")
    if model is None:
        # Attempt fallback reload if artifact was generated recently
        model = load_inference_model()
        model_container["model"] = model

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CNN model is not loaded or checkpoint artifact is missing. Run training pipeline first.",
        )

    # Validate file presence
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload payload.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file payload is empty.",
        )

    batch_tensor, _ = preprocess_image_bytes(content)

    # Run inference
    preds = model.predict(batch_tensor, verbose=0)[0]

    # Rank top 3 categories
    top_indices = np.argsort(preds)[::-1][:3]
    top_items = []
    for rank_idx, idx in enumerate(top_indices, start=1):
        prob = float(preds[idx])
        top_items.append(
            PredictionItem(
                rank=rank_idx,
                class_name=CLASS_NAMES[idx],
                probability=round(prob, 4),
                percentage=f"{prob * 100:.2f}%",
            )
        )

    top_idx = int(top_indices[0])
    return ClassificationResponse(
        filename=file.filename,
        top_prediction=CLASS_NAMES[top_idx],
        top_confidence=round(float(preds[top_idx]), 4),
        top_predictions=top_items,
        image_dimensions=[IMG_HEIGHT, IMG_WIDTH, 3],
    )


@app.post("/explain", tags=["Explainability"])
async def explain_image(file: UploadFile = File(...)):
    """
    Compute Grad-CAM heatmap for uploaded image and return prediction metadata
    along with base64-encoded visual saliency overlay.
    """
    model = model_container.get("model")
    if model is None:
        model = load_inference_model()
        model_container["model"] = model

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CNN model is not loaded. Train the model first.",
        )

    content = await file.read()
    batch_tensor, display_img = preprocess_image_bytes(content)

    preds = model.predict(batch_tensor, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_label = CLASS_NAMES[top_idx]
    confidence = float(preds[top_idx])

    # Compute Grad-CAM
    heatmap = make_gradcam_heatmap(batch_tensor, model, pred_index=top_idx)
    _, overlay = overlay_gradcam(display_img, heatmap, alpha=0.45)

    # Encode overlay image to Base64 PNG
    success, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode Grad-CAM overlay image.",
        )

    overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "filename": file.filename,
        "predicted_class": top_label,
        "confidence": round(confidence, 4),
        "overlay_base64": f"data:image/png;base64,{overlay_base64}",
    }

