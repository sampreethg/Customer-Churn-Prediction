"""
Global Environment Configuration and Model Artifact Locations.

Centralizes filesystem paths, runtime device mapping (CUDA vs CPU), and fallback settings
for the unified multi-domain AI Portfolio Hub.
"""

import os
from pathlib import Path
import torch

# Root directories
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = BASE_DIR / "models"

# Model Artifact Locations
CHURN_MODEL_PATH: Path = Path(os.getenv("CHURN_MODEL_PATH", str(MODELS_DIR / "best_churn_model.joblib")))
VISION_MODEL_PATH: Path = Path(os.getenv("VISION_MODEL_PATH", str(MODELS_DIR / "best_cnn_model.keras")))
TRANSFORMER_PATH: Path = Path(os.getenv("TRANSFORMER_PATH", str(MODELS_DIR / "fine_tuned_distilbert")))
RL_MODEL_PATH: Path = Path(os.getenv("RL_MODEL_PATH", str(MODELS_DIR / "best_per_d3qn.pt")))

# Compute Device
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thresholds & Label Constants
CHURN_THRESHOLD: float = 0.50
CIFAR10_CLASSES: list[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
NLP_CLASSES: list[str] = ["NEGATIVE", "POSITIVE"]

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

