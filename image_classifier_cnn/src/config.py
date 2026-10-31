"""
Configuration and Hyperparameter Constants.

Defines input image dimensions, CIFAR-10 class labels, model architecture hyperparameters,
training callbacks configuration, and filesystem artifact paths.
"""

from pathlib import Path

# Image Dimensions & Spatial Properties
IMG_HEIGHT: int = 32
IMG_WIDTH: int = 32
CHANNELS: int = 3
INPUT_SHAPE: tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
NUM_CLASSES: int = 10

# CIFAR-10 Class Label Mapping
CLASS_NAMES: list[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

import os

# Training Hyperparameters
BATCH_SIZE: int = 64
EPOCHS: int = int(os.getenv("EPOCHS", "20"))
LEARNING_RATE: float = 0.001
RANDOM_SEED: int = 42
VAL_SPLIT: float = 0.2

# Early Stopping & LR Scheduling Parameters
EARLY_STOPPING_PATIENCE: int = 5
REDUCE_LR_PATIENCE: int = 3
REDUCE_LR_FACTOR: float = 0.5
MIN_LR: float = 1e-5

# Directory Paths & Artifact Files
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = BASE_DIR / "models"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"

MODEL_SAVE_PATH: Path = MODELS_DIR / "best_cnn_model.keras"
TRAINING_HISTORY_PATH: Path = ARTIFACTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH: Path = ARTIFACTS_DIR / "confusion_matrix.png"
GRADCAM_SAMPLE_PATH: Path = ARTIFACTS_DIR / "gradcam_sample.png"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

