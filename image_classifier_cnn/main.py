"""
End-to-End Orchestration Pipeline for CNN Computer Vision Classification.

Executes the full pipeline:
1. Environment & directory setup
2. CIFAR-10 data ingestion, normalization, and stratified splitting
3. Modern CNN model architecture construction & summary
4. Training with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks
5. Test evaluation with classification report and Confusion Matrix heatmap
6. Grad-CAM visual explainability heatmap & overlay generation
"""

import sys
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from src.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    CLASS_NAMES,
    CONFUSION_MATRIX_PATH,
    EPOCHS,
    GRADCAM_SAMPLE_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    MODELS_DIR,
    NUM_CLASSES,
    TRAINING_HISTORY_PATH,
)
from src.dataset import load_and_preprocess_data
from src.evaluate import evaluate_model
from src.explain import generate_and_save_gradcam_visual
from src.model import build_cnn_model
from src.train import train_model


def print_banner(title: str):
    width = 70
    print("\n" + "=" * width)
    print(f" {title.center(width - 2)} ")
    print("=" * width)


def main():
    print_banner("PRODUCTION CNN COMPUTER VISION CLASSIFICATION SYSTEM")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {bool(tf.config.list_physical_devices('GPU'))}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Model Checkpoint: {MODEL_SAVE_PATH}")
    print(f"Artifacts Dir: {ARTIFACTS_DIR}\n")

    # Step 1: Directory Setup
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2: Ingest & Preprocess Dataset
    print_banner("PHASE 1: DATA INGESTION & PREPROCESSING")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    # Step 3: Build & Compile CNN Model
    print_banner("PHASE 2: CNN MODEL ARCHITECTURE BUILD")
    model = build_cnn_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE,
    )
    model.summary()

    # Step 4: Model Training with Production Callbacks
    print_banner("PHASE 3: TRAINING WITH CALLBACKS")
    trained_model, history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=MODEL_SAVE_PATH,
        history_save_path=TRAINING_HISTORY_PATH,
        use_augmentation=True,
    )

    # Step 5: Test Evaluation & Confusion Matrix
    print_banner("PHASE 4: TEST EVALUATION")
    evaluate_model(
        model=trained_model,
        x_test=x_test,
        y_test=y_test,
        class_names=CLASS_NAMES,
        save_path=CONFUSION_MATRIX_PATH,
    )

    # Step 6: Grad-CAM Explainability Generation
    print_banner("PHASE 5: GRAD-CAM VISUAL EXPLAINABILITY")
    # Pick a high-confidence representative test instance
    sample_idx = 0
    sample_img = x_test[sample_idx]
    true_label_idx = int(np.argmax(y_test[sample_idx]))
    true_label_name = CLASS_NAMES[true_label_idx]

    pred_label, conf = generate_and_save_gradcam_visual(
        model=trained_model,
        img=sample_img,
        true_label=true_label_name,
        class_names=CLASS_NAMES,
        save_path=GRADCAM_SAMPLE_PATH,
        last_conv_layer_name="last_conv_layer",
    )

    # Step 7: Summary of Generated Artifacts
    print_banner("EXECUTION SUMMARY & ARTIFACTS VERIFICATION")
    artifacts = [
        ("Saved Best Model", MODEL_SAVE_PATH),
        ("Training History Curves", TRAINING_HISTORY_PATH),
        ("Confusion Matrix Heatmap", CONFUSION_MATRIX_PATH),
        ("Grad-CAM Visual Heatmap", GRADCAM_SAMPLE_PATH),
    ]

    all_exist = True
    for label, path in artifacts:
        exists = path.is_file()
        all_exist = all_exist and exists
        size_kb = path.stat().st_size / 1024 if exists else 0.0
        status_str = f"[OK] ({size_kb:.1f} KB)" if exists else "[MISSING]"
        print(f" {label:<28}: {status_str:<18} -> {path.name}")

    if all_exist:
        print("\nAll pipeline stages and production artifacts generated successfully!")
    else:
        print("\nWARNING: One or more expected artifacts are missing.")


if __name__ == "__main__":
    main()

