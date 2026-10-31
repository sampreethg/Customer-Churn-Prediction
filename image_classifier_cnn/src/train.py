"""
Model Compilation, Callbacks Configuration, and Training Loop.

Executes model training with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks,
and generates training history diagnostic visualizations.
"""

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    MIN_LR,
    MODEL_SAVE_PATH,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    TRAINING_HISTORY_PATH,
)
from .dataset import create_tf_dataset


def plot_training_history(
    history: tf.keras.callbacks.History,
    save_path: Path = TRAINING_HISTORY_PATH,
) -> None:
    """
    Generate and save a 2-panel chart comparing Training vs. Validation Loss and Accuracy.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        History object returned by model.fit().
    save_path : Path
        Filesystem location to save PNG diagnostic chart.
    """
    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Loss Curves
    axes[0].plot(
        epochs_range,
        history.history["loss"],
        label="Training Loss",
        color="#2563eb",
        linewidth=2.2,
        marker="o",
        markersize=4,
    )
    axes[0].plot(
        epochs_range,
        history.history["val_loss"],
        label="Validation Loss",
        color="#dc2626",
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=4,
    )
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(frameon=True, facecolor="#f8fafc")

    # Panel 2: Accuracy Curves
    axes[1].plot(
        epochs_range,
        history.history["accuracy"],
        label="Training Accuracy",
        color="#16a34a",
        linewidth=2.2,
        marker="o",
        markersize=4,
    )
    axes[1].plot(
        epochs_range,
        history.history["val_accuracy"],
        label="Validation Accuracy",
        color="#9333ea",
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=4,
    )
    axes[1].set_title("Classification Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True, facecolor="#f8fafc")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training history visualization exported to: {save_path.resolve()}")


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    model_save_path: Path = MODEL_SAVE_PATH,
    history_save_path: Path = TRAINING_HISTORY_PATH,
    use_augmentation: bool = True,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Configure callbacks and train the CNN model.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled Keras model instance.
    x_train : np.ndarray
        Training image tensor of shape (N, 32, 32, 3).
    y_train : np.ndarray
        Training target matrix (one-hot).
    x_val : np.ndarray
        Validation image tensor.
    y_val : np.ndarray
        Validation target matrix (one-hot).
    epochs : int, default=20
        Maximum training epochs.
    batch_size : int, default=64
        Batch size for gradient updates.
    model_save_path : Path
        Filepath for saving best checkpoint.
    history_save_path : Path
        Filepath for saving diagnostic loss/accuracy plot.
    use_augmentation : bool, default=True
        Whether to apply spatial data augmentation.

    Returns
    -------
    Tuple[tf.keras.Model, tf.keras.callbacks.History]
        Trained model (with best weights restored) and training history.
    """
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n" + "=" * 60)
    print(" STAGE: MODEL TRAINING LOOP")
    print("=" * 60)
    print(f"Max Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Data Augmentation: {'Enabled' if use_augmentation else 'Disabled'}")
    print(f"Checkpoint Target: {model_save_path.resolve()}\n")

    # Build tf.data pipelines for optimized throughput
    train_ds = create_tf_dataset(
        x_train,
        y_train,
        batch_size=batch_size,
        is_training=True,
        augment=use_augmentation,
    )
    val_ds = create_tf_dataset(
        x_val,
        y_val,
        batch_size=batch_size,
        is_training=False,
        augment=False,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save diagnostic curves
    plot_training_history(history, save_path=history_save_path)

    return model, history

