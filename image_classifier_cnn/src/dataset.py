"""
Dataset Ingestion, Preprocessing, and Augmentation Pipeline.

Handles automated download of CIFAR-10, pixel normalization to [0.0, 1.0],
stratified train/validation splitting, one-hot encoding, and data augmentation layers.
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from .config import (
    BATCH_SIZE,
    NUM_CLASSES,
    RANDOM_SEED,
    VAL_SPLIT,
)


def get_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Construct Keras data augmentation pipeline with spatial transformations.

    Returns
    -------
    tf.keras.Sequential
        Augmentation pipeline applying horizontal flips, subtle rotations, and zoom.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", name="random_flip"),
            layers.RandomRotation(0.1, fill_mode="nearest", name="random_rotation"),
            layers.RandomZoom(0.1, fill_mode="nearest", name="random_zoom"),
        ],
        name="augmentation_pipeline",
    )


def load_and_preprocess_data(
    val_split: float = VAL_SPLIT,
    random_seed: int = RANDOM_SEED,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Download CIFAR-10, normalize pixels to [0.0, 1.0], split into train/val/test,
    and convert labels to one-hot categorical vectors.

    Parameters
    ----------
    val_split : float, default=0.2
        Fraction of full training set allocated to validation.
    random_seed : int, default=42
        Reproducibility seed for train/val split.

    Returns
    -------
    Tuple[Tuple[x_train, y_train], Tuple[x_val, y_val], Tuple[x_test, y_test]]
        Normalized arrays with one-hot encoded targets.
    """
    print("Loading CIFAR-10 dataset via tf.keras.datasets...")
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.cifar10.load_data()

    # Normalize uint8 [0, 255] to float32 [0.0, 1.0]
    x_train_norm = x_train_raw.astype(np.float32) / 255.0
    x_test_norm = x_test_raw.astype(np.float32) / 255.0
    y_train_flat = y_train_raw.flatten()
    y_test_ints = y_test_raw.flatten()

    x_test = x_test_norm
    import os
    sample_limit = os.getenv("SAMPLE_LIMIT")
    if sample_limit:
        limit = int(sample_limit)
        x_train_norm = x_train_norm[:limit]
        y_train_flat = y_train_flat[:limit]
        test_limit = max(100, limit // 5)
        x_test = x_test[:test_limit]
        y_test_ints = y_test_ints[:test_limit]

    # Stratified train/val split on raw integer labels
    x_train, x_val, y_train_ints, y_val_ints = train_test_split(
        x_train_norm,
        y_train_flat,
        test_size=val_split,
        random_state=random_seed,
        stratify=y_train_flat,
    )

    # Convert targets to one-hot categorical matrices
    y_train = tf.keras.utils.to_categorical(y_train_ints, num_classes=NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(y_val_ints, num_classes=NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test_ints, num_classes=NUM_CLASSES)

    print("Data ingestion complete:")
    print(f" Training samples: {x_train.shape[0]} images | shape: {x_train.shape}")
    print(f" Validation samples: {x_val.shape[0]} images | shape: {x_val.shape}")
    print(f" Testing samples: {x_test.shape[0]} images | shape: {x_test.shape}")
    print(f" Pixel range: [{x_train.min():.1f}, {x_train.max():.1f}]")
    print(f" Target shape: {y_train.shape} (One-Hot Encoded)")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = BATCH_SIZE,
    is_training: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Build optimized tf.data.Dataset with optional on-the-fly augmentation,
    shuffling, batching, and prefetching.

    Parameters
    ----------
    x : np.ndarray
        Image array of shape (N, 32, 32, 3).
    y : np.ndarray
        One-hot label array of shape (N, 10).
    batch_size : int, default=64
        Batch dimension.
    is_training : bool, default=False
        Whether to shuffle and apply augmentations.
    augment : bool, default=False
        Whether to invoke data augmentation pipeline.

    Returns
    -------
    tf.data.Dataset
        Batched and prefetched dataset pipeline.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(x), 10000), seed=RANDOM_SEED)

    if is_training and augment:
        augmentation_layer = get_augmentation_pipeline()
        dataset = dataset.map(
            lambda img, lbl: (augmentation_layer(img, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

