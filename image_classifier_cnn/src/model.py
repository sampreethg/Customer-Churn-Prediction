"""
Convolutional Neural Network (CNN) Architecture Definition.

Constructs a modern multi-block CNN tailored for 32x32x3 image classification
with Batch Normalization, Dropout regularization, spatial pooling, and Softmax dense head.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from .config import INPUT_SHAPE, LEARNING_RATE, NUM_CLASSES


def build_cnn_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build, assemble, and compile the production CNN classification model.

    Architecture Layout:
    -------------------
    - Block 1: Conv2D(32, 3x3, same) -> BatchNorm -> Conv2D(32, 3x3) -> MaxPool(2x2) -> Dropout(0.25)
    - Block 2: Conv2D(64, 3x3, same) -> BatchNorm -> Conv2D(64, 3x3) -> MaxPool(2x2) -> Dropout(0.25)
    - Block 3: Conv2D(128, 3x3, same) -> BatchNorm -> MaxPool(2x2) -> Dropout(0.30)
    - Dense Head: Flatten -> Dense(256, relu) -> BatchNorm -> Dropout(0.50) -> Dense(num_classes, softmax)

    Parameters
    ----------
    input_shape : Tuple[int, int, int], default=(32, 32, 3)
        Height, width, and color channels of the input images.
    num_classes : int, default=10
        Number of output target categories.
    learning_rate : float, default=0.001
        Initial learning rate for the Adam optimizer.

    Returns
    -------
    tf.keras.Model
        Compiled Keras functional/sequential model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    # --- Feature Extraction Block 1 ---
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Conv2D(32, (3, 3), padding="valid", activation="relu", name="conv1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="dropout1")(x)

    # --- Feature Extraction Block 2 ---
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Conv2D(64, (3, 3), padding="valid", activation="relu", name="conv2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="dropout2")(x)

    # --- Feature Extraction Block 3 (Salient Feature Extraction Layer) ---
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="last_conv_layer")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="dropout3")(x)

    # --- Dense Classification Head ---
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Dropout(0.50, name="dropout4")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="production_cifar10_cnn")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

