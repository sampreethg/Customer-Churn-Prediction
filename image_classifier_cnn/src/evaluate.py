"""
Model Evaluation and Diagnostic Reporting Module.

Evaluates test set performance across Accuracy, Macro Precision, Macro Recall, and F1-Score,
outputs detailed per-category metrics, and generates a labeled Confusion Matrix heatmap.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf

from .config import CLASS_NAMES, CONFUSION_MATRIX_PATH


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Path = CONFUSION_MATRIX_PATH,
) -> None:
    """
    Generate and save a seaborn heatmap confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True integer class indices.
    y_pred : np.ndarray
        Predicted integer class indices.
    class_names : List[str]
        Category label strings.
    save_path : Path
        Filepath for saving PNG plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalized confusion matrix (percentages)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="#cbd5e1",
    )
    plt.title("CIFAR-10 Test Confusion Matrix (Counts)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("Ground Truth Class", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix visualization exported to: {save_path.resolve()}")


def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Path = CONFUSION_MATRIX_PATH,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of test data performance.

    Parameters
    ----------
    model : tf.keras.Model
        Trained CNN model.
    x_test : np.ndarray
        Test image array of shape (N, 32, 32, 3).
    y_test : np.ndarray
        One-hot or integer encoded ground truth labels.
    class_names : List[str]
        List of category names.
    save_path : Path
        Filesystem location to save confusion matrix chart.

    Returns
    -------
    Dict[str, float]
        Dictionary with overall accuracy, macro precision, recall, and f1-score.
    """
    print("\n" + "=" * 60)
    print("              STAGE: MODEL TEST EVALUATION")
    print("=" * 60)

    # Convert one-hot to class indices if needed
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.flatten()

    # Generate model predictions
    y_pred_probs = model.predict(x_test, batch_size=64, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate overall macro metrics
    acc = float(np.mean(y_true == y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    print(f"\n[Overall Performance Metrics on Test Set ({len(y_true)} samples)]")
    print(f"  Accuracy:        {acc * 100:.2f}%")
    print(f"  Macro Precision: {prec:.4f}")
    print(f"  Macro Recall:    {rec:.4f}")
    print(f"  Macro F1-Score:  {f1:.4f}")

    print("\n[Detailed Per-Category Classification Report]")
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report_str)

    # Plot Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=save_path)

    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
    }

