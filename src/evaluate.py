"""
Model Evaluation and Diagnostic Reporting Module.

Calculates holdout test performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
formats confusion matrices and classification reports, and generates visual diagnostic plots.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from .config import ARTIFACTS_DIR


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> dict[str, float]:
    """
    Evaluate trained pipeline on unseen holdout test set and export diagnostics.

    Parameters
    ----------
    model : Pipeline
        Trained Scikit-Learn pipeline.
    X_test : pd.DataFrame
        Unseen test feature set.
    y_test : pd.Series
        Ground-truth binary test labels.
    artifacts_dir : Path | str
        Destination directory for exported visualization charts.

    Returns
    -------
    dict[str, float]
        Dictionary of key evaluation metrics.
    """
    print("\n" + "=" * 60)
    print(" STAGE 2: MODEL EVALUATION & DIAGNOSTICS")
    print("=" * 60)

    # Predictions and Probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metric Calculations
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
    }

    # Console Metrics Presentation (ASCII for cross-platform Windows compatibility)
    print("\n[Holdout Test Performance]")
    print(f" * Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f" * Precision: {prec:.4f} ({prec * 100:.2f}%)")
    print(f" * Recall: {rec:.4f} ({rec * 100:.2f}%)")
    print(f" * F1-Score: {f1:.4f}")
    print(f" * ROC-AUC: {roc_auc:.4f}")

    # Confusion Matrix Presentation
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n[Confusion Matrix Breakdown]")
    print(" +---------------------+---------------------+")
    print(" | True Negatives (TN) | False Positives(FP) |")
    print(f" | {tn:<19} | {fp:<19} |")
    print(" +---------------------+---------------------+")
    print(" | False Negatives(FN) | True Positives (TP) |")
    print(f" | {fn:<19} | {tp:<19} |")
    print(" +---------------------+---------------------+")

    # Classification Report
    print("\n[Detailed Classification Report]")
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Retained (0)", "Churned (1)"],
        digits=4,
    )
    print(report)

    # Save Diagnostic Plots
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_evaluation_plots(cm, y_test, y_prob, roc_auc, out_dir)

    return metrics


def _save_evaluation_plots(
    cm: np.ndarray,
    y_test: pd.Series,
    y_prob: np.ndarray,
    roc_auc: float,
    artifacts_dir: Path,
) -> None:
    """
    Save Confusion Matrix heatmap and ROC Curve plots to artifacts folder.
    Uses Matplotlib if available, with a pure Pillow fallback for high-res rendering.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"],
            annot_kws={"size": 14, "weight": "bold"},
        )
        plt.title("Holdout Confusion Matrix", fontsize=14, pad=12, weight="bold")
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()
        cm_path = artifacts_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"[Artifact Saved] Confusion matrix plot: {cm_path.resolve()}")

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Chance")
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
        plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
        plt.title(
            "Receiver Operating Characteristic (ROC) Curve", fontsize=14, pad=12, weight="bold"
        )
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = artifacts_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"[Artifact Saved] ROC curve visualization: {roc_path.resolve()}")
        return

    except ImportError: # pragma: no cover
        pass

    # Pure Pillow High-Resolution Rendering Fallback
    from PIL import Image, ImageDraw

    # 1. Confusion Matrix Plot via Pillow
    cm_img = Image.new("RGB", (700, 600), "#f8fafc")
    draw = ImageDraw.Draw(cm_img)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    # Title
    draw.text((70, 30), "Holdout Test Confusion Matrix", fill="#0f172a")
    draw.text((70, 55), f"Total Test Samples: {total} (Stratified 20% Holdout)", fill="#64748b")

    # Cells
    cell_w, cell_h = 240, 180
    start_x, start_y = 120, 120

    cells = [
        (0, 0, tn, f"TN: {tn}", f"({tn / total * 100:.1f}%)", "#dbeafe", "#1e40af"),
        (1, 0, fp, f"FP: {fp}", f"({fp / total * 100:.1f}%)", "#fee2e2", "#991b1b"),
        (0, 1, fn, f"FN: {fn}", f"({fn / total * 100:.1f}%)", "#fef3c7", "#92400e"),
        (1, 1, tp, f"TP: {tp}", f"({tp / total * 100:.1f}%)", "#dcfce7", "#166534"),
    ]

    for col, row, _count, title, pct, bg_color, text_color in cells:
        x0 = start_x + col * cell_w
        y0 = start_y + row * cell_h
        x1 = x0 + cell_w - 10
        y1 = y0 + cell_h - 10
        draw.rectangle([x0, y0, x1, y1], fill=bg_color, outline="#cbd5e1", width=2)
        draw.text((x0 + 40, y0 + 50), title, fill=text_color)
        draw.text((x0 + 40, y0 + 90), f"Share: {pct}", fill="#475569")

    # Axis labels
    draw.text((start_x + 60, start_y - 30), "Predicted Retained", fill="#334155")
    draw.text((start_x + cell_w + 50, start_y - 30), "Predicted Churned", fill="#334155")
    draw.text((15, start_y + 70), "Actual Retained", fill="#334155")
    draw.text((15, start_y + cell_h + 70), "Actual Churned", fill="#334155")

    cm_path = artifacts_dir / "confusion_matrix.png"
    cm_img.save(cm_path)
    print(f"[Artifact Saved] Confusion matrix plot: {cm_path.resolve()}")

    # 2. ROC Curve Plot via Pillow
    roc_img = Image.new("RGB", (800, 600), "#ffffff")
    draw = ImageDraw.Draw(roc_img)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    draw.text((80, 25), "Receiver Operating Characteristic (ROC) Curve", fill="#0f172a")
    draw.text(
        (80, 50), f"Model: Random Forest Pipeline | ROC-AUC = {roc_auc:.4f}", fill="#2563eb"
    )

    # Plot boundaries
    plot_x0, plot_y0 = 100, 100
    plot_x1, plot_y1 = 720, 500

    # Axes
    draw.line([(plot_x0, plot_y1), (plot_x1, plot_y1)], fill="#475569", width=2)
    draw.line([(plot_x0, plot_y0), (plot_x0, plot_y1)], fill="#475569", width=2)

    # Gridlines and baseline
    draw.line([(plot_x0, plot_y1), (plot_x1, plot_y0)], fill="#94a3b8", width=2) # 45 deg chance
    draw.text((plot_x1 - 180, plot_y0 + 20), "-- Chance (AUC=0.50)", fill="#94a3b8")

    # Plot curve points
    curve_points = []
    for fp_val, tp_val in zip(fpr, tpr, strict=False):
        px = plot_x0 + fp_val * (plot_x1 - plot_x0)
        py = plot_y1 - tp_val * (plot_y1 - plot_y0)
        curve_points.append((px, py))

    draw.line(curve_points, fill="#2563eb", width=3)
    draw.text(
        (plot_x0 + 200, plot_y1 + 35), "False Positive Rate (1 - Specificity)", fill="#334155"
    )
    draw.text((20, plot_y0 + 190), "True\nPositive\nRate", fill="#334155")

    roc_path = artifacts_dir / "roc_curve.png"
    roc_img.save(roc_path)
    print(f"[Artifact Saved] ROC curve visualization: {roc_path.resolve()}")
