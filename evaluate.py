"""Shared evaluation helpers: metric dict + confusion-matrix / ROC plots."""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from config import RESULTS_DIR


# ── Metric dictionary ────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Return a flat JSON-serialisable dict of all reported metrics."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro")),
        "precision_spam":  float(precision_score(y_true, y_pred, pos_label=1)),
        "recall_spam":     float(recall_score(y_true, y_pred, pos_label=1)),
        "f1_spam":         float(f1_score(y_true, y_pred, pos_label=1)),
        "auc":             float(auc(fpr, tpr)),
    }


# ── Plots ────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """Save a labelled confusion-matrix heatmap."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        cbar=False, linewidths=1, linecolor="black",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_roc_curve(y_true, y_prob, model_name: str):
    """Save an ROC curve with AUC annotation."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve – {model_name}", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"{model_name}_roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
