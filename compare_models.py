"""
Load test-set metrics from both models and produce a side-by-side
bar chart.  Must be run after both train_baseline.py and train_lstm.py.

Saves
-----
results/model_comparison.png

Usage
-----
    python compare_models.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR

METRIC_KEYS   = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc"]
METRIC_LABELS = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)",
                 "F1\n(macro)", "AUC-ROC"]


def load_metrics(name: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{name}_metrics.json")
    with open(path) as fh:
        return json.load(fh)


def main():
    baseline = load_metrics("baseline")
    lstm     = load_metrics("lstm")

    x     = np.arange(len(METRIC_KEYS))
    width = 0.33

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_b = ax.bar(x - width / 2,
                    [baseline[k] for k in METRIC_KEYS], width,
                    label="TF-IDF + LR", color="steelblue", edgecolor="white")
    bars_l = ax.bar(x + width / 2,
                    [lstm[k]     for k in METRIC_KEYS], width,
                    label="Bi-LSTM",     color="coral",    edgecolor="white")

    # value labels on top of each bar
    for bars in (bars_b, bars_l):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison on Test Set", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.80, 1.04)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # ── console summary ───────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  MODEL COMPARISON  –  Test Set")
    print("=" * 56)
    print(f"{'Metric':<22} {'TF-IDF + LR':>15} {'Bi-LSTM':>15}")
    print("-" * 56)
    for label, key in zip(METRIC_LABELS, METRIC_KEYS):
        clean = label.replace("\n", " ")
        print(f"{clean:<22} {baseline[key]:>15.4f} {lstm[key]:>15.4f}")
    print("=" * 56)

    winner = "TF-IDF + LR" if baseline["f1_macro"] >= lstm["f1_macro"] else "Bi-LSTM"
    print(f"\n  Best model (by macro F1): {winner}")


if __name__ == "__main__":
    main()
