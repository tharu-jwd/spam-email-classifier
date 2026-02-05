"""
Baseline model: TF-IDF vectors + Logistic Regression.

Class imbalance is handled via class_weight='balanced'.

Saves
-----
models/baseline_tfidf.pkl          – fitted TfidfVectorizer
models/baseline_lr.pkl             – fitted LogisticRegression
results/baseline_metrics.json      – test-set metric dict
results/baseline_confusion_matrix.png
results/baseline_roc_curve.png

Usage
-----
    python train_baseline.py
"""
import json
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from config import (
    DATA_DIR, LR_C, LR_MAX_ITER, MODELS_DIR, RESULTS_DIR, SEED,
    TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE, TFIDF_SUBLINEAR_TF,
)
from evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from preprocess import preprocess_df


def main():
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── load & preprocess ─────────────────────────────────────────
    print("Loading data …")
    train = preprocess_df(pd.read_csv(os.path.join(DATA_DIR, "train.csv")))
    val   = preprocess_df(pd.read_csv(os.path.join(DATA_DIR, "validation.csv")))
    test  = preprocess_df(pd.read_csv(os.path.join(DATA_DIR, "test.csv")))

    # ── TF-IDF ────────────────────────────────────────────────────
    print("Fitting TF-IDF …")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        min_df=TFIDF_MIN_DF,
    )
    X_train = tfidf.fit_transform(train["cleaned_text"])
    X_val   = tfidf.transform(val["cleaned_text"])
    X_test  = tfidf.transform(test["cleaned_text"])

    y_train = train["label"].values
    y_val   = val["label"].values
    y_test  = test["label"].values

    # ── Logistic Regression ───────────────────────────────────────
    print("Training Logistic Regression …")
    model = LogisticRegression(
        C=LR_C,
        max_iter=LR_MAX_ITER,
        class_weight="balanced",   # ← handles class imbalance
        solver="lbfgs",
        random_state=SEED,
    )
    model.fit(X_train, y_train)

    # ── Evaluate on val then test ─────────────────────────────────
    for split_name, X, y in [("Validation", X_val, y_val),
                             ("Test",       X_test, y_test)]:
        preds = model.predict(X)
        print(f"\n{'='*50}")
        print(f" {split_name} results")
        print("=" * 50)
        print(classification_report(y, preds, target_names=["Ham", "Spam"]))

    # ── Final test plots & metrics ────────────────────────────────
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]   # P(spam)
    test_metrics = compute_metrics(y_test, test_preds, test_probs)

    plot_confusion_matrix(y_test, test_preds, "baseline")
    plot_roc_curve(y_test, test_probs, "baseline")

    # ── Persist ───────────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "baseline_tfidf.pkl"), "wb") as fh:
        pickle.dump(tfidf, fh)
    with open(os.path.join(MODELS_DIR, "baseline_lr.pkl"), "wb") as fh:
        pickle.dump(model, fh)
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as fh:
        json.dump(test_metrics, fh, indent=2)

    print("\nBaseline model + artifacts saved to models/ and results/")


if __name__ == "__main__":
    main()
