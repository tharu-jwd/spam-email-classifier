"""
Classify one or more emails as ham / spam using the trained models.

Usage
-----
    # single text on the command line
    python predict.py --text "Congratulations! You have won a free prize!"

    # read texts from a file (one per line)
    python predict.py --file sample_emails.txt

    # interactive REPL
    python predict.py --interactive

    # choose which model(s) to use  (default: both)
    python predict.py --text "…" --model lstm
"""
import argparse
import os
import pickle
import sys

import numpy as np

from config import MODELS_DIR
from preprocess import clean_text

_LABEL = {0: "HAM", 1: "SPAM"}


# ══════════════════════════════════════════════════════════════════
# Loaders  –  lazy so we only pull in torch when the LSTM is used
# ══════════════════════════════════════════════════════════════════
def _load_baseline():
    tfidf_path = os.path.join(MODELS_DIR, "baseline_tfidf.pkl")
    lr_path    = os.path.join(MODELS_DIR, "baseline_lr.pkl")
    if not os.path.exists(tfidf_path) or not os.path.exists(lr_path):
        print("ERROR: Baseline model not found. Run 'python train_baseline.py' first.")
        sys.exit(1)
    with open(tfidf_path, "rb") as fh:
        tfidf = pickle.load(fh)
    with open(lr_path, "rb") as fh:
        model = pickle.load(fh)
    return tfidf, model


def _load_lstm():
    import torch                                    # noqa: F811
    from train_lstm import SpamLSTM, Vocabulary     # noqa: F811  (defines classes)

    vocab_path = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
    model_path = os.path.join(MODELS_DIR, "lstm_best.pt")
    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        print("ERROR: LSTM model not found. Run 'python train_lstm.py' first.")
        sys.exit(1)

    with open(vocab_path, "rb") as fh:
        vocab = pickle.load(fh)

    model = SpamLSTM(vocab_size=len(vocab))
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return vocab, model


# ══════════════════════════════════════════════════════════════════
# Prediction helpers
# ══════════════════════════════════════════════════════════════════
def predict_baseline(texts):
    tfidf, model = _load_baseline()
    cleaned = [clean_text(t) for t in texts]
    X       = tfidf.transform(cleaned)
    preds   = model.predict(X)
    probs   = model.predict_proba(X)[:, 1]   # P(spam)
    return preds, probs


def predict_lstm(texts):
    import torch                                    # noqa: F811
    vocab, model = _load_lstm()
    cleaned = [clean_text(t) for t in texts]
    ids     = torch.tensor([vocab.encode(t) for t in cleaned], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
        probs  = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    return preds, probs


# ══════════════════════════════════════════════════════════════════
# Pretty-print
# ══════════════════════════════════════════════════════════════════
def _print_results(model_name, texts, preds, probs):
    print(f"\n  [{model_name}]")
    for i, (text, pred, prob) in enumerate(zip(texts, preds, probs), 1):
        confidence = float(max(prob, 1 - prob))
        snippet    = text[:70] + (" …" if len(text) > 70 else "")
        print(f"    #{i}  {_LABEL[int(pred)]:>4}  ({confidence:.1%} confidence)")
        if len(texts) > 1:
            print(f"         {snippet}")


def classify(texts, model_choice: str = "both"):
    if model_choice in ("baseline", "both"):
        preds, probs = predict_baseline(texts)
        _print_results("TF-IDF + Logistic Regression", texts, preds, probs)
    if model_choice in ("lstm", "both"):
        preds, probs = predict_lstm(texts)
        _print_results("Bi-LSTM", texts, preds, probs)


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Spam / Ham Classifier")
    parser.add_argument("--text",        help="Email text to classify")
    parser.add_argument("--file",        help="Path to file (one email per line)")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter texts one at a time (type 'quit' to stop)")
    parser.add_argument("--model", default="both",
                        choices=["baseline", "lstm", "both"],
                        help="Which model(s) to use  (default: both)")
    args = parser.parse_args()

    if args.interactive:
        print("Spam Classifier  –  type 'quit' to exit")
        print("-" * 45)
        while True:
            raw = input("\nEmail text: ").strip()
            if raw.lower() in ("quit", "exit", "q"):
                break
            if raw:
                classify([raw], args.model)
    elif args.text:
        classify([args.text], args.model)
    elif args.file:
        with open(args.file) as fh:
            texts = [line.strip() for line in fh if line.strip()]
        if not texts:
            print("No texts found in file.")
            sys.exit(1)
        classify(texts, args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
