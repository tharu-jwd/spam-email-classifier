"""
Advanced model: Bi-directional LSTM.

Architecture
    Embedding  →  BiLSTM  →  concat last hidden (fwd + bwd)
    →  Dropout  →  Linear(1)   [raw logit; sigmoid applied externally]

Class imbalance is handled via pos_weight in BCEWithLogitsLoss.
The checkpoint with the best validation F1 is kept; it is loaded
before final test evaluation.

Saves
-----
models/lstm_best.pt                – best state_dict
models/lstm_vocab.pkl              – Vocabulary object (needed at inference)
results/lstm_metrics.json
results/lstm_confusion_matrix.png
results/lstm_roc_curve.png
results/lstm_training_history.png

Usage
-----
    python train_lstm.py
"""
import json
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE, DATA_DIR, DROPOUT, EMBED_DIM, EPOCHS, HIDDEN_DIM,
    LSTM_LR, MAX_SEQ_LEN, MODELS_DIR, NUM_LAYERS, RESULTS_DIR, SEED,
    VOCAB_SIZE,
)
from evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from preprocess import preprocess_df

# ── reproducibility ───────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════
# Vocabulary
# ══════════════════════════════════════════════════════════════════
class Vocabulary:
    """Simple word→index mapping built exclusively from training data."""

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, max_size: int = VOCAB_SIZE):
        self.max_size  = max_size
        self.word2idx  = {"<PAD>": 0, "<UNK>": 1}

    # ----------------------------------------------------------
    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        for word, _ in counter.most_common(self.max_size - 2):
            self.word2idx[word] = len(self.word2idx)

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN):
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        ids += [self.PAD_IDX] * (max_len - len(ids))   # pad to fixed length
        return ids

    def __len__(self):
        return len(self.word2idx)


# ══════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════
class SpamDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts  = texts
        self.labels = labels
        self.vocab  = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx])
        return (
            torch.tensor(ids,             dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


# ══════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════
class SpamLSTM(nn.Module):
    """Embedding → BiLSTM → dropout → FC(1).  Output is a raw logit."""

    def __init__(self, vocab_size,
                 embed_dim  = EMBED_DIM,
                 hidden_dim = HIDDEN_DIM,
                 num_layers = NUM_LAYERS,
                 dropout    = DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, 1)   # *2 for bidirectional

    def forward(self, x):                          # x : (B, seq_len)
        emb      = self.dropout(self.embedding(x)) # (B, seq_len, E)
        _, (h, _) = self.lstm(emb)                 # h : (2*layers, B, H)
        # concatenate the last hidden state from each direction
        h = torch.cat((h[-2], h[-1]), dim=1)       # (B, 2*H)
        return self.fc(self.dropout(h)).squeeze(1) # (B,)


# ══════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss                        = 0.0
    all_preds, all_probs, all_labels  = [], [], []

    for X, y in loader:
        X, y   = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * X.size(0)

        probs = torch.sigmoid(logits)
        all_preds.extend((probs > 0.5).long().cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return (
        total_loss / len(loader.dataset),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels),
    )


# ══════════════════════════════════════════════════════════════════
# Training-history plot
# ══════════════════════════════════════════════════════════════════
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", markersize=4, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_f1"], "g-o", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 (macro)")
    axes[1].set_title("Validation F1 Score")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "lstm_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────
    print("Loading & preprocessing …")
    train_df = preprocess_df(pd.read_csv(os.path.join(DATA_DIR, "train.csv")))
    val_df   = preprocess_df(pd.read_csv(os.path.join(DATA_DIR, "validation.csv")))
    test_df  = preprocess_df(pd.read_csv(os.path.join(DATA_DIR, "test.csv")))

    # ── vocabulary (train only) ───────────────────────────────────
    print("Building vocabulary …")
    vocab = Vocabulary(max_size=VOCAB_SIZE)
    vocab.build(train_df["cleaned_text"].tolist())
    print(f"  Vocabulary size: {len(vocab)}")

    # ── dataloaders ───────────────────────────────────────────────
    def _ds(df):
        return SpamDataset(
            df["cleaned_text"].tolist(),
            df["label"].tolist(),
            vocab,
        )

    train_loader = DataLoader(_ds(train_df), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(_ds(val_df),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(_ds(test_df),  batch_size=BATCH_SIZE)

    # ── class-imbalance weight ────────────────────────────────────
    n_spam     = int(train_df["label"].sum())
    n_ham      = len(train_df) - n_spam
    pos_weight = torch.tensor([n_ham / n_spam], dtype=torch.float, device=device)
    print(f"  pos_weight = {pos_weight.item():.2f}  (ham / spam ratio)")

    # ── model / optimiser / loss ──────────────────────────────────
    model     = SpamLSTM(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── training loop ─────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = 0.0

    print(f"\nTraining for {EPOCHS} epochs …")
    for epoch in range(1, EPOCHS + 1):
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_preds, v_probs, v_labels = eval_epoch(
            model, val_loader, criterion, device
        )
        v_f1 = compute_metrics(v_labels, v_preds, v_probs)["f1_macro"]

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_f1"].append(v_f1)

        marker = ""
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(),
                       os.path.join(MODELS_DIR, "lstm_best.pt"))
            marker = "  <-- best"

        print(f"  Epoch {epoch:>2}/{EPOCHS}  "
              f"train_loss={t_loss:.4f}  "
              f"val_loss={v_loss:.4f}  "
              f"val_f1={v_f1:.4f}{marker}")

    # ── training curves ───────────────────────────────────────────
    plot_history(history)

    # ── load best checkpoint, evaluate on test ────────────────────
    print("\nLoading best checkpoint …")
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, "lstm_best.pt"),
        map_location=device,
        weights_only=True,
    ))
    _, test_preds, test_probs, test_labels = eval_epoch(
        model, test_loader, criterion, device
    )

    print("\n" + "=" * 50)
    print(" Test results")
    print("=" * 50)
    print(classification_report(test_labels, test_preds,
                                target_names=["Ham", "Spam"]))

    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    plot_confusion_matrix(test_labels, test_preds, "lstm")
    plot_roc_curve(test_labels, test_probs, "lstm")

    # ── persist ───────────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "lstm_vocab.pkl"), "wb") as fh:
        pickle.dump(vocab, fh)
    with open(os.path.join(RESULTS_DIR, "lstm_metrics.json"), "w") as fh:
        json.dump(test_metrics, fh, indent=2)

    print("\nLSTM model + artifacts saved.")


if __name__ == "__main__":
    main()
