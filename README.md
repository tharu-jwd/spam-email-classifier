# Spam Email Classifier

Binary classifier that labels messages as **ham** (legitimate) or
**spam**.  Two models are trained on the same data and compared head-to-head.

| Model | Description |
|---|---|
| **Baseline** | TF-IDF (uni + bigrams) → Logistic Regression |
| **Advanced** | Learned Embedding → Bi-directional LSTM |

---

## Project Layout

```
├── config.py              # All hyper-parameters and paths in one place
├── download_data.py       # Fetch dataset via kagglehub, split into CSVs
├── preprocess.py          # Shared text-cleaning pipeline
├── evaluate.py            # Metrics + confusion-matrix / ROC-curve plots
├── train_baseline.py      # TF-IDF + Logistic Regression (scikit-learn)
├── train_lstm.py          # Bi-LSTM (PyTorch)
├── compare_models.py      # Side-by-side bar chart of both models
├── predict.py             # Classify new emails from the command line
├── run_all.py             # Single command to run the full pipeline
├── sample_emails.txt      # Example emails for quick inference testing
├── requirements.txt
│
├── data/                  # Created at runtime – raw data + train/val/test CSVs
├── models/                # Saved model artefacts (.pkl, .pt)
└── results/               # PNG plots + JSON metrics
```

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (download → train → compare)
python run_all.py

# 4. Classify new emails
python predict.py --text "Congratulations! You have won a free iPhone!"
python predict.py --file sample_emails.txt
python predict.py --interactive   # REPL mode
```

> **Tip** – to skip the LSTM and only train the baseline (faster first run):
> `python run_all.py --skip-lstm`

---

## Dataset – Spam Emails (Kaggle)

| Property | Value |
|---|---|
| Source | [abdallahwagih/spam-emails](https://www.kaggle.com/datasets/abdallahwagih/spam-emails) via `kagglehub` |
| Size | 5 572 messages (after de-duplication) |
| Imbalance | ≈ 13 % spam / 87 % ham |
| Split | 70 / 15 / 15 — train / validation / test (stratified) |

### Handling class imbalance

| Model | Strategy |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Bi-LSTM | `pos_weight` in `BCEWithLogitsLoss` (= ham count / spam count) |

---

## Preprocessing Pipeline  (`preprocess.py`)

Applied to every split **before** any model sees the text:

1. **Lower-case** the entire message
2. **Replace** URLs → `url`, phone numbers → `phone`, currency amounts → `money`, remaining numbers → `num`
3. **Strip** all punctuation
4. **Tokenise** on whitespace
5. **Remove** English stop-words and single-character tokens
6. **Stem** every token with the Porter stemmer

---

## Models in Detail

### Baseline – TF-IDF + Logistic Regression  (`train_baseline.py`)

* **TfidfVectorizer** – up to 50 000 features, unigrams + bigrams, sub-linear TF scaling, minimum document frequency of 2.
* **LogisticRegression** – L-BFGS solver, `class_weight='balanced'`, regularisation `C = 1.0`.
* Serialised with `pickle` → `models/baseline_tfidf.pkl` + `models/baseline_lr.pkl`.

### Advanced – Bi-LSTM  (`train_lstm.py`)

| Component | Details |
|---|---|
| Embedding | 128-dim, learned, PAD token masked |
| BiLSTM | 1 layer, 128 hidden units per direction |
| Dropout | 0.3 – applied after embedding and before the FC layer |
| Output | Linear(256 → 1) raw logit; sigmoid gives P(spam) |

* Vocabulary: 30 000 most frequent tokens from **training set only**.
* Sequences padded / truncated to 200 tokens.
* Optimiser: Adam, lr = 0.001; gradient clipping at 1.0.
* Best checkpoint (highest validation macro-F1) is restored before the final test evaluation.
* Serialised with `torch.save` → `models/lstm_best.pt` + `models/lstm_vocab.pkl`.

---

## Results

> Run `python run_all.py` to reproduce.  All plots land in `results/`.

| Metric | TF-IDF + LR | Bi-LSTM |
|---|---|---|
| Accuracy | — | — |
| Precision (macro) | — | — |
| Recall (macro) | — | — |
| F1 (macro) | — | — |
| AUC-ROC | — | — |

*After training, read `results/baseline_metrics.json` and `results/lstm_metrics.json` for exact numbers, or open `results/model_comparison.png` for the chart.*

### Generated plots

| File | What it shows |
|---|---|
| `baseline_confusion_matrix.png` | Baseline confusion matrix |
| `baseline_roc_curve.png` | Baseline ROC curve |
| `lstm_confusion_matrix.png` | LSTM confusion matrix |
| `lstm_roc_curve.png` | LSTM ROC curve |
| `lstm_training_history.png` | LSTM loss & val-F1 across epochs |
| `model_comparison.png` | Side-by-side bar chart |

---

## Inference  (`predict.py`)

```bash
# single text
python predict.py --text "You owe taxes …"

# batch – one email per line
python predict.py --file sample_emails.txt

# interactive loop
python predict.py --interactive

# use only one model
python predict.py --text "…" --model baseline
python predict.py --text "…" --model lstm
```

---

## Configuration  (`config.py`)

Every tuneable value – split ratios, TF-IDF settings, LSTM architecture,
learning rates – lives in `config.py`.  Change a value there and re-run the
relevant training script; no other file needs editing.

---

## Requirements

See `requirements.txt`.  Tested with **Python 3.9+**.  A GPU is *not* required;
the LSTM trains comfortably on CPU with this dataset size.
