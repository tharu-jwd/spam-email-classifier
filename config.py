"""Central configuration – every tuneable value lives here."""
import os

# ── Directories ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42

# ── Data split  (train / val / test = 70 / 15 / 15) ─────────────
TEST_SIZE  = 0.30   # first split  → 30 % goes to temp
VAL_RATIO  = 0.50   # second split → 50 % of temp becomes val

# ── Preprocessing ────────────────────────────────────────────────
MAX_SEQ_LEN = 200   # max tokens kept for LSTM input

# ── Baseline: TF-IDF + Logistic Regression ───────────────────────
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE  = (1, 2)   # unigrams + bigrams
TFIDF_SUBLINEAR_TF = True
TFIDF_MIN_DF       = 2

LR_C        = 1.0
LR_MAX_ITER = 1000

# ── LSTM ──────────────────────────────────────────────────────────
VOCAB_SIZE  = 30_000
EMBED_DIM   = 128
HIDDEN_DIM  = 128
NUM_LAYERS  = 1
BATCH_SIZE  = 64
EPOCHS      = 15
LSTM_LR     = 0.001
DROPOUT     = 0.3
