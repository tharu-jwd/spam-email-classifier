"""
Text-preprocessing utilities shared by every model.

Pipeline applied by clean_text():
    1. Lower-case
    2. Replace URLs / phones / currency / numbers with tokens
    3. Strip punctuation
    4. Tokenise on whitespace
    5. Remove English stop-words and single-char tokens
    6. Porter-stem every remaining token

The resulting string is what TF-IDF sees and what the LSTM vocabulary
is built on.
"""
import re
import string

import nltk

# download required NLTK data silently (no-ops if already present)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords   # noqa: E402
from nltk.stem import PorterStemmer # noqa: E402

_STOP    = set(stopwords.words("english"))
_STEMMER = PorterStemmer()

# ── compiled regex patterns ──────────────────────────────────────
_URL_RE    = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PHONE_RE  = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_MONEY_RE  = re.compile(r"[$£€]\s?\d[\d,.]*")
_NUM_RE    = re.compile(r"\b\d+\b")
_PUNCT     = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Return a stemmed, stop-word-free string ready for featurisation."""
    text = text.lower()
    text = _URL_RE.sub("url", text)
    text = _PHONE_RE.sub("phone", text)
    text = _MONEY_RE.sub("money", text)
    text = _NUM_RE.sub("num", text)
    text = text.translate(_PUNCT)

    tokens = [
        _STEMMER.stem(tok)
        for tok in text.split()
        if tok not in _STOP and len(tok) > 1
    ]
    return " ".join(tokens)


def preprocess_df(df):
    """Return a copy of *df* with a new ``cleaned_text`` column."""
    df = df.copy()
    df["cleaned_text"] = df["text"].apply(clean_text)
    return df
