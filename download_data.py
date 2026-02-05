"""
Download the Spam Emails dataset via kagglehub, deduplicate it,
and write a stratified 70 / 15 / 15 split into
data/train.csv, data/validation.csv, data/test.csv.

Usage:
    python download_data.py
"""
import os

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, SEED, TEST_SIZE, VAL_RATIO

KAGGLE_DATASET = "abdallahwagih/spam-emails"


def download_and_extract():
    """Pull the dataset via kagglehub; returns path to the downloaded files."""
    print(f"Downloading dataset '{KAGGLE_DATASET}' via kagglehub …")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Download complete → {path}")
    return path


def load_and_clean(dataset_path):
    """Read the CSV, encode labels (ham=0 / spam=1), drop dupes."""
    csv_path = os.path.join(dataset_path, "spam.csv")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Category": "label", "Message": "text"})
    df = df[df["label"].isin({"ham", "spam"})].reset_index(drop=True)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows → {len(df)} samples.")
    return df


def split_and_save(df):
    """Stratified 70 / 15 / 15 split → CSV files in data/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    train, temp = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=VAL_RATIO, stratify=temp["label"], random_state=SEED
    )

    for name, subset in [("train", train), ("validation", val), ("test", test)]:
        subset.to_csv(os.path.join(DATA_DIR, f"{name}.csv"), index=False)
        spam_pct = subset["label"].mean() * 100
        print(f"  {name:>12}: {len(subset):>5} samples  ({spam_pct:.1f} % spam)")


def main():
    dataset_path = download_and_extract()
    df = load_and_clean(dataset_path)
    print(f"\nDataset: {len(df)} samples | "
          f"spam = {df['label'].sum()} ({df['label'].mean()*100:.1f} %)")
    print("Splitting …")
    split_and_save(df)
    print("Done. Data files are in data/")


if __name__ == "__main__":
    main()
