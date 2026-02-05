"""
Run the full pipeline end-to-end.

Steps
-----
    1. Download & split data
    2. Train baseline  (TF-IDF + LR)
    3. Train Bi-LSTM              (skip with --skip-lstm)
    4. Compare models             (only when both are trained)

Usage
-----
    python run_all.py                # full pipeline
    python run_all.py --skip-lstm    # baseline only (faster first run)
"""
import argparse
import subprocess
import sys

STEPS = [
    ("download_data.py",   "Step 1 – Download & split data"),
    ("train_baseline.py",  "Step 2 – Train baseline (TF-IDF + LR)"),
]

LSTM_STEPS = [
    ("train_lstm.py",      "Step 3 – Train Bi-LSTM"),
    ("compare_models.py",  "Step 4 – Compare models"),
]


def run(script: str, title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM training (baseline only)")
    args = parser.parse_args()

    steps = STEPS + ([] if args.skip_lstm else LSTM_STEPS)
    for script, title in steps:
        run(script, title)

    print("\n" + "=" * 60)
    print("  All done!  Plots and metrics are in results/")
    print("  Run  python predict.py --interactive  to classify emails")
    print("=" * 60)


if __name__ == "__main__":
    main()
