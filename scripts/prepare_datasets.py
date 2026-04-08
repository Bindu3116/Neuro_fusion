"""
prepare_datasets.py
====================
One-time data-preparation script for the NeuroFusionGPT project.

What this script does
---------------------
1. Merges ecg_train.csv + ecg_test.csv into one combined ECG dataset.
2. Splits ECG (combined) into three non-overlapping parts:
       60 % → ecg_train.csv       (used for model training)
       20 % → ecg_val.csv         (used for model validation)
       20 % → ecg_real.csv        (kept for real-world / professor demo)
3. Splits eeg_data.csv the same way:
       60 % → eeg_train.csv
       20 % → eeg_val.csv
       20 % → eeg_real.csv
4. Saves all six files into  datasets/split/

Usage
-----
    python scripts/prepare_datasets.py

    # Optionally change the base data directory:
    python scripts/prepare_datasets.py --data_dir /path/to/datasets

After running this script
-------------------------
- Upload datasets/split/ecg_train.csv, ecg_val.csv,
                         eeg_train.csv, eeg_val.csv
  to Google Drive for Colab training.
- Keep  datasets/split/ecg_real.csv  and  eeg_real.csv  for the demo.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ECG_FEATURES = 187          # number of feature columns in ECG files
ECG_LABEL_COL = ECG_FEATURES  # 0-indexed → last column (index 187)
RANDOM_STATE = 42

VAL_RATIO  = 0.20   # 20 % of the whole dataset  → validation
REAL_RATIO = 0.20   # 20 % of the whole dataset  → real-world demo

# train ends up being the rest: 100 - 20 - 20 = 60 %


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def split_dataset(X, y, val_ratio=VAL_RATIO, real_ratio=REAL_RATIO,
                  random_state=RANDOM_STATE):
    """
    Three-way stratified split → train / val / real.

    Parameters
    ----------
    X : ndarray, shape (N, F)
    y : ndarray, shape (N,)

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_real, y_real)
    """
    n = len(y)

    # ---- Step 1: carve off real-world data (20 % of whole) ----
    real_size = real_ratio          # fraction of the whole dataset
    X_dev, X_real, y_dev, y_real = train_test_split(
        X, y,
        test_size=real_size,
        random_state=random_state,
        stratify=y,
    )

    # ---- Step 2: split remaining 80 % into 60 % train / 20 % val ----
    # val_ratio relative to the *whole* dataset → relative to dev set:
    val_of_dev = val_ratio / (1.0 - real_size)   # 0.20 / 0.80 = 0.25
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev,
        test_size=val_of_dev,
        random_state=random_state,
        stratify=y_dev,
    )

    # Sanity check
    total = len(y_train) + len(y_val) + len(y_real)
    assert total == n, f"Row count mismatch: {total} != {n}"

    train_pct = len(y_train) / n * 100
    val_pct   = len(y_val)   / n * 100
    real_pct  = len(y_real)  / n * 100
    print(f"  Total {n:,} rows → "
          f"train {len(y_train):,} ({train_pct:.1f} %)  "
          f"val {len(y_val):,} ({val_pct:.1f} %)  "
          f"real {len(y_real):,} ({real_pct:.1f} %)")

    return (X_train, y_train), (X_val, y_val), (X_real, y_real)


def save_ecg(X, y, path):
    """Save ECG split: no header, features + label in last column."""
    arr = np.column_stack([X, y])
    df  = pd.DataFrame(arr)
    df.to_csv(path, header=False, index=False)
    print(f"  Saved {len(df):,} rows → {path}")


def save_eeg(X, y, feat_cols, path):
    """Save EEG split: with header, feature columns + 'y' column."""
    df = pd.DataFrame(X, columns=feat_cols)
    df["y"] = y.astype(int)
    df.to_csv(path, header=True, index=False)
    print(f"  Saved {len(df):,} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare 60/20/20 dataset splits.")
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets"),
        help="Directory containing ecg_train.csv, ecg_test.csv, eeg_data.csv",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir  = os.path.join(data_dir, "split")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # ECG
    # ------------------------------------------------------------------
    print("\n[1/2] Processing ECG data …")

    ecg_train_path = os.path.join(data_dir, "ecg_train.csv")
    ecg_test_path  = os.path.join(data_dir, "ecg_test.csv")

    for p in (ecg_train_path, ecg_test_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: file not found: {p}")

    print("  Loading ecg_train.csv …")
    ecg_tr = pd.read_csv(ecg_train_path, header=None)
    print("  Loading ecg_test.csv …")
    ecg_te = pd.read_csv(ecg_test_path,  header=None)

    ecg_all = pd.concat([ecg_tr, ecg_te], ignore_index=True)
    print(f"  Merged ECG: {len(ecg_all):,} rows × {ecg_all.shape[1]} cols")

    X_ecg = ecg_all.iloc[:, :ECG_FEATURES].values.astype(np.float32)
    y_ecg = ecg_all.iloc[:, ECG_LABEL_COL].values.astype(int)
    print(f"  ECG label distribution: { {k: int((y_ecg==k).sum()) for k in sorted(set(y_ecg))} }")

    (X_ecg_tr, y_ecg_tr), (X_ecg_val, y_ecg_val), (X_ecg_real, y_ecg_real) = \
        split_dataset(X_ecg, y_ecg)

    save_ecg(X_ecg_tr,   y_ecg_tr,   os.path.join(out_dir, "ecg_train.csv"))
    save_ecg(X_ecg_val,  y_ecg_val,  os.path.join(out_dir, "ecg_val.csv"))
    save_ecg(X_ecg_real, y_ecg_real, os.path.join(out_dir, "ecg_real.csv"))

    # ------------------------------------------------------------------
    # EEG
    # ------------------------------------------------------------------
    print("\n[2/2] Processing EEG data …")

    eeg_path = os.path.join(data_dir, "eeg_data.csv")
    if not os.path.exists(eeg_path):
        sys.exit(f"ERROR: file not found: {eeg_path}")

    print("  Loading eeg_data.csv …")
    eeg_all  = pd.read_csv(eeg_path)
    feat_cols = [c for c in eeg_all.columns if c != "y"]
    X_eeg    = eeg_all[feat_cols].values.astype(np.float32)
    y_eeg    = eeg_all["y"].values.astype(int)
    print(f"  EEG rows: {len(eeg_all):,}   features: {len(feat_cols)}")
    print(f"  EEG label distribution: { {k: int((y_eeg==k).sum()) for k in sorted(set(y_eeg))} }")

    (X_eeg_tr, y_eeg_tr), (X_eeg_val, y_eeg_val), (X_eeg_real, y_eeg_real) = \
        split_dataset(X_eeg, y_eeg)

    save_eeg(X_eeg_tr,   y_eeg_tr,   feat_cols, os.path.join(out_dir, "eeg_train.csv"))
    save_eeg(X_eeg_val,  y_eeg_val,  feat_cols, os.path.join(out_dir, "eeg_val.csv"))
    save_eeg(X_eeg_real, y_eeg_real, feat_cols, os.path.join(out_dir, "eeg_real.csv"))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n✅  Done!  Files saved to:", out_dir)
    print(
        "\nNext steps:"
        "\n  1. Upload the following files to Google Drive  datasets/bindu/:"
        "\n       ecg_train.csv   ecg_val.csv"
        "\n       eeg_train.csv   eeg_val.csv"
        "\n  2. Run the updated bindu_training.ipynb notebook."
        "\n  3. Keep ecg_real.csv and eeg_real.csv for the professor demo."
    )


if __name__ == "__main__":
    main()
