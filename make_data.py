"""make_data.py -- Download MNIST and prepare the train/val/test splits.

Project 7 (Handwritten Sudoku Reader and Solver).

Reproducible: a fixed seed is used. The prepared dataset is cached as
`data/mnist_prepared.pkl` so subsequent runs do not re-download.

Run:
    python make_data.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Allow `python make_data.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sudoku_solver.data import (DATA_DIR, PREPARED_PATH, file_sha1,
                                  load_mnist, save_prepared)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download + prepare MNIST.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the train/val split (default 42).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached.")
    args = parser.parse_args()

    if os.path.exists(PREPARED_PATH) and not args.force:
        print(f"[make_data] Already prepared: {PREPARED_PATH}")
        print(f"[make_data] Delete it or pass --force to rebuild.")
        return 0

    print(f"[make_data] Downloading MNIST (digit 0 will be dropped per spec)...")
    data = load_mnist(seed=args.seed)

    n_tr, n_val, n_te = (len(data["X_train"]), len(data["X_val"]),
                          len(data["X_test"]))
    print(f"[make_data] Train: {n_tr}  Val: {n_val}  Test: {n_te}")
    print(f"[make_data] Label balance (train):",
          {int(d): int((data["y_train"] == d).sum()) for d in range(1, 10)})

    save_prepared(data, PREPARED_PATH)
    sha = file_sha1(PREPARED_PATH)
    print(f"[make_data] Saved to {PREPARED_PATH}")
    print(f"[make_data] SHA1: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
