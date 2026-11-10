"""train.py -- Train the digit classifier (and the LR baseline).

Project 7.

Trains the MLP classifier on the prepared MNIST dataset, runs temperature
scaling on the validation set, and reports test accuracy, confusion matrix
and Expected Calibration Error (ECE) compared with logistic regression.

Run:
    python -m sudoku_solver.train
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow `python -m sudoku_solver.train` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sudoku_solver.data import MODELS_DIR, PREPARED_PATH, load_prepared
from sudoku_solver.nn import (DigitClassifier, LogisticRegressionBaseline,
                                _ece, accuracy_for, confusion_matrix_for)

MODEL_PATH = os.path.join(MODELS_DIR, "digit_classifier.pkl")
BASELINE_PATH = os.path.join(MODELS_DIR, "lr_baseline.pkl")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")


def _plot_confusion(cm: np.ndarray, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(9)); ax.set_xticklabels(range(1, 10))
    ax.set_yticks(range(9)); ax.set_yticklabels(range(1, 10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for r in range(9):
        for c in range(9):
            color = "white" if cm[r, c] > cm.max() * 0.6 else "black"
            ax.text(c, r, int(cm[r, c]), ha="center", va="center",
                    color=color, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_calibration(clf, X, y, title, path):
    """Reliability diagram: binned accuracy vs. confidence."""
    P = clf.predict_proba(X)
    pred = P.argmax(axis=1)
    conf = P.max(axis=1)
    targets = (y - 1).astype(np.int64)
    correct = (pred == targets).astype(np.float32)
    bins = np.linspace(0, 1, 11)
    accs, confs = [], []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            accs.append(np.nan); confs.append((lo + hi) / 2)
        else:
            accs.append(correct[mask].mean())
            confs.append((lo + hi) / 2)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(confs, accs, width=0.08, alpha=0.7, label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    print("[train] Loading prepared MNIST...")
    data = load_prepared(PREPARED_PATH)
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_te, y_te = data["X_test"], data["y_test"]

    print(f"[train] X_train={X_tr.shape}, X_val={X_val.shape}, X_test={X_te.shape}")

    # 1. Train the MLP.
    print("[train] Training MLP...")
    t0 = time.time()
    clf = DigitClassifier(
        hidden=args.hidden, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed,
    )
    clf.fit(X_tr, y_tr, X_val, y_val, verbose=True)
    train_time = time.time() - t0
    print(f"[train] MLP trained in {train_time:.1f}s")

    # 2. Calibrate on the validation set.
    print("[train] Calibrating (temperature scaling)...")
    cal = clf.calibrate(X_val, y_val)
    print(f"[train] ECE before scaling: {cal.ece_before:.4f}")
    print(f"[train] ECE after  scaling: {cal.ece_after:.4f} (T={cal.temperature:.3f})")

    # 3. Test accuracy.
    acc_mlp = clf.score(X_te, y_te)
    print(f"[train] MLP test accuracy: {acc_mlp:.4f}")

    # 4. Logistic regression baseline.
    print("[train] Training logistic regression baseline...")
    lr = LogisticRegressionBaseline(seed=args.seed)
    lr.fit(X_tr, y_tr)
    acc_lr = lr.score(X_te, y_te)
    print(f"[train] LR test accuracy: {acc_lr:.4f}")

    # 5. Confusion matrices and ECE for both.
    cm_mlp = confusion_matrix_for(clf, X_te, y_te)
    cm_lr = confusion_matrix_for(lr, X_te, y_te)
    # ECE: compute via logits/proba against (y-1).
    P_mlp = clf.predict_proba(X_te)
    P_lr = lr.predict_proba(X_te)
    ece_mlp = _ece(np.log(P_mlp + 1e-12), (y_te - 1).astype(np.int64))
    ece_lr = _ece(np.log(P_lr + 1e-12), (y_te - 1).astype(np.int64))
    print(f"[train] MLP ECE: {ece_mlp:.4f}")
    print(f"[train] LR  ECE: {ece_lr:.4f}")

    os.makedirs(FIG_DIR, exist_ok=True)
    _plot_confusion(cm_mlp, f"MLP confusion (acc={acc_mlp:.3f})",
                   os.path.join(FIG_DIR, "nn_confusion_mlp.png"))
    _plot_confusion(cm_lr, f"LR confusion (acc={acc_lr:.3f})",
                   os.path.join(FIG_DIR, "nn_confusion_lr.png"))
    _plot_calibration(clf, X_te, y_te,
                      f"MLP reliability (ECE={ece_mlp:.3f})",
                      os.path.join(FIG_DIR, "nn_reliability_mlp.png"))
    _plot_calibration(lr, X_te, y_te,
                      f"LR reliability (ECE={ece_lr:.3f})",
                      os.path.join(FIG_DIR, "nn_reliability_lr.png"))

    # 6. Save the model + a small JSON summary.
    clf.save(MODEL_PATH)
    print(f"[train] Saved model -> {MODEL_PATH}")
    import json
    summary = {
        "mlp_accuracy": acc_mlp, "lr_accuracy": acc_lr,
        "mlp_ece": ece_mlp, "lr_ece": ece_lr,
        "ece_before_scaling": cal.ece_before,
        "ece_after_scaling": cal.ece_after,
        "temperature": cal.temperature,
        "train_time_s": train_time,
        "epochs": args.epochs, "seed": args.seed,
    }
    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
