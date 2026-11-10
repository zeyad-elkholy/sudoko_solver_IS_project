"""nn.py -- Neural-network classifier for handwritten Sudoku digits (1-9).

Project 7.

Two classifiers are trained and compared:
- `DigitClassifier` -- a PyTorch MLP (3 hidden layers, dropout) with logits
  converted to calibrated probabilities via temperature scaling (optional).
- `LogisticRegressionBaseline` -- a multinomial logistic regression that the
  spec requires as a comparison baseline.

Both expose `predict_proba(images)` returning (N, 9) probabilities over the
digits 1..9 (in that order) and `predict(images)` returning argmax + the
probability vector.

Probability order convention: the columns of every probability vector are
always [P(1), P(2), ..., P(9)].
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional torch import -- fall back to sklearn MLPClassifier if torch is
# not available, so the project remains runnable in minimal environments.
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover -- exercised only when torch is missing
    _HAS_TORCH = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from .data import MODELS_DIR, NN_CLASSES


# ---------------------------------------------------------------------------
# PyTorch MLP
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class _MLP(nn.Module):
        def __init__(self, hidden=128, dropout=0.2):
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, 64)
            self.out = nn.Linear(64, 9)  # 9 classes: digits 1..9
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.drop(F.relu(self.fc1(x)))
            x = self.drop(F.relu(self.fc2(x)))
            x = F.relu(self.fc3(x))
            return self.out(x)


@dataclass
class CalibrationResult:
    temperature: float
    ece_before: float  # expected calibration error before scaling
    ece_after: float


class DigitClassifier:
    """Trained digit classifier with optional temperature scaling.

    Pickled metadata stores: weights, scaler params, temperature.
    """

    def __init__(self, hidden: int = 128, dropout: float = 0.2, lr: float = 1e-3,
                 epochs: int = 12, batch_size: int = 128, seed: int = 42,
                 device: str = "cpu"):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.scaler_mean_: Optional[np.ndarray] = None
        self.scaler_std_: Optional[np.ndarray] = None
        self.temperature: float = 1.0
        self.model_: Optional[object] = None  # nn.Module
        self._classes = NN_CLASSES  # [1..9]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        # Standardise using stored stats (or compute them on fit).
        if self.scaler_mean_ is None:
            self.scaler_mean_ = X.mean(axis=0, keepdims=True)
            self.scaler_std_ = X.std(axis=0, keepdims=True) + 1e-8
        return (X - self.scaler_mean_) / self.scaler_std_

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            verbose: bool = True) -> dict:
        if not _HAS_TORCH:
            return self._fit_sklearn(X_train, y_train, X_val, y_val, verbose)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_tr = self._preprocess(X_train)
        # Labels are 1..9; subtract 1 for cross-entropy.
        y_tr = (y_train - 1).astype(np.int64)
        Xt = torch.from_numpy(X_tr).float()
        yt = torch.from_numpy(y_tr).long()
        dl = DataLoader(TensorDataset(Xt, yt),
                        batch_size=self.batch_size, shuffle=True)

        self.model_ = _MLP(hidden=self.hidden, dropout=self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_acc": []}
        for ep in range(self.epochs):
            self.model_.train()
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                tot += loss.item() * xb.size(0)
            tr_loss = tot / len(Xt)
            history["train_loss"].append(tr_loss)

            if X_val is not None and y_val is not None:
                val_acc = self.score(X_val, y_val)
                history["val_acc"].append(val_acc)
                if verbose:
                    print(f"[nn] epoch {ep+1:2d}/{self.epochs}  "
                          f"train_loss={tr_loss:.4f}  val_acc={val_acc:.4f}")
            elif verbose:
                print(f"[nn] epoch {ep+1:2d}/{self.epochs}  train_loss={tr_loss:.4f}")
        return history

    def _fit_sklearn(self, X_train, y_train, X_val, y_val, verbose):
        """Fallback if torch is unavailable."""
        from sklearn.neural_network import MLPClassifier
        X_tr = self._preprocess(X_train)
        clf = MLPClassifier(hidden_layer_sizes=(128, 128, 64), alpha=1e-4,
                            max_iter=40, random_state=self.seed, verbose=verbose)
        clf.fit(X_tr, y_train)
        self._sk_model = clf
        return {"train_loss": [], "val_acc": []}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _logits(self, X: np.ndarray) -> np.ndarray:
        Xp = self._preprocess(X)
        if not _HAS_TORCH:
            # sklearn MLP returns proba directly; we won't go through logits.
            return self._sk_model.predict_proba(Xp)
        with torch.no_grad():
            Xt = torch.from_numpy(Xp).float().to(self.device)
            logits = self.model_(Xt).cpu().numpy()
        return logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 9) probability matrix over classes [1..9]."""
        if not _HAS_TORCH:
            Xp = self._preprocess(X)
            return self._sk_model.predict_proba(Xp)  # already proba over 1..9

        logits = self._logits(X) / max(self.temperature, 1e-8)
        # numerically stable softmax
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (predicted_digits (N,), max_probabilities (N,))."""
        P = self.predict_proba(X)
        idx = P.argmax(axis=1)
        digits = np.array([self._classes[i] for i in idx], dtype=np.int64)
        return digits, P.max(axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        digits, _ = self.predict(X)
        return float((digits == y).mean())

    # ------------------------------------------------------------------
    # Calibration: temperature scaling on a held-out set.
    # ------------------------------------------------------------------

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray,
                  n_steps: int = 50) -> CalibrationResult:
        """Find a temperature T minimising Expected Calibration Error (ECE)
        on a held-out set, then store it for inference."""
        logits = self._logits(X_val)  # (N, 9)
        targets = (y_val - 1).astype(np.int64)
        ece_before = _ece(logits / 1.0, targets)

        best_T, best_ece = 1.0, ece_before
        for T in np.linspace(0.1, 5.0, n_steps):
            e = _ece(logits / T, targets)
            if e < best_ece:
                best_ece, best_T = e, float(T)
        self.temperature = best_T
        return CalibrationResult(temperature=best_T,
                                  ece_before=ece_before,
                                  ece_after=best_ece)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "hidden": self.hidden, "dropout": self.dropout, "lr": self.lr,
            "epochs": self.epochs, "batch_size": self.batch_size, "seed": self.seed,
            "scaler_mean_": self.scaler_mean_, "scaler_std_": self.scaler_std_,
            "temperature": self.temperature,
            "classes": self._classes,
        }
        if _HAS_TORCH and self.model_ is not None:
            state["weights"] = self.model_.state_dict()
            state["backend"] = "torch"
        else:
            state["weights"] = self._sk_model
            state["backend"] = "sklearn"
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "DigitClassifier":
        with open(path, "rb") as f:
            state = pickle.load(f)
        clf = cls(hidden=state["hidden"], dropout=state["dropout"],
                  lr=state["lr"], epochs=state["epochs"],
                  batch_size=state["batch_size"], seed=state["seed"])
        clf.scaler_mean_ = state["scaler_mean_"]
        clf.scaler_std_ = state["scaler_std_"]
        clf.temperature = state["temperature"]
        clf._classes = state["classes"]
        if state["backend"] == "torch" and _HAS_TORCH:
            clf.model_ = _MLP(hidden=clf.hidden, dropout=clf.dropout)
            clf.model_.load_state_dict(state["weights"])
            clf.model_.eval()
        else:
            clf._sk_model = state["weights"]
        return clf


# ---------------------------------------------------------------------------
# Logistic regression baseline (per spec)
# ---------------------------------------------------------------------------

class LogisticRegressionBaseline:
    """Multinomial logistic regression baseline."""

    def __init__(self, max_iter: int = 200, C: float = 1.0, seed: int = 42):
        self.model = LogisticRegression(max_iter=max_iter, C=C,
                                         random_state=seed, n_jobs=-1)
        self.scaler = StandardScaler()
        self._classes = NN_CLASSES

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xf = X.reshape(X.shape[0], -1).astype(np.float32)
        self.scaler.fit(Xf)
        self.model.fit(self.scaler.transform(Xf), y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = X.reshape(X.shape[0], -1).astype(np.float32)
        return self.model.predict_proba(self.scaler.transform(Xf))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = self.predict_proba(X)
        idx = P.argmax(axis=1)
        digits = np.array([self._classes[i] for i in idx], dtype=np.int64)
        return digits, P.max(axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        d, _ = self.predict(X)
        return float((d == y).mean())


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _ece(probs_or_logits: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    `probs_or_logits` is (N, K). We softmax-normalise first so this works for
    both raw logits and already-normalised probabilities.
    """
    P = probs_or_logits.copy()
    P = P - P.max(axis=1, keepdims=True)
    P = np.exp(P)
    P = P / P.sum(axis=1, keepdims=True)
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == targets).astype(np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(P)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        c = conf[mask].mean()
        ece += (mask.sum() / N) * abs(acc - c)
    return float(ece)


def confusion_matrix_for(clf, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    digits, _ = clf.predict(X)
    # confusion_matrix expects 0-based labels; our classes are 1..9 so we
    # pass labels explicitly.
    return confusion_matrix(y, digits, labels=NN_CLASSES)


def accuracy_for(clf, X: np.ndarray, y: np.ndarray) -> float:
    return float(accuracy_score(y, clf.predict(X)[0]))
