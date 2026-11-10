"""data.py -- MNIST loading, preprocessing, and Sudoku-puzzle/mosaic generation.

Project 7 (Handwritten Sudoku Reader and Solver).

Real vs generated (per project spec):
- REAL: MNIST digit images and labels.
- GENERATED: Sudoku puzzles (full grid by randomised backtracking, then remove
  cells while solution stays unique); 9x9 image mosaics; the injected hard
  (low-confidence) digits that simulate an OCR misread.

All generated artefacts use a documented rule and a fixed seed.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import pickle
import struct
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# Resolve project paths relative to this file so tests work from any cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MNIST_IMAGE_SIZE = 28
MNIST_N_PIXELS = MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE

# Per spec, the NN target is digit 1-9 (digit 0 dropped). Digit 0 never appears
# in a Sudoku cell, so we specialise the classifier to 1..9.
NN_CLASSES = list(range(1, 10))  # [1, 2, ..., 9]

# Mirror URLs for MNIST. The original Yann LeCun page is intermittently
# unavailable, so we fall back through several public mirrors.
MNIST_URLS = {
    "train_images": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    ],
    "train_labels": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    ],
    "test_images": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    ],
    "test_labels": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ],
}


# ---------------------------------------------------------------------------
# MNIST download + parse (IDX format)
# ---------------------------------------------------------------------------

def _download_with_fallback(urls: list[str], dest_path: str) -> None:
    """Download `dest_path` from the first URL in `urls` that succeeds."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest_path, "wb") as f:
                f.write(r.read())
            return
        except Exception as e:  # noqa: BLE001 -- try the next mirror
            last_err = e
            continue
    raise RuntimeError(
        f"Could not download MNIST file from any mirror. Last error: {last_err}"
    )


def _idx_images(path_gz: str) -> np.ndarray:
    with gzip.open(path_gz, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic for images: {magic}"
        buf = f.read(num * rows * cols)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
        return arr


def _idx_labels(path_gz: str) -> np.ndarray:
    with gzip.open(path_gz, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic for labels: {magic}"
        return np.frombuffer(f.read(num), dtype=np.uint8)


def load_mnist(seed: int = 42) -> dict:
    """Load MNIST, drop digit 0 (per spec), scale pixels to 0..1, and split
    the official 60k train into 50k train / 10k validation.

    Returns a dict with keys: X_train, y_train, X_val, y_val, X_test, y_test.
    Each X is float32 in [0,1] of shape (N, 28, 28); each y is int64 in 1..9.

    The split is deterministic given `seed`.
    """
    files = {}
    for key, urls in MNIST_URLS.items():
        dest = os.path.join(DATA_DIR, os.path.basename(urls[0]))
        if not os.path.exists(dest):
            _download_with_fallback(urls, dest)
        files[key] = dest

    train_imgs = _idx_images(files["train_images"])
    train_lbls = _idx_labels(files["train_labels"])
    test_imgs = _idx_images(files["test_images"])
    test_lbls = _idx_labels(files["test_labels"])

    # Drop digit 0 (project spec: target digit 1-9).
    tr_mask = train_lbls != 0
    te_mask = test_lbls != 0
    X_tr_full = train_imgs[tr_mask].astype(np.float32) / 255.0
    y_tr_full = train_lbls[tr_mask].astype(np.int64)
    X_te = test_imgs[te_mask].astype(np.float32) / 255.0
    y_te = test_lbls[te_mask].astype(np.int64)

    # Deterministic 50k / 10k split of the (filtered) training set.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_tr_full))
    val_size = min(10_000, len(X_tr_full) // 6)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    return {
        "X_train": X_tr_full[tr_idx],
        "y_train": y_tr_full[tr_idx],
        "X_val": X_tr_full[val_idx],
        "y_val": y_tr_full[val_idx],
        "X_test": X_te,
        "y_test": y_te,
    }


# ---------------------------------------------------------------------------
# Sudoku puzzle generation
# ---------------------------------------------------------------------------

def _empty_grid() -> list[list[int]]:
    return [[0] * 9 for _ in range(9)]


def _valid_placement(grid: list[list[int]], row: int, col: int, num: int) -> bool:
    if num in grid[row]:
        return False
    if num in (grid[r][col] for r in range(9)):
        return False
    br, bc = (row // 3) * 3, (col // 3) * 3
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if grid[r][c] == num:
                return False
    return True


def _fill_grid_randomised(grid: list[list[int]], rng: np.random.Generator) -> bool:
    """Fill an empty grid by randomised backtracking -- produces a full valid
    Sudoku solution."""
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                nums = list(range(1, 10))
                rng.shuffle(nums)
                for num in nums:
                    if _valid_placement(grid, r, c, num):
                        grid[r][c] = num
                        if _fill_grid_randomised(grid, rng):
                            return True
                        grid[r][c] = 0
                return False
    return True


def _count_solutions(grid: list[list[int]], cap: int = 2) -> int:
    """Count solutions of `grid` up to `cap` (used to test uniqueness)."""
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                count = 0
                for num in range(1, 10):
                    if _valid_placement(grid, r, c, num):
                        grid[r][c] = num
                        count += _count_solutions(grid, cap - count)
                        grid[r][c] = 0
                        if count >= cap:
                            return count
                return count
    return 1  # all cells filled -> 1 solution


def generate_solution(rng: np.random.Generator) -> np.ndarray:
    """Generate one full valid 9x9 Sudoku solution."""
    grid = _empty_grid()
    _fill_grid_randomised(grid, rng)
    return np.array(grid, dtype=np.int8)


def generate_puzzle(
    rng: np.random.Generator,
    n_givens: int = 30,
    max_remove_attempts: int = 200,
    return_solution: bool = True,
):
    """Generate a Sudoku puzzle with exactly `n_givens` clues (or as close as
    the uniqueness constraint allows). The solution stays unique.

    Returns the puzzle (9x9 int8) by default, or (puzzle, solution) tuple if
    `return_solution=True`. Returning the solution alongside the puzzle is
    essential because the puzzle is generated *from* a randomly-built
    solution; calling `generate_solution()` separately would produce an
    unrelated grid.
    """
    sol = generate_solution(rng)
    puzzle = sol.copy()
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)

    removed = 0
    target_remove = 81 - n_givens
    for r, c in cells:
        if removed >= target_remove:
            break
        backup = int(puzzle[r, c])
        puzzle[r, c] = 0
        # Test uniqueness on a copy.
        test = puzzle.tolist()
        n = _count_solutions(test, cap=2)
        if n == 1:
            removed += 1
        else:
            puzzle[r, c] = backup  # keep clue, not unique

    if return_solution:
        return puzzle, sol
    return puzzle


# ---------------------------------------------------------------------------
# Mosaic generation: build a 9x9 image-mosaic of MNIST digits representing
# the puzzle (blank cells -> white image).
# ---------------------------------------------------------------------------

@dataclass
class Mosaic:
    """A 9x9 mosaic of MNIST digit images plus metadata.

    Attributes:
        puzzle:        9x9 int array of the puzzle (0 = blank).
        solution:      9x9 int array of the unique solution.
        digit_indices: 9x9 int array; for each cell, the index in the MNIST
                       image array that was placed there (-1 for blank).
        images:        list of (28,28) float arrays, indexed by
                       `digit_indices` (excluding -1).
        true_labels:   list of true digit labels (1-9) for each image.
        injected:      list of (r, c, original_digit, injected_digit)
                       describing injected hard digits (simulated misreads).
        givens_mask:   9x9 bool array, True where the cell is a given clue.
    """

    puzzle: np.ndarray
    solution: np.ndarray
    digit_indices: np.ndarray
    images: list
    true_labels: list
    injected: list
    givens_mask: np.ndarray

    def image_for_cell(self, r: int, c: int) -> Optional[np.ndarray]:
        idx = int(self.digit_indices[r, c])
        return None if idx < 0 else self.images[idx]


def build_mosaic(
    puzzle: np.ndarray,
    solution: np.ndarray,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    rng: np.random.Generator,
    n_injected_misreads: int = 0,
    givens_mask: Optional[np.ndarray] = None,
    hard_pool: Optional[dict] = None,
) -> Mosaic:
    """Build an image mosaic for `puzzle` using MNIST images drawn from
    `X_pool`/`y_pool`. Cells that are 0 (blank) get no image.

    If `n_injected_misreads > 0`, that many givens have their image replaced
    with a "hard" MNIST image -- an image whose true label matches the cell's
    correct digit, but which the trained NN naturally misclassifies. This
    implements the project spec's "inject hard (low-confidence) test digits"
    rule. The puzzle's stored value (and the solution) are unchanged: the
    image is genuinely of the correct digit, but the NN will misread it.

    `hard_pool` (optional) maps a true digit X to a list of dicts:
        {"image": img, "predicted": Y, "proba": (N,9) array, "img_idx": i}
    where Y != X (the NN's top-1 prediction is wrong). If `hard_pool` is
    None, injected misreads are simulated by swapping to an image of a
    random different digit (the original behaviour -- less realistic, kept
    for backwards compatibility with the tests).

    The `givens_mask` records the original clue positions so the solver
    pipeline knows which cells are "givens" vs. blanks.
    """
    if givens_mask is None:
        givens_mask = puzzle > 0

    images: list = []
    true_labels: list = []
    digit_indices = np.full((9, 9), -1, dtype=np.int32)
    injected: list = []

    # Index MNIST pool by label for fast lookup.
    by_label = {d: np.where(y_pool == d)[0] for d in range(1, 10)}

    # First pass: place the correct image in every given cell.
    puzzle_view = puzzle.copy()
    for r in range(9):
        for c in range(9):
            if puzzle_view[r, c] == 0:
                continue
            d = int(puzzle_view[r, c])
            pool_idx = int(rng.choice(by_label[d]))
            images.append(X_pool[pool_idx])
            true_labels.append(d)
            digit_indices[r, c] = len(images) - 1

    # Second pass: inject misreads by swapping in a HARD image of the SAME
    # digit (an image the NN naturally misclassifies). The puzzle's true
    # solution at this cell is unchanged.
    if n_injected_misreads > 0:
        given_cells = [(r, c) for r in range(9) for c in range(9) if givens_mask[r, c]]
        rng.shuffle(given_cells)
        for r, c in given_cells[:n_injected_misreads]:
            orig = int(puzzle_view[r, c])
            if hard_pool is not None and orig in hard_pool and len(hard_pool[orig]) > 0:
                # Pick a hard image of digit `orig` -- the NN will misread it.
                pick = int(rng.choice(len(hard_pool[orig])))
                entry = hard_pool[orig][pick]
                images.append(entry["image"])
                true_labels.append(orig)
                digit_indices[r, c] = len(images) - 1
                # puzzle_view[r, c] stays as orig (the true solution digit).
                # The NN will misread this image as entry["predicted"].
                injected.append((r, c, orig, int(entry["predicted"])))
            else:
                # Fallback (used when no hard pool is available): swap to a
                # random different digit. Less realistic but kept for the
                # unit tests that don't have a classifier available.
                alts = [d for d in range(1, 10) if d != orig]
                new_d = int(rng.choice(alts))
                pool_idx = int(rng.choice(by_label[new_d]))
                images.append(X_pool[pool_idx])
                true_labels.append(new_d)
                digit_indices[r, c] = len(images) - 1
                puzzle_view[r, c] = new_d
                injected.append((r, c, orig, new_d))

    return Mosaic(
        puzzle=puzzle_view,
        solution=solution,
        digit_indices=digit_indices,
        images=images,
        true_labels=true_labels,
        injected=injected,
        givens_mask=givens_mask,
    )


def build_hard_image_pool(classifier, X: np.ndarray, y: np.ndarray,
                           n_per_digit: int = 20) -> dict:
    """Scan `X`/`y` with `classifier` and build a pool of *hard* images per
    true digit -- images whose true label is d but the classifier's top-1
    prediction is some other digit. Used to inject realistic misreads into
    the puzzle mosaic.

    Returns a dict {digit: list of {"image", "predicted", "proba", "img_idx"}}.
    """
    # Predict in batches to keep memory bounded.
    pool: dict[int, list] = {d: [] for d in range(1, 10)}
    batch = 1024
    for start in range(0, len(X), batch):
        xb = X[start:start + batch]
        yb = y[start:start + batch]
        P = classifier.predict_proba(xb)
        preds = P.argmax(axis=1) + 1  # classes are 1..9
        for i in range(len(xb)):
            true = int(yb[i])
            pred = int(preds[i])
            if true == pred:
                continue
            if len(pool[true]) < n_per_digit * 3:
                pool[true].append({
                    "image": xb[i],
                    "predicted": pred,
                    "proba": P[i],
                    "img_idx": start + i,
                })
    # Trim to n_per_digit per digit.
    for d in pool:
        rng_local = np.random.default_rng(d)
        if len(pool[d]) > n_per_digit:
            idx = rng_local.choice(len(pool[d]), size=n_per_digit, replace=False)
            pool[d] = [pool[d][i] for i in idx]
    return pool


# ---------------------------------------------------------------------------
# Brightness rule for detecting blank cells (per spec: "Blank cells are
# detected by a simple brightness rule, not by the NN").
# ---------------------------------------------------------------------------

BLANK_BRIGHTNESS_THRESHOLD = 0.10  # mean pixel intensity below this => blank


def is_blank_image(img: np.ndarray, threshold: float = BLANK_BRIGHTNESS_THRESHOLD) -> bool:
    """Return True if `img` is detected as a blank cell.

    MNIST digits are centred strokes on a black background; the mean pixel
    intensity of a real digit is well above 0.10. Blank cells (a white image
    representing "no clue") are detected by mean intensity below threshold.
    NOTE: in this project we represent blank cells by *not placing an image*
    in the mosaic, so the rule here is used only as a sanity check / for the
    pipeline that consumes a photographed mosaic.
    """
    return float(img.mean()) < threshold


# ---------------------------------------------------------------------------
# Hash / verification helper (used by make_data.py)
# ---------------------------------------------------------------------------

def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Convenience save/load for the prepared dataset
# ---------------------------------------------------------------------------

PREPARED_PATH = os.path.join(DATA_DIR, "mnist_prepared.pkl")


def save_prepared(data: dict, path: str = PREPARED_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_prepared(path: str = PREPARED_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
