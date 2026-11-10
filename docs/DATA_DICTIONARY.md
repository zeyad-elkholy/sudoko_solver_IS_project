# Data Dictionary -- Project 7 (Handwritten Sudoku Reader and Solver)

This document describes every column / field in every data structure used by
the project, whether the data is **real** (from MNIST) or **generated**, and
(if generated) the exact rule and seed used to produce it.

## 1. MNIST dataset (REAL)

Downloaded by `make_data.py` from
`https://ossci-datasets.s3.amazonaws.com/mnist/` (mirror of the original
Yann LeCun page). Cached locally at `data/mnist_prepared.pkl`.

| Field       | Type           | Source    | Notes                                                                 |
|-------------|----------------|-----------|-----------------------------------------------------------------------|
| `X_train`   | float32 (N,28,28) | real     | 50,000 training images (after dropping digit 0), pixels scaled to [0,1] |
| `y_train`   | int64 (N,)     | real      | digit labels 1..9 (digit 0 dropped per spec)                          |
| `X_val`     | float32 (10000,28,28) | real | validation split (10k of the original 60k training set)               |
| `y_val`     | int64 (10000,)  | real      | validation labels                                                     |
| `X_test`    | float32 (9020,28,28) | real | the official 10k test set (digit-0 rows removed)                      |
| `y_test`    | int64 (9020,)   | real      | test labels                                                           |

*Real vs generated:* pixels and labels are real MNIST.
*Preprocessing:* pixels divided by 255 to [0,1]; digit-0 rows dropped
(Sudoku cells are 1..9 only); 50k/10k split is deterministic with `seed=42`
using `numpy.random.default_rng(42).permutation`.

## 2. Sudoku puzzle (GENERATED)

Produced by `sudoku_solver/data.py:generate_puzzle(rng, n_givens)`.

| Field      | Type        | Source     | Generation rule                                                |
|------------|-------------|------------|----------------------------------------------------------------|
| `puzzle`   | int8 (9,9)  | generated  | random full grid by randomised backtracking; cells removed while uniqueness holds |
| `solution` | int8 (9,9)  | generated  | the unique full solution to `puzzle`                           |

*Uniqueness check:* `_count_solutions(cap=2)` returns 1 for the puzzle to
qualify as valid (only puzzles with a unique solution are kept).
*Determinism:* `rng` is a `numpy.random.default_rng(seed)`; the same seed
produces the same puzzle every time.

## 3. Mosaic (GENERATED from REAL MNIST images)

Produced by `sudoku_solver/data.py:build_mosaic(...)`. Each given cell is
filled with a randomly chosen MNIST image of the correct digit; blank cells
get no image.

| Field            | Type                | Source              | Notes                                          |
|------------------|---------------------|---------------------|------------------------------------------------|
| `puzzle`         | int8 (9,9)          | generated (copy of input puzzle; misread cells unchanged when using `hard_pool`) | the puzzle as the NN sees it |
| `solution`       | int8 (9,9)          | generated           | the original unique solution                    |
| `digit_indices`  | int32 (9,9)         | generated           | for each cell, the index in `images` (-1 = blank) |
| `images`         | list of float32 (28,28) | real MNIST       | the actual MNIST images placed at given cells   |
| `true_labels`    | list of int         | real                | the true digit label of each image              |
| `injected`       | list of (r,c,orig,new) | generated       | records which cells were swapped with a "hard" image (orig = correct digit, new = NN's predicted digit) |
| `givens_mask`    | bool (9,9)          | generated           | True where the cell is a given clue            |

## 4. Hard-image pool (GENERATED index of REAL images)

Produced by `sudoku_solver/data.py:build_hard_image_pool(classifier, X, y,
n_per_digit=30)`. Scans the MNIST test set with the trained classifier and
keeps, per true digit, the first 30 images whose classifier top-1 prediction
is *wrong* (a different digit). Used to inject realistic misreads.

| Field        | Type                | Source | Notes                                                  |
|--------------|---------------------|--------|--------------------------------------------------------|
| `image`      | float32 (28,28)     | real   | a real MNIST image whose true label is `d`             |
| `predicted` | int                 | real   | the classifier's (wrong) top-1 prediction               |
| `proba`      | float32 (9,)        | real   | the classifier's full probability vector for this image |
| `img_idx`    | int                 | real   | the image's index in the MNIST test set                |

*Rule (per spec):* "inject hard (low-confidence) test digits using a
documented rule" -- the rule is "pick a real MNIST image whose true label
matches the cell's correct digit but whose trained-classifier top-1
prediction is some other digit". The image is genuinely of the correct
digit; the NN naturally misreads it.

## 5. Cell readings (GENERATED by the NN)

Produced by `sudoku_solver/integration.py:read_mosaic(classifier, mosaic,
top_k=3)`. The classifier is run on each given cell's image; the top-k
candidates are stored.

| Field         | Type                              | Source        | Notes                                            |
|---------------|-----------------------------------|---------------|--------------------------------------------------|
| `r`, `c`      | int                               | generated     | cell coordinates                                 |
| `candidates`  | tuple of (digit, prob) sorted desc | real (NN output) | the top-k digit candidates with their probabilities |
| `current_digit` | int (property)                  | real          | the NN's top-1 prediction (`candidates[0][0]`)   |
| `current_prob`  | float (property)                 | real          | the NN's top-1 probability                       |

## 6. Correction (GENERATED by the search)

Produced by `sudoku_solver/search.py`. A correction is a single
"change cell (r,c) from its current reading to a different digit".

| Field        | Type | Source                                          | Notes                                              |
|--------------|------|-------------------------------------------------|----------------------------------------------------|
| `r`, `c`     | int  | generated (chosen by the search)                | cell coordinates                                   |
| `new_digit`  | int  | generated (one of the top-k alternatives)        | the digit the cell is changed to                   |
| `cost`       | float | generated (= -log(p_new / p_current), never neg) | the drop in log-probability caused by the change  |

## 7. Experiment outputs (GENERATED)

CSV files in `results/`, plots in `figures/`. Each experiment records per-seed
and aggregate (mean ± std over 5 seeds) values. See `results/*.csv` for the
exact column lists per experiment.

## Reproducibility notes

* All randomness flows through `numpy.random.default_rng(seed)`.
* MNIST split seed = 42 (configurable via `make_data.py --seed`).
* Training seed = 42 (configurable via `python -m sudoku_solver.train --seed`).
* Experiments use seeds 1..5 (configurable via `eval.py`'s `seeds` argument).
